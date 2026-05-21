from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import os


def _debug_print(*args, **kwargs):
    if os.environ.get("ECHO_VERBOSE", "0") == "1":
        print(*args, **kwargs)

flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def rope_cut(freqs, start_frame, f, transition_frames=3, transition_offset=45):
    """
    Apply rope cut for scene transitions.
    
    Args:
        freqs: Frequency tensor, shape [1024, C / num_heads / 2]
        start_frame: Starting frame index (integer)
        f: Number of frames (integer)
        transition_frames: Latent block size (integer, default=3)
        transition_offset: Temporal offset added to the tail frames.
        
    Returns:
        temporal_freqs: Concatenated temporal frequencies
    """
    prefix_len = max(0, f - transition_frames)
    jump_len = min(f, transition_frames)
    suffix_start = start_frame + prefix_len + int(transition_offset)
    max_start = max(0, freqs[0].shape[0] - jump_len)
    suffix_start = max(0, min(suffix_start, max_start))

    starting_group = freqs[0][start_frame:start_frame + prefix_len]
    final_group = freqs[0][suffix_start:suffix_start + jump_len]
    if prefix_len == 0:
        return final_group
    return torch.cat([starting_group, final_group], dim=0)


def causal_rope_apply(
    x,
    grid_sizes,
    freqs,
    start_frame=0,
    scene_cut=False,
    rope_jump_value=None,
    rope_jump_frames=0,
):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # @hidir: becomes 4680 x 12 x 32
        
        # ⭐
        if scene_cut or (rope_jump_value is not None and rope_jump_frames > 0):
            temporal_freqs = rope_cut(
                freqs,
                start_frame,
                f,
                transition_frames=rope_jump_frames if rope_jump_frames > 0 else 3,
                transition_offset=rope_jump_value if rope_jump_value is not None else 45,
            )
        else:
            temporal_freqs = freqs[0][start_frame:start_frame + f]

        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1) # @hidir: becomes 4680 x 1 x 64

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
        
    result = torch.stack(output).type_as(x)
    return result

def causal_rope_apply_t_only(
    x,
    grid_sizes,
    freqs,
    start_frame=0,
    scene_cut=False,
    rope_jump_value=None,
    rope_jump_frames=0,
):
    """
    Apply RoPE only on the temporal axis.

    Args:
        x: [B, seq_len, num_heads, head_dim]
        grid_sizes: [B, 3], each item is (f, h, w)
        freqs: Precomputed complex RoPE frequencies
        start_frame: Temporal start frame
        scene_cut: Whether to apply the scene-transition temporal jump

    Returns:
        Tensor with the same shape as x.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )  # [seq_len, n, c]

        if scene_cut or (rope_jump_value is not None and rope_jump_frames > 0):
            temporal_freqs = rope_cut(
                freqs,
                start_frame,
                f,
                transition_frames=rope_jump_frames if rope_jump_frames > 0 else 3,
                transition_offset=rope_jump_value if rope_jump_value is not None else 45,
            )
        else:
            temporal_freqs = freqs[0][start_frame:start_frame + f]

        height_freqs = freqs[1].new_ones((h, freqs[1].shape[1]))
        width_freqs = freqs[2].new_ones((w, freqs[2].shape[1]))
        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            height_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
            width_freqs.view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)
        output.append(x_i)

    result = torch.stack(output).type_as(x)
    return result

def causal_rope_apply_hw_only(x, grid_sizes, freqs):
    """
    Apply RoPE only on height and width axes.

    x: [B, seq, num_heads, head_dim]
    grid_sizes: [B, 3], each item is (f, h, w)
    freqs: Precomputed complex RoPE frequencies
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )

        temporal_freqs = freqs[0].new_ones((f, freqs[0].shape[1]))

        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)
        output.append(x_i)
    result = torch.stack(output).type_as(x)
    return result

def _to_complex_pairs(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Head dim must be even, got {x.shape[-1]}")
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())

def _build_geometric_offsets(max_len: int, device: torch.device) -> torch.Tensor:
    if max_len < 1:
        return torch.tensor([1.0], device=device, dtype=torch.float32)
    offsets = []
    value = 1
    while value <= max_len:
        offsets.append(float(value))
        value *= 2
    if len(offsets) == 0:
        offsets.append(1.0)
    return torch.tensor(offsets, device=device, dtype=torch.float32)

def _mkv_update_history(kv_cache, new_k, new_v, new_abs, new_spatial, R):
    new_k = new_k.detach()
    new_v = new_v.detach()
    new_abs = new_abs.detach()
    if "history_k" not in kv_cache or kv_cache["history_k"] is None:
        kv_cache["history_k"] = new_k
        kv_cache["history_v"] = new_v
        kv_cache["history_abs_frame_idx"] = new_abs
        kv_cache["history_spatial_idx"] = new_spatial

        kv_cache["history_topc_select_counts"] = torch.zeros(
            (new_k.shape[0], new_k.shape[1]), dtype=torch.long, device=new_k.device
        )
    else:
        kv_cache["history_k"] = torch.cat([kv_cache["history_k"], new_k], dim=1)
        kv_cache["history_v"] = torch.cat([kv_cache["history_v"], new_v], dim=1)
        kv_cache["history_abs_frame_idx"] = torch.cat([kv_cache["history_abs_frame_idx"], new_abs], dim=1)
        kv_cache["history_spatial_idx"] = torch.cat([kv_cache["history_spatial_idx"], new_spatial], dim=1)
        kv_cache["history_topc_select_counts"] = torch.cat([
            kv_cache["history_topc_select_counts"],
            torch.zeros((new_k.shape[0], new_k.shape[1]), dtype=torch.long, device=new_k.device)
        ], dim=1)

    if kv_cache["history_k"].shape[1] > R:
        trim = kv_cache["history_k"].shape[1] - R
        kv_cache["history_k"] = kv_cache["history_k"][:, trim:]
        kv_cache["history_v"] = kv_cache["history_v"][:, trim:]
        kv_cache["history_abs_frame_idx"] = kv_cache["history_abs_frame_idx"][:, trim:]
        kv_cache["history_spatial_idx"] = kv_cache["history_spatial_idx"][:, trim:]
        kv_cache["history_topc_select_counts"] = kv_cache["history_topc_select_counts"][:, trim:]


class PCConfig:
    def __init__(self,
                 enable=True,
                 history_capacity = 1560 * 18,
                 window=1560 * 3,
                 fusion="sum",
                 keep_sinks=True,
                 topc_max_reuse = 8,
                 max_atten_size = 21* 1560,
                 sink_frames = 12,
                 rolling_cycle = 4,
                 compressed_frames = 3,
                 recent_frames = 3,
                 use_amplitude_compensation=True,
                 use_drift_gate=True,
        drift_gate_lambda=3.0):
        """
        Fixed-quota PC configuration:
        1. The sink segment size is self.sink_size * frame_seqlen.
        2. The compressed middle segment keeps compressed_frames frames.
        3. The recent segment keeps recent_frames frames.
        """
        self.enable = bool(enable)
        self.history_capacity = int(history_capacity)
        self.window = int(window)
        self.fusion = fusion
        self.keep_sinks = bool(keep_sinks)

        self.max_atten_size = max_atten_size

        self.rolling_cycle = min(int(self.max_atten_size / 3), rolling_cycle)
        self.topc_max_reuse = max(0, int(topc_max_reuse))
        self.sink_frames =  max(0, int(sink_frames))

        self.compressed_frames = max(0, int(compressed_frames))
        self.recent_frames = max(1, int(recent_frames))
        self.use_amplitude_compensation = bool(use_amplitude_compensation)
        self.use_drift_gate = bool(use_drift_gate)
        self.drift_gate_lambda = max(0.0, float(drift_gate_lambda))


class CausalWanSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 block_id=-1,
                 qk_norm=True,
                 eps=1e-6,
                 PC: PCConfig | None = None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.PC = PC or PCConfig(enable = True)

        self.local_attn_size = local_attn_size
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.sink_size = sink_size

        self.block_id = block_id
        self.qk_norm = qk_norm
        self.eps = eps

        if self.block_id == 0:
            _debug_print("\n---- Attention cache init ----")
            _debug_print(f"Sink frames: {self.PC.sink_frames}, compressed frames: {self.PC.compressed_frames}, recent frames: {self.PC.recent_frames}")
            _debug_print(f"Amplitude compensation: {self.PC.use_amplitude_compensation}, drift gate: {self.PC.use_drift_gate}, gate lambda: {self.PC.drift_gate_lambda}")
            _debug_print(f"Max KV capacity: {self.PC.max_atten_size}")
      
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


    def _ensure_cache_buffers(self, kv_cache):

        dtype = kv_cache["k"].dtype
        device = kv_cache["k"].device
        if "history_k" not in kv_cache or kv_cache["history_k"] is None:
            kv_cache["history_k"] = None
        if "history_v" not in kv_cache or kv_cache["history_v"] is None:
            kv_cache["history_v"] = None
        if "history_abs_frame_idx" not in kv_cache or kv_cache["history_abs_frame_idx"] is None:
            kv_cache["history_abs_frame_idx"] = None
        if "history_spatial_idx" not in kv_cache or kv_cache["history_spatial_idx"] is None:
            kv_cache["history_spatial_idx"] = None
        if "history_topc_select_counts" not in kv_cache or kv_cache["history_topc_select_counts"] is None:
            kv_cache["history_topc_select_counts"] = None

        if "k_original" not in kv_cache or kv_cache["k_original"] is None:
            kv_cache["k_original"] = torch.zeros_like(kv_cache["k"])
        if "v_original" not in kv_cache or kv_cache["v_original"] is None: 
            kv_cache["v_original"] = torch.zeros_like(kv_cache["v"])

        if "ori_start_ptr" not in kv_cache or kv_cache["ori_start_ptr"] is None:
            kv_cache["ori_start_ptr"] = 0
        if "ori_write_ptr" not in kv_cache or kv_cache["ori_write_ptr"] is None:
            kv_cache["ori_write_ptr"] = None

        if "k_raw" not in kv_cache or kv_cache["k_raw"] is None:
            kv_cache["k_raw"] = torch.zeros_like(kv_cache["k"])

        if "abs_frame_idx" not in kv_cache or kv_cache["abs_frame_idx"] is None:
            kv_cache["abs_frame_idx"] = torch.full(
                (kv_cache["k"].shape[0], kv_cache["k"].shape[1]),
                -1,
                dtype=torch.long,
                device=kv_cache["k"].device,
            )

        if "spatial_idx" not in kv_cache or kv_cache["spatial_idx"] is None:
            kv_cache["spatial_idx"] = torch.full(
                (kv_cache["k"].shape[0], kv_cache["k"].shape[1]),
                -1,
                dtype=torch.long,
                device=kv_cache["k"].device,
            )

        calib_shape = (self.num_heads, self.head_dim // 2)
        if "q_calib_sum" not in kv_cache or kv_cache["q_calib_sum"] is None:
            kv_cache["q_calib_sum"] = torch.zeros(calib_shape, dtype=torch.complex64, device=device)
        if "q_calib_abs_sum" not in kv_cache or kv_cache["q_calib_abs_sum"] is None:
            kv_cache["q_calib_abs_sum"] = torch.zeros(calib_shape, dtype=torch.float32, device=device)
        if "q_calib_mean" not in kv_cache or kv_cache["q_calib_mean"] is None:
            kv_cache["q_calib_mean"] = torch.zeros(calib_shape, dtype=torch.complex64, device=device)
        if "q_calib_abs_mean" not in kv_cache or kv_cache["q_calib_abs_mean"] is None:
            kv_cache["q_calib_abs_mean"] = torch.zeros(calib_shape, dtype=torch.float32, device=device)
        if "q_calib_token_count" not in kv_cache or kv_cache["q_calib_token_count"] is None:
            kv_cache["q_calib_token_count"] = torch.zeros((), dtype=torch.float64, device=device)
        if "q_calib_ready" not in kv_cache or kv_cache["q_calib_ready"] is None:
            kv_cache["q_calib_ready"] = torch.tensor(False, dtype=torch.bool, device=device)


    def _ensure_score_metadata(self, device):
        freq_dim = self.head_dim // 2
        temporal_count = freq_dim - 2 * (freq_dim // 3)
        spatial_count = freq_dim // 3

        if temporal_count <= 0 or spatial_count < 0:
            raise ValueError(
                f"Invalid frequency split: freq_dim={freq_dim}, temporal={temporal_count}, spatial={spatial_count}"
            )
        if hasattr(self, "omega") and self.omega is not None and self.omega.device == device:
            return
        
        def _axis_omega(count: int) -> torch.Tensor:
            if count <= 0:
                return torch.empty(0, device=device, dtype=torch.float32)
            idx = torch.arange(count, device=device, dtype=torch.float32)
            return 1.0 / torch.pow(10000.0, idx / float(count))

        omega_t = _axis_omega(temporal_count)
        omega_h = _axis_omega(spatial_count)
        omega_w = _axis_omega(spatial_count)

        self.omega = torch.cat([omega_t, omega_h, omega_w], dim=0)

        self.temporal_mask = torch.zeros(freq_dim, device=device, dtype=torch.float32)
        self.temporal_mask[:temporal_count] = 1.0

        self.freq_scale_sq = torch.ones(freq_dim, device=device, dtype=torch.float32)
        self.offsets = _build_geometric_offsets(self.max_attention_size // 1560, device=device)

    def _update_q_calibration(self, kv_cache, q, current_end):
        if kv_cache["q_calib_ready"].item():
            return
        if current_end > self.max_attention_size:
            return

        q_complex = _to_complex_pairs(q.detach().to(torch.float32))
        kv_cache["q_calib_sum"] += q_complex.sum(dim=(0, 1)).to(torch.complex64)
        kv_cache["q_calib_abs_sum"] += q_complex.abs().sum(dim=(0, 1)).to(torch.float32)
        kv_cache["q_calib_token_count"] += float(q.shape[0] * q.shape[1])
        if self.block_id == 0:
            _debug_print(f"Q calibration tokens={int(kv_cache['q_calib_token_count'].item())}")

        if current_end >= self.max_attention_size:
            denom = kv_cache["q_calib_token_count"].clamp_min(1.0).to(torch.float32)
            kv_cache["q_calib_mean"] = kv_cache["q_calib_sum"] / denom
            kv_cache["q_calib_abs_mean"] = kv_cache["q_calib_abs_sum"] / denom
            kv_cache["q_calib_ready"].fill_(True)
            if self.block_id == 0:
                _debug_print(f"Q calibration ready: tokens={int(kv_cache['q_calib_token_count'].item())}")


    def _apply_relative_rope(self, q, k_raw, v, grid_sizes, freqs, current_end_frame ,local_end_index, num_new_frames, frame_seqlen, max_attention_frames,scene_cut):
        window_len_tokens = min(local_end_index, self.max_attention_size)
        window_len_frames = max(1, window_len_tokens // frame_seqlen)
        
        q_start_frame = max(0, window_len_frames - num_new_frames)
        
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame = q_start_frame,scene_cut = scene_cut).type_as(v)
        
        if self.block_id == 0:
            _debug_print(f"Q temporal RoPE: {q_start_frame}->{q_start_frame + grid_sizes[0][0]}")

        grid_sizes_full = grid_sizes.clone()

        if  current_end_frame <= max_attention_frames:
            grid_sizes_full[0][0] = min(window_len_frames, max_attention_frames)
            roped_key = causal_rope_apply(k_raw, grid_sizes_full, freqs, start_frame = 0,scene_cut = scene_cut).type_as(v)
            if self.block_id == 0:
                _debug_print(f"Full temporal RoPE: 0->{grid_sizes_full[0][0]}")
        else:
            grid_sizes_sink =  grid_sizes.clone()
            grid_sizes_comp =  grid_sizes.clone()
            grid_sizes_other = grid_sizes.clone()

            grid_sizes_sink[0][0] = min( window_len_frames,self.PC.sink_frames)
            grid_sizes_comp[0][0] = min( window_len_frames,self.PC.compressed_frames)
            other_frames = self.PC.recent_frames + 3 
            grid_sizes_other[0][0] = min( window_len_frames,other_frames)

            k_raw_sink = k_raw[:, 0: self.PC.sink_frames * frame_seqlen]
            k_raw_comp = k_raw[:, self.PC.sink_frames * frame_seqlen : (self.PC.sink_frames + self.PC.compressed_frames) * frame_seqlen]
            k_raw_other = k_raw[:, (self.PC.sink_frames + self.PC.compressed_frames) * frame_seqlen:
                                (self.PC.sink_frames + self.PC.compressed_frames + other_frames) * frame_seqlen]
            
            roped_key_sink = causal_rope_apply(
                k_raw_sink,
                grid_sizes_sink,         
                freqs,
                start_frame=0,
                scene_cut=scene_cut
            ).type_as(v)
            
            if self.PC.compressed_frames > 0:
                roped_key_comp = causal_rope_apply_t_only(
                    k_raw_comp,
                    grid_sizes_comp,         
                    freqs,
                    start_frame=self.PC.sink_frames,
                    scene_cut=scene_cut
                ).type_as(v)
          
            roped_key_other = causal_rope_apply(
                k_raw_other,
                grid_sizes_other,
                freqs,
                start_frame = self.PC.sink_frames + self.PC.compressed_frames,
                scene_cut=scene_cut
            ).type_as(v)

            if self.block_id == 0:
                _debug_print(f"Sink temporal RoPE: 0->{grid_sizes_sink[0][0]}")
                if self.PC.compressed_frames  > 0:
                    _debug_print(f"Compressed temporal RoPE: {self.PC.sink_frames}->{self.PC.sink_frames + grid_sizes_comp[0][0]}")
                _debug_print(f"Other temporal RoPE: {self.PC.sink_frames + self.PC.compressed_frames}->{self.PC.sink_frames + self.PC.compressed_frames + grid_sizes_other[0][0]}")

            if self.PC.compressed_frames > 0:
                roped_key = torch.cat([roped_key_sink, roped_key_comp, roped_key_other], dim=1)
            else:
                roped_key = torch.cat([roped_key_sink, roped_key_other], dim=1)

        return roped_query, roped_key, q_start_frame


    def _compute_pc_scores(self, kv_cache, grid_sizes, freqs, frame_seqlen,h_mode = 'k_pre', q_recent=None):
        
        history_k = None
        history_abs = None
        if h_mode == 'generate': 
            history_k = kv_cache.get("history_k", None)
            history_abs = kv_cache.get("history_abs_frame_idx", None)
        elif  h_mode == 'original': 
            history_k = kv_cache.get("k_original", None)
            history_abs = kv_cache.get("abs_frame_idx", None)
        elif h_mode == 'k_pre':
            history_k = kv_cache.get("k_raw", None)
            history_abs = kv_cache.get("abs_frame_idx", None)
        

        q_calib_ready = kv_cache.get("q_calib_ready", None)
        q_mean = kv_cache.get("q_calib_mean", None)
        q_abs_mean = kv_cache.get("q_calib_abs_mean", None)

        if history_k is None or history_abs is None or q_calib_ready is None or (not q_calib_ready.item()) or q_mean is None or q_abs_mean is None:
            return None, 0

        history_len = history_k.shape[1]
        if history_len <= 0:
            return None, 0

        self._ensure_score_metadata(history_k.device)

        fixed_recent_tokens = self.PC.recent_frames * frame_seqlen
        recent_tokens = min(fixed_recent_tokens, history_len)
        if recent_tokens <= 0:
            return None, 0

        history_k_complex = _to_complex_pairs(history_k.to(torch.float32))[0]

        q_mean = q_mean.to(device=history_k.device, dtype=torch.complex64)
        q_abs_mean = q_abs_mean.to(device=history_k.device, dtype=torch.float32)
        q_mean_abs = torch.abs(q_mean)
        k_abs = torch.abs(history_k_complex)

        relative = q_mean.unsqueeze(0) * torch.conj(history_k_complex)

        phi = torch.atan2(relative.imag, relative.real)
        
        amp = q_mean_abs.unsqueeze(0) * k_abs

        if self.PC.use_amplitude_compensation:
            extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
        else:
            extra = torch.zeros_like(k_abs)

        if self.PC.use_amplitude_compensation and self.PC.use_drift_gate and q_recent is not None:
            q_recent_complex = _to_complex_pairs(q_recent.detach().to(torch.float32))
            q_recent_mean = q_recent_complex.mean(dim=(0, 1)).to(
                device=history_k.device,
                dtype=torch.complex64,
            )

            recent_vec = torch.view_as_real(q_recent_mean).flatten()
            calib_vec = torch.view_as_real(q_mean).flatten()

            drift_similarity = torch.nn.functional.cosine_similarity(
                recent_vec,
                calib_vec,
                dim=0,
                eps=1e-6,
            ).clamp(-1.0, 1.0)

            drift_gate = torch.exp(-self.PC.drift_gate_lambda * (1.0 - drift_similarity))
            extra = extra * drift_gate

        key_frame = history_abs[0, :history_len].to(device=history_k.device, dtype=torch.float32)
        delta_t = (
            float(history_abs[0, history_len - 1].item() + 1) - key_frame
        ).unsqueeze(1) + self.offsets.unsqueeze(0)

        phase = (
            delta_t.unsqueeze(1).unsqueeze(-1)
            * self.temporal_mask.view(1, 1, 1, -1)
            * self.omega.view(1, 1, 1, -1)
            + phi.unsqueeze(2)
        )
        base_scores = (
            amp.unsqueeze(2)
            * self.freq_scale_sq.view(1, 1, 1, -1)
            * torch.cos(phase)
        ).sum(dim=-1)

        additive = (
            extra * self.freq_scale_sq.view(1, 1, -1)
        ).sum(dim=-1, keepdim=True)
        
        combined = base_scores + additive

        if self.PC.fusion == "max":
            token_scores = combined.max(dim=2).values.max(dim=1).values
        else:
            token_scores = combined.mean(dim=2).mean(dim=1)

        fused = token_scores.unsqueeze(0).to(torch.float32)
        fused[:, max(0, history_len - recent_tokens):] = -float("inf")
        # fused [1, history_len]
        return fused, recent_tokens


    def _select_history_middle(self, kv_cache, fused_history, frame_seqlen):

        history_len = fused_history.shape[1]
        fixed_compressed_tokens = self.PC.compressed_frames * frame_seqlen
        candidate_len = history_len

        top_c = min(fixed_compressed_tokens, candidate_len)
        self.top_c = top_c

        if top_c <= 0:
            return torch.tensor([], device=fused_history.device, dtype=torch.long)
        
        history_counts = kv_cache.get("history_topc_select_counts", None)
        candidate_idx = torch.arange(0, history_len, device=fused_history.device)
        scores_b = fused_history[0]

        if history_counts is not None and self.PC.topc_max_reuse > 0:
            counts_b = history_counts[0]
            valid_mask = counts_b < self.PC.topc_max_reuse
            if not torch.any(valid_mask):
                return torch.tensor([], device=fused_history.device, dtype=torch.long)
            
            allowed_scores = scores_b[valid_mask]
            allowed_idx = candidate_idx[valid_mask]
            k_eff = min(top_c, allowed_idx.numel())
            if k_eff <= 0:
                return torch.tensor([], device=fused_history.device, dtype=torch.long)
            _, top_local = torch.topk(allowed_scores, k=k_eff, dim=0)
            selected = torch.sort(allowed_idx[top_local])[0]
        else:
            k_eff = min(top_c, candidate_idx.numel())
            _, top_local = torch.topk(scores_b, k=k_eff, dim=0)
            selected = torch.sort(candidate_idx[top_local])[0]
        
        return selected


    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        timestep,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            self._ensure_cache_buffers(kv_cache)

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            cache_start_frame = (cache_start // frame_seqlen) if cache_start is not None else 0
            scene_local_start_frame = max(0, current_start_frame - cache_start_frame)
            use_scene_local_timing = bool(kv_cache.get("use_scene_local_timing", False))
            rope_position_mode = str(kv_cache.get("rope_position_mode", "local")).lower()
            scene_timing_offset_frames = int(
                kv_cache.get("scene_timing_offset_frames", torch.tensor([0], device=q.device)).item()
            )

            rope_jump_active = bool(kv_cache.get("rope_jump_active", False))
            rope_jump_value = int(
                kv_cache.get("rope_jump_value", torch.tensor([0], device=q.device)).item()
            ) if rope_jump_active else None
            rope_jump_frames = int(
                kv_cache.get("rope_jump_frames", torch.tensor([0], device=q.device)).item()
            ) if rope_jump_active else 0
            rope_jump_enabled = (
                rope_jump_active
                and rope_jump_value is not None
                and rope_jump_frames > 0
            )

            num_new_tokens = q.shape[1]
            num_new_frames = num_new_tokens // frame_seqlen

            if bool(kv_cache.get("collect_q_stats", False)):
                q_stats = q.detach().to(torch.float32)
                kv_cache["q_sum"] += q_stats.sum(dim=(0, 1))
                kv_cache["q_count"] += float(q_stats.shape[0] * q_stats.shape[1])

            if bool(kv_cache.get("record_scene_candidates", False)):
                candidate_start = int(kv_cache["candidate_token_count"].item())
                candidate_capacity = int(kv_cache["candidate_k"].shape[0])
                candidate_tokens = min(num_new_tokens * q.shape[0], candidate_capacity - candidate_start)
                if candidate_tokens > 0:
                    candidate_end = candidate_start + candidate_tokens
                    flat_k = k.detach().reshape(-1, k.shape[-2], k.shape[-1])
                    flat_v = v.detach().reshape(-1, v.shape[-2], v.shape[-1])
                    kv_cache["candidate_k"][candidate_start:candidate_end] = flat_k[:candidate_tokens].to(
                        kv_cache["candidate_k"].dtype
                    )
                    kv_cache["candidate_v"][candidate_start:candidate_end] = flat_v[:candidate_tokens].to(
                        kv_cache["candidate_v"].dtype
                    )
                    kv_cache["candidate_token_count"] = torch.tensor(
                        [candidate_end], dtype=torch.long, device=kv_cache["candidate_token_count"].device
                    )

            current_end = current_start + num_new_tokens
            current_end_frame = current_end // frame_seqlen
            if use_scene_local_timing:
                pc_start_frame = scene_local_start_frame + scene_timing_offset_frames
            else:
                pc_start_frame = current_start_frame
            pc_end_frame = pc_start_frame + num_new_frames
            pc_end_tokens = pc_end_frame * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            
            max_attention_frames = self.max_attention_size // frame_seqlen

            sink_frames = self.PC.sink_frames
            sink_tokens = sink_frames * frame_seqlen
            compress_frames = self.PC.compressed_frames
            recent_frames = self.PC.recent_frames
          
            if timestep == 1000 and self.block_id == 0:
                print(f"Frame {int(current_start_frame)} -> {int(current_end_frame)}")

   
         
            new_abs = torch.arange(
                current_start_frame,
                current_start_frame + num_new_frames,
                device=k.device,
                dtype=torch.long,
            ).repeat_interleave(frame_seqlen).unsqueeze(0)
            new_spatial = torch.arange(
                frame_seqlen,
                device=k.device,
                dtype=torch.long,
            ).repeat(num_new_frames).unsqueeze(0)

            history_capacity = max(self.PC.history_capacity, self.max_attention_size) if self.PC.enable else self.max_attention_size
            
            # if current_end > kv_cache["global_end_index"].item():
            if timestep == 0 and self.PC.enable:
                self._update_q_calibration(kv_cache, q, pc_end_tokens)
                _mkv_update_history(kv_cache, k, v, new_abs, new_spatial, R=history_capacity)
   
            #----------------------------------------
            rolled_condition = self.local_attn_size != -1  and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache['local_end_index'] > kv_cache_size)
            available_slots = kv_cache_size - kv_cache['local_end_index']

            if self.block_id == 0:
            # if timestep == 1000 and self.block_id == 0:
                _debug_print(f"Rolling check: rolled_condition={rolled_condition}, PC.enable={self.PC.enable}, available_slots={available_slots}")

            if rolled_condition and self.PC.enable:  
               
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = max(0, sink_tokens - num_evicted_tokens)

                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                if num_rolled_tokens > 0:
                    src_slice = slice(num_evicted_tokens, num_evicted_tokens + num_rolled_tokens)
                    dst_slice = slice(0, num_rolled_tokens)
                    kv_cache["k"][:, dst_slice] = kv_cache["k"][:, src_slice].clone()
                    kv_cache["v"][:, dst_slice] = kv_cache["v"][:, src_slice].clone()
                    kv_cache["abs_frame_idx"][:, dst_slice] = kv_cache["abs_frame_idx"][:, src_slice].clone()
                    kv_cache["spatial_idx"][:, dst_slice] = kv_cache["spatial_idx"][:, src_slice].clone()
                    if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                        kv_cache["local_token_weights"][dst_slice] = kv_cache["local_token_weights"][src_slice].clone()
                        kv_cache["local_token_decay_mask"][dst_slice] = kv_cache["local_token_decay_mask"][src_slice].clone()
                        kv_cache["local_token_decay_rates"][dst_slice] = kv_cache["local_token_decay_rates"][src_slice].clone()
                
                current_start_frame_RS = None
                cycle = int( sink_frames / 3 )
                rolling_cycle = self.PC.rolling_cycle
                
                if self.block_id == 0:
                # if timestep == 1000 and self.block_id == 0:
                    _debug_print("---- Prepare rolling sink ----")
                    _debug_print(f"Sink shift: source {int(num_evicted_tokens/1560)} - {int((num_evicted_tokens + num_rolled_tokens)/1560)} to target 0 - {int(num_rolled_tokens/1560)}")
                    _debug_print(f"Evicted frames: {int(num_new_tokens/1560)}, shifted sink frames: {int(num_rolled_tokens/1560)}")
                    _debug_print(f"❤️ local_start_index:{int(local_start_index / 1560)},\t -> local_end_index:{int(local_end_index / 1560)}")
                
                if pc_end_frame - 3 >= max_attention_frames:
                    current_start_frame_RS = pc_start_frame - (max_attention_frames - rolling_cycle * 3)
                    reverse_window_size = rolling_cycle * 3
                    reverse = (current_start_frame_RS // reverse_window_size) % 2 == 1
                    if reverse:
                        right = reverse_window_size - current_start_frame_RS % reverse_window_size 
                        left = right - 3
                    else:
                        left = current_start_frame_RS % reverse_window_size
                        right = left + 3
                    if right <= max_attention_frames:
                        re_k_cache = kv_cache["k_original"][:, left * frame_seqlen:right * frame_seqlen].clone()
                        re_v_cache = kv_cache["v_original"][:, left * frame_seqlen:right * frame_seqlen].clone()
                        if reverse:
                            re_k_cache = re_k_cache.flip(dims=[1])
                            re_v_cache = re_v_cache.flip(dims=[1])                 
                            for re_block_idx in range(3):
                                re_k_cache[:, re_block_idx*frame_seqlen:(re_block_idx + 1)*frame_seqlen, :, :] = \
                                    re_k_cache[:, re_block_idx*frame_seqlen:(re_block_idx + 1)*frame_seqlen, :, :].flip(dims=[1])
                                re_v_cache[:, re_block_idx*frame_seqlen:(re_block_idx + 1)*frame_seqlen, :, :] = \
                                    re_v_cache[:, re_block_idx*frame_seqlen:(re_block_idx + 1)*frame_seqlen, :, :].flip(dims=[1])
                            
                        insert_left = (cycle - 1) * 3 * frame_seqlen
                        insert_right = cycle * 3 * frame_seqlen
                        
                        if insert_right <= kv_cache_size:
                            kv_cache["k"][:, insert_left:insert_right] = re_k_cache
                            kv_cache["v"][:, insert_left:insert_right] = re_v_cache
                            if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                                kv_cache["local_token_weights"][insert_left:insert_right] = 1.0
                                kv_cache["local_token_decay_mask"][insert_left:insert_right] = False
                                kv_cache["local_token_decay_rates"][insert_left:insert_right] = 1.0
    

                    if self.block_id == 0:
                    # if timestep ==1000 and self.block_id == 0:
                        _debug_print("---- Start rolling sink ----")
                        _debug_print(f"❤️ current_start_frame_RS:{int(current_start_frame_RS)}")
                        _debug_print(f"Rolling window: {left} -> {right}")
                        _debug_print(f"Insert window: {int(insert_left/1560)} -> {int(insert_right/1560)}")

                
                num_tail_evicted_tokens = num_new_tokens
                num_tail_rolled_tokens = max(0,recent_frames * frame_seqlen)
                if num_tail_rolled_tokens > 0:
                    tail_start =  sink_tokens + compress_frames * frame_seqlen
                    src_slice = slice(tail_start + num_tail_evicted_tokens , tail_start + num_tail_evicted_tokens + num_tail_rolled_tokens)
                    dst_slice = slice(tail_start,  tail_start + num_tail_rolled_tokens)

                    if self.block_id == 0:
                    # if timestep ==1000 and self.block_id == 0:
                        _debug_print(f"Tail shift source: {int((tail_start + num_tail_evicted_tokens)/1560)} -> {int((tail_start + num_tail_evicted_tokens + num_tail_rolled_tokens)/1560)}")
                        _debug_print(f"Tail shift target: {int(tail_start/1560)} -> {int((tail_start + num_tail_rolled_tokens)/1560)}")
                    kv_cache["k"][:, dst_slice] = kv_cache["k"][:, src_slice].clone()
                    kv_cache["v"][:, dst_slice] = kv_cache["v"][:, src_slice].clone()
                    kv_cache["abs_frame_idx"][:, dst_slice] = kv_cache["abs_frame_idx"][:, src_slice].clone()
                    kv_cache["spatial_idx"][:, dst_slice] = kv_cache["spatial_idx"][:, src_slice].clone()
                    if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                        kv_cache["local_token_weights"][dst_slice] = kv_cache["local_token_weights"][src_slice].clone()
                        kv_cache["local_token_decay_mask"][dst_slice] = kv_cache["local_token_decay_mask"][src_slice].clone()
                        kv_cache["local_token_decay_rates"][dst_slice] = kv_cache["local_token_decay_rates"][src_slice].clone()


                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
                kv_cache["abs_frame_idx"][:, local_start_index:local_end_index] = new_abs
                kv_cache["spatial_idx"][:, local_start_index:local_end_index] = new_spatial
                if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                    kv_cache["local_token_weights"][local_start_index:local_end_index] = 1.0
                    kv_cache["local_token_decay_mask"][local_start_index:local_end_index] = False
                    kv_cache["local_token_decay_rates"][local_start_index:local_end_index] = 1.0

                if self.block_id == 0:
                    # if timestep ==1000 and self.block_id == 0:
                    _debug_print(f"Insert latest frames: {int(local_start_index/1560)} -> {int(local_end_index/1560)}")

                if compress_frames > 0:
                    fused_history, R_eff = self._compute_pc_scores(
                        kv_cache, grid_sizes, freqs, frame_seqlen, h_mode="generate", q_recent=q
                    )
                    if fused_history is not None and R_eff > 0:
                        middle_idx = self._select_history_middle(kv_cache, fused_history, frame_seqlen)

                        history_k = kv_cache["history_k"]
                        history_v = kv_cache["history_v"]
                        history_abs = kv_cache["history_abs_frame_idx"]
                        history_spatial = kv_cache["history_spatial_idx"]

                        if middle_idx.numel() > 0:
                            gather_middle = middle_idx.view(1, -1, 1, 1).expand(
                                1, middle_idx.numel(), history_k.shape[2], history_k.shape[3]
                            )
                            middle_k = torch.gather(history_k, dim=1, index=gather_middle)
                            middle_v = torch.gather(history_v, dim=1, index=gather_middle)
                            middle_abs = history_abs[:, middle_idx]
                            middle_spatial = history_spatial[:, middle_idx]

                            if self.PC.topc_max_reuse > 0:
                                kv_cache["history_topc_select_counts"][:, middle_idx] += 1
                        else:
                            middle_k = history_k[:, :0]
                            middle_v = history_v[:, :0]
                            middle_abs = history_abs[:, :0]
                            middle_spatial = history_spatial[:, :0]

                        middle_len = middle_k.shape[1]
                        middle_start = sink_tokens
                        middle_end = sink_tokens + middle_len
                        kv_cache["k"][:, middle_start:middle_end] = middle_k
                        kv_cache["v"][:, middle_start:middle_end] = middle_v
                        kv_cache["abs_frame_idx"][:, middle_start:middle_end] = middle_abs
                        kv_cache["spatial_idx"][:, middle_start:middle_end] = middle_spatial
                        if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                            kv_cache["local_token_weights"][middle_start:middle_end] = 1.0
                            kv_cache["local_token_decay_mask"][middle_start:middle_end] = False
                            kv_cache["local_token_decay_rates"][middle_start:middle_end] = 1.0

                        if self.block_id == 0:
                            _debug_print(f"Fill compressed memory: length={int(middle_len/1560)}, range={int(middle_start/1560)} -> {int(middle_end/1560)}")

            elif rolled_condition:
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                fallback_sink_tokens = int(
                    kv_cache.get(
                        "sink_token_count",
                        torch.tensor([self.sink_size * frame_seqlen], device=q.device),
                    ).item()
                )
                fallback_sink_tokens = min(fallback_sink_tokens, kv_cache["local_end_index"].item())
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - fallback_sink_tokens
                num_rolled_tokens = max(0, num_rolled_tokens)

                if num_rolled_tokens > 0:
                    src_slice = slice(
                        fallback_sink_tokens + num_evicted_tokens,
                        fallback_sink_tokens + num_evicted_tokens + num_rolled_tokens,
                    )
                    dst_slice = slice(fallback_sink_tokens, fallback_sink_tokens + num_rolled_tokens)
                    kv_cache["k"][:, dst_slice] = kv_cache["k"][:, src_slice].clone()
                    kv_cache["v"][:, dst_slice] = kv_cache["v"][:, src_slice].clone()
                    kv_cache["abs_frame_idx"][:, dst_slice] = kv_cache["abs_frame_idx"][:, src_slice].clone()
                    kv_cache["spatial_idx"][:, dst_slice] = kv_cache["spatial_idx"][:, src_slice].clone()
                    if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                        kv_cache["local_token_weights"][dst_slice] = kv_cache["local_token_weights"][src_slice].clone()
                        kv_cache["local_token_decay_mask"][dst_slice] = kv_cache["local_token_decay_mask"][src_slice].clone()
                        kv_cache["local_token_decay_rates"][dst_slice] = kv_cache["local_token_decay_rates"][src_slice].clone()

                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
                kv_cache["abs_frame_idx"][:, local_start_index:local_end_index] = new_abs
                kv_cache["spatial_idx"][:, local_start_index:local_end_index] = new_spatial
                if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                    kv_cache["local_token_weights"][local_start_index:local_end_index] = 1.0
                    kv_cache["local_token_decay_mask"][local_start_index:local_end_index] = False
                    kv_cache["local_token_decay_rates"][local_start_index:local_end_index] = 1.0

                if self.block_id == 0:
                    _debug_print(
                        f"PC disabled: sink={int(fallback_sink_tokens/frame_seqlen)}, "
                        f"roll={int(num_rolled_tokens/frame_seqlen)}, "
                        f"insert:{int(local_start_index/frame_seqlen)} -> {int(local_end_index/frame_seqlen)}"
                    )

            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                if timestep == 1000 and self.block_id == 0:
                    _debug_print(f"KV space available: {int(local_start_index/1560)} -> {int(local_end_index/1560)}")
          
                  
                elif  timestep !=1000 and self.block_id == 0:
                    _debug_print(f"Recompute timestep: {int(local_start_index/1560)} -> {int(local_end_index/1560)}")
      
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
                kv_cache["abs_frame_idx"][:, local_start_index:local_end_index] = new_abs
                kv_cache["spatial_idx"][:, local_start_index:local_end_index] = new_spatial
                if torch.is_tensor(kv_cache.get("local_token_weights", None)):
                    kv_cache["local_token_weights"][local_start_index:local_end_index] = 1.0
                    kv_cache["local_token_decay_mask"][local_start_index:local_end_index] = False
                    kv_cache["local_token_decay_rates"][local_start_index:local_end_index] = 1.0

            if self.PC.enable and pc_end_tokens <= self.max_attention_size:
                kv_cache["k_original"][:, local_start_index:local_end_index] = k
                kv_cache["v_original"][:, local_start_index:local_end_index] = v

            window_start = max(0, local_end_index - self.max_attention_size)
            key_win_raw = kv_cache["k"][:, window_start:local_end_index]
            val_win = kv_cache["v"][:, window_start:local_end_index]

            grid_sizes_full = grid_sizes.clone()
            grid_sizes_full[0][0] = min(local_end_index // frame_seqlen, max_attention_frames)
            if rope_position_mode == "tail":
                relative_start_frame = max_attention_frames - num_new_frames
            elif use_scene_local_timing:
                relative_start_frame = scene_local_start_frame + scene_timing_offset_frames
                relative_start_frame = min(relative_start_frame, max_attention_frames - num_new_frames)
            else:
                relative_start_frame = current_start_frame if current_start_frame < max_attention_frames else max_attention_frames - num_new_frames
            relative_start_frame = max(0, relative_start_frame)

            roped_query = causal_rope_apply(
                q,
                grid_sizes,
                freqs,
                start_frame=relative_start_frame,
                scene_cut=False,
                rope_jump_value=rope_jump_value if rope_jump_enabled else None,
                rope_jump_frames=rope_jump_frames if rope_jump_enabled else 0,
            )
            roped_key = causal_rope_apply(
                key_win_raw,
                grid_sizes_full,
                freqs,
                start_frame=max_attention_frames - grid_sizes_full[0][0] if rope_position_mode == "tail" else 0,
                scene_cut=False,
                rope_jump_value=rope_jump_value if rope_jump_enabled else None,
                rope_jump_frames=rope_jump_frames if rope_jump_enabled else 0,
            )

            local_weights = kv_cache.get("local_token_weights", None)
            if torch.is_tensor(local_weights):
                local_weights = local_weights[window_start:local_end_index].to(roped_key.dtype).view(1, -1, 1, 1)
                roped_key = roped_key * local_weights
                val_win = val_win * local_weights
            if self.block_id == 0 and timestep == 1000:
                _debug_print(f"Q temporal RoPE: {relative_start_frame} -> {relative_start_frame + grid_sizes[0][0]}")
                key_start_frame = max_attention_frames - grid_sizes_full[0][0] if rope_position_mode == "tail" else 0
                _debug_print(f"K temporal RoPE: {key_start_frame} -> {key_start_frame + grid_sizes_full[0][0]}")
                if rope_jump_enabled:
                    _debug_print(f"💥 rope_jump={rope_jump_value}, frames={rope_jump_frames}")
            x = attention(
                roped_query,
                roped_key,
                val_win,
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)  
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 block_id=-1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 PC: PCConfig | None = None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.PC = PC or PCConfig(enable=True)

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, block_id, qk_norm, eps,PC=getattr(self, "PC", None))
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        timestep,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, timestep, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 pc_config=None,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        pc_kwargs = dict(pc_config or {})
        pc_kwargs.setdefault(
            "max_atten_size",
            32760 if local_attn_size == -1 else local_attn_size * 1560,
        )
        self.PC = PCConfig(**pc_kwargs)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, block_id, qk_norm, cross_attn_norm, eps, PC=self.PC)
            for block_id in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024*8, d - 4 * (d // 6)),
            rope_params(1024*8, 2 * (d // 6)),
            rope_params(1024*8, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            _debug_print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            _debug_print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            _debug_print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            _debug_print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            _debug_print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            timestep=t[0,0].item(),
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
