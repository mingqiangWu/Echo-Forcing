"""Fused FP64 RoPE: retain reference precision without materializing large temporaries."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _rope_kernel(X, FREQ, Y, L, H, W, START, PREFIX, SUFFIX,
                     XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
                     HEADS: tl.constexpr, PAIRS: tl.constexpr, SEQ,
                     JUMP: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < L * HEADS * PAIRS
        pair = i % PAIRS
        head = i // PAIRS % HEADS
        token = i // (HEADS * PAIRS)
        frame = token // (H * W)
        height = token // W % H
        width = token % W
        temporal = START + frame
        if JUMP:
            temporal = tl.where(frame >= PREFIX, SUFFIX + frame - PREFIX, temporal)
        spatial = PAIRS // 3
        row = tl.where(pair < PAIRS - 2 * spatial, temporal,
                       tl.where(pair < PAIRS - spatial, height, width))
        offset = token * XS0 + head * XS1 + pair * 2 * XS2
        xr = tl.load(X + offset, valid, other=0).to(tl.float64)
        xi = tl.load(X + offset + XS2, valid, other=0).to(tl.float64)
        rotate = valid & (token < SEQ)
        fr = tl.load(FREQ + row * (2 * PAIRS) + 2 * pair, rotate, other=1)
        fi = tl.load(FREQ + row * (2 * PAIRS) + 2 * pair + 1, rotate, other=0)
        yr = xr * fr - xi * fi
        yi = xr * fi + xi * fr
        yr = tl.where(token < SEQ, yr, xr)
        yi = tl.where(token < SEQ, yi, xi)
        # Match PyTorch's output conversion through float32. Triton 3.1 does
        # not lower a direct FP64 -> BF16/FP16 store.
        tl.store(Y + i * 2, yr.to(tl.float32), valid)
        tl.store(Y + i * 2 + 1, yi.to(tl.float32), valid)


def fused_rope(x, grid_sizes, freqs, start_frame=0, scene_cut=False,
               rope_jump_value=None, rope_jump_frames=0):
    """Return None for unsupported inputs so the caller can use its reference path."""
    if (triton is None or not x.is_cuda or freqs.device != x.device or freqs.dtype != torch.complex128
            or (torch.is_grad_enabled() and (x.requires_grad or freqs.requires_grad))
            or not freqs.is_contiguous() or x.shape[-1] % 2
            or freqs.shape[1] != x.shape[-1] // 2
            or x.dtype not in (torch.bfloat16, torch.float16)):
        return None
    result = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    start = int(start_frame)
    jump = scene_cut or (rope_jump_value is not None and rope_jump_frames > 0)
    for batch, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq = frames * height * width
        if seq > x.shape[1] or height > freqs.shape[0] or width > freqs.shape[0]:
            return None
        prefix = max(0, frames - (rope_jump_frames if rope_jump_frames > 0 else 3))
        suffix = min(max(0, start + prefix + int(rope_jump_value if rope_jump_value is not None else 45)),
                     max(0, freqs.shape[0] - (frames - prefix)))
        if start < 0 or start + (prefix if jump else frames) > freqs.shape[0]:
            return None
        _rope_kernel[(triton.cdiv(x.shape[1] * x.shape[2] * (x.shape[3] // 2), 256),)](
            x[batch], torch.view_as_real(freqs), result[batch], x.shape[1], height, width,
            start, prefix, suffix, x.stride(1), x.stride(2), x.stride(3),
            x.shape[2], x.shape[3] // 2, seq, bool(jump), 256,
            enable_fp_fusion=False, num_warps=4)
    return result
