from typing import List, Optional, Tuple
import math
import os
import re
import torch

from utils.interactive import parse_scene_segments
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation


def _debug_print(*args, **kwargs):
    if os.environ.get("ECHO_VERBOSE", "0") == "1":
        print(*args, **kwargs)


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.inter_cfg = args.interactive
        self.scene_pool = []
        compress_candidate_cfg = getattr(self.inter_cfg, "compress_candidates", None)

        self.compress_candidate_stride = int(
            getattr(compress_candidate_cfg, "stride", 1)
        )
        self.compress_candidate_source_frames = int(
            getattr(compress_candidate_cfg, "source_frames", 0)
        )
        self.compress_candidate_storage_frames = int(
            getattr(compress_candidate_cfg, "storage_frames", 3)
        )
        self.compress_candidate_base_frame_weight = float(
            getattr(compress_candidate_cfg, "base_frame_weight", 0.9)
        )
        self.compress_mode = str(
            getattr(compress_candidate_cfg, "mode", "token_select")
        ).lower()
        self.default_fps = int(self.inter_cfg.scene_prompt.fps)
        self.default_blocks_per_scene = int(self.inter_cfg.scene_prompt.default_blocks_per_scene)
        self.rope_transition_frames = int(self.inter_cfg.rope.transition_frames)
        self.max_scene_pool_size = int(self.inter_cfg.pool.max_size)

        _debug_print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _seconds_to_blocks(self, duration_seconds: Optional[float]) -> int:
        if duration_seconds is None:
            return self.default_blocks_per_scene
        blocks = int((duration_seconds * self.default_fps) / (4 * self.num_frame_per_block))
        return max(1, blocks)

    def _mode_config(self, mode_name: str):
        return getattr(self.inter_cfg.modes, mode_name)

    def _compute_prompt_feature(self, conditional_dict: dict) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"][0].detach()
        valid_mask = prompt_embeds.abs().sum(dim=-1) > 0
        if bool(valid_mask.any()):
            prompt_embeds = prompt_embeds[valid_mask]
        pooled = prompt_embeds.mean(dim=0)

        return torch.nn.functional.normalize(pooled.to(torch.float32), dim=0, eps=1e-6).cpu()

    def _split_conditional_dict(self, conditional_dict: dict) -> List[dict]:
        prompt_embeds = conditional_dict["prompt_embeds"]
        return [
            {"prompt_embeds": prompt_embeds[index:index + 1]}
            for index in range(prompt_embeds.shape[0])
        ]

    def _compute_recall_prompt_features(self, scene_prompts: List[str]) -> List[torch.Tensor]:
        recall_cfg = self.inter_cfg.modes.recall
        match_method = str(getattr(recall_cfg, "match_method", "Similar")).lower()
        stop_words = {
            "a", "an", "and", "are", "as", "at", "back", "be", "been", "before",
            "by", "for", "from", "has", "have", "he", "her", "his", "in", "into",
            "is", "it", "its", "of", "on", "or", "over", "same", "she", "shot",
            "static", "the", "their", "them", "this", "to", "under", "while",
            "with", "wearing", "young", "now", "seen",
        }

        tokenized_prompts = []
        document_frequency = {}
        for prompt in scene_prompts:
            tokens = [
                token
                for token in re.findall(r"[a-zA-Z][a-zA-Z']+", prompt.lower())
                if len(token) > 2 and token not in stop_words
            ]
            tokenized_prompts.append(tokens)
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        if match_method == "idf":
            vocab = sorted(document_frequency)
            if not vocab:
                return [torch.zeros(1, dtype=torch.float32) for _ in scene_prompts]

            vocab_index = {token: index for index, token in enumerate(vocab)}
            num_documents = max(1, len(scene_prompts))
            features = []

            for tokens in tokenized_prompts:
                feature = torch.zeros(len(vocab), dtype=torch.float32)
                token_counts = {}
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1

                for token, count in token_counts.items():
                    df = document_frequency[token]
                    idf = math.log((1.0 + num_documents) / (1.0 + df)) + 1.0
                    tf = 1.0 + math.log(float(count))
                    feature[vocab_index[token]] = tf * idf

                features.append(torch.nn.functional.normalize(feature, dim=0, eps=1e-6))

            _debug_print("[Recall] prompt match method: IDF")
            return features

        if match_method != "similar":
            _debug_print(f"[Recall] unknown prompt match method '{match_method}', fallback to Similar.")

        num_documents = max(1, len(scene_prompts))
        common_threshold = max(2, int(num_documents * 0.7 + 0.999))
        common_tokens = {
            token
            for token, count in document_frequency.items()
            if count >= common_threshold
        }

        filtered_prompts = []
        for original_prompt, tokens in zip(scene_prompts, tokenized_prompts):
            kept_tokens = [
                token
                for token in tokens
                if token not in common_tokens
            ]
            filtered_prompts.append(" ".join(kept_tokens) if kept_tokens else original_prompt)

        recall_conditional_dicts = self._split_conditional_dict(
            self.text_encoder(text_prompts=filtered_prompts)
        )
        _debug_print("[Recall] prompt match method: Similar")
        return [
            self._compute_prompt_feature(conditional_dict)
            for conditional_dict in recall_conditional_dicts
        ]

    def _compute_similarity_rope_jump(self, similarity: float, scored_entries, mode_cfg) -> int:
        min_jump = int(mode_cfg.rope_jump.min_value)
        max_jump = int(mode_cfg.rope_jump.max_value)

        if not scored_entries:
            return max_jump

        candidate_gaps = [
            max(0.0, 1.0 - float(entry["similarity"]))
            for entry in scored_entries
        ]
        selected_gap = max(0.0, 1.0 - float(similarity))

        gap_sum = sum(candidate_gaps)
        if gap_sum <= 1e-6:
            normalized_gap = 0.0
        else:
            normalized_gap = selected_gap / gap_sum
        normalized_gap = max(0.0, min(1.0, normalized_gap))
        jump_value = min_jump + normalized_gap * (max_jump - min_jump)
        return int(round(jump_value))



    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
        # ================================
        # Interactive Video Generation
        # ================================
        scene_segments = parse_scene_segments(text_prompts[0], self.inter_cfg)
        scene_prompts = [segment.prompt for segment in scene_segments]
        scene_block_counts = [self._seconds_to_blocks(segment.duration_seconds) for segment in scene_segments]
        scene_transition_modes = [segment.transition_mode for segment in scene_segments]
        active_transition_modes = scene_transition_modes[:-1]
        needs_recall_matching = "recall" in active_transition_modes
        needs_compress_memory = any(
            bool(getattr(self._mode_config(mode_name), "use_old_memory", True))
            and str(getattr(self._mode_config(mode_name), "old_memory_source", "")) == "compress"
            for mode_name in active_transition_modes
        )
        conditional_dict_list = self._split_conditional_dict(
            self.text_encoder(text_prompts=scene_prompts)
        )
        if needs_recall_matching:
            scene_prompt_features = self._compute_recall_prompt_features(scene_prompts)
        else:
            scene_prompt_features = [torch.zeros(1, dtype=torch.float32) for _ in scene_prompts]
        
        
        self.scene_pool = []
        registered_select_indices = set()
        registered_compress_indices = set()

        smooth_select_anchor_layers = None
        smooth_select_anchor_scene_index = None

        smooth_compress_anchor_layers = None
        smooth_compress_anchor_scene_index = None

        pending_smooth_select_anchor_capture = True
        pending_smooth_compress_anchor_capture = True

        scene_block_boundaries = []
        boundary_transition_modes = {}
        cumulative_blocks = 0

        for scene_index, block_count in enumerate(scene_block_counts[:-1]):
            cumulative_blocks += block_count
            scene_block_boundaries.append(cumulative_blocks)
            boundary_transition_modes[cumulative_blocks] = scene_transition_modes[scene_index]

        _debug_print("Scene summary:")
        for scene_index, (prompt, blocks, mode_name) in enumerate(
            zip(scene_prompts, scene_block_counts, scene_transition_modes)
        ):
            duration_seconds = (blocks * 4 * self.num_frame_per_block) / self.default_fps
            _debug_print(
                f"Scene {scene_index + 1}: {blocks} blocks ({duration_seconds:.2f}s) "
                f"[next={mode_name}] - '{prompt[:50]}...'"
            )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)


        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )


        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                
                self.kv_cache1[block_index]["scene_cut"] = False
                self.kv_cache1[block_index]["rope_jump_active"] = False
                self.kv_cache1[block_index]["rope_jump_value"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache1[block_index]["rope_jump_frames"] = torch.tensor(
                    [self.rope_transition_frames], dtype=torch.long, device=noise.device
                )

                self.kv_cache1[block_index]["active_transition_mode"] = "smooth"
                self.kv_cache1[block_index]["use_scene_local_timing"] = False 
                self.kv_cache1[block_index]["scene_timing_offset_frames"].zero_()
                
                self.kv_cache1[block_index]["decay_active"] = False
                self.kv_cache1[block_index]["local_token_weights"].fill_(1.0)
                self.kv_cache1[block_index]["local_token_decay_mask"].zero_()
                self.kv_cache1[block_index]["local_token_decay_rates"].fill_(1.0)
                
                self.kv_cache1[block_index]["sink_token_count"] = torch.tensor(
                    [self.generator.model.sink_size * self.frame_seq_length],
                    dtype=torch.long,
                    device=noise.device
                )
                self.kv_cache1[block_index]["transition_evict_steps_remaining"].zero_()
                self.kv_cache1[block_index]["transition_evict_start_index"].zero_()
                self.kv_cache1[block_index]["transition_evict_token_count"].zero_()

                self.kv_cache1[block_index]["q_sum"].zero_()
                self.kv_cache1[block_index]["q_count"].zero_()
                self.kv_cache1[block_index]["collect_q_stats"] = False

                self.kv_cache1[block_index]["record_scene_candidates"] = False
                self.kv_cache1[block_index]["candidate_token_count"].zero_()
                self.kv_cache1[block_index]["old_memory_token_count"].zero_()

                self.kv_cache1[block_index]["old_memory_decay_start"].zero_()
                self.kv_cache1[block_index]["old_memory_decay_token_count"].zero_()

                self.kv_cache1[block_index]["select_token_count"].zero_()

                for key in (
                    "history_k",
                    "history_v",
                    "history_abs_frame_idx",
                    "history_spatial_idx",
                    "history_topc_select_counts",
                    "win_q_raw",
                ):
                    if key in self.kv_cache1[block_index]:
                        self.kv_cache1[block_index][key] = None
                for key in (
                    "k_original",
                    "v_original",
                    "k_raw",
                    "q_calib_sum",
                    "q_calib_abs_sum",
                    "q_calib_mean",
                    "q_calib_abs_mean",
                    "q_calib_token_count",
                ):
                    value = self.kv_cache1[block_index].get(key, None)
                    if torch.is_tensor(value):
                        value.zero_()
                for key in ("abs_frame_idx", "spatial_idx"):
                    value = self.kv_cache1[block_index].get(key, None)
                    if torch.is_tensor(value):
                        value.fill_(-1)
                q_ready = self.kv_cache1[block_index].get("q_calib_ready", None)
                if torch.is_tensor(q_ready):
                    q_ready.fill_(False)
                self.kv_cache1[block_index]["ori_start_ptr"] = 0
                self.kv_cache1[block_index]["ori_write_ptr"] = None


        # Step 2: Cache context feature
        current_start_frame = 0
        current_scene_start_frame = 0
        if initial_latent is not None:
            # Use the first scene's conditional dict for initial latent processing
            initial_conditional_dict = conditional_dict_list[0]
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=initial_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=current_scene_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=initial_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=current_scene_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block
        current_scene_start_frame = current_start_frame

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop @hidir: Inference  enters here
        all_num_frames = [self.num_frame_per_block] * num_blocks 
        # all_num_frames = [self.num_frame_per_block * num_blocks]
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        # ------------------------------------------------------------ #
        
        def _clear_scene_collection_state():
            for cache in self.kv_cache1:
                cache["q_sum"].zero_()
                cache["q_count"].zero_()
                cache["candidate_token_count"].zero_()
                cache["collect_q_stats"] = False
                cache["record_scene_candidates"] = False

        def _clear_old_memory():
            for cache in self.kv_cache1:
                cache["old_memory_token_count"].zero_()

        def _reset_pc_state_for_scene(cache, old_memory_token_count: int):
            if cache.get("k_original", None) is None:
                cache["k_original"] = torch.zeros_like(cache["k"])
            if cache.get("v_original", None) is None:
                cache["v_original"] = torch.zeros_like(cache["v"])
            if cache.get("k_raw", None) is None:
                cache["k_raw"] = torch.zeros_like(cache["k"])
            if cache.get("abs_frame_idx", None) is None:
                cache["abs_frame_idx"] = torch.full(
                    (cache["k"].shape[0], cache["k"].shape[1]),
                    -1,
                    dtype=torch.long,
                    device=cache["k"].device,
                )
            if cache.get("spatial_idx", None) is None:
                cache["spatial_idx"] = torch.full(
                    (cache["k"].shape[0], cache["k"].shape[1]),
                    -1,
                    dtype=torch.long,
                    device=cache["k"].device,
                )

            for key in (
                "history_k",
                "history_v",
                "history_abs_frame_idx",
                "history_spatial_idx",
                "history_topc_select_counts",
                "win_q_raw",
            ):
                if key in cache:
                    cache[key] = None

            for key in (
                "k_original",
                "v_original",
                "k_raw",
                "q_calib_sum",
                "q_calib_abs_sum",
                "q_calib_mean",
                "q_calib_abs_mean",
                "q_calib_token_count",
            ):
                value = cache.get(key, None)
                if torch.is_tensor(value):
                    value.zero_()

            for key in ("abs_frame_idx", "spatial_idx"):
                value = cache.get(key, None)
                if torch.is_tensor(value):
                    value.fill_(-1)

            q_ready = cache.get("q_calib_ready", None)
            if torch.is_tensor(q_ready):
                q_ready.fill_(False)

            cache["ori_start_ptr"] = 0
            cache["ori_write_ptr"] = None

            old_memory_token_count = min(old_memory_token_count, int(cache["local_end_index"].item()))
            if old_memory_token_count <= 0:
                return

            cache["k_original"][:, :old_memory_token_count] = cache["k"][:, :old_memory_token_count].clone()
            cache["v_original"][:, :old_memory_token_count] = cache["v"][:, :old_memory_token_count].clone()

            old_memory_frames = (old_memory_token_count + self.frame_seq_length - 1) // self.frame_seq_length
            frame_ids = torch.arange(old_memory_frames, dtype=torch.long, device=cache["k"].device)
            frame_ids = frame_ids.repeat_interleave(self.frame_seq_length)[:old_memory_token_count].unsqueeze(0)
            spatial_ids = torch.arange(self.frame_seq_length, dtype=torch.long, device=cache["k"].device)
            spatial_ids = spatial_ids.repeat(old_memory_frames)[:old_memory_token_count].unsqueeze(0)
            cache["abs_frame_idx"][:, :old_memory_token_count] = frame_ids
            cache["spatial_idx"][:, :old_memory_token_count] = spatial_ids

        def _reset_transition_evict_state(cache):
            cache["transition_evict_steps_remaining"].zero_()
            cache["transition_evict_start_index"].zero_()
            cache["transition_evict_token_count"].zero_()

        def _drop_kv_span(cache, start_index: int, token_count: int) -> Tuple[int, int]:
            local_end_index = int(cache["local_end_index"].item())
            if token_count <= 0 or start_index >= local_end_index:
                return local_end_index, 0

            drop_end_index = min(local_end_index, start_index + token_count)
            dropped_tokens = max(0, drop_end_index - start_index)
            remaining_tokens = local_end_index - drop_end_index

            if remaining_tokens > 0:
                cache["k"][:, start_index:start_index + remaining_tokens] = cache["k"][
                    :, drop_end_index:local_end_index
                ].clone()
                cache["v"][:, start_index:start_index + remaining_tokens] = cache["v"][
                    :, drop_end_index:local_end_index
                ].clone()
                for original_key in ("k_original", "v_original", "k_raw"):
                    original_value = cache.get(original_key, None)
                    if torch.is_tensor(original_value):
                        original_value[:, start_index:start_index + remaining_tokens] = original_value[
                            :, drop_end_index:local_end_index
                        ].clone()
                cache["abs_frame_idx"][:, start_index:start_index + remaining_tokens] = cache["abs_frame_idx"][
                    :, drop_end_index:local_end_index
                ].clone()
                cache["spatial_idx"][:, start_index:start_index + remaining_tokens] = cache["spatial_idx"][
                    :, drop_end_index:local_end_index
                ].clone()
                cache["local_token_weights"][start_index:start_index + remaining_tokens] = cache["local_token_weights"][
                    drop_end_index:local_end_index
                ].clone()
                cache["local_token_decay_mask"][start_index:start_index + remaining_tokens] = cache["local_token_decay_mask"][
                    drop_end_index:local_end_index
                ].clone()
                cache["local_token_decay_rates"][start_index:start_index + remaining_tokens] = cache["local_token_decay_rates"][
                    drop_end_index:local_end_index
                ].clone()

            new_local_end_index = local_end_index - dropped_tokens
            for original_key in ("k_original", "v_original", "k_raw"):
                original_value = cache.get(original_key, None)
                if torch.is_tensor(original_value):
                    original_value[:, new_local_end_index:local_end_index] = 0
            cache["abs_frame_idx"][:, new_local_end_index:local_end_index] = -1
            cache["spatial_idx"][:, new_local_end_index:local_end_index] = -1
            cache["local_token_weights"][new_local_end_index:local_end_index] = 1.0
            cache["local_token_decay_mask"][new_local_end_index:local_end_index] = False
            cache["local_token_decay_rates"][new_local_end_index:local_end_index] = 1.0
            return new_local_end_index, dropped_tokens

        def _evict_old_memory():
           
            for idx, cache in enumerate(self.kv_cache1):
                remaining_steps = int(cache["transition_evict_steps_remaining"].item())
                if remaining_steps <= 0:
                    continue

                remaining_steps -= 1
                cache["transition_evict_steps_remaining"] = torch.tensor(
                    [remaining_steps], dtype=torch.long, device=cache["k"].device
                )
                if remaining_steps > 0:
                    continue

                start_index = int(cache["transition_evict_start_index"].item())
                token_count = int(cache["transition_evict_token_count"].item())
                if idx == 0: 
                    _debug_print(f"Clear old_memory: start={start_index / 1560}, frames={token_count / 1560}")
                new_local_end_index, dropped_tokens = _drop_kv_span(cache, start_index, token_count)

                cache["local_end_index"] = torch.tensor(
                    [new_local_end_index], dtype=torch.long, device=cache["k"].device
                )
                dropped_frames = dropped_tokens // self.frame_seq_length
                current_offset = int(cache["scene_timing_offset_frames"].item())
                cache["scene_timing_offset_frames"] = torch.tensor(
                    [max(0, current_offset - dropped_frames)],
                    dtype=torch.long,
                    device=cache["k"].device,
                )

                cache["old_memory_decay_start"].zero_()
                cache["old_memory_decay_token_count"].zero_()
                cache["decay_active"] = False
                cache["local_token_decay_mask"].zero_()
                cache["local_token_decay_rates"].fill_(1.0)
                
                _reset_transition_evict_state(cache)

            

        def _update_decay_rates_from_diff(current_num_frames: int, block_index: int, scene_index: int):
            if current_start_frame != current_scene_start_frame:
                return

            current_block_token_count = current_num_frames * self.frame_seq_length

            for layer_index, cache in enumerate(self.kv_cache1):
                mode_name = cache.get("active_transition_mode", "smooth")
                mode_cfg = self._mode_config(mode_name)
                decay_cfg = mode_cfg.decay
                if str(decay_cfg.strategy) != "diff":
                    continue

                min_decay = float(decay_cfg.min_rate)
                max_decay = float(decay_cfg.max_rate)
                decay_target = str(decay_cfg.target)

                if decay_target == "old_recent":
                    if not cache.get("decay_active", False):
                        continue
                    target_start = int(cache["old_memory_decay_start"].item())
                    target_token_count = int(cache["old_memory_decay_token_count"].item())
                    reference_token_count = int(cache["select_token_count"].item())
                else:
                    if not cache.get("decay_active", False):
                        continue
                    target_start = int(cache["old_memory_decay_start"].item())
                    target_token_count = int(cache["old_memory_decay_token_count"].item())
                    reference_token_count = target_token_count

                if target_token_count <= 0 or reference_token_count <= 0:
                    continue

                ref_tokens = min(self.frame_seq_length, reference_token_count, current_block_token_count)
                if ref_tokens <= 0:
                    continue

                local_end_index = int(cache["local_end_index"].item())
                current_block_start = local_end_index - current_block_token_count
                if current_block_start < 0:
                    continue

                current_first_frame_k = cache["k"][
                    :, current_block_start:current_block_start + ref_tokens
                ].to(torch.float32)

                if decay_target == "old_recent":
                    reference_sink_k = cache["select_k"][:, :ref_tokens].to(torch.float32)
                else:
                    reference_sink_k = cache["k"][:, :ref_tokens].to(torch.float32)

                current_flat = current_first_frame_k.flatten(2)
                reference_flat = reference_sink_k.flatten(2)
                current_norm = torch.nn.functional.normalize(current_flat, dim=-1, eps=1e-6)
                reference_norm = torch.nn.functional.normalize(reference_flat, dim=-1, eps=1e-6)
                
                token_diff = 1.0 - (current_norm * reference_norm).sum(dim=-1)
                token_diff = token_diff.mean(dim=0)

                diff_min = token_diff.min()
                diff_max = token_diff.max()
                if float((diff_max - diff_min).item()) > 1e-6:
                    normalized_diff = (token_diff - diff_min) / (diff_max - diff_min)
                else:
                    normalized_diff = torch.zeros_like(token_diff)

                token_decay_rates = max_decay - (max_decay - min_decay) * normalized_diff
                token_decay_rates = token_decay_rates.clamp(min=min_decay, max=max_decay)

                repeat_count = (target_token_count + ref_tokens - 1) // ref_tokens
                expanded_rates = token_decay_rates.repeat(repeat_count)[:target_token_count]

                cache["local_token_decay_rates"][
                    target_start:target_start + target_token_count
                ] = expanded_rates.to(cache["local_token_decay_rates"].dtype)

        def _capture_select_from_cache(current_num_frames: int):
            
            if current_start_frame != current_scene_start_frame:
                return
            
            select_token_capacity = int(self.generator.model.sink_size * self.frame_seq_length)
            if select_token_capacity <= 0:
                return

            current_block_token_count = current_num_frames * self.frame_seq_length
            capture_tokens = min(select_token_capacity, current_block_token_count)
            if capture_tokens <= 0:
                return

            _debug_print("Captured external select memory")
            for cache in self.kv_cache1:
                local_end_index = int(cache["local_end_index"].item())
                current_block_start = local_end_index - current_block_token_count
                if current_block_start < 0:
                    continue

                cache["select_k"][:, :capture_tokens] = cache["k"][
                    :, current_block_start:current_block_start + capture_tokens
                ].clone()
                cache["select_v"][:, :capture_tokens] = cache["v"][
                    :, current_block_start:current_block_start + capture_tokens
                ].clone()
                cache["select_token_count"] = torch.tensor(
                    [capture_tokens], dtype=torch.long, device=cache["k"].device
                )

        def _store_smooth_select_anchor_from_select(scene_index: int):
            nonlocal smooth_select_anchor_layers, smooth_select_anchor_scene_index

            anchor_layers = []
            has_valid_sink = False
            for cache in self.kv_cache1:
                token_count = min(int(cache["select_token_count"].item()), self.frame_seq_length)
                if token_count <= 0:
                    anchor_layers.append(None)
                    continue

                has_valid_sink = True
                anchor_layers.append({
                    "token_count": token_count,
                    "k": cache["select_k"][:, :token_count].detach().cpu(),
                    "v": cache["select_v"][:, :token_count].detach().cpu(),
                })

            if has_valid_sink:
                smooth_select_anchor_layers = anchor_layers
                smooth_select_anchor_scene_index = scene_index

        def _store_smooth_compress_anchor_from_layers(scene_index: int, pool_layers):
            nonlocal smooth_compress_anchor_layers, smooth_compress_anchor_scene_index
            nonlocal pending_smooth_compress_anchor_capture

            if not pending_smooth_compress_anchor_capture:
                return

            anchor_layers = []
            has_valid_sink = False
            for layer_entry in pool_layers:
                if layer_entry is None:
                    anchor_layers.append(None)
                    continue

                token_count = min(int(layer_entry["token_count"]), self.frame_seq_length)
                if token_count <= 0:
                    anchor_layers.append(None)
                    continue

                has_valid_sink = True
                anchor_layer = {
                    "token_count": token_count,
                    "k": layer_entry["k"][:token_count].detach().cpu(),
                    "v": layer_entry["v"][:token_count].detach().cpu(),
                }
                anchor_layers.append(anchor_layer)

            if has_valid_sink:
                smooth_compress_anchor_layers = anchor_layers
                smooth_compress_anchor_scene_index = scene_index
                pending_smooth_compress_anchor_capture = False

        def _apply_decay():
            for cache in self.kv_cache1:
                if not cache.get("decay_active", False):
                    continue

                decay_mask = cache["local_token_decay_mask"]
                if not bool(decay_mask.any()):
                    cache["decay_active"] = False
                    continue

                decay_rates = cache["local_token_decay_rates"][decay_mask].to(
                    cache["local_token_weights"].dtype
                )
                cache["local_token_weights"][decay_mask] *= decay_rates

        def _set_scene_collection_flags(collect_q_stats: bool, record_scene_candidates: bool):
            for cache in self.kv_cache1:
                cache["collect_q_stats"] = collect_q_stats
                cache["record_scene_candidates"] = record_scene_candidates

        def _compute_compress_recall(device, mode: Optional[str] = None):
            active_mode = (self.compress_mode if mode is None else str(mode)).lower()
         
            if active_mode not in {"token_select", "stitch", "score_weighted"}:
                _debug_print(
                    f"[Recall:compress] unknown compress mode '{active_mode}', "
                    "fallback to token_select."
                )
                active_mode = "token_select"

            candidate_token_count = int(self.kv_cache1[0]["candidate_token_count"].item())
            if candidate_token_count < self.frame_seq_length:
                return False, 0, None
            available_candidate_frames = candidate_token_count // self.frame_seq_length
            if available_candidate_frames <= 0:
                return False, 0, None

            candidate_source_frames = int(self.compress_candidate_source_frames)
            
            if candidate_source_frames <= 0:
                candidate_source_frames = available_candidate_frames
            candidate_source_frames = min(candidate_source_frames, available_candidate_frames)

            _debug_print(f"Compress candidate frames: {candidate_source_frames}")

            sampled_frame_indices = torch.arange(
                0,
                candidate_source_frames,
                max(1, int(self.compress_candidate_stride)),
                dtype=torch.long,
                device=device,
            )

            sampled_frame_indices = sampled_frame_indices[
                sampled_frame_indices < available_candidate_frames
            ]

            num_candidate_frames = int(sampled_frame_indices.numel())
            if num_candidate_frames <= 0:
                return False, 0, None

            valid_layers = 0
            selected_recall_layers = [None] * len(self.kv_cache1)
            base_weight = max(0.0, min(1.0, self.compress_candidate_base_frame_weight))
            residual_weight = 1.0 - base_weight
            token_positions = torch.arange(self.frame_seq_length, device=device)

            for layer_index, cache in enumerate(self.kv_cache1):
                q_count = float(cache["q_count"].item())
                if q_count <= 0:
                    continue
                q_mean = (cache["q_sum"] / max(q_count, 1.0)).to(torch.float32)
                
                candidate_k = cache["candidate_k"][:candidate_token_count].view(
                    available_candidate_frames, self.frame_seq_length, 12, 128
                ).index_select(0, sampled_frame_indices).to(torch.float32)
                candidate_v = cache["candidate_v"][:candidate_token_count].view(
                    available_candidate_frames, self.frame_seq_length, 12, 128
                ).index_select(0, sampled_frame_indices).to(torch.float32)
                per_head_scores = (candidate_k * q_mean.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

                layer_scores = per_head_scores.mean(dim=-1)
                valid_layers += 1

                if active_mode == "score_weighted" and num_candidate_frames > 1:
                    normalized_layer_scores = torch.softmax(layer_scores, dim=0)

                    token_weights = normalized_layer_scores.unsqueeze(-1).unsqueeze(-1)
                    weighted_k = (candidate_k * token_weights).sum(dim=0)
                    weighted_v = (candidate_v * token_weights).sum(dim=0)

                    base_k = candidate_k[0]
                    base_v = candidate_v[0]

                    # ⭐⭐
                    selected_k = base_k * base_weight + weighted_k * residual_weight
                    selected_v = base_v * base_weight + weighted_v * residual_weight
                    selected_recall_layers[layer_index] = {
                        "k": selected_k,
                        "v": selected_v,
                    }
                else:
                    best_frame_indices = layer_scores.argmax(dim=0)
                    flat_indices = best_frame_indices * self.frame_seq_length + token_positions
                    candidate_k_flat = candidate_k.reshape(-1, 12, 128)
                    candidate_v_flat = candidate_v.reshape(-1, 12, 128)
                    selected_recall_layers[layer_index] = {
                        "k": candidate_k_flat.index_select(0, flat_indices),
                        "v": candidate_v_flat.index_select(0, flat_indices),
                    }

            if valid_layers == 0:
                return False, 0, None

            return True, num_candidate_frames, selected_recall_layers

        def _scene_pool_entry(scene_index: int):
            for entry in self.scene_pool:
                if entry["scene_index"] == scene_index:
                    return entry

            entry = {
                "scene_index": scene_index,
                "prompt": scene_prompts[scene_index],
                "prompt_feature": scene_prompt_features[scene_index].clone(),
                "select_layers": None,
                "compress_layers": None,
            }
            self.scene_pool.append(entry)
            if len(self.scene_pool) > self.max_scene_pool_size:
                self.scene_pool = self.scene_pool[-self.max_scene_pool_size:]
            return entry

        def _register_scene_pool_select(scene_index: int):
            if scene_index in registered_select_indices:
                return

            pool_layers = []
            has_valid_sink = False
            for cache in self.kv_cache1:
                token_count = int(cache["select_token_count"].item())
                if token_count <= 0:
                    pool_layers.append(None)
                    continue

                has_valid_sink = True
                pool_layers.append({
                    "token_count": token_count,
                    "k": cache["select_k"][:, :token_count].detach().cpu(),
                    "v": cache["select_v"][:, :token_count].detach().cpu(),
                })

            if not has_valid_sink:
                raise RuntimeError(
                    f"[Scene Pool] scene {scene_index + 1} has no valid select source."
                )

            _scene_pool_entry(scene_index)["select_layers"] = pool_layers
            registered_select_indices.add(scene_index)

        def _register_scene_pool_compress(scene_index: int, device):
            if scene_index in registered_compress_indices:
                return

            success, num_candidate_frames, selected_recall_layers = _compute_compress_recall(
                device,
                mode=self.compress_mode,
            )
            if not success or selected_recall_layers is None:
                raise RuntimeError(
                    f"[Scene Pool] scene {scene_index + 1} compress source unavailable."
                )

            pool_layers = []
            has_valid_sink = False
            for layer_entry in selected_recall_layers:
                if layer_entry is None:
                    pool_layers.append(None)
                    continue

                token_count = min(int(layer_entry["k"].shape[0]), self.frame_seq_length)
                if token_count <= 0:
                    pool_layers.append(None)
                    continue

                has_valid_sink = True
                pool_layers.append({
                    "token_count": token_count,
                    "k": layer_entry["k"][:token_count].detach().cpu(),
                    "v": layer_entry["v"][:token_count].detach().cpu(),
                })

            if not has_valid_sink:
                raise RuntimeError(
                    f"[Scene Pool] scene {scene_index + 1} compress source has no valid layer."
                )

            _store_smooth_compress_anchor_from_layers(scene_index, pool_layers)

            _scene_pool_entry(scene_index)["compress_layers"] = pool_layers
            registered_compress_indices.add(scene_index)

            _debug_print(
                f"[Scene Pool] registered scene {scene_index + 1} compress source "
                f"with {num_candidate_frames} candidate frames."
            )

        def _select_scene_pool_entry(scene_index: int, source_type: str):
            layer_key = f"{source_type}_layers"
            pool = [entry for entry in self.scene_pool if entry.get(layer_key) is not None]

            if not pool:
                return None, None, []

            current_feature = scene_prompt_features[scene_index]
            best_similarity = None
            best_entry = None
            scored_entries = []
            for entry in pool:
                similarity = float(torch.dot(current_feature, entry["prompt_feature"]).item())
                scored_entries.append({
                    "scene_index": entry["scene_index"],
                    "prompt": entry["prompt"],
                    "similarity": similarity,
                })
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if scored_entries:
                min_similarity = min(item["similarity"] for item in scored_entries)
                max_similarity = max(item["similarity"] for item in scored_entries)
                similarity_range = max(max_similarity - min_similarity, 1e-6)
                for item in scored_entries:
                    item["normalized_similarity"] = (
                        (item["similarity"] - min_similarity) / similarity_range
                    )
            scored_entries.sort(key=lambda item: item["similarity"], reverse=True)
            return best_entry, best_similarity, scored_entries

        def _manual_recall_scene_index(scene_index: int):
            manual_scene_ids = getattr(self.inter_cfg.modes.recall, "manual_recall_scene_ids", [])
            if scene_index >= len(manual_scene_ids):
                return None

            manual_scene_id = manual_scene_ids[scene_index]
            if manual_scene_id is None:
                return None

            manual_scene_id = int(manual_scene_id)
            if manual_scene_id <= 0:
                return None

            return manual_scene_id - 1

        def _select_manual_scene_pool_entry(scene_index: int, source_type: str):
            manual_scene_index = _manual_recall_scene_index(scene_index)
            if manual_scene_index is None:
                return None

            layer_key = f"{source_type}_layers"
            for entry in self.scene_pool:
                if entry["scene_index"] == manual_scene_index and entry.get(layer_key) is not None:
                    return entry

            return "missing"

        def _old_recent_token_budget(mode_cfg) -> Tuple[int, int]:
            if not bool(mode_cfg.use_old_recent):
                return 0, 0

            old_recent_frames = max(0, int(getattr(mode_cfg, "old_recent_frames", 0)))
            old_recent_tokens = old_recent_frames * self.frame_seq_length
            return old_recent_frames, old_recent_tokens

        def _stage_transition_memory(
            cache,
            source_k,
            source_v,
            token_count: int,
            device,
        ) -> int:
            cache["old_memory_token_count"].zero_()

            if token_count <= 0 or source_k is None or source_v is None:
                return 0
            
            cache["old_memory_k"][:, :token_count] = source_k[:, :token_count].to(
                device=device, dtype=cache["old_memory_k"].dtype
            )
            cache["old_memory_v"][:, :token_count] = source_v[:, :token_count].to(
                device=device, dtype=cache["old_memory_v"].dtype
            )
            cache["old_memory_token_count"] = torch.tensor([token_count], dtype=torch.long, device=device)
            return token_count

        def _load_resink_memory(device):
            memory_built = False
            for cache in self.kv_cache1:
                local_end_index = int(cache["local_end_index"].item())
                available_sink_tokens = min(
                    int(cache["sink_token_count"].item()),
                    local_end_index,
                )
                token_count = min(available_sink_tokens, self.frame_seq_length)
                staged_tokens = _stage_transition_memory(
                    cache,
                    cache["k"][:, :token_count].clone() if token_count > 0 else None,
                    cache["v"][:, :token_count].clone() if token_count > 0 else None,
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            return memory_built, None

        def _load_select_memory(device):
            memory_built = False
            for cache in self.kv_cache1:
                token_count = min(int(cache["select_token_count"].item()), self.frame_seq_length)
                staged_tokens = _stage_transition_memory(
                    cache,
                    cache["select_k"][:, :token_count].clone() if token_count > 0 else None,
                    cache["select_v"][:, :token_count].clone() if token_count > 0 else None,
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            if not memory_built:
                raise RuntimeError("[Recall:select] requested select memory, but no valid select is available.")
            return memory_built, None

       
        def _load_smooth_select_anchor_memory(device):
            if smooth_select_anchor_layers is None:
                return _load_select_memory(device)

            memory_built = False
            for cache, layer_entry in zip(self.kv_cache1, smooth_select_anchor_layers):
                token_count = min(int(layer_entry["token_count"]), self.frame_seq_length) if layer_entry is not None else 0
                staged_tokens = _stage_transition_memory(
                    cache,
                    None if layer_entry is None else layer_entry["k"],
                    None if layer_entry is None else layer_entry["v"],
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            if memory_built and smooth_select_anchor_scene_index is not None:
                _debug_print(f"[Smooth] reuse select anchor from scene {smooth_select_anchor_scene_index + 1}.")
            return memory_built, None

        def _load_smooth_compress_anchor_memory(device):

            # smooth_compress_anchor_layers = None
            if smooth_compress_anchor_layers is None:
                _debug_print("Load previous compress memory for smooth transition")
                memory_built, similarity = _load_compress_memory(device)
                
                if memory_built:
                    return memory_built, similarity
                return _load_smooth_select_anchor_memory(device)

            _debug_print("Reuse compress memory for smooth transition")
            memory_built = False
            for cache, layer_entry in zip(self.kv_cache1, smooth_compress_anchor_layers):
                token_count = min(int(layer_entry["token_count"]), self.frame_seq_length) if layer_entry is not None else 0
                staged_tokens = _stage_transition_memory(
                    cache,
                    None if layer_entry is None else layer_entry["k"],
                    None if layer_entry is None else layer_entry["v"],
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            if memory_built and smooth_compress_anchor_scene_index is not None:
                _debug_print(f"[Smooth] reuse compress anchor from scene {smooth_compress_anchor_scene_index + 1}.")
            return memory_built, None

        def _load_compress_memory(device):
            compress_entries = [
                entry for entry in self.scene_pool
                if entry.get("compress_layers") is not None
            ]
            if not compress_entries:
                raise RuntimeError("[Recall:compress] requested compress memory, but scene_pool has no compress source.")

            selected_entry = compress_entries[-1]
            _debug_print(
                f"[Recall:compress] selected previous scene {selected_entry['scene_index'] + 1}: "
                f"prompt='{selected_entry['prompt'][:50]}'"
            )

            memory_built = False
            for cache, layer_entry in zip(self.kv_cache1, selected_entry["compress_layers"]):
                token_count = min(int(layer_entry["token_count"]), self.frame_seq_length) if layer_entry is not None else 0
                staged_tokens = _stage_transition_memory(
                    cache,
                    None if layer_entry is None else layer_entry["k"],
                    None if layer_entry is None else layer_entry["v"],
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            if not memory_built:
                raise RuntimeError(
                    f"[Recall:compress] selected scene {selected_entry['scene_index'] + 1} "
                    "has no valid compress memory layers."
                )
            return memory_built, None

        def _load_recall_from_scene_pool(device, scene_index: int, source_type: str):
            if source_type not in {"compress", "select"}:
                raise RuntimeError(
                    f"[Recall] unsupported scene_pool source '{source_type}'. Expected 'compress' or 'select'."
                )

            selected_entry, similarity, scored_entries = _select_scene_pool_entry(scene_index, source_type)
            manual_selected_entry = _select_manual_scene_pool_entry(scene_index, source_type)
            selected_source = f"scene_pool.{source_type}"
            layer_key = f"{source_type}_layers"

            manual_scene_index = _manual_recall_scene_index(scene_index)

            _debug_print(f"[Recall] current scene {scene_index + 1}: '{scene_prompts[scene_index]}'")
            if scored_entries:
                _debug_print("[Recall] candidate similarities:")
                for candidate in scored_entries:
                    _debug_print(
                        f"  - pool scene {candidate['scene_index'] + 1}: "
                        f"raw_sim={candidate['similarity']:.4f}, "
                        f"norm_sim={candidate['normalized_similarity']:.4f} | "
                        f"prompt='{candidate['prompt'][:10]}'"
                    )
                if selected_entry is not None:
                    selected_normalized_similarity = next(
                        (
                            item["normalized_similarity"]
                            for item in scored_entries
                            if item["scene_index"] == selected_entry["scene_index"]
                        ),
                        None,
                    )
                    _debug_print(
                        f"[Recall] auto selected {selected_source} scene {selected_entry['scene_index'] + 1}: "
                        f"raw_sim={0.0 if similarity is None else similarity:.4f}, "
                        f"norm_sim={0.0 if selected_normalized_similarity is None else selected_normalized_similarity:.4f}"
                    )
            else:
                _debug_print(f"[Recall] candidate similarities: {selected_source} is empty.")

            if manual_selected_entry == "missing":
                raise RuntimeError(
                    f"[Recall] manual target scene {manual_scene_index + 1} is not in "
                    f"{selected_source}; cannot load requested recall memory."
                )
            elif manual_selected_entry is not None:
                # 😡😡😡
                selected_entry = manual_selected_entry
                similarity = next(
                    (
                        item["similarity"]
                        for item in scored_entries
                        if item["scene_index"] == selected_entry["scene_index"]
                    ),
                    None,
                )
                normalized_similarity = next(
                    (
                        item["normalized_similarity"]
                        for item in scored_entries
                        if item["scene_index"] == selected_entry["scene_index"]
                    ),
                    None,
                )
                _debug_print(
                    f"[Recall] manual override {selected_source} scene {selected_entry['scene_index'] + 1}: "
                    f"raw_sim={0.0 if similarity is None else similarity:.4f}, "
                    f"norm_sim={0.0 if normalized_similarity is None else normalized_similarity:.4f} | "
                    f"prompt='{selected_entry['prompt']}'"
                )

            if selected_entry is None:
                if manual_scene_index is not None:
                    raise RuntimeError(
                        f"[Recall] manual target scene {manual_scene_index + 1} unavailable "
                        f"in {selected_source}; cannot load requested recall memory."
                    )
                raise RuntimeError(
                    f"[Recall] {selected_source} is empty; cannot load recall memory "
                    f"for scene {scene_index + 1}."
                )

            if manual_selected_entry is None:
                _debug_print(
                    f"[Recall] selected {selected_source} scene {selected_entry['scene_index'] + 1}: "
                    f"raw_sim={0.0 if similarity is None else similarity:.4f} | "
                    f"prompt='{selected_entry['prompt']}'"
                )

            memory_built = False
            
            for cache, layer_entry in zip(self.kv_cache1, selected_entry[layer_key]):
                token_count = min(int(layer_entry["token_count"]), self.frame_seq_length) if layer_entry is not None else 0
                staged_tokens = _stage_transition_memory(
                    cache,
                    None if layer_entry is None else layer_entry["k"],
                    None if layer_entry is None else layer_entry["v"],
                    token_count,
                    device,
                )
                memory_built = memory_built or staged_tokens > 0

            if not memory_built:
                raise RuntimeError(
                    f"[Recall] selected {selected_source} scene "
                    f"{selected_entry['scene_index'] + 1} has no valid memory layers."
                )

            return memory_built, similarity, scored_entries


        def KV_scene_control(transition_mode: str, device, scene_index: int):

            mode_cfg = self._mode_config(transition_mode)
            decay_cfg = mode_cfg.decay
            rope_cfg = mode_cfg.rope_jump
            use_old_memory = bool(getattr(mode_cfg, "use_old_memory", True))

            if not use_old_memory:
                _clear_old_memory()
                memory_built = False
                similarity = None
                scored_entries = []
                if transition_mode == "recall":
                    _debug_print("[Recall] old_memory disabled by config, skip loading recall memory.")
            
            elif transition_mode == "recall":
                memory_source = str(mode_cfg.old_memory_source)
                memory_built, similarity, scored_entries = _load_recall_from_scene_pool(
                    device, scene_index, source_type=memory_source
                )

            else:
                if transition_mode == "smooth":
                    memory_source = str(mode_cfg.old_memory_source)
                    if memory_source == "resink":
                        memory_built, similarity = _load_resink_memory(device)
                    elif memory_source == "compress":
                        memory_built, similarity = _load_smooth_compress_anchor_memory(device)
                    elif memory_source == "select":
                        memory_built, similarity = _load_smooth_select_anchor_memory(device)
                    else:
                        raise RuntimeError(
                            f"[{transition_mode}] unsupported old_memory source '{memory_source}'."
                        )
                    
                elif transition_mode == "hardcut":
                    memory_source = str(mode_cfg.old_memory_source)
                    if memory_source == "select":
                        memory_built, similarity = _load_select_memory(device)
                    elif memory_source == "compress":
                        memory_built, similarity = _load_compress_memory(device)
                    elif memory_source == "resink":
                        memory_built, similarity = _load_resink_memory(device)
                    else:
                        raise RuntimeError(
                            f"[{transition_mode}] unsupported old_memory source '{memory_source}'."
                        )
                    
                scored_entries = []

            if str(rope_cfg.strategy) == "similarity":
                rope_jump_value = self._compute_similarity_rope_jump(
                    0.0 if similarity is None else similarity,
                    scored_entries,
                    mode_cfg,
                )
            else:
                rope_jump_value = int(rope_cfg.value)

            old_recent_frames, old_recent_token_budget = _old_recent_token_budget(mode_cfg)

            for i, cache in enumerate(self.kv_cache1):
                self.crossattn_cache[i]["is_init"] = False
                local_end_index = int(cache["local_end_index"].item())
                
                old_memory_capacity = int(cache["old_memory_k"].shape[1])
                old_recall_tokens = min(
                    int(cache["old_memory_token_count"].item()),
                    old_memory_capacity,
                ) if memory_built else 0
                old_recent_tokens = 0

                if transition_mode == "smooth" and old_recent_token_budget > 0:
                    old_recent_tokens = min(
                        old_recent_token_budget,
                        local_end_index,
                    )

                old_memory_write_index = 0
                if old_recall_tokens > 0:
                    cache["k"][:, :old_recall_tokens] = cache["old_memory_k"][:, :old_recall_tokens].clone()
                    cache["v"][:, :old_recall_tokens] = cache["old_memory_v"][:, :old_recall_tokens].clone()
                    old_memory_write_index += old_recall_tokens

                if transition_mode == "smooth" and old_recent_tokens > 0:
                    old_recent_start = local_end_index - old_recent_tokens
                    # 😡
                    cache["k"][:, old_memory_write_index:old_memory_write_index + old_recent_tokens] = cache["k"][
                        :, old_recent_start:local_end_index
                    ].clone()
                    cache["v"][:, old_memory_write_index:old_memory_write_index + old_recent_tokens] = cache["v"][
                        :, old_recent_start:local_end_index
                    ].clone()
                    old_memory_write_index += old_recent_tokens

                cache["local_end_index"] = torch.tensor([old_memory_write_index], dtype=torch.long, device=device)
                _reset_pc_state_for_scene(cache, old_memory_write_index)
                
                cache["scene_cut"] = True
                cache["rope_jump_active"] = True
                cache["rope_jump_value"] = torch.tensor([rope_jump_value], dtype=torch.long, device=device)
                cache["rope_jump_frames"] = torch.tensor(
                    [self.rope_transition_frames], dtype=torch.long, device=device
                )
                cache["active_transition_mode"] = transition_mode
                cache["rope_position_mode"] = str(
                    getattr(mode_cfg, "rope_position_mode", "local")
                )
                cache["use_scene_local_timing"] = bool(
                    getattr(mode_cfg, "use_scene_local_timing", True)
                )


                cache["scene_timing_offset_frames"] = torch.tensor(
                    [old_memory_write_index // self.frame_seq_length], dtype=torch.long, device=device
                )
                cache["local_token_weights"].fill_(1.0)
                cache["local_token_decay_mask"].zero_()
                cache["local_token_decay_rates"].fill_(1.0)

                cache["old_memory_decay_start"].zero_()
                cache["old_memory_decay_token_count"].zero_()
                cache["decay_active"] = False
                _reset_transition_evict_state(cache)

                cache["old_memory_token_count"].zero_()

                if transition_mode == "smooth":
                    decay_target = str(decay_cfg.target)
                    if decay_target == "old_recent":
                        start = old_recall_tokens
                        decay_token_count = old_recent_tokens
                    elif decay_target == "old_recall":
                        start = 0
                        decay_token_count = old_recall_tokens
                    elif decay_target == "old_memory":
                        start = 0
                        decay_token_count = old_recall_tokens + old_recent_tokens
                    else:
                        start = 0
                        decay_token_count = 0
                    end = start + decay_token_count

                if transition_mode == "smooth" and decay_token_count > 0:
                    cache["local_token_decay_mask"][start:end] = True
                    cache["old_memory_decay_start"] = torch.tensor([start], dtype=torch.long, device=device)
                    cache["old_memory_decay_token_count"] = torch.tensor(
                        [decay_token_count], dtype=torch.long, device=device
                    )
                    cache["decay_active"] = True

                    if str(decay_cfg.strategy) == "fixed":
                        cache["local_token_decay_rates"][start:end] = float(decay_cfg.fixed_rate)

                    evict_steps = int(getattr(decay_cfg, "clear_after_blocks", old_recent_frames))
                    if evict_steps > 0:
                        evict_token_count = old_recall_tokens + old_recent_tokens
                        cache["transition_evict_steps_remaining"] = torch.tensor(
                            [evict_steps], dtype=torch.long, device=device
                        )
                        cache["transition_evict_start_index"] = torch.tensor(
                            [0], dtype=torch.long, device=device
                        )
                        cache["transition_evict_token_count"] = torch.tensor(
                            [evict_token_count], dtype=torch.long, device=device
                        )

                if transition_mode in {"hardcut", "recall"} and old_recall_tokens > 0 and str(decay_cfg.target) == "old_memory":
                    first_block_weight = float(getattr(decay_cfg, "first_block_weight", 1.0))
                    first_block_weight = max(0.0, min(1.0, first_block_weight))
                    cache["local_token_weights"][:old_recall_tokens] = first_block_weight
                    cache["local_token_decay_mask"][:old_recall_tokens] = True

                    cache["old_memory_decay_start"] = torch.tensor([0], dtype=torch.long, device=device)
                    cache["old_memory_decay_token_count"] = torch.tensor(
                        [old_recall_tokens], dtype=torch.long, device=device
                    )
                    cache["decay_active"] = True

                    if str(decay_cfg.strategy) == "fixed":
                        cache["local_token_decay_rates"][:old_recall_tokens] = float(decay_cfg.fixed_rate)
                    evict_steps = int(getattr(decay_cfg, "clear_after_blocks", 1))
                    if evict_steps > 0:
                        cache["transition_evict_steps_remaining"] = torch.tensor(
                            [evict_steps], dtype=torch.long, device=device
                        )
                        cache["transition_evict_start_index"] = torch.tensor(
                            [0], dtype=torch.long, device=device
                        )
                        cache["transition_evict_token_count"] = torch.tensor(
                            [old_recall_tokens], dtype=torch.long, device=device
                        )

            _clear_scene_collection_state()
        
 
        for current_block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()
            # Determine which scene this block belongs to

            scene_index = 0
            for boundary in scene_block_boundaries:
                if current_block_index < boundary:
                    break
                scene_index += 1
            conditional_dict = conditional_dict_list[scene_index]
            
            if current_block_index in scene_block_boundaries:
                transition_mode = boundary_transition_modes[current_block_index]
                KV_scene_control(
                    transition_mode,
                    noise.device,
                    scene_index=scene_index,
                )

                current_scene_start_frame = current_start_frame

                if transition_mode != "smooth":
                    smooth_select_anchor_layers = None
                    smooth_select_anchor_scene_index = None
                    smooth_compress_anchor_layers = None
                    smooth_compress_anchor_scene_index = None
                    pending_smooth_select_anchor_capture = True
                    pending_smooth_compress_anchor_capture = True
                print(f"Scene switch -> scene {scene_index + 1}: {transition_mode}")
            else:
                n_layers = len(self.crossattn_cache)
                for i in range(n_layers):
                    self.kv_cache1[i]['scene_cut'] = False
                    self.kv_cache1[i]['rope_jump_active'] = False

            # ---------------------------------------------------------------- #
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=current_scene_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=current_scene_start_frame * self.frame_seq_length,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            _set_scene_collection_flags(needs_compress_memory, needs_compress_memory)

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=current_scene_start_frame * self.frame_seq_length,
            )

            _update_decay_rates_from_diff(
                current_num_frames=current_num_frames,
                block_index=current_block_index,
                scene_index=scene_index,
            )

            _capture_select_from_cache(current_num_frames)

            if current_start_frame == current_scene_start_frame and pending_smooth_select_anchor_capture:
                _store_smooth_select_anchor_from_select(scene_index)
                pending_smooth_select_anchor_capture = False

            _register_scene_pool_select(scene_index)

            is_scene_tail_block = (
                (current_block_index + 1) in scene_block_boundaries
                or current_block_index == len(all_num_frames) - 1
            )
            if needs_compress_memory and is_scene_tail_block:
                _register_scene_pool_compress(scene_index, noise.device)

            _set_scene_collection_flags(False, False)

            if progress_callback is not None and is_scene_tail_block:
                scene_start_frame = int(current_scene_start_frame)
                scene_end_frame = int(current_start_frame + current_num_frames)
                progress_callback({
                    "scene_index": int(scene_index),
                    "scene_prompt": scene_prompts[scene_index],
                    "transition_mode": scene_transition_modes[scene_index],
                    "start_frame": scene_start_frame,
                    "end_frame": scene_end_frame,
                    "latents": output[:, scene_start_frame:scene_end_frame].detach(),
                })
            
            _apply_decay()
            

            _evict_old_memory()

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            _debug_print("Profiling results:")
            _debug_print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            _debug_print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                _debug_print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            _debug_print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            _debug_print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    # def _initialize_compressed_kv_cache(self, batch_size, dtype, device):
    #     """
    #     Initialize a Per-GPU compressed KV cache for the Wan model.
    #     """
    #     kv_cache1 = []
    #     if self.local_attn_size != -1:
    #         # Use the local attention size to compute the KV cache size
    #         kv_cache_size = self.local_attn_size * self.frame_seq_length
    #     else:
    #         # Use the default KV cache size
    #         kv_cache_size = 32760

    #     for _ in range(self.num_transformer_blocks):
    #         kv_cache1.append({
    #             "compressed_kv": torch.zeros([batch_size, kv_cache_size, 1088], dtype=dtype, device=device),
    #             "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
    #             "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
    #         })

    #     self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        for layer_index in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),

                "scene_cut": False,
                "rope_jump_active": False,
                "rope_jump_value": torch.tensor([0], dtype=torch.long, device=device),
                "rope_jump_frames": torch.tensor([self.rope_transition_frames], dtype=torch.long, device=device),
                "active_transition_mode": "smooth",
                "rope_position_mode": "local",

                "use_scene_local_timing": False,
                "scene_timing_offset_frames": torch.tensor([0], dtype=torch.long, device=device),

                "sink_token_count": torch.tensor(
                    [self.generator.model.sink_size * self.frame_seq_length],
                    dtype=torch.long,
                    device=device
                ),
                "transition_evict_steps_remaining": torch.tensor([0], dtype=torch.long, device=device),
                "transition_evict_start_index": torch.tensor([0], dtype=torch.long, device=device),
                "transition_evict_token_count": torch.tensor([0], dtype=torch.long, device=device),

                "collect_q_stats": False,
                "q_sum": torch.zeros([12, 128], dtype=torch.float32, device=device),
                "q_count": torch.tensor([0.0], dtype=torch.float32, device=device),
                
                #--------------------------compress recall------------------------------
                #-----------------------------------------------------------------
                "record_scene_candidates": False,

                "candidate_k": torch.zeros(
                    [self.compress_candidate_storage_frames * self.frame_seq_length, 12, 128],
                    dtype=dtype,
                    device=device
                ),
                "candidate_v": torch.zeros(
                    [self.compress_candidate_storage_frames * self.frame_seq_length, 12, 128],
                    dtype=dtype,
                    device=device
                ),
                "candidate_token_count": torch.tensor([0], dtype=torch.long, device=device),
   
                "old_memory_k": torch.zeros(
                    [batch_size, self.frame_seq_length, 12, 128], dtype=dtype, device=device
                ),
                "old_memory_v": torch.zeros(
                    [batch_size, self.frame_seq_length, 12, 128], dtype=dtype, device=device
                ),
                "old_memory_token_count": torch.tensor([0], dtype=torch.long, device=device),
                
                # ✅ select recall
                "select_k": torch.zeros(
                    [batch_size, self.generator.model.sink_size * self.frame_seq_length, 12, 128],
                    dtype=dtype,
                    device=device
                ),
                "select_v": torch.zeros(
                    [batch_size, self.generator.model.sink_size * self.frame_seq_length, 12, 128],
                    dtype=dtype,
                    device=device
                ),
                "select_token_count": torch.tensor([0], dtype=torch.long, device=device),

                "decay_active": False,
                "local_token_weights": torch.ones([kv_cache_size], dtype=torch.float32, device=device),
                "local_token_decay_mask": torch.zeros([kv_cache_size], dtype=torch.bool, device=device),
                "local_token_decay_rates": torch.ones([kv_cache_size], dtype=torch.float32, device=device),
                
              
                "old_memory_decay_start": torch.tensor([0], dtype=torch.long, device=device),
                "old_memory_decay_token_count": torch.tensor([0], dtype=torch.long, device=device),

         
                "layer_index": layer_index,

            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _parse_scene_durations(self, prompt: str) -> Tuple[List[str], List[int], List[str]]:
        scene_segments = parse_scene_segments(prompt, self.inter_cfg)
        prompt_texts = [segment.prompt for segment in scene_segments]
        block_counts = [self._seconds_to_blocks(segment.duration_seconds) for segment in scene_segments]
        transition_modes = [segment.transition_mode for segment in scene_segments]
        return prompt_texts, block_counts, transition_modes
