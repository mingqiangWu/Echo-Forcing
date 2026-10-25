"""Keep VAE continuity within smooth shots; reset at hard/recall boundaries."""
import torch


def decode_scene_segment(vae, latents, *, first_scene, target_frames=None, context_latents=None,
                         return_in_cpu=False):
    """Decode a scene, then optionally fit its declared output frame count.

    A new hard/recall shot pads its first decoded frame to preserve timing.
    Smooth live previews instead replay real same-shot context and crop it,
    without adding a cold-start hold at the prompt boundary.
    """
    if latents.ndim != 5 or latents.shape[1] < 1:
        raise ValueError("Expected nonempty [B, T, C, H, W] scene latents")
    length = latents.shape[1]
    context_length = 0
    if context_latents is not None and context_latents.shape[1] > 0:
        context_length = context_latents.shape[1]
        latents = torch.cat((context_latents, latents), dim=1)
    decoded = vae.decode_to_pixel(
        latents, use_cache=False, return_in_cpu=return_in_cpu)
    if context_length:
        decoded = decoded[:, 4 * context_length - 3:]
    elif not first_scene:
        decoded = torch.cat((decoded[:, :1].expand(-1, 3, -1, -1, -1), decoded), dim=1)
    expected = 4 * length - (3 if first_scene and not context_length else 0)
    if decoded.shape[1] != expected:
        raise RuntimeError(
            f"Wan VAE scene decode returned {decoded.shape[1]} frames; expected {expected}")
    return fit_scene_frames(decoded, target_frames)


def fit_scene_frames(video, target_frames):
    """Fit one scene in time using nearest frames, never blending across a cut."""
    if target_frames is None or target_frames == video.shape[1]:
        return video
    if not isinstance(target_frames, int) or target_frames < 1:
        raise ValueError("target_frames must be a positive integer")
    indices = torch.linspace(0, video.shape[1] - 1, target_frames, device=video.device).round().long()
    return video.index_select(1, indices)


def decode_scene_sequence(vae, latents, scene_starts, *, scene_frame_counts=None, transition_modes=None,
                          return_in_cpu=False):
    """Decode actual latent boundaries and optionally align declared durations."""
    starts = list(scene_starts)
    length = latents.shape[1]
    if (not starts or starts[0] != 0 or any(not isinstance(s, int) for s in starts)
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or starts[-1] >= length):
        raise ValueError("scene_starts must begin at zero and increase within the latent sequence")
    if scene_frame_counts is None:
        scene_frame_counts = [None] * len(starts)
    if len(scene_frame_counts) != len(starts):
        raise ValueError("Expected one output frame count per scene")
    if transition_modes is None:
        transition_modes = ["hardcut"] * (len(starts) - 1)
    if len(transition_modes) != len(starts) - 1 or any(
            mode not in {"smooth", "hardcut", "recall"} for mode in transition_modes):
        raise ValueError("Expected one valid transition mode per boundary")
    # Adjacent smooth prompts belong to one continuous VAE decode. Hard and
    # recall start a new shot, retaining their existing cold-start frame padding.
    group_indices = [0] + [i + 1 for i, mode in enumerate(transition_modes) if mode != "smooth"]
    videos = []
    for group_index, first_index in enumerate(group_indices):
        stop_index = group_indices[group_index + 1] if group_index + 1 < len(group_indices) else len(starts)
        start = starts[first_index]
        end = starts[stop_index] if stop_index < len(starts) else length
        decoded = decode_scene_segment(
            vae, latents[:, start:end], first_scene=(first_index == 0),
            return_in_cpu=return_in_cpu)
        offset = 3 if first_index == 0 else 0
        for index in range(first_index, stop_index):
            begin_pixel = 0 if index == first_index else 4 * (starts[index] - start) - offset
            end_pixel = (4 * (starts[index + 1] - start) - offset
                         if index + 1 < stop_index else decoded.shape[1])
            videos.append(fit_scene_frames(decoded[:, begin_pixel:end_pixel], scene_frame_counts[index]))
    return videos[0] if len(videos) == 1 else torch.cat(videos, dim=1)
