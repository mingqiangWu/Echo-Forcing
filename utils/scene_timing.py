"""Cumulative scene timing on the generation-block and output-frame grids."""
import math


def scene_timing(durations, fps, block_size, default_blocks, *, first_block_frames=None,
                 initial_latent_frames=0):
    """Return block counts and exact declared output lengths (None if unspecified).

    Round cumulative boundaries rather than each duration, accounting for Wan's
    first-frame compression. Every scene receives at least one generation block.
    """
    first = block_size if first_block_frames is None else first_block_frames
    elapsed = 0.0
    previous_blocks = previous_pixels = 0
    counts, targets = [], []
    for duration in durations:
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError("Scene durations must be finite and positive")
        elapsed += duration if duration is not None else default_blocks * 4 * block_size / fps
        pixels = max(previous_pixels + 1, math.floor(elapsed * fps + 0.5))
        ideal = ((pixels + 3) / 4 - initial_latent_frames - first) / block_size + 1
        boundary = max(previous_blocks + 1, math.floor(ideal + 0.5))
        counts.append(boundary - previous_blocks)
        targets.append(pixels - previous_pixels if duration is not None else None)
        previous_blocks, previous_pixels = boundary, pixels
    return counts, targets
