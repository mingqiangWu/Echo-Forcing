import torch


class OnlineFrameResampler:
    """Emit the same nearest frames as offline fitting, without retaining a scene."""
    def __init__(self, source_frames, target_frames, device="cpu"):
        if source_frames < 1 or target_frames < 1:
            raise ValueError("Frame counts must be positive")
        self.indices = torch.linspace(0, source_frames - 1, target_frames,
                                      device=device).round().long().cpu()
        self.source_cursor = self.output_cursor = 0

    def push(self, frames):
        end = self.source_cursor + frames.shape[1]
        output_end = int(torch.searchsorted(self.indices, end).item())
        selected = self.indices[self.output_cursor:output_end] - self.source_cursor
        if selected.numel() and int(selected.min()) < 0:
            raise RuntimeError("Frame resampler lost a source frame")
        result = frames.index_select(1, selected.to(frames.device))
        self.source_cursor, self.output_cursor = end, output_end
        return result


class IncrementalDecoder:
    """One owner per VAE session; retain smooth caches and reset at hard/recall."""
    def __init__(self, vae):
        self.vae = vae
        self.scene_index = None
        self.resampler = None
        self.first_block = True

    def push(self, latents, *, scene_index, incoming_mode, native_frames, target_frames):
        new_scene = scene_index != self.scene_index
        cold_start = new_scene and (self.scene_index is None or incoming_mode != "smooth")
        if new_scene:
            if cold_start:
                self.vae.model.clear_cache()
            self.resampler = OnlineFrameResampler(native_frames,
                                                  target_frames or native_frames,
                                                  latents.device)
            self.scene_index = scene_index
        pixels = self.vae.decode_to_pixel(latents, use_cache=True)
        if cold_start and not self.first_block:
            pixels = torch.cat((pixels[:, :1].expand(-1, 3, -1, -1, -1), pixels), dim=1)
        self.first_block = False
        return self.resampler.push(pixels)

    def close(self):
        self.vae.model.clear_cache()
