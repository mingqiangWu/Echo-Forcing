import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

import torch

from inference_demo import EchoForcingDemoEngine, calculate_latent_frames_from_duration
from utils.interactive import parse_total_duration
from utils.misc import set_seed
from streaming.control import StreamControl
from streaming.decoder import IncrementalDecoder
from streaming.writer import ChunkWriter


class BlockHandler:
    def __init__(self, vae, writer, emit, control, started, fps=16):
        self.decoder = IncrementalDecoder(vae)
        self.writer, self.emit, self.control = writer, emit, control
        self.started, self.fps = started, fps
        self.frames = 0
        self.first_frame_seconds = None
        self.blocks = []
        self.boundaries = []

    def on_block(self, latents, **metadata):
        started = time.perf_counter()
        if metadata["scene_index"] != self.decoder.scene_index:
            self.boundaries.append(dict(scene=metadata["scene_index"], frame=self.frames,
                                        mode=metadata["incoming_mode"], prompt=metadata["prompt"]))
        pixels = self.decoder.push(latents, **{key: metadata[key] for key in
                                  ("scene_index", "incoming_mode", "native_frames", "target_frames")})
        remaining = max(0, int(self.control.max_seconds * self.fps) - self.frames)
        pixels = pixels[:, :remaining]
        arrays = ((pixels[0].permute(0, 2, 3, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if len(arrays):
            self.frames += len(arrays)
            now = time.perf_counter()
            if self.first_frame_seconds is None:
                self.first_frame_seconds = now - self.started
            self.blocks.append(dict(block=metadata["block_index"], scene=metadata["scene_index"],
                                    frames=len(arrays), elapsed=now-self.started,
                                    decode_seconds=now-started))
            self.writer.put(arrays)
            self.emit(dict(type="preview", image=arrays[-1], frames=self.frames,
                           elapsed=now-self.started, fps=self.frames/max(now-self.started, 1e-6)))
        if self.frames >= int(self.control.max_seconds * self.fps):
            self.control.stop()

    def finish(self):
        self.decoder.close()


class AsyncBlockHandler:
    """Optional second GPU decodes while the generator starts its next block."""
    def __init__(self, handler, device):
        self.handler, self.device = handler, torch.device(device)
        self.queue = queue.Queue(maxsize=1)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            torch.cuda.set_device(self.device)
            with torch.inference_mode(), self.handler.decoder.vae.decode_lock:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    latents, metadata = item
                    self.handler.on_block(latents, **metadata)
                self.handler.finish()
        except BaseException as exc:
            self.error = exc

    def _put(self, item):
        while True:
            if self.error is not None:
                raise RuntimeError("Decoder worker failed") from self.error
            try:
                self.queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def on_block(self, latents, **metadata):
        self._put((latents.to(self.device), metadata))

    def finish(self):
        self._put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Decoder worker failed") from self.error


class StreamingEngine(EchoForcingDemoEngine):
    def __init__(self, decoder_device=None, **kwargs):
        super().__init__(output_root=".streaming_cache", **kwargs)
        self.decoder_device = decoder_device

    def load(self):
        pipeline, config, device = super().load()
        if self.decoder_device is not None:
            destination = torch.device(self.decoder_device)
            if next(pipeline.vae.parameters()).device != destination:
                pipeline.vae.to(destination)
        return pipeline, config, device

    def run(self, prompt_text, control, emit, *, two_steps=False, seed=0, chunk_frames=24,
            directory=None):
        with self._lock, torch.inference_mode():
            torch.cuda.set_device(0)
            loaded_at = time.perf_counter()
            emit(dict(type="status", message="Loading the model; it will be reused for later runs"))
            pipeline, config, device = self.load()
            load_seconds = time.perf_counter() - loaded_at
            if control.stopped:
                return dict(final_path=None, cancelled=True, frames=0)
            set_seed(int(seed))
            fps = int(config.interactive.scene_prompt.fps)
            duration = parse_total_duration(prompt_text, config.interactive)
            if duration is None or duration <= 0 or duration > control.max_seconds:
                raise ValueError(f"Specify each scene's duration. Total duration must be at most {control.max_seconds}s")
            n = calculate_latent_frames_from_duration(duration, fps, 4,
                    int(config.num_frame_per_block), bool(config.independent_first_frame), False)
            noise = torch.randn([1, n, 16, 60, 104], device=device, dtype=torch.bfloat16)
            directory = Path(directory) if directory else self.output_root / uuid.uuid4().hex
            directory.mkdir(parents=True, exist_ok=True)
            writer = ChunkWriter(directory, emit, fps, int(chunk_frames))
            started = time.perf_counter()
            emit(dict(type="status", message="Model ready. Generating and decoding video blocks"))
            control.on_ack = emit
            handler = BlockHandler(pipeline.vae, writer, emit, control, started, fps)
            vae_device = next(pipeline.vae.parameters()).device
            asynchronous = vae_device != next(pipeline.generator.parameters()).device
            worker = AsyncBlockHandler(handler, vae_device) if asynchronous else handler
            original_steps = pipeline.denoising_step_list
            if two_steps:
                pipeline.denoising_step_list = original_steps[::2]
            error = None
            result = {}
            try:
                result = pipeline.inference(noise=noise, text_prompts=[prompt_text],
                            low_memory=self.low_memory, stream_handler=worker, stream_control=control)
            except BaseException as exc:
                error = exc
            finally:
                pipeline.denoising_step_list = original_steps
                try:
                    worker.finish()
                except BaseException as exc:
                    error = error or exc
                final_path = writer.finish()
            elapsed = time.perf_counter() - started
            steady = None
            if len(handler.blocks) > 2:
                warm = handler.blocks[1]
                frames_after_warmup = sum(row["frames"] for row in handler.blocks[2:])
                steady = frames_after_warmup / max(handler.blocks[-1]["elapsed"]-warm["elapsed"], 1e-6)
            result.update(final_path=final_path, directory=str(directory), frames=writer.frames,
                          fps=fps, video_seconds=writer.frames/fps, elapsed_seconds=elapsed,
                          load_seconds=load_seconds, first_frame_seconds=handler.first_frame_seconds,
                          first_chunk_seconds=(writer.first_chunk_at-started) if writer.first_chunk_at else None,
                          throughput_fps=writer.frames/max(elapsed, 1e-6), steady_fps=steady,
                          two_steps=two_steps, decoder_device=str(vae_device),
                          boundaries=handler.boundaries, blocks_detail=handler.blocks)
            (directory / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            if error:
                raise RuntimeError(f"Streaming failed; generated segments remain in {directory}") from error
            return result

    def stream(self, prompt_text, control, **kwargs):
        events = queue.Queue(maxsize=16)

        def emit(event):
            while not control.disconnected.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except queue.Full:
                    if event["type"] == "preview":
                        return

        def task():
            try:
                result = self.run(prompt_text, control, emit, **kwargs)
                emit(dict(type="done", result=result))
            except BaseException:
                emit(dict(type="error", message=traceback.format_exc()))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()
        try:
            while True:
                try:
                    event = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                yield event
                if event["type"] in {"done", "error"}:
                    break
        finally:
            control.disconnected.set()
            control.stop()
