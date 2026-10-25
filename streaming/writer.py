import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import av
import numpy as np


def encode_chunk(path, frames, fps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "20"}
        stream.codec_context.thread_count = 2
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class ChunkWriter:
    """CPU encoding overlaps GPU work, with bounded buffering and atomic chunks."""
    def __init__(self, directory, emit, fps=16, chunk_frames=24):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.emit, self.fps, self.chunk_frames = emit, fps, chunk_frames
        self.queue = queue.Queue(maxsize=3)
        self.paths, self.error, self.frames = [], None, 0
        self.first_chunk_at = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def put(self, frames):
        while True:
            self.check()
            try:
                self.queue.put(frames, timeout=0.2)
                return
            except queue.Full:
                pass

    def check(self):
        if self.error is not None:
            raise RuntimeError("Video encoder failed") from self.error

    def _save(self, frames):
        path = self.directory / f"chunk_{len(self.paths):06d}.mp4"
        encode_chunk(path, frames, self.fps)
        self.paths.append(path)
        self.frames += len(frames)
        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()
        self.emit(dict(type="chunk", path=str(path), frames=self.frames,
                       video_seconds=self.frames / self.fps))

    def _worker(self):
        pending = None
        try:
            while True:
                frames = self.queue.get()
                if frames is None:
                    break
                pending = frames if pending is None else np.concatenate((pending, frames))
                while len(pending) >= self.chunk_frames:
                    self._save(pending[:self.chunk_frames])
                    pending = pending[self.chunk_frames:]
            if pending is not None and len(pending):
                self._save(pending)
        except BaseException as exc:
            self.error = exc

    def finish(self):
        self.put(None)
        self.thread.join()
        self.check()
        if not self.paths:
            return None
        listing = self.directory / "concat.txt"
        listing.write_text("".join(f"file '{path.name}'\n" for path in self.paths))
        final = self.directory / "final.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
        return str(final)
