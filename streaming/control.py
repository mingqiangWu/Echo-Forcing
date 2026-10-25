import math
import threading
import time
import uuid


class StreamControl:
    def __init__(self, interactive=True, max_seconds=180):
        self.interactive = interactive
        self.max_seconds = max_seconds
        self._stop = threading.Event()
        self.disconnected = threading.Event()
        self._lock = threading.Lock()
        self._pending = None
        self.on_ack = lambda event: None

    @property
    def stopped(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def switch(self, prompt, duration, mode):
        if not self.interactive or self.stopped:
            raise ValueError("This run is not accepting prompt switches")
        if not prompt.strip() or mode not in {"smooth", "hardcut", "recall"}:
            raise ValueError("Enter a prompt and select a valid transition")
        duration = float(duration)
        if not math.isfinite(duration) or not 1 <= duration <= 120:
            raise ValueError("Duration must be between 1 and 120 seconds")
        command = dict(id=uuid.uuid4().hex, prompt=prompt.strip(), duration=duration,
                       mode=mode, submitted=time.perf_counter())
        with self._lock:
            self._pending = command  # The latest request supersedes unsent edits.
        return command["id"]

    def take_switch(self):
        with self._lock:
            command, self._pending = self._pending, None
        return command

    def acknowledge(self, command, block_index):
        self.on_ack(dict(type="switch", command_id=command["id"], mode=command["mode"],
                         prompt=command["prompt"], block=block_index,
                         response_seconds=time.perf_counter() - command["submitted"]))
