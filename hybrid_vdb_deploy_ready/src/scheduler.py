from __future__ import annotations
import threading
import time
from typing import Callable


class RepeatedJob(threading.Thread):
    """Very small utility to run `fn` every `interval_sec` seconds."""

    def __init__(self, interval_sec: float, fn: Callable[[], None]):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.fn = fn
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.interval)
            try:
                self.fn()
            except Exception:
                # In a demo we don't want to crash the whole process; in a
                # real system this would be logged properly.
                pass

    def stop(self) -> None:
        self._stop.set()
