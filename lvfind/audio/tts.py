"""Non-blocking text output / TTS queue."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Full, Queue
from threading import Thread


class TTSUnavailableError(RuntimeError):
    """Raised when a TTS engine is unavailable."""


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """TTS/output configuration."""

    engine: str = "terminal"
    queue_size: int = 8

    @classmethod
    def from_mapping(cls, data: dict[str, object] | None) -> TTSConfig:
        data = data or {}
        return cls(
            engine=str(data.get("engine", "terminal")),
            queue_size=int(data.get("queue_size", 8)),
        )


class QueuedTextSpeaker:
    """A non-blocking text output worker suitable for TTS adapters."""

    _STOP = object()

    def __init__(
        self,
        sink: Callable[[str], None] = print,
        queue_size: int = 8,
    ) -> None:
        self._sink = sink
        self._queue: Queue[str | object] = Queue(maxsize=queue_size)
        self._worker: Thread | None = None

    def start(self) -> None:
        """Start the background output worker."""

        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(target=self._run, name="lvfind-tts", daemon=True)
        self._worker.start()

    def say(self, message: str) -> bool:
        """Queue a message without blocking. Returns False if the queue is full."""

        self.start()
        try:
            self._queue.put_nowait(message)
        except Full:
            return False
        return True

    def stop(self, drain: bool = True) -> None:
        """Stop the worker, optionally waiting for queued messages first."""

        if self._worker is None:
            return
        if drain:
            self._queue.join()
        self._queue.put(self._STOP)
        self._worker.join(timeout=2)
        self._worker = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._sink(str(item))
            finally:
                self._queue.task_done()


def build_tts(config: TTSConfig) -> QueuedTextSpeaker:
    """Create a non-blocking text speaker from config."""

    engine = config.engine.casefold()
    if engine in {"terminal", "text", "mock"}:
        return QueuedTextSpeaker(queue_size=config.queue_size)
    raise TTSUnavailableError(f"Unsupported TTS engine: {config.engine}")
