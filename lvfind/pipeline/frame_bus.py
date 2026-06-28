"""Latest-frame-only frame handoff."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from typing import Final

from lvfind.sim.base import FramePacket


@dataclass(frozen=True, slots=True)
class FrameBusStats:
    """Snapshot of frame bus counters."""

    stale_frames_dropped: int


class LatestFrameBus:
    """A capacity-one frame bus that replaces stale, unread frames."""

    capacity: Final[int] = 1

    def __init__(self) -> None:
        self._condition = Condition()
        self._frame: FramePacket | None = None
        self._stale_frames_dropped = 0

    @property
    def stale_frames_dropped(self) -> int:
        """Number of unread frames replaced by newer frames."""

        with self._condition:
            return self._stale_frames_dropped

    def stats(self) -> FrameBusStats:
        """Return a stable snapshot of current counters."""

        with self._condition:
            return FrameBusStats(stale_frames_dropped=self._stale_frames_dropped)

    def push(self, frame: FramePacket) -> None:
        """Publish a frame, replacing any unread frame already waiting."""

        with self._condition:
            if self._frame is not None:
                self._stale_frames_dropped += 1
            self._frame = frame
            self._condition.notify()

    def read(self) -> FramePacket | None:
        """Consume and return the latest frame, or None if the bus is empty."""

        with self._condition:
            frame = self._frame
            self._frame = None
            return frame

    def wait_for_frame(self, timeout_s: float | None = None) -> FramePacket | None:
        """Wait until a frame is available, then consume it."""

        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout=timeout_s)
            frame = self._frame
            self._frame = None
            return frame
