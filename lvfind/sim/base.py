"""Simulator frame source primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class FramePacket:
    """One camera frame plus optional simulator side channels."""

    frame_id: int
    timestamp_ms: float
    rgb: Any | None = None
    depth: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulatorFrameSource(Protocol):
    """Common interface for simulator-backed camera sources."""

    def start(self) -> None:
        """Start the simulator runtime."""

    def stop(self) -> None:
        """Stop the simulator runtime and release resources."""

    def get_latest_frame(self) -> FramePacket:
        """Capture and return the latest simulator frame."""
