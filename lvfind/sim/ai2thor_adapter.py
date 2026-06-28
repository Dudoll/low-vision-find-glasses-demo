"""AI2-THOR simulator adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from lvfind.sim.base import FramePacket


class AI2ThorUnavailableError(RuntimeError):
    """Raised when the optional AI2-THOR dependency is not installed."""


class AI2ThorStartupError(RuntimeError):
    """Raised when AI2-THOR is installed but cannot start."""


@dataclass(frozen=True, slots=True)
class AI2ThorConfig:
    """Configuration for the AI2-THOR frame source."""

    scene: str = "FloorPlan201"
    width: int = 640
    height: int = 480
    fov: int = 90
    fps: int = 5
    render_depth: bool = True
    render_class_image: bool = False
    render_object_image: bool = False


class AI2ThorFrameSource:
    """Capture first-person RGB and optional depth frames from AI2-THOR."""

    def __init__(self, config: AI2ThorConfig) -> None:
        self.config = config
        self._controller: Any | None = None
        self._frame_id = 0

    def start(self) -> None:
        """Start AI2-THOR for the configured scene."""

        controller_cls = _load_controller_class()
        try:
            self._controller = controller_cls(
                scene=self.config.scene,
                width=self.config.width,
                height=self.config.height,
                fieldOfView=self.config.fov,
                renderDepthImage=self.config.render_depth,
                renderClassImage=self.config.render_class_image,
                renderObjectImage=self.config.render_object_image,
            )
        except Exception as exc:
            message = (
                "AI2-THOR is installed but failed to start. If this happened while "
                "downloading the simulator build, rerun the same command; AI2-THOR "
                "will retry its Linux build download. Original error: "
                f"{type(exc).__name__}: {exc}"
            )
            raise AI2ThorStartupError(message) from exc

    def stop(self) -> None:
        """Stop AI2-THOR if it was started."""

        if self._controller is None:
            return
        self._controller.stop()
        self._controller = None

    def get_latest_frame(self) -> FramePacket:
        """Advance the simulator by one passive step and return a frame packet."""

        if self._controller is None:
            raise RuntimeError("AI2-THOR frame source has not been started. Call start() first.")

        event = self._controller.step(action="Pass")
        self._frame_id += 1
        return FramePacket(
            frame_id=self._frame_id,
            timestamp_ms=time.time_ns() / 1_000_000,
            rgb=getattr(event, "frame", None),
            depth=getattr(event, "depth_frame", None),
            metadata=getattr(event, "metadata", {}) or {},
        )


def _load_controller_class() -> Any:
    try:
        from ai2thor.controller import Controller
    except ImportError as exc:
        message = (
            "AI2-THOR is required for the simulator demo but is not installed. "
            "Install it with `python -m pip install '.[sim]'` from the repository root, "
            "or install directly with `python -m pip install ai2thor`."
        )
        raise AI2ThorUnavailableError(message) from exc
    return Controller
