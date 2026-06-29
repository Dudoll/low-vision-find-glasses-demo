"""Stateful, non-blocking cue policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing dependency installs.
    yaml = None


VERY_NEAR = "very_near"
NEAR = "near"
MID = "mid"
FAR = "far"
TOO_FAR = "too_far"

DIRECTION_LABELS = {
    "left": "左边",
    "front": "正前方",
    "right": "右边",
}


@dataclass(frozen=True, slots=True)
class DistanceBandConfig:
    """Distance thresholds for cue urgency."""

    very_near_m: float = 0.5
    near_m: float = 1.0
    mid_m: float = 2.0
    far_m: float = 3.0


@dataclass(frozen=True, slots=True)
class CuePolicyConfig:
    """Configurable cue policy thresholds and message templates."""

    min_interval_ms: int = 800
    same_message_interval_ms: int = 2000
    distance_change_threshold_m: float = 0.3
    distance_bands: DistanceBandConfig = field(default_factory=DistanceBandConfig)
    messages: dict[str, str] = field(
        default_factory=lambda: {
            "first_found": "发现{target}，在{direction}，约{distance}米",
            "closer": "{target}更近了，在{direction}",
            "very_near": "{target}很近，在{direction}",
            "direction_changed": "{target}在{direction}",
            "lost": "暂时没有看到{target}",
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CuePolicyConfig:
        """Load cue policy config from YAML."""

        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load cue policy YAML. "
                "Install project dependencies with `python -m pip install -e .`."
            )

        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> CuePolicyConfig:
        """Build cue policy config from parsed YAML-like data."""

        cue = data.get("cue", {})
        bands = data.get("distance_bands", {})
        messages = cls().messages | dict(data.get("messages", {}))
        return cls(
            min_interval_ms=int(cue.get("min_interval_ms", 800)),
            same_message_interval_ms=int(cue.get("same_message_interval_ms", 2000)),
            distance_change_threshold_m=float(cue.get("distance_change_threshold_m", 0.3)),
            distance_bands=DistanceBandConfig(
                very_near_m=float(bands.get("very_near_m", 0.5)),
                near_m=float(bands.get("near_m", 1.0)),
                mid_m=float(bands.get("mid_m", 2.0)),
                far_m=float(bands.get("far_m", 3.0)),
            ),
            messages=messages,
        )


@dataclass(frozen=True, slots=True)
class CueDecision:
    """A non-blocking cue decision returned to output adapters."""

    should_emit: bool
    message: str | None = None
    reason: str | None = None
    distance_band: str | None = None


@dataclass(frozen=True, slots=True)
class _EmittedCueState:
    target: str
    direction: str
    distance_m: float | None
    distance_band: str | None
    timestamp_ms: float


class CuePolicy:
    """Stateful cue throttling for target detections."""

    def __init__(self, config: CuePolicyConfig | None = None) -> None:
        self.config = config or CuePolicyConfig()
        self._last_emitted: _EmittedCueState | None = None

    def reset(self) -> None:
        """Forget previous cue state."""

        self._last_emitted = None

    def update_detection(
        self,
        target: str,
        direction: str,
        distance_m: float | None,
        timestamp_ms: float | None = None,
    ) -> CueDecision:
        """Return whether a detection should emit a cue."""

        now_ms = _now_ms() if timestamp_ms is None else float(timestamp_ms)
        distance_band = distance_band_for(distance_m, self.config.distance_bands)
        reason = self._reason_to_emit(target, direction, distance_m, distance_band, now_ms)

        if reason is None:
            return CueDecision(should_emit=False, distance_band=distance_band)

        self._last_emitted = _EmittedCueState(
            target=target,
            direction=direction,
            distance_m=distance_m,
            distance_band=distance_band,
            timestamp_ms=now_ms,
        )
        return CueDecision(
            should_emit=True,
            message=self._format_message(reason, target, direction, distance_m),
            reason=reason,
            distance_band=distance_band,
        )

    def _reason_to_emit(
        self,
        target: str,
        direction: str,
        distance_m: float | None,
        distance_band: str | None,
        now_ms: float,
    ) -> str | None:
        last = self._last_emitted
        if last is None or last.target != target:
            return "first_found"
        if direction != last.direction:
            return "direction_changed"
        if _is_significantly_closer(
            previous_distance_m=last.distance_m,
            current_distance_m=distance_m,
            threshold_m=self.config.distance_change_threshold_m,
        ):
            if distance_band == VERY_NEAR:
                return "very_near"
            return "closer"
        if _is_repeat_due(last, direction, distance_band, now_ms, self.config):
            return "repeat"
        return None

    def _format_message(
        self,
        reason: str,
        target: str,
        direction: str,
        distance_m: float | None,
    ) -> str:
        message_key = "first_found" if reason == "repeat" else reason
        template = self.config.messages.get(message_key, self.config.messages["first_found"])
        return template.format(
            target=target,
            direction=DIRECTION_LABELS.get(direction, direction),
            distance=_format_distance(distance_m),
        )


def distance_band_for(
    distance_m: float | None,
    config: DistanceBandConfig | None = None,
) -> str | None:
    """Return the configured distance band for a distance in meters."""

    if distance_m is None:
        return None

    bands = config or DistanceBandConfig()
    if distance_m < bands.very_near_m:
        return VERY_NEAR
    if distance_m < bands.near_m:
        return NEAR
    if distance_m < bands.mid_m:
        return MID
    if distance_m < bands.far_m:
        return FAR
    return TOO_FAR


def _is_significantly_closer(
    previous_distance_m: float | None,
    current_distance_m: float | None,
    threshold_m: float,
) -> bool:
    if previous_distance_m is None or current_distance_m is None:
        return False
    return previous_distance_m - current_distance_m >= threshold_m


def _is_repeat_due(
    last: _EmittedCueState,
    direction: str,
    distance_band: str | None,
    now_ms: float,
    config: CuePolicyConfig,
) -> bool:
    if direction != last.direction or distance_band != last.distance_band:
        return False

    elapsed_ms = now_ms - last.timestamp_ms
    return (
        elapsed_ms >= config.same_message_interval_ms
        and elapsed_ms >= config.min_interval_ms
    )


def _format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "未知"
    return f"{distance_m:.1f}"


def _now_ms() -> float:
    return time.monotonic_ns() / 1_000_000
