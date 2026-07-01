"""Wake word detector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class WakeWordUnavailableError(RuntimeError):
    """Raised when a wake word engine dependency or model is unavailable."""


class WakeWordDetector(Protocol):
    """Common wake word detector interface."""

    def detect(self, audio_frame: Any) -> bool:
        """Return whether the wake word was detected in one audio frame."""


@dataclass(frozen=True, slots=True)
class WakeWordConfig:
    """Wake word engine configuration."""

    engine: str = "manual"
    model_path: str | None = None
    threshold: float = 0.5

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> WakeWordConfig:
        data = data or {}
        return cls(
            engine=str(data.get("engine", "manual")),
            model_path=data.get("model_path"),
            threshold=float(data.get("threshold", 0.5)),
        )


class ManualWakeWordDetector:
    """A deterministic wake word detector for demos and tests."""

    def detect(self, audio_frame: Any) -> bool:
        """Always treat the next command as awake."""

        return True


class OpenWakeWordDetector:
    """openWakeWord adapter loaded only when explicitly configured."""

    def __init__(self, config: WakeWordConfig) -> None:
        if config.model_path is not None and not Path(config.model_path).exists():
            raise WakeWordUnavailableError(f"openWakeWord model file not found: {config.model_path}")

        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise WakeWordUnavailableError(
                "openWakeWord is required for wake word detection. Install it separately "
                "and set voice.wake_word.model_path in configs/app.yaml."
            ) from exc

        kwargs: dict[str, Any] = {}
        if config.model_path is not None:
            kwargs["wakeword_models"] = [config.model_path]
        self._model = Model(**kwargs)
        self._threshold = config.threshold

    def detect(self, audio_frame: Any) -> bool:
        predictions = self._model.predict(audio_frame)
        return any(float(score) >= self._threshold for score in predictions.values())


def build_wake_word_detector(config: WakeWordConfig) -> WakeWordDetector:
    """Create a wake word detector from config."""

    engine = config.engine.casefold()
    if engine in {"manual", "none"}:
        return ManualWakeWordDetector()
    if engine == "openwakeword":
        return OpenWakeWordDetector(config)
    raise WakeWordUnavailableError(f"Unsupported wake word engine: {config.engine}")
