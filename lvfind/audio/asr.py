"""ASR adapters for voice commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol


class ASRUnavailableError(RuntimeError):
    """Raised when an ASR engine dependency, binary, or model is unavailable."""


@dataclass(frozen=True, slots=True)
class ASRResult:
    """One ASR transcription result."""

    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ASREngine(Protocol):
    """Common ASR interface."""

    def transcribe(self, audio: Any) -> ASRResult:
        """Transcribe audio into text."""


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """ASR engine configuration."""

    engine: str = "stdin"
    model_path: str | None = None
    sample_rate: int = 16_000
    whisper_cpp_binary: str | None = None
    language: str | None = "zh"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ASRConfig:
        data = data or {}
        return cls(
            engine=str(data.get("engine", "stdin")),
            model_path=data.get("model_path"),
            sample_rate=int(data.get("sample_rate", 16_000)),
            whisper_cpp_binary=data.get("whisper_cpp_binary"),
            language=data.get("language", "zh"),
        )


class TextASR:
    """ASR adapter for already-transcribed text."""

    def transcribe(self, audio: Any) -> ASRResult:
        return ASRResult(text=str(audio).strip())


class VoskASR:
    """Vosk streaming ASR adapter."""

    def __init__(self, config: ASRConfig) -> None:
        if config.model_path is None or not Path(config.model_path).exists():
            raise ASRUnavailableError(
                "Vosk model file or directory not found. Set voice.asr.model_path in configs/app.yaml."
            )

        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise ASRUnavailableError(
                "Vosk is required for this ASR engine. Install vosk and configure "
                "voice.asr.model_path."
            ) from exc

        self._recognizer = KaldiRecognizer(Model(config.model_path), config.sample_rate)

    def transcribe(self, audio: Any) -> ASRResult:
        if not isinstance(audio, bytes):
            raise TypeError("VoskASR expects 16-bit PCM audio bytes.")

        self._recognizer.AcceptWaveform(audio)
        payload = json.loads(self._recognizer.Result() or "{}")
        return ASRResult(
            text=str(payload.get("text", "")).strip(),
            confidence=payload.get("confidence"),
            metadata=payload,
        )


class WhisperCppASR:
    """whisper.cpp command-line ASR adapter."""

    def __init__(self, config: ASRConfig) -> None:
        if config.whisper_cpp_binary is None or not Path(config.whisper_cpp_binary).exists():
            raise ASRUnavailableError(
                "whisper.cpp binary not found. Set voice.asr.whisper_cpp_binary in configs/app.yaml."
            )
        if config.model_path is None or not Path(config.model_path).exists():
            raise ASRUnavailableError(
                "whisper.cpp model file not found. Set voice.asr.model_path in configs/app.yaml."
            )

        self._binary = config.whisper_cpp_binary
        self._model_path = config.model_path
        self._language = config.language

    def transcribe(self, audio: Any) -> ASRResult:
        audio_path = Path(str(audio))
        if not audio_path.exists():
            raise ASRUnavailableError(f"Audio file not found for whisper.cpp: {audio_path}")

        command = [self._binary, "-m", self._model_path, "-f", str(audio_path), "-nt"]
        if self._language:
            command.extend(["-l", self._language])
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return ASRResult(text=completed.stdout.strip(), metadata={"stderr": completed.stderr})


def build_asr(config: ASRConfig) -> ASREngine:
    """Create an ASR engine from config."""

    engine = config.engine.casefold()
    if engine in {"stdin", "text", "manual"}:
        return TextASR()
    if engine == "vosk":
        return VoskASR(config)
    if engine in {"whisper_cpp", "whisper.cpp"}:
        return WhisperCppASR(config)
    raise ASRUnavailableError(f"Unsupported ASR engine: {config.engine}")
