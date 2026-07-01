"""Audio, ASR, wake word, and TTS adapters."""

from lvfind.audio.asr import (
    ASRConfig,
    ASREngine,
    ASRResult,
    ASRUnavailableError,
    TextASR,
    VoskASR,
    WhisperCppASR,
    build_asr,
)
from lvfind.audio.tts import TTSConfig, TTSUnavailableError, QueuedTextSpeaker, build_tts
from lvfind.audio.wakeword import (
    ManualWakeWordDetector,
    OpenWakeWordDetector,
    WakeWordConfig,
    WakeWordDetector,
    WakeWordUnavailableError,
    build_wake_word_detector,
)

__all__ = [
    "ASRConfig",
    "ASREngine",
    "ASRResult",
    "ASRUnavailableError",
    "ManualWakeWordDetector",
    "OpenWakeWordDetector",
    "QueuedTextSpeaker",
    "TTSConfig",
    "TTSUnavailableError",
    "TextASR",
    "VoskASR",
    "WakeWordConfig",
    "WakeWordDetector",
    "WakeWordUnavailableError",
    "WhisperCppASR",
    "build_asr",
    "build_tts",
    "build_wake_word_detector",
]
