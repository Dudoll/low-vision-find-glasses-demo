import unittest

from lvfind.audio.asr import ASRConfig, ASRUnavailableError, TextASR, VoskASR, WhisperCppASR
from lvfind.audio.tts import QueuedTextSpeaker
from lvfind.audio.wakeword import (
    ManualWakeWordDetector,
    OpenWakeWordDetector,
    WakeWordConfig,
    WakeWordUnavailableError,
    build_wake_word_detector,
)


class AudioInterfaceTests(unittest.TestCase):
    def test_manual_wake_word_detector_always_detects(self) -> None:
        detector = ManualWakeWordDetector()

        self.assertTrue(detector.detect(audio_frame=None))
        self.assertIsInstance(build_wake_word_detector(WakeWordConfig()), ManualWakeWordDetector)

    def test_openwakeword_missing_model_fails_clearly(self) -> None:
        with self.assertRaisesRegex(WakeWordUnavailableError, "model file not found"):
            OpenWakeWordDetector(
                WakeWordConfig(engine="openwakeword", model_path="/tmp/not-a-wakeword.onnx")
            )

    def test_text_asr_returns_stripped_text(self) -> None:
        result = TextASR().transcribe("  帮我找手机  ")

        self.assertEqual(result.text, "帮我找手机")

    def test_vosk_missing_model_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ASRUnavailableError, "Vosk model"):
            VoskASR(ASRConfig(engine="vosk", model_path="/tmp/not-a-vosk-model"))

    def test_whisper_cpp_missing_binary_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ASRUnavailableError, "whisper.cpp binary"):
            WhisperCppASR(
                ASRConfig(
                    engine="whisper_cpp",
                    whisper_cpp_binary="/tmp/not-whisper-cpp",
                    model_path="/tmp/not-whisper-model.bin",
                )
            )

    def test_queued_text_speaker_is_nonblocking_and_drains(self) -> None:
        messages: list[str] = []
        speaker = QueuedTextSpeaker(sink=messages.append, queue_size=2)

        self.assertTrue(speaker.say("hello"))
        speaker.stop(drain=True)

        self.assertEqual(messages, ["hello"])


if __name__ == "__main__":
    unittest.main()
