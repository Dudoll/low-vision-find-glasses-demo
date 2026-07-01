import unittest
from pathlib import Path

from lvfind.intent.object_vocab import ObjectVocabulary
from lvfind.intent.parser import (
    CHANGE_TARGET_INTENT,
    FIND_OBJECT_INTENT,
    NO_MATCH_INTENT,
    RESTART_INTENT,
    STOP_SEARCH_INTENT,
    parse_voice_command,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "objects_zh.yaml"


class VoiceCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocab = ObjectVocabulary.from_yaml(CONFIG_PATH)

    def test_find_object_commands_reuse_intent_parser(self) -> None:
        command = parse_voice_command("帮我找手机", self.vocab)

        self.assertEqual(command.intent, FIND_OBJECT_INTENT)
        self.assertEqual(command.target, "手机")
        self.assertEqual(command.raw_target, "手机")

    def test_stop_command(self) -> None:
        command = parse_voice_command("停止寻找", self.vocab)

        self.assertEqual(command.intent, STOP_SEARCH_INTENT)
        self.assertIsNone(command.target)

    def test_change_target_command(self) -> None:
        command = parse_voice_command("换一个目标", self.vocab)

        self.assertEqual(command.intent, CHANGE_TARGET_INTENT)

    def test_restart_command(self) -> None:
        command = parse_voice_command("重新开始", self.vocab)

        self.assertEqual(command.intent, RESTART_INTENT)

    def test_unsupported_command(self) -> None:
        command = parse_voice_command("今天天气怎么样", self.vocab)

        self.assertEqual(command.intent, NO_MATCH_INTENT)
        self.assertFalse(command.is_match)


if __name__ == "__main__":
    unittest.main()
