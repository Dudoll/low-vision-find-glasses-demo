import unittest
from pathlib import Path

from lvfind.intent.object_vocab import ObjectVocabulary, normalize_text


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "objects_zh.yaml"


class ObjectVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = ObjectVocabulary.from_yaml(CONFIG_PATH)

    def test_loads_canonical_objects_from_yaml(self) -> None:
        self.assertIn("手机", self.vocab.canonical_names)
        self.assertIn("杯子", self.vocab.canonical_names)
        self.assertIn("遥控器", self.vocab.canonical_names)

    def test_finds_canonical_name_by_chinese_alias(self) -> None:
        self.assertEqual(self.vocab.find_canonical("电话"), "手机")
        self.assertEqual(self.vocab.find_canonical("水杯"), "杯子")

    def test_finds_canonical_name_by_english_alias(self) -> None:
        self.assertEqual(self.vocab.find_canonical("PHONE"), "手机")
        self.assertEqual(self.vocab.find_canonical("remote control"), "遥控器")

    def test_returns_detector_labels_for_known_object(self) -> None:
        self.assertEqual(self.vocab.detector_labels_for("手机"), ("cell phone",))
        self.assertEqual(self.vocab.detector_labels_for("桌子"), ("dining table",))

    def test_unknown_object_returns_clean_empty_values(self) -> None:
        self.assertIsNone(self.vocab.find_canonical("火星基地"))
        self.assertEqual(self.vocab.detector_labels_for("火星基地"), ())

    def test_finds_longest_alias_in_text(self) -> None:
        match = self.vocab.find_alias_match("请帮我找笔记本电脑")

        self.assertIsNotNone(match)
        self.assertEqual(match.canonical_name, "笔记本电脑")
        self.assertEqual(match.alias, "笔记本电脑")

    def test_normalization_ignores_case_spaces_and_punctuation(self) -> None:
        self.assertEqual(normalize_text(" Remote Control! "), "remotecontrol")


if __name__ == "__main__":
    unittest.main()
