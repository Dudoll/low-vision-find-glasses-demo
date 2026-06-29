import unittest
from pathlib import Path

from lvfind.intent import FIND_OBJECT_INTENT, NO_MATCH_INTENT
from lvfind.intent.object_vocab import ObjectVocabulary
from lvfind.intent.parser import parse_find_object_intent


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "objects_zh.yaml"


class IntentParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocab = ObjectVocabulary.from_yaml(CONFIG_PATH)

    def assertFinds(self, text: str, target: str, raw_target: str | None = None) -> None:
        result = parse_find_object_intent(text, self.vocab)

        self.assertEqual(result.intent, FIND_OBJECT_INTENT)
        self.assertEqual(result.target, target)
        self.assertEqual(result.raw_target, raw_target or target)
        self.assertTrue(result.is_match)

    def assertNoMatch(self, text: str) -> None:
        result = parse_find_object_intent(text, self.vocab)

        self.assertEqual(result.intent, NO_MATCH_INTENT)
        self.assertIsNone(result.target)
        self.assertIsNone(result.raw_target)
        self.assertFalse(result.is_match)

    def test_phase_examples_parse(self) -> None:
        self.assertFinds("帮我找手机", "手机")
        self.assertFinds("找一下杯子", "杯子")
        self.assertFinds("我要找遥控器", "遥控器")
        self.assertFinds("手机在哪里", "手机")
        self.assertFinds("帮我看看杯子在哪里", "杯子")

    def test_additional_chinese_commands_parse(self) -> None:
        self.assertFinds("请帮我找一下水杯", "杯子", "水杯")
        self.assertFinds("找找我的鼠标", "鼠标")
        self.assertFinds("电脑在哪儿", "笔记本电脑", "电脑")
        self.assertFinds("键盘在哪里？", "键盘")
        self.assertFinds("背包在哪", "背包")
        self.assertFinds("帮我找餐桌", "桌子", "餐桌")
        self.assertFinds("床铺在哪里", "床", "床铺")

    def test_direct_known_object_text_is_supported(self) -> None:
        self.assertFinds("手机", "手机")
        self.assertFinds("水杯", "杯子", "水杯")

    def test_result_can_be_serialized_to_dict(self) -> None:
        result = parse_find_object_intent("帮我找手机", self.vocab)

        self.assertEqual(
            result.to_dict(),
            {"intent": "find_object", "target": "手机", "raw_target": "手机"},
        )

    def test_unknown_object_returns_no_match(self) -> None:
        self.assertNoMatch("帮我找火星基地")

    def test_unsupported_text_returns_no_match(self) -> None:
        self.assertNoMatch("今天天气怎么样")
        self.assertNoMatch("停止寻找")


if __name__ == "__main__":
    unittest.main()
