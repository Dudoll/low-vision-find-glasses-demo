import unittest
from pathlib import Path

from lvfind.cue.policy import (
    FAR,
    MID,
    NEAR,
    TOO_FAR,
    VERY_NEAR,
    CuePolicy,
    CuePolicyConfig,
    distance_band_for,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cue_policy.yaml"


class CuePolicyTests(unittest.TestCase):
    def test_distance_bands(self) -> None:
        self.assertEqual(distance_band_for(0.4), VERY_NEAR)
        self.assertEqual(distance_band_for(0.5), NEAR)
        self.assertEqual(distance_band_for(1.0), MID)
        self.assertEqual(distance_band_for(2.0), FAR)
        self.assertEqual(distance_band_for(3.0), TOO_FAR)
        self.assertIsNone(distance_band_for(None))

    def test_loads_config_from_yaml(self) -> None:
        config = CuePolicyConfig.from_yaml(CONFIG_PATH)

        self.assertEqual(config.min_interval_ms, 800)
        self.assertEqual(config.same_message_interval_ms, 2000)
        self.assertEqual(config.distance_change_threshold_m, 0.3)
        self.assertIn("direction_changed", config.messages)

    def test_first_detection_emits_immediately(self) -> None:
        policy = CuePolicy()

        decision = policy.update_detection("手机", "front", 1.2, timestamp_ms=0)

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "first_found")
        self.assertEqual(decision.distance_band, MID)
        self.assertIn("手机", decision.message or "")
        self.assertIn("正前方", decision.message or "")

    def test_identical_detection_is_throttled_then_repeated(self) -> None:
        policy = CuePolicy()
        policy.update_detection("手机", "front", 1.2, timestamp_ms=0)

        suppressed = policy.update_detection("手机", "front", 1.2, timestamp_ms=1000)
        repeated = policy.update_detection("手机", "front", 1.2, timestamp_ms=2500)

        self.assertFalse(suppressed.should_emit)
        self.assertTrue(repeated.should_emit)
        self.assertEqual(repeated.reason, "repeat")

    def test_direction_change_emits(self) -> None:
        policy = CuePolicy()
        policy.update_detection("手机", "front", 1.2, timestamp_ms=0)

        decision = policy.update_detection("手机", "left", 1.2, timestamp_ms=200)

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "direction_changed")
        self.assertIn("左边", decision.message or "")

    def test_significantly_closer_emits(self) -> None:
        policy = CuePolicy()
        policy.update_detection("手机", "front", 1.4, timestamp_ms=0)

        decision = policy.update_detection("手机", "front", 1.0, timestamp_ms=200)

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "closer")

    def test_very_near_closer_message_uses_very_near_reason(self) -> None:
        policy = CuePolicy()
        policy.update_detection("手机", "front", 0.9, timestamp_ms=0)

        decision = policy.update_detection("手机", "front", 0.4, timestamp_ms=200)

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "very_near")
        self.assertEqual(decision.distance_band, VERY_NEAR)

    def test_new_target_emits_immediately(self) -> None:
        policy = CuePolicy()
        policy.update_detection("手机", "front", 1.2, timestamp_ms=0)

        decision = policy.update_detection("杯子", "front", 1.2, timestamp_ms=100)

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "first_found")


if __name__ == "__main__":
    unittest.main()
