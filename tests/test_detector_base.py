import unittest
from pathlib import Path

from lvfind.intent.object_vocab import ObjectVocabulary
from lvfind.vision.detector_base import (
    Detection,
    DetectorUnavailableError,
    MockDetector,
    best_detection,
    detector_labels_for_target,
    filter_detections_by_labels,
)
from lvfind.vision.yolo_detector import YOLODetector, YOLODetectorConfig


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "objects_zh.yaml"


class DetectorBaseTests(unittest.TestCase):
    def test_detection_validates_result_shape(self) -> None:
        detection = Detection(label="cup", confidence=0.8, bbox=(1, 2, 3, 4))

        self.assertEqual(detection.label, "cup")
        self.assertEqual(detection.confidence, 0.8)
        self.assertEqual(detection.bbox, (1.0, 2.0, 3.0, 4.0))

    def test_detection_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            Detection(label="cup", confidence=1.2, bbox=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            Detection(label="cup", confidence=0.8, bbox=(1, 2, 1, 4))

    def test_target_labels_come_from_vocabulary(self) -> None:
        vocab = ObjectVocabulary.from_yaml(CONFIG_PATH)

        self.assertEqual(detector_labels_for_target(vocab, "手机"), ("cell phone",))
        self.assertEqual(detector_labels_for_target(vocab, "杯子"), ("cup",))
        self.assertEqual(detector_labels_for_target(vocab, "未知"), ())

    def test_filters_detections_by_target_labels(self) -> None:
        detections = [
            Detection(label="Cup", confidence=0.7, bbox=(0, 0, 10, 10)),
            Detection(label="cell phone", confidence=0.9, bbox=(20, 0, 30, 10)),
            Detection(label="book", confidence=0.6, bbox=(40, 0, 50, 10)),
        ]

        filtered = filter_detections_by_labels(detections, ("cup", "bottle"))

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].label, "Cup")

    def test_best_detection_returns_highest_confidence(self) -> None:
        low = Detection(label="cup", confidence=0.4, bbox=(0, 0, 10, 10))
        high = Detection(label="cup", confidence=0.9, bbox=(20, 0, 30, 10))

        self.assertIs(best_detection([low, high]), high)
        self.assertIsNone(best_detection([]))

    def test_mock_detector_returns_configured_detections(self) -> None:
        detection = Detection(label="cup", confidence=0.7, bbox=(0, 0, 10, 10))
        detector = MockDetector([detection])

        self.assertEqual(detector.detect(image=None), [detection])

    def test_yolo_missing_model_fails_clearly(self) -> None:
        with self.assertRaisesRegex(DetectorUnavailableError, "YOLO model file not found"):
            YOLODetector(YOLODetectorConfig(model_path="/tmp/not-a-real-yolo-model.onnx"))


if __name__ == "__main__":
    unittest.main()
