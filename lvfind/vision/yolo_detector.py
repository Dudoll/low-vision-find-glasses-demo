"""Optional ONNXRuntime YOLO detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lvfind.vision.detector_base import Detection, DetectorUnavailableError


COCO_LABELS: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass(frozen=True, slots=True)
class YOLODetectorConfig:
    """Config for a YOLO ONNX detector."""

    model_path: str | Path
    input_size: int = 416
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    labels: tuple[str, ...] = field(default_factory=lambda: COCO_LABELS)


class YOLODetector:
    """YOLO detector backed by ONNXRuntime."""

    def __init__(self, config: YOLODetectorConfig) -> None:
        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise DetectorUnavailableError(
                f"YOLO model file not found: {model_path}. "
                "Set vision.model_path in configs/app.yaml or pass --model-path."
            )

        cv2, ort = _load_optional_dependencies()
        self._cv2 = cv2
        self._session = ort.InferenceSession(
            str(model_path),
            providers=_providers_for_device(config.device),
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: Any) -> list[Detection]:
        """Run YOLO inference and return detections in original image coordinates."""

        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[2] < 3:
            raise ValueError("YOLODetector expects an RGB image with shape HxWx3.")

        original_height, original_width = image_array.shape[:2]
        input_tensor = self._preprocess(image_array)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        candidates = _parse_yolo_output(
            outputs[0],
            labels=self.config.labels,
            conf_threshold=self.config.conf_threshold,
            original_width=original_width,
            original_height=original_height,
            input_size=self.config.input_size,
        )
        return _non_max_suppression(candidates, self.config.iou_threshold)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = self._cv2.resize(
            image[:, :, :3],
            (self.config.input_size, self.config.input_size),
            interpolation=self._cv2.INTER_LINEAR,
        )
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise DetectorUnavailableError(
            "OpenCV is required for YOLO detection. Install it with "
            "`python -m pip install -e '.[vision]'`."
        ) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DetectorUnavailableError(
            "ONNXRuntime is required for YOLO detection. Install it with "
            "`python -m pip install -e '.[vision]'`."
        ) from exc
    return cv2, ort


def _providers_for_device(device: str) -> list[str]:
    if device.casefold() == "cpu":
        return ["CPUExecutionProvider"]
    if device.casefold() in {"cuda", "gpu"}:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise DetectorUnavailableError(f"Unsupported YOLO device: {device!r}. Use 'cpu' or 'cuda'.")


def _parse_yolo_output(
    output: Any,
    labels: Sequence[str],
    conf_threshold: float,
    original_width: int,
    original_height: int,
    input_size: int,
) -> list[Detection]:
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise DetectorUnavailableError(f"Unsupported YOLO output shape: {predictions.shape}.")
    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= len(labels) + 5:
        predictions = predictions.T

    detections: list[Detection] = []
    feature_count = predictions.shape[1]
    has_objectness = feature_count != len(labels) + 4
    class_offset = 5 if has_objectness else 4
    if feature_count <= class_offset:
        raise DetectorUnavailableError(f"Unsupported YOLO output feature count: {feature_count}.")

    scale_x = original_width / float(input_size)
    scale_y = original_height / float(input_size)
    for prediction in predictions:
        class_scores = prediction[class_offset:]
        if class_scores.size == 0:
            continue
        class_id = int(np.argmax(class_scores))
        if class_id >= len(labels):
            continue

        score = float(class_scores[class_id])
        if has_objectness:
            score *= float(prediction[4])
        if score < conf_threshold:
            continue

        center_x, center_y, width, height = (float(value) for value in prediction[:4])
        x1 = max(0.0, (center_x - width / 2.0) * scale_x)
        y1 = max(0.0, (center_y - height / 2.0) * scale_y)
        x2 = min(float(original_width), (center_x + width / 2.0) * scale_x)
        y2 = min(float(original_height), (center_y + height / 2.0) * scale_y)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            Detection(
                label=labels[class_id],
                confidence=score,
                bbox=(x1, y1, x2, y2),
                metadata={"class_id": class_id},
            )
        )
    return detections


def _non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float,
) -> list[Detection]:
    if not detections:
        return []

    remaining = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            detection
            for detection in remaining
            if detection.label != current.label or _iou(current.bbox, detection.bbox) < iou_threshold
        ]
    return kept


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)
