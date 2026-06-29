"""Detector result structures, matching helpers, and mock detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from lvfind.intent.object_vocab import ObjectVocabulary


BBox = tuple[float, float, float, float]


class DetectorError(RuntimeError):
    """Base class for detector failures."""


class DetectorUnavailableError(DetectorError):
    """Raised when an optional detector dependency or model is unavailable."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector result in image coordinates."""

    label: str
    confidence: float
    bbox: BBox
    mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four values.")
        x1, y1, x2, y2 = (float(value) for value in self.bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in the range [0, 1].")
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))
        object.__setattr__(self, "confidence", float(self.confidence))


class ObjectDetector(Protocol):
    """Common detector interface."""

    def detect(self, image: Any) -> list[Detection]:
        """Return detections for one image."""


class MockDetector:
    """Fast deterministic detector for early demos and tests."""

    def __init__(self, detections: Sequence[Detection] | None = None) -> None:
        self._detections = list(detections or ())

    def detect(self, image: Any) -> list[Detection]:
        """Return configured detections without inspecting the image."""

        return list(self._detections)


def detector_labels_for_target(
    vocabulary: ObjectVocabulary,
    canonical_target: str,
) -> tuple[str, ...]:
    """Return detector labels configured for a canonical target."""

    return vocabulary.detector_labels_for(canonical_target)


def filter_detections_by_labels(
    detections: Sequence[Detection],
    detector_labels: Sequence[str],
) -> list[Detection]:
    """Keep detections whose label is in the target detector-label set."""

    allowed_labels = {_normalize_label(label) for label in detector_labels}
    if not allowed_labels:
        return []
    return [
        detection
        for detection in detections
        if _normalize_label(detection.label) in allowed_labels
    ]


def best_detection(detections: Sequence[Detection]) -> Detection | None:
    """Return the highest-confidence detection, or None."""

    if not detections:
        return None
    return max(detections, key=lambda detection: detection.confidence)


def _normalize_label(label: str) -> str:
    return " ".join(label.casefold().split())
