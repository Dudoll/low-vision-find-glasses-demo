"""Benchmark metric records, writers, and summary calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from lvfind.cue.policy import distance_band_for


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FrameMetricRecord:
    """Per-frame benchmark metrics."""

    frame_id: int
    timestamp_ms: float
    target: str
    detected: bool
    confidence: float | None = None
    bbox: BBox | None = None
    distance_m: float | None = None
    direction: str | None = None
    latency_total_ms: float = 0.0
    latency_capture_ms: float = 0.0
    latency_preprocess_ms: float = 0.0
    latency_inference_ms: float = 0.0
    latency_postprocess_ms: float = 0.0
    latency_distance_ms: float = 0.0
    latency_cue_ms: float = 0.0
    stale_frame_drop_count: int = 0
    expected_present: bool | None = None
    expected_direction: str | None = None
    expected_distance_band: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Aggregate benchmark metrics."""

    frame_count: int
    detected_count: int
    expected_present_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    fps_avg: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    stale_frame_count: int
    recall: float | None
    precision: float | None
    false_positives_per_minute: float | None
    direction_accuracy: float | None
    distance_band_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV/JSON-friendly summary mapping."""

        return asdict(self)


def summarize_frame_metrics(records: Sequence[FrameMetricRecord]) -> BenchmarkSummary:
    """Compute benchmark summary metrics from per-frame records."""

    frame_count = len(records)
    detected_records = [record for record in records if record.detected]
    expected_present_records = [record for record in records if record.expected_present is True]
    true_positive_records = [
        record for record in records if record.detected and record.expected_present is True
    ]
    false_positive_records = [
        record for record in records if record.detected and record.expected_present is False
    ]
    false_negative_records = [
        record for record in records if not record.detected and record.expected_present is True
    ]

    latencies = [record.latency_total_ms for record in records]
    duration_s = _duration_s(records)
    direction_accuracy = _direction_accuracy(true_positive_records)
    distance_band_accuracy = _distance_band_accuracy(true_positive_records)

    return BenchmarkSummary(
        frame_count=frame_count,
        detected_count=len(detected_records),
        expected_present_count=len(expected_present_records),
        true_positive_count=len(true_positive_records),
        false_positive_count=len(false_positive_records),
        false_negative_count=len(false_negative_records),
        fps_avg=(frame_count - 1) / duration_s if duration_s and frame_count > 1 else None,
        latency_p50_ms=percentile_nearest_rank(latencies, 50),
        latency_p95_ms=percentile_nearest_rank(latencies, 95),
        latency_p99_ms=percentile_nearest_rank(latencies, 99),
        stale_frame_count=max((record.stale_frame_drop_count for record in records), default=0),
        recall=_safe_ratio(len(true_positive_records), len(expected_present_records)),
        precision=_safe_ratio(len(true_positive_records), len(detected_records)),
        false_positives_per_minute=(
            len(false_positive_records) / (duration_s / 60.0) if duration_s else None
        ),
        direction_accuracy=direction_accuracy,
        distance_band_accuracy=distance_band_accuracy,
    )


def percentile_nearest_rank(values: Sequence[float], percentile: int) -> float | None:
    """Return nearest-rank percentile for a non-empty sequence."""

    if not values:
        return None
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in the range [0, 100].")

    sorted_values = sorted(float(value) for value in values)
    if percentile == 0:
        return sorted_values[0]

    rank = max(1, _ceil(percentile / 100.0 * len(sorted_values)))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def write_metrics_jsonl(path: str | Path, records: Iterable[FrameMetricRecord]) -> None:
    """Write per-frame metrics as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def read_metrics_jsonl(path: str | Path) -> list[FrameMetricRecord]:
    """Read JSONL metrics back into frame records."""

    records: list[FrameMetricRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            bbox = payload.get("bbox")
            if bbox is not None:
                payload["bbox"] = tuple(bbox)
            records.append(FrameMetricRecord(**payload))
    return records


def write_summary_csv(path: str | Path, summary: BenchmarkSummary) -> None:
    """Write a one-row benchmark summary CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = summary.to_dict()
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_data))
        writer.writeheader()
        writer.writerow(summary_data)


def write_config_snapshot(path: str | Path, config_text: str) -> None:
    """Write the benchmark configuration snapshot."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_text, encoding="utf-8")


def _duration_s(records: Sequence[FrameMetricRecord]) -> float | None:
    if len(records) < 2:
        return None
    duration_ms = records[-1].timestamp_ms - records[0].timestamp_ms
    if duration_ms <= 0:
        return None
    return duration_ms / 1000.0


def _direction_accuracy(records: Sequence[FrameMetricRecord]) -> float | None:
    evaluable = [
        record
        for record in records
        if record.expected_direction is not None and record.direction is not None
    ]
    if not evaluable:
        return None
    correct = sum(record.direction == record.expected_direction for record in evaluable)
    return correct / len(evaluable)


def _distance_band_accuracy(records: Sequence[FrameMetricRecord]) -> float | None:
    evaluable = [
        record
        for record in records
        if record.expected_distance_band is not None and record.distance_m is not None
    ]
    if not evaluable:
        return None
    correct = sum(
        distance_band_for(record.distance_m) == record.expected_distance_band
        for record in evaluable
    )
    return correct / len(evaluable)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1
