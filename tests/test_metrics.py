import csv
import unittest

from lvfind.pipeline.metrics import (
    FrameMetricRecord,
    percentile_nearest_rank,
    read_metrics_jsonl,
    summarize_frame_metrics,
    write_metrics_jsonl,
    write_summary_csv,
)


class MetricsTests(unittest.TestCase):
    def test_percentile_nearest_rank(self) -> None:
        self.assertEqual(percentile_nearest_rank([30, 10, 20], 50), 20.0)
        self.assertEqual(percentile_nearest_rank([10, 20, 30, 40], 95), 40.0)
        self.assertIsNone(percentile_nearest_rank([], 50))

    def test_summary_metrics(self) -> None:
        records = [
            FrameMetricRecord(
                frame_id=1,
                timestamp_ms=0,
                target="杯子",
                detected=True,
                confidence=0.9,
                bbox=(0, 0, 10, 10),
                distance_m=0.8,
                direction="front",
                latency_total_ms=20,
                stale_frame_drop_count=0,
                expected_present=True,
                expected_direction="front",
                expected_distance_band="near",
            ),
            FrameMetricRecord(
                frame_id=2,
                timestamp_ms=200,
                target="杯子",
                detected=False,
                latency_total_ms=30,
                stale_frame_drop_count=1,
                expected_present=True,
            ),
            FrameMetricRecord(
                frame_id=3,
                timestamp_ms=400,
                target="杯子",
                detected=True,
                confidence=0.4,
                bbox=(0, 0, 10, 10),
                latency_total_ms=40,
                stale_frame_drop_count=1,
                expected_present=False,
            ),
        ]

        summary = summarize_frame_metrics(records)

        self.assertEqual(summary.frame_count, 3)
        self.assertEqual(summary.true_positive_count, 1)
        self.assertEqual(summary.false_negative_count, 1)
        self.assertEqual(summary.false_positive_count, 1)
        self.assertEqual(summary.stale_frame_count, 1)
        self.assertEqual(summary.recall, 0.5)
        self.assertEqual(summary.precision, 0.5)
        self.assertEqual(summary.direction_accuracy, 1.0)
        self.assertEqual(summary.distance_band_accuracy, 1.0)
        self.assertEqual(summary.latency_p95_ms, 40.0)

    def test_jsonl_roundtrip_and_summary_csv(self) -> None:
        with self.subTest("jsonl"):
            records = [
                FrameMetricRecord(
                    frame_id=1,
                    timestamp_ms=0,
                    target="手机",
                    detected=True,
                    confidence=0.7,
                    bbox=(1, 2, 3, 4),
                    latency_total_ms=12,
                )
            ]
            jsonl_path = self.tmp_path("metrics.jsonl")
            write_metrics_jsonl(jsonl_path, records)

            loaded = read_metrics_jsonl(jsonl_path)
            self.assertEqual(loaded, records)

        with self.subTest("csv"):
            summary_path = self.tmp_path("summary.csv")
            write_summary_csv(summary_path, summarize_frame_metrics(records))

            with summary_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["frame_count"], "1")

    def tmp_path(self, name: str):
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.mkdtemp())
        return directory / name


if __name__ == "__main__":
    unittest.main()
