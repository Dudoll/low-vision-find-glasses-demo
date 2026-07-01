import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkAppTests(unittest.TestCase):
    def test_mock_benchmark_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "test-run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "apps" / "benchmark.py"),
                    "--targets",
                    "cup,book",
                    "--episodes",
                    "2",
                    "--run-id",
                    run_id,
                    "--output-dir",
                    tmpdir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("frames=4", completed.stdout)
            run_dir = Path(tmpdir) / run_id
            metrics_path = run_dir / "metrics.jsonl"
            summary_path = run_dir / "summary.csv"
            config_path = run_dir / "config_snapshot.yaml"

            self.assertTrue(metrics_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(config_path.exists())

            records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["target"], "杯子")

            with summary_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["frame_count"], "4")
            self.assertEqual(rows[0]["recall"], "1.0")
            self.assertEqual(rows[0]["precision"], "1.0")

    def test_unknown_target_exits_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "apps" / "benchmark.py"),
                    "--targets",
                    "火星基地",
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Unknown benchmark target", completed.stderr)


if __name__ == "__main__":
    unittest.main()
