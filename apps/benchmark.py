"""Run benchmark episodes and write JSONL/CSV metrics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from lvfind.intent.object_vocab import ObjectVocabulary  # noqa: E402
from lvfind.pipeline.metrics import (  # noqa: E402
    FrameMetricRecord,
    summarize_frame_metrics,
    write_config_snapshot,
    write_metrics_jsonl,
    write_summary_csv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark episodes.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "app.yaml"))
    parser.add_argument("--objects-config", default=str(REPO_ROOT / "configs" / "objects_zh.yaml"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--targets", default="cup,bottle,book,laptop")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        choices=("mock",),
        default="mock",
        help="Benchmark data source. Phase 6 implements deterministic mock episodes.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than 0")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be greater than 0")

    app_config = _load_config(args.config)
    vocabulary = ObjectVocabulary.from_yaml(args.objects_config)
    targets = _resolve_targets(args.targets, vocabulary)
    run_id = args.run_id or _default_run_id()
    run_dir = Path(args.output_dir) / run_id

    records = _generate_mock_records(
        targets=targets,
        episodes=args.episodes,
        target_fps=args.target_fps,
    )
    summary = summarize_frame_metrics(records)

    write_metrics_jsonl(run_dir / "metrics.jsonl", records)
    write_summary_csv(run_dir / "summary.csv", summary)
    write_config_snapshot(
        run_dir / "config_snapshot.yaml",
        _config_snapshot_text(args, app_config, targets),
    )

    print(f"run_id={run_id}")
    print(f"metrics={run_dir / 'metrics.jsonl'}")
    print(f"summary={run_dir / 'summary.csv'}")
    print(
        "frames={frames} fps_avg={fps} latency_p95_ms={latency} recall={recall} precision={precision}".format(
            frames=summary.frame_count,
            fps=_fmt(summary.fps_avg),
            latency=_fmt(summary.latency_p95_ms),
            recall=_fmt(summary.recall),
            precision=_fmt(summary.precision),
        )
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_targets(targets_csv: str, vocabulary: ObjectVocabulary) -> tuple[str, ...]:
    targets: list[str] = []
    for raw_target in targets_csv.split(","):
        raw_target = raw_target.strip()
        if not raw_target:
            continue
        match = vocabulary.find_alias_match(raw_target)
        if match is None:
            raise SystemExit(f"Unknown benchmark target: {raw_target}")
        targets.append(match.canonical_name)

    if not targets:
        raise SystemExit("--targets must contain at least one known object")
    return tuple(targets)


def _generate_mock_records(
    targets: Sequence[str],
    episodes: int,
    target_fps: float,
) -> list[FrameMetricRecord]:
    records: list[FrameMetricRecord] = []
    frame_interval_ms = 1000.0 / target_fps
    directions = ("left", "front", "right")
    distances = (0.4, 0.8, 1.4, 2.4)
    distance_bands = ("very_near", "near", "mid", "far")

    frame_id = 0
    for episode_index in range(episodes):
        for target_index, target in enumerate(targets):
            frame_id += 1
            direction = directions[(episode_index + target_index) % len(directions)]
            distance_index = (episode_index + target_index) % len(distances)
            distance_m = distances[distance_index]
            latency_capture_ms = 5.0
            latency_preprocess_ms = 2.0
            latency_inference_ms = 1.0
            latency_postprocess_ms = 1.0
            latency_distance_ms = 1.0
            latency_cue_ms = 0.5
            latency_total_ms = (
                latency_capture_ms
                + latency_preprocess_ms
                + latency_inference_ms
                + latency_postprocess_ms
                + latency_distance_ms
                + latency_cue_ms
            )

            records.append(
                FrameMetricRecord(
                    frame_id=frame_id,
                    timestamp_ms=(frame_id - 1) * frame_interval_ms,
                    target=target,
                    detected=True,
                    confidence=0.9,
                    bbox=(200.0, 120.0, 360.0, 300.0),
                    distance_m=distance_m,
                    direction=direction,
                    latency_total_ms=latency_total_ms,
                    latency_capture_ms=latency_capture_ms,
                    latency_preprocess_ms=latency_preprocess_ms,
                    latency_inference_ms=latency_inference_ms,
                    latency_postprocess_ms=latency_postprocess_ms,
                    latency_distance_ms=latency_distance_ms,
                    latency_cue_ms=latency_cue_ms,
                    stale_frame_drop_count=0,
                    expected_present=True,
                    expected_direction=direction,
                    expected_distance_band=distance_bands[distance_index],
                )
            )
    return records


def _config_snapshot_text(
    args: argparse.Namespace,
    app_config: dict[str, Any],
    targets: Sequence[str],
) -> str:
    snapshot = {
        "benchmark": {
            "mode": args.mode,
            "run_id": args.run_id,
            "targets": list(targets),
            "episodes": args.episodes,
            "target_fps": args.target_fps,
        },
        "app_config": app_config,
    }
    return yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
