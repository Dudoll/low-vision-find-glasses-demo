# Benchmark Plan

## Goal

Measure whether the demo satisfies the 5 FPS and 200 ms frame-processing target.

## Metrics

Record per frame:

- `frame_id`
- `timestamp_ms`
- `target`
- `detected`
- `confidence`
- `bbox`
- `distance_m`
- `direction`
- `latency_total_ms`
- `latency_capture_ms`
- `latency_preprocess_ms`
- `latency_inference_ms`
- `latency_postprocess_ms`
- `latency_distance_ms`
- `latency_cue_ms`
- `stale_frame_drop_count`

## Summary metrics

Compute:

- Average FPS.
- P50 latency.
- P95 latency.
- P99 latency.
- Stale frame count.
- Recall.
- Precision.
- False positives per minute.
- Direction accuracy.
- Distance band accuracy.

## Target thresholds

MVP target:

```text
fps_avg >= 5
latency_p50 < 100 ms
latency_p95 < 200 ms
stale_frame_count = 0 during normal load
```

## Output files

Use:

```text
runs/<run_id>/metrics.jsonl
runs/<run_id>/summary.csv
runs/<run_id>/config_snapshot.yaml
```

## Benchmark command

Suggested command:

```bash
python apps/benchmark.py --targets cup,bottle,book,laptop --episodes 1000
```
