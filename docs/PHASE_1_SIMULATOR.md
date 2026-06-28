# Phase 1: Simulator Frame Source

## Goal

Implement the simulator frame source skeleton and latest-frame-only frame bus.

Do not implement detector inference, ASR, wake word, or TTS in this phase.

## Files to create or modify

- `lvfind/sim/base.py`
- `lvfind/sim/ai2thor_adapter.py`
- `lvfind/pipeline/frame_bus.py`
- `apps/sim_demo.py`
- `configs/sim_ai2thor.yaml`
- `tests/test_frame_bus.py`

## Requirements

### Frame packet

Create a frame packet data structure containing:

- `frame_id`
- `timestamp_ms`
- `rgb`
- `depth`
- `metadata`

The `rgb`, `depth`, and `metadata` fields may be optional for tests.

### Frame bus

Implement latest-frame-only behavior:

- Capacity is exactly 1.
- Pushing a new frame replaces the previous unconsumed frame.
- Reading returns the latest frame.
- Track how many stale frames were dropped.

### Simulator adapter

The AI2-THOR adapter should expose:

- `start()`
- `stop()`
- `get_latest_frame()`

If AI2-THOR is missing, fail with a clear error message that tells the user how to install it.

### CLI demo

`apps/sim_demo.py` should support:

```bash
python apps/sim_demo.py --scene FloorPlan201 --fps 5 --width 640 --height 480
```

It should print per-frame information and not require a detector.

## Tests

Add tests for:

- Latest frame replacement.
- Stale frame drop count.
- Empty bus behavior.

Run:

```bash
python -m pytest tests/test_frame_bus.py
```

## Definition of done

- The app imports successfully.
- The frame bus tests pass.
- AI2-THOR missing dependency error is clear.
- No object detection code is added.
