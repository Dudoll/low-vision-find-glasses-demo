# Phase 3: Distance, Direction, and Cue Policy

## Goal

Implement the core perception-to-cue logic without integrating a real detector.

## Files to create or modify

- `lvfind/vision/direction.py`
- `lvfind/vision/distance.py`
- `lvfind/cue/policy.py`
- `configs/cue_policy.yaml`
- `tests/test_direction.py`
- `tests/test_distance.py`
- `tests/test_cue_policy.py`

## Requirements

### Direction estimation

Input:

- bbox: `(x1, y1, x2, y2)`
- image width
- image height

Output one of:

- `left`
- `front`
- `right`

Rule:

```text
bbox center x / image_width < 0.33 -> left
0.33 to 0.66 -> front
> 0.66 -> right
```

### Distance estimation

Input:

- bbox
- depth frame

Use the center region of bbox and compute median depth.

The implementation should handle:

- Empty depth frame.
- Invalid bbox.
- NaN depth values.
- Zero or negative depth values.

### Cue policy

The cue policy should:

- Emit a cue immediately on first detection.
- Throttle repeated identical cues.
- Emit again if the target gets significantly closer.
- Emit again if direction changes.
- Avoid blocking the vision loop.

Distance bands:

- `very_near`: < 0.5 m
- `near`: 0.5 - 1.0 m
- `mid`: 1.0 - 2.0 m
- `far`: 2.0 - 3.0 m
- `too_far`: > 3.0 m

## Tests

Run:

```bash
python -m pytest tests/test_direction.py tests/test_distance.py tests/test_cue_policy.py
```

## Definition of done

- Direction logic is deterministic.
- Distance logic uses median center crop.
- Cue policy is stateful and throttled.
- No TTS implementation is required.
