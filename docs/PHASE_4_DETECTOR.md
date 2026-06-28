# Phase 4: Detector Integration

## Goal

Add a detector abstraction and integrate YOLO as the bounded-vocabulary default detector.

## Files to create or modify

- `lvfind/vision/detector_base.py`
- `lvfind/vision/yolo_detector.py`
- `apps/sim_demo.py`
- `configs/app.yaml`
- `tests/test_detector_base.py`

## Requirements

### Detector interface

Define a detector interface that returns:

- label
- confidence score
- bbox
- optional mask
- optional raw model metadata

### YOLO implementation

Support:

- model path from config
- input size from config
- confidence threshold from config
- IoU threshold from config
- device selection from config

The detector should not hardcode Chinese object names.

### Target matching

Target matching should use `objects_zh.yaml`:

```text
canonical target -> detector labels -> filter detections
```

### Visualization

In demo mode, draw:

- bbox
- label
- confidence
- distance
- direction
- latency

## Tests

At minimum, add tests for detector result structures and target label matching.

## Definition of done

- Detector interface is isolated.
- YOLO is optional and fails clearly if dependencies are missing.
- The pipeline can still run in mock mode without YOLO.
