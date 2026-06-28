# Low Vision Indoor Object Finding Glasses Demo

A simulator-first open-source demo for blind or low-vision indoor object finding.

The intended flow:

1. Wake the system by voice.
2. Say the object to find.
3. A first-person camera stream captures frames at 5 FPS.
4. The vision pipeline detects the target object within 200 ms per frame.
5. The system emits direction and distance-aware cues.

## Current priority

Build the simulator MVP first.

Do not start with real glasses hardware or full open-vocabulary detection.

## Repository guide

Codex should read:

1. `AGENTS.md`
2. `docs/PROJECT_PLAN.md`
3. The relevant phase document in `docs/`

## Suggested first command

After dependencies are implemented:

```bash
python apps/sim_demo.py --target 杯子 --scene FloorPlan201 --show
```

## Safety note

This project is an experimental object-finding assistant. It is not a medical device, not a navigation system, and not an obstacle-avoidance safety system.
