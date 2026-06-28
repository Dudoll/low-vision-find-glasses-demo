# AGENTS.md

## Project

This repository implements a personal open-source demo for low-vision / blind indoor object finding with camera glasses.

The first target is a simulator-based MVP:

- AI2-THOR provides first-person RGB/depth frames.
- Camera pipeline runs at 5 FPS.
- Vision processing must finish within 200 ms per frame.
- The system detects a target object requested by voice.
- If the target appears, it emits directional and distance-aware cues.
- The first implementation should prefer a bounded object vocabulary over fully open-vocabulary detection.

## Read this first

Before implementing any feature, read:

1. `docs/PROJECT_PLAN.md`
2. The phase document relevant to the task, for example:
   - `docs/PHASE_1_SIMULATOR.md`
   - `docs/PHASE_2_VISION.md`
   - `docs/PHASE_3_DISTANCE_CUE.md`
   - `docs/PHASE_4_VOICE.md`
   - `docs/BENCHMARK.md`

If these files conflict, follow `AGENTS.md` first, then the phase document, then `docs/PROJECT_PLAN.md`.

## Architecture constraints

Use this module layout:

- `lvfind/audio/` for wake word, ASR, and TTS.
- `lvfind/intent/` for target-object parsing and vocabulary mapping.
- `lvfind/sim/` for simulator adapters.
- `lvfind/vision/` for detection, distance estimation, tracking.
- `lvfind/cue/` for audio/vibration cue policy.
- `lvfind/pipeline/` for realtime frame pipeline and metrics.
- `apps/` for runnable demos.
- `configs/` for YAML config files.
- `tests/` for unit and integration tests.

Do not put core logic directly inside `apps/`. Apps should compose library modules.

## MVP constraints

Prioritize Phase 1 and Phase 2 before adding advanced features.

Do not implement open-vocabulary detection as the default path in the MVP. Use a bounded object vocabulary first.

The frame pipeline must be latest-frame-only:

- Queue size must be 1.
- If processing falls behind, drop stale frames.
- Never accumulate old camera frames.

The visual pipeline must record per-frame latency metrics.

## Performance target

The simulator camera target is 5 FPS.

Each frame should satisfy:

- P50 processing latency < 100 ms if possible.
- P95 processing latency < 200 ms.
- No stale-frame backlog.

## Coding conventions

Use Python 3.10+.

Prefer:

- dataclasses or pydantic-style typed config objects.
- clear module boundaries.
- small testable functions.
- YAML config for thresholds and object vocabulary.
- JSONL metrics for frame-level logs.

Avoid:

- hardcoded object names in detector logic.
- blocking TTS inside the vision loop.
- global mutable state unless isolated behind runtime objects.
- adding heavyweight models to the default path without a config switch.

## Test commands

When changing pure Python logic, run:

```bash
python -m pytest tests
```

When changing formatting-sensitive code, prefer:

```bash
python -m ruff check .
python -m ruff format .
```

If these tools are not installed, update `pyproject.toml` or document the missing dependency.

## Definition of done

A task is done only if:

1. The code runs or the limitation is explicitly documented.
2. Relevant tests are added or updated.
3. The command used for verification is reported.
4. New config keys are documented.
5. The change does not violate the 5 FPS / 200 ms realtime design.
