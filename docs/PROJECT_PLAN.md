# Project Plan: Low Vision Indoor Object Finding Glasses Demo

## 1. Goal

Build a personal open-source simulator-first demo for blind or low-vision indoor object finding.

The intended product flow is:

1. The user wakes the glasses by voice.
2. The user says the target object name, such as “帮我找手机”.
3. The simulated glasses camera captures first-person images at 5 FPS.
4. The vision pipeline processes each frame within 200 ms.
5. If the target appears in the frame, the system emits an audio or vibration cue.
6. The cue becomes stronger or more frequent as the target gets closer.

The project should first run in a simulator. Real glasses or mobile hardware should come later.

## 2. Product scope

### MVP scope

The MVP should support:

- AI2-THOR simulator as the first-person indoor environment.
- RGB frame capture at 5 FPS.
- Optional depth frame capture for distance estimation.
- Manual target input from CLI.
- Bounded object vocabulary.
- Object detection through a small YOLO model or a mocked detector during early stages.
- Direction estimation from bounding-box center.
- Distance estimation from depth frame.
- Audio cue or terminal-based vibration mock.
- Per-frame latency metrics.
- Benchmark mode.

### Non-goals for MVP

Do not start with:

- Real glasses hardware.
- Full navigation.
- Obstacle avoidance.
- Cloud inference.
- Fully open-vocabulary detection as the default path.
- Complex app UI.

## 3. Recommended architecture

```text
                +--------------------+
                | Wake Word Detector |
                | openWakeWord       |
                +---------+----------+
                          |
                          v
                +--------------------+
                | ASR                |
                | Vosk / whisper.cpp |
                +---------+----------+
                          |
                          v
                +--------------------+
                | Intent Parser      |
                | find object name   |
                +---------+----------+
                          |
                          v
+-------------+   frame   +--------------------+
| Simulator   |---------->| Frame Pipeline      |
| AI2-THOR    | 5 FPS     | latest-frame only   |
+-------------+           +---------+----------+
                                    |
                                    v
                          +--------------------+
                          | Object Detector    |
                          | YOLO / fallback    |
                          +---------+----------+
                                    |
                                    v
                          +--------------------+
                          | Distance Estimator |
                          | depth / bbox / GT  |
                          +---------+----------+
                                    |
                                    v
                          +--------------------+
                          | Cue Policy         |
                          | audio / vibration  |
                          +--------------------+
```

## 4. Module layout

```text
low-vision-find-glasses-demo/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── app.yaml
│   ├── objects_zh.yaml
│   ├── cue_policy.yaml
│   └── sim_ai2thor.yaml
├── apps/
│   ├── sim_demo.py
│   ├── voice_sim_demo.py
│   ├── replay_demo.py
│   └── benchmark.py
├── lvfind/
│   ├── audio/
│   │   ├── wakeword.py
│   │   ├── asr.py
│   │   └── tts.py
│   ├── intent/
│   │   ├── parser.py
│   │   └── object_vocab.py
│   ├── sim/
│   │   ├── base.py
│   │   └── ai2thor_adapter.py
│   ├── vision/
│   │   ├── detector_base.py
│   │   ├── yolo_detector.py
│   │   ├── distance.py
│   │   ├── direction.py
│   │   └── tracker.py
│   ├── cue/
│   │   ├── policy.py
│   │   ├── audio_cue.py
│   │   └── vibration_mock.py
│   ├── pipeline/
│   │   ├── frame_bus.py
│   │   ├── runtime.py
│   │   └── metrics.py
│   └── utils/
│       ├── logging.py
│       └── time_utils.py
├── scripts/
│   ├── export_yolo_onnx.py
│   ├── quantize_onnx.py
│   ├── collect_ai2thor_dataset.py
│   └── run_latency_test.py
├── tests/
└── docs/
```

## 5. Key design decisions

### 5.1 Use latest-frame-only processing

The camera captures at 5 FPS, so one frame arrives every 200 ms.

The processing queue must not accumulate old frames. If the vision pipeline is still busy and a new frame arrives, the old frame should be replaced.

Correct behavior:

```text
queue_size = 1
new_frame arrives
if queue is full:
    drop old frame
push new frame
```

This prevents the user from hearing cues about stale images.

### 5.2 Start with bounded vocabulary

The first version should support a fixed list of indoor objects, for example:

- 手机
- 杯子
- 瓶子
- 书
- 笔记本电脑
- 鼠标
- 键盘
- 背包
- 椅子
- 桌子
- 沙发
- 床

The vocabulary maps Chinese aliases to detector labels.

Example:

```yaml
手机:
  aliases: ["手机", "电话", "cell phone", "phone"]
  detector_labels: ["cell phone"]
```

### 5.3 Keep open-vocabulary detection as fallback

Grounding DINO, YOLO-World, or similar models may be added later, but they should not be the MVP default path.

Suggested rule:

```text
Known object -> YOLO every frame
Unknown object -> open-vocabulary fallback at low frequency
```

### 5.4 Distance estimation

In simulator mode, use depth frame first.

Given a bounding box:

```text
x1, y1, x2, y2
```

Take the center region of the box and compute median depth:

```text
center crop = bbox center 30% x 30%
distance = median(depth[center crop])
```

Avoid using the full bbox because edges often include background.

### 5.5 Direction estimation

Use bounding-box center:

```text
cx = (x1 + x2) / 2
nx = cx / image_width
```

Rule:

```text
nx < 0.33      -> left
0.33 <= nx <= 0.66 -> front
nx > 0.66      -> right
```

The first version should not over-describe direction. Use short cues:

- 左边
- 正前方
- 右边

### 5.6 Cue policy

The cue policy should be stateful and throttled.

Avoid speaking every frame.

Suggested policy:

- First detection: immediate cue.
- Same direction and same distance band: repeat no more than once every 2 seconds.
- Distance becomes significantly closer: emit cue.
- Target disappears for a while: optionally say “暂时没有看到目标”.

Distance bands:

```text
very_near: < 0.5 m
near:      0.5 - 1.0 m
mid:       1.0 - 2.0 m
far:       2.0 - 3.0 m
too_far:   > 3.0 m
```

## 6. Performance budget

Target camera rate:

```text
5 FPS
```

Frame interval:

```text
200 ms
```

Budget per frame:

| Stage | Target |
|---|---:|
| frame acquisition | 5 - 20 ms |
| resize / normalize | 2 - 8 ms |
| inference | 40 - 130 ms |
| postprocess | 5 - 20 ms |
| distance estimate | 1 - 10 ms |
| cue decision | < 2 ms |
| logging | < 10 ms |
| total P95 | < 200 ms |

## 7. Development phases

### Phase 1: simulator frame source

Goal:

- Start AI2-THOR.
- Capture first-person RGB and optional depth frames.
- Implement latest-frame-only frame bus.
- Provide `apps/sim_demo.py`.
- Add latency logging skeleton.

### Phase 2: object vocabulary and intent parser

Goal:

- Parse text commands such as “帮我找手机”.
- Map aliases to canonical object names.
- Map canonical object names to detector labels.
- Add unit tests for Chinese commands.

### Phase 3: distance, direction, and cue policy

Goal:

- Compute left/front/right from bbox.
- Compute distance from depth frame.
- Emit throttled audio or vibration mock cues.
- Add unit tests.

### Phase 4: detector integration

Goal:

- Add detector interface.
- Add YOLO detector implementation.
- Support model path and thresholds through config.
- Draw detection boxes in demo mode.
- Log confidence, bbox, distance, direction, and latency.

### Phase 5: voice interaction

Goal:

- Add wake word detector.
- Add ASR.
- Add command parsing from speech.
- Support commands:
  - 开始寻找
  - 停止寻找
  - 换一个目标
  - 重新开始

### Phase 6: benchmark

Goal:

- Run automated episodes in simulator.
- Measure recall, precision, latency, stale frame count, and direction accuracy.
- Produce JSONL and summary CSV.

### Phase 7: hardware migration

Goal:

- Add real camera adapter.
- Add video replay mode.
- Add Android or phone companion path later.
- Add vibration hardware abstraction.

## 8. MVP acceptance criteria

The first meaningful MVP is:

```bash
python apps/sim_demo.py --target 杯子 --scene FloorPlan201 --show
```

Expected behavior:

- AI2-THOR starts.
- Frames are captured at 5 FPS.
- The target object is detected or simulated.
- Direction is estimated as left/front/right.
- Distance is estimated from depth.
- A cue is emitted when the target appears.
- Metrics are written to JSONL.

Benchmark target:

```text
fps_avg >= 5
latency_p50 < 100 ms
latency_p95 < 200 ms
stale_frame_count = 0
```

## 9. Suggested Codex workflow

Do not ask Codex to implement the whole project at once.

Use small tasks:

1. Create project skeleton.
2. Implement frame bus.
3. Implement intent parser.
4. Implement distance/direction.
5. Implement cue policy.
6. Add simulator adapter.
7. Add detector interface.
8. Add YOLO detector.
9. Add benchmark.

Each task should specify:

- Files to create or modify.
- Requirements.
- Files not to touch.
- Tests to add.
- Commands to run.
- Definition of done.

## 10. Safety and privacy notes

This project should be described as an object-finding assistant, not a navigation or obstacle-avoidance safety device.

Default behavior should be local inference. Do not upload camera frames unless explicitly configured.

The README should clearly state:

- The demo is experimental.
- It is not a medical device.
- It is not a navigation safety system.
- It should not be relied on for avoiding hazards.
