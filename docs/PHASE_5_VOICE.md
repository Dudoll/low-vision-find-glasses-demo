# Phase 5: Voice Interaction

## Goal

Add voice wake word and ASR integration.

This phase should be added only after the simulator and vision pipeline work.

## Files to create or modify

- `lvfind/audio/wakeword.py`
- `lvfind/audio/asr.py`
- `lvfind/audio/tts.py`
- `apps/voice_sim_demo.py`
- `configs/app.yaml`

## Requirements

### Wake word

Support a configurable wake word engine.

First target:

- `openWakeWord`

The implementation should fail clearly if the dependency or model is missing.

### ASR

Support a configurable ASR engine.

First options:

- `vosk`
- `whisper.cpp` wrapper or command adapter

The parser should receive plain text from ASR and reuse `lvfind.intent.parser`.

### Commands

Support:

- 帮我找 X
- 找一下 X
- 我要找 X
- X 在哪里
- 停止寻找
- 换一个目标
- 重新开始

### TTS / output

TTS must not block the vision loop.

Use a queue or asynchronous output worker.

## Definition of done

- Voice mode can start a target-finding session.
- Vision loop remains independent from audio output.
- Missing model files produce clear messages.
