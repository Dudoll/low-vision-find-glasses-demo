"""Voice-command demo using text ASR by default."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from lvfind.audio.asr import ASRConfig, ASRUnavailableError, build_asr  # noqa: E402
from lvfind.audio.tts import TTSConfig, TTSUnavailableError, build_tts  # noqa: E402
from lvfind.audio.wakeword import (  # noqa: E402
    WakeWordConfig,
    WakeWordUnavailableError,
    build_wake_word_detector,
)
from lvfind.intent.object_vocab import ObjectVocabulary  # noqa: E402
from lvfind.intent.parser import (  # noqa: E402
    CHANGE_TARGET_INTENT,
    FIND_OBJECT_INTENT,
    NO_MATCH_INTENT,
    RESTART_INTENT,
    STOP_SEARCH_INTENT,
    VoiceCommandResult,
    parse_voice_command,
)


@dataclass(slots=True)
class VoiceSession:
    """Current target-finding session state."""

    active: bool = False
    target: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse voice commands for the simulator demo.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "app.yaml"))
    parser.add_argument("--objects-config", default=str(REPO_ROOT / "configs" / "objects_zh.yaml"))
    parser.add_argument(
        "--command",
        action="append",
        default=None,
        help="Command text to process once. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    app_config = _load_config(args.config)
    voice_config = app_config.get("voice", {})

    try:
        wake_word = build_wake_word_detector(
            WakeWordConfig.from_mapping(voice_config.get("wake_word"))
        )
        asr = build_asr(ASRConfig.from_mapping(voice_config.get("asr")))
        speaker = build_tts(TTSConfig.from_mapping(voice_config.get("tts")))
    except (ASRUnavailableError, TTSUnavailableError, WakeWordUnavailableError) as exc:
        raise SystemExit(str(exc)) from exc

    vocabulary = ObjectVocabulary.from_yaml(args.objects_config)
    session = VoiceSession()
    speaker.start()
    try:
        for command_text in _iter_command_text(args.command):
            if not wake_word.detect(command_text):
                continue
            asr_result = asr.transcribe(command_text)
            command = parse_voice_command(asr_result.text, vocabulary)
            message = _apply_voice_command(command, session)
            speaker.say(message)
            print(
                "command={command} target={target} active={active} text={text}".format(
                    command=command.intent,
                    target=session.target or "none",
                    active=str(session.active).lower(),
                    text=asr_result.text,
                ),
                flush=True,
            )
    finally:
        speaker.stop(drain=True)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _iter_command_text(commands: Sequence[str] | None) -> Iterable[str]:
    if commands:
        yield from commands
        return

    print("Enter recognized text commands. Press Ctrl-D to stop.", flush=True)
    for line in sys.stdin:
        text = line.strip()
        if text:
            yield text


def _apply_voice_command(command: VoiceCommandResult, session: VoiceSession) -> str:
    if command.intent == FIND_OBJECT_INTENT and command.target is not None:
        session.target = command.target
        session.active = True
        return f"开始寻找{command.target}"

    if command.intent == STOP_SEARCH_INTENT:
        session.active = False
        return "已停止寻找"

    if command.intent == CHANGE_TARGET_INTENT:
        session.active = False
        session.target = None
        return "请说新的目标"

    if command.intent == RESTART_INTENT:
        if session.target is None:
            session.active = False
            return "还没有目标"
        session.active = True
        return f"重新开始寻找{session.target}"

    if command.intent == NO_MATCH_INTENT:
        return "没有识别到支持的命令"

    return f"暂不支持的命令: {command.intent}"


if __name__ == "__main__":
    main()
