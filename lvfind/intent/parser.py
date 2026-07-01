"""Text intent parser for target-object finding commands."""

from __future__ import annotations

from dataclasses import dataclass

from lvfind.intent.object_vocab import ObjectVocabulary, normalize_text


FIND_OBJECT_INTENT = "find_object"
NO_MATCH_INTENT = "no_match"
STOP_SEARCH_INTENT = "stop_search"
CHANGE_TARGET_INTENT = "change_target"
RESTART_INTENT = "restart"

_FIND_MARKERS = (
    "找",
    "寻找",
    "在哪",
    "在哪里",
    "哪儿",
    "哪里",
    "位置",
    "看见",
    "看到",
)
_STOP_MARKERS = ("停止寻找", "停止找", "停止搜索", "停一下", "停止")
_CHANGE_TARGET_MARKERS = ("换一个目标", "换个目标", "更换目标", "换目标")
_RESTART_MARKERS = ("重新开始", "重新寻找", "重新找", "重来")


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Structured parser result."""

    intent: str
    target: str | None = None
    raw_target: str | None = None

    @property
    def is_match(self) -> bool:
        """Whether this result matched a supported intent."""

        return self.intent != NO_MATCH_INTENT

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""

        return {
            "intent": self.intent,
            "target": self.target,
            "raw_target": self.raw_target,
        }


@dataclass(frozen=True, slots=True)
class VoiceCommandResult:
    """Structured voice command parser result."""

    intent: str
    target: str | None = None
    raw_target: str | None = None

    @property
    def is_match(self) -> bool:
        """Whether this result matched a supported command."""

        return self.intent != NO_MATCH_INTENT

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""

        return {
            "intent": self.intent,
            "target": self.target,
            "raw_target": self.raw_target,
        }


def parse_find_object_intent(text: str, vocabulary: ObjectVocabulary) -> IntentResult:
    """Parse a plain-text target finding command."""

    normalized = normalize_text(text)
    if not normalized:
        return IntentResult(intent=NO_MATCH_INTENT)

    alias_match = vocabulary.find_alias_match(text)
    if alias_match is None:
        return IntentResult(intent=NO_MATCH_INTENT)

    if _is_direct_object(normalized, vocabulary) or _looks_like_find_request(normalized):
        return IntentResult(
            intent=FIND_OBJECT_INTENT,
            target=alias_match.canonical_name,
            raw_target=alias_match.alias,
        )
    return IntentResult(intent=NO_MATCH_INTENT)


def parse_voice_command(text: str, vocabulary: ObjectVocabulary) -> VoiceCommandResult:
    """Parse find-object and session-control voice commands."""

    normalized = normalize_text(text)
    if not normalized:
        return VoiceCommandResult(intent=NO_MATCH_INTENT)

    if _contains_any(normalized, _STOP_MARKERS):
        return VoiceCommandResult(intent=STOP_SEARCH_INTENT)
    if _contains_any(normalized, _CHANGE_TARGET_MARKERS):
        return VoiceCommandResult(intent=CHANGE_TARGET_INTENT)
    if _contains_any(normalized, _RESTART_MARKERS):
        return VoiceCommandResult(intent=RESTART_INTENT)

    find_result = parse_find_object_intent(text, vocabulary)
    return VoiceCommandResult(
        intent=find_result.intent,
        target=find_result.target,
        raw_target=find_result.raw_target,
    )


def _is_direct_object(normalized_text: str, vocabulary: ObjectVocabulary) -> bool:
    return vocabulary.find_canonical(normalized_text) is not None


def _looks_like_find_request(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _FIND_MARKERS)


def _contains_any(normalized_text: str, markers: tuple[str, ...]) -> bool:
    return any(normalize_text(marker) in normalized_text for marker in markers)
