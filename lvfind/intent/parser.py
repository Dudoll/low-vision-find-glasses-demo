"""Text intent parser for target-object finding commands."""

from __future__ import annotations

from dataclasses import dataclass

from lvfind.intent.object_vocab import ObjectVocabulary, normalize_text


FIND_OBJECT_INTENT = "find_object"
NO_MATCH_INTENT = "no_match"

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


def _is_direct_object(normalized_text: str, vocabulary: ObjectVocabulary) -> bool:
    return vocabulary.find_canonical(normalized_text) is not None


def _looks_like_find_request(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _FIND_MARKERS)
