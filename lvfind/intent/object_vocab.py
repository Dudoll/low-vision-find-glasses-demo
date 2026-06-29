"""Bounded object vocabulary and alias lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing dependency installs.
    yaml = None


@dataclass(frozen=True, slots=True)
class ObjectEntry:
    """One canonical object and the detector labels it maps to."""

    canonical_name: str
    aliases: tuple[str, ...]
    detector_labels: tuple[str, ...]
    typical_width_m: float | None = None


@dataclass(frozen=True, slots=True)
class AliasMatch:
    """A vocabulary alias found in text."""

    canonical_name: str
    alias: str


class ObjectVocabulary:
    """Load and query a bounded object vocabulary."""

    def __init__(self, entries: list[ObjectEntry]) -> None:
        self._entries = {entry.canonical_name: entry for entry in entries}
        self._alias_to_canonical: dict[str, str] = {}
        self._alias_display: dict[str, str] = {}

        for entry in entries:
            for alias in entry.aliases:
                normalized_alias = normalize_text(alias)
                if not normalized_alias:
                    continue
                existing = self._alias_to_canonical.get(normalized_alias)
                if existing is not None and existing != entry.canonical_name:
                    raise ValueError(
                        f"Alias {alias!r} maps to both {existing!r} and "
                        f"{entry.canonical_name!r}"
                    )
                self._alias_to_canonical[normalized_alias] = entry.canonical_name
                self._alias_display[normalized_alias] = alias

        self._aliases_by_length = sorted(
            self._alias_to_canonical,
            key=len,
            reverse=True,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ObjectVocabulary:
        """Load vocabulary entries from a YAML file."""

        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load object vocabulary YAML. "
                "Install project dependencies with `python -m pip install -e .`."
            )

        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ObjectVocabulary:
        """Build a vocabulary from parsed YAML-like data."""

        objects = data.get("objects", data)
        if not isinstance(objects, dict):
            raise ValueError("Object vocabulary must contain an 'objects' mapping.")

        entries: list[ObjectEntry] = []
        for canonical_name, raw_entry in objects.items():
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Object entry {canonical_name!r} must be a mapping.")

            aliases = _string_tuple(raw_entry.get("aliases", ()))
            if canonical_name not in aliases:
                aliases = (str(canonical_name), *aliases)

            detector_labels = _string_tuple(raw_entry.get("detector_labels", ()))
            typical_width = raw_entry.get("typical_width_m")
            entries.append(
                ObjectEntry(
                    canonical_name=str(canonical_name),
                    aliases=aliases,
                    detector_labels=detector_labels,
                    typical_width_m=float(typical_width) if typical_width is not None else None,
                )
            )
        return cls(entries)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        """Return known canonical object names."""

        return tuple(self._entries)

    def get_entry(self, canonical_name: str) -> ObjectEntry | None:
        """Return the vocabulary entry for a canonical object."""

        return self._entries.get(canonical_name)

    def find_canonical(self, alias: str) -> str | None:
        """Return a canonical object name for an exact alias, or None."""

        return self._alias_to_canonical.get(normalize_text(alias))

    def find_alias_match(self, text: str) -> AliasMatch | None:
        """Find the longest known alias mentioned in text."""

        normalized_text = normalize_text(text)
        if not normalized_text:
            return None

        exact_match = self._alias_to_canonical.get(normalized_text)
        if exact_match is not None:
            return AliasMatch(
                canonical_name=exact_match,
                alias=self._alias_display[normalized_text],
            )

        for normalized_alias in self._aliases_by_length:
            if normalized_alias in normalized_text:
                return AliasMatch(
                    canonical_name=self._alias_to_canonical[normalized_alias],
                    alias=self._alias_display[normalized_alias],
                )
        return None

    def detector_labels_for(self, canonical_name: str) -> tuple[str, ...]:
        """Return detector labels for a canonical object, or an empty tuple."""

        entry = self.get_entry(canonical_name)
        if entry is None:
            return ()
        return entry.detector_labels


def normalize_text(text: str) -> str:
    """Normalize user text and aliases for vocabulary matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a string list, got {type(values).__name__}.")
    return tuple(str(value) for value in values)
