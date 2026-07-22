from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
from typing import Callable


SUPPORTED_DEDUPLICATION_MODES = frozenset(
    {
        "none",
        "exact",
        "normalized",
        "similarity",
    }
)


@dataclass(frozen=True)
class DuplicateMatch:
    """Describes a previously seen context item that matches a new item."""

    item_id: str
    mode: str
    similarity: float | None = None


@dataclass(frozen=True)
class _SeenText:
    item_id: str
    normalized: str


def default_normalize_text(text: str) -> str:
    """
    Canonicalize text for normalized and similarity deduplication.

    The default normalizer applies Unicode NFKC normalization, Unicode-aware
    case folding, leading/trailing whitespace removal, and whitespace collapse.
    It intentionally does not remove timestamps, identifiers, numbers, or
    punctuation because those values may be semantically important.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class Deduplicator:
    """
    Stateful, deterministic deduplication for one context build operation.

    `find_duplicate` checks a candidate against previously retained candidates.
    `remember` must be called only when the candidate remains eligible for
    selection. This avoids allowing a low-scoring dropped item to suppress a
    later eligible item with the same content.
    """

    def __init__(
        self,
        *,
        mode: str,
        similarity_threshold: float,
        custom_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        normalized_mode = mode.lower().strip()
        if normalized_mode not in SUPPORTED_DEDUPLICATION_MODES:
            supported = ", ".join(sorted(SUPPORTED_DEDUPLICATION_MODES))
            raise ValueError(
                f"unsupported deduplication mode {mode!r}; expected one of: {supported}"
            )
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.mode = normalized_mode
        self.similarity_threshold = similarity_threshold
        self.custom_normalizer = custom_normalizer
        self._exact_hashes: dict[str, str] = {}
        self._normalized_hashes: dict[str, str] = {}
        self._similarity_candidates: list[_SeenText] = []

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _normalize(self, text: str) -> str:
        if self.custom_normalizer is not None:
            text = self.custom_normalizer(text)
            if not isinstance(text, str):
                raise TypeError("deduplication_normalizer must return a string")
        return default_normalize_text(text)

    def find_duplicate(self, text: str) -> DuplicateMatch | None:
        if self.mode == "none":
            return None

        if self.mode == "exact":
            duplicate_id = self._exact_hashes.get(self._digest(text))
            if duplicate_id is None:
                return None
            return DuplicateMatch(item_id=duplicate_id, mode="exact", similarity=1.0)

        normalized = self._normalize(text)
        normalized_digest = self._digest(normalized)
        duplicate_id = self._normalized_hashes.get(normalized_digest)
        if duplicate_id is not None:
            return DuplicateMatch(
                item_id=duplicate_id,
                mode="normalized",
                similarity=1.0,
            )

        if self.mode == "normalized":
            return None

        for candidate in self._similarity_candidates:
            # A large length difference cannot reach a high similarity ratio.
            longer_length = max(len(normalized), len(candidate.normalized))
            if longer_length == 0:
                ratio = 1.0
            else:
                shorter_length = min(len(normalized), len(candidate.normalized))
                length_upper_bound = (2.0 * shorter_length) / (
                    len(normalized) + len(candidate.normalized)
                )
                if length_upper_bound < self.similarity_threshold:
                    continue
                ratio = SequenceMatcher(
                    None,
                    normalized,
                    candidate.normalized,
                    autojunk=False,
                ).ratio()

            if ratio >= self.similarity_threshold:
                return DuplicateMatch(
                    item_id=candidate.item_id,
                    mode="similarity",
                    similarity=ratio,
                )

        return None

    def remember(self, item_id: str, text: str) -> None:
        if self.mode == "none":
            return

        if self.mode == "exact":
            self._exact_hashes.setdefault(self._digest(text), item_id)
            return

        normalized = self._normalize(text)
        self._normalized_hashes.setdefault(self._digest(normalized), item_id)
        self._similarity_candidates.append(
            _SeenText(item_id=item_id, normalized=normalized)
        )
