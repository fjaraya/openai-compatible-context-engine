from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from .models import ContextItem
from .tokenizers import Tokenizer


class Reducer(Protocol):
    def reduce(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        ...


@dataclass(frozen=True)
class QueryAwareSentenceReducer:
    """
    Extracts sentences with the highest lexical overlap with the query.
    It is deterministic and model-free.
    """

    minimum_sentence_chars: int = 20
    preserve_order: bool = True

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[\w\-]{3,}", text, flags=re.UNICODE)
        }

    def reduce(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        text = item.render()
        if tokenizer.count(text) <= target_tokens:
            return text

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", text)
            if len(s.strip()) >= self.minimum_sentence_chars
        ]
        if not sentences:
            return None

        query_terms = self._terms(query)
        ranked: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            terms = self._terms(sentence)
            overlap = len(query_terms.intersection(terms))
            density = overlap / max(1, len(terms))
            score = overlap + density
            ranked.append((score, index, sentence))

        ranked.sort(key=lambda value: (value[0], -value[1]), reverse=True)

        selected: list[tuple[int, str]] = []
        consumed = 0
        for score, index, sentence in ranked:
            sentence_tokens = tokenizer.count(sentence)
            if consumed + sentence_tokens <= target_tokens:
                selected.append((index, sentence))
                consumed += sentence_tokens

        if not selected:
            return None

        if self.preserve_order:
            selected.sort(key=lambda value: value[0])

        return "\n".join(sentence for _, sentence in selected)


@dataclass(frozen=True)
class HeadTailReducer:
    head_ratio: float = 0.7
    marker: str = "\n...[context reduced]...\n"

    def reduce(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        if target_tokens <= 0:
            return None

        text = item.render()
        if tokenizer.count(text) <= target_tokens:
            return text

        marker_tokens = tokenizer.count(self.marker)
        remaining = target_tokens - marker_tokens
        if remaining <= 1:
            return tokenizer.truncate(text, target_tokens)

        head_tokens = max(1, int(remaining * self.head_ratio))
        tail_tokens = max(1, remaining - head_tokens)

        head = tokenizer.truncate(text, head_tokens)
        reverse_tail = tokenizer.truncate(text[::-1], tail_tokens)
        tail = reverse_tail[::-1]
        reduced = head + self.marker + tail

        while tokenizer.count(reduced) > target_tokens and tail:
            tail = tail[1:]
            reduced = head + self.marker + tail

        return reduced or None
