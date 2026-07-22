from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .models import ContextItem
from .reducers import HeadTailReducer, QueryAwareSentenceReducer
from .tokenizers import Tokenizer


SUPPORTED_COMPRESSION_MODES = {"none", "extractive", "llm", "custom"}


class Compressor(Protocol):
    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        ...


@dataclass(frozen=True)
class ExtractiveCompressor:
    """Deterministic, dependency-free query-aware compression."""

    sentence_reducer: QueryAwareSentenceReducer = field(
        default_factory=QueryAwareSentenceReducer
    )
    fallback_reducer: HeadTailReducer = field(default_factory=HeadTailReducer)

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        reduced = self.sentence_reducer.reduce(
            item=item,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            query=query,
        )
        if reduced and tokenizer.count(reduced) <= target_tokens:
            return reduced
        return self.fallback_reducer.reduce(
            item=item,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            query=query,
        )


@dataclass(frozen=True)
class CallableCompressor:
    """Adapter for application-provided compression functions."""

    function: Callable[[ContextItem, int, Tokenizer, str], str | None]

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        return self.function(item, target_tokens, tokenizer, query)


@dataclass(frozen=True)
class OpenAIChatCompressor:
    """
    LLM compressor for clients exposing `client.chat.completions.create(...)`.

    The client may be the official OpenAI Python client or another compatible
    client object. The package does not import OpenAI at runtime.
    """

    client: Any
    model: str
    system_prompt: str = (
        "Compress the supplied context while preserving facts, identifiers, "
        "numbers, constraints, exceptions, and uncertainty relevant to the "
        "user query. Do not add new information. Return only the compressed "
        "context, without commentary or markdown fences."
    )
    temperature: float | None = 0.0
    extra_request_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def compress(
        self,
        item: ContextItem,
        target_tokens: int,
        tokenizer: Tokenizer,
        query: str,
    ) -> str | None:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"TARGET MAXIMUM TOKENS: {target_tokens}\n\n"
                        f"USER QUERY:\n{query}\n\n"
                        f"CONTEXT ITEM ID: {item.id}\n"
                        f"CONTEXT CATEGORY: {item.category}\n\n"
                        f"CONTEXT TO COMPRESS:\n{item.render()}"
                    ),
                },
            ],
            "max_tokens": target_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        request.update(dict(self.extra_request_kwargs))

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            return None
        return str(content).strip() or None
