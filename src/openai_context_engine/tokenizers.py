from __future__ import annotations

from typing import Callable, Protocol


class Tokenizer(Protocol):
    def count(self, text: str) -> int:
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        ...


class ApproximateTokenizer:
    """
    Dependency-free fallback.

    `characters_per_token=4` is only an estimate. Use the model's real tokenizer
    for hard production limits.
    """

    def __init__(self, characters_per_token: float = 4.0):
        if characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive")
        self.characters_per_token = characters_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int((len(text) + self.characters_per_token - 1) // self.characters_per_token))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        max_chars = int(max_tokens * self.characters_per_token)
        return text[:max_chars]


class CallableTokenizer:
    def __init__(
        self,
        count_fn: Callable[[str], int],
        truncate_fn: Callable[[str, int], str],
    ):
        self._count_fn = count_fn
        self._truncate_fn = truncate_fn

    def count(self, text: str) -> int:
        return int(self._count_fn(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        return self._truncate_fn(text, max_tokens)


class TiktokenTokenizer:
    def __init__(self, model: str | None = None, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                'Install optional dependency: pip install "openai-compatible-context-engine[tiktoken]"'
            ) from exc

        if model:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoding = tiktoken.get_encoding(encoding_name)
        else:
            self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self._encoding.encode(text)
        return self._encoding.decode(tokens[:max_tokens])


class TransformersTokenizer:
    def __init__(self, model_or_path: str, **kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Install optional dependency: pip install "openai-compatible-context-engine[transformers]"'
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_or_path, **kwargs)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return self._tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=True,
        )
