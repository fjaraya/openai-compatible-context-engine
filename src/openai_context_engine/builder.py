from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

from .compressors import (
    Compressor,
    ExtractiveCompressor,
    SUPPORTED_COMPRESSION_MODES,
)
from .deduplication import Deduplicator, SUPPORTED_DEDUPLICATION_MODES
from .exceptions import BudgetConfigurationError, PinnedItemOverflowError
from .models import (
    AuditEntry,
    CompressionDecision,
    CompressionEntry,
    ContextBundle,
    ContextItem,
    ContextPolicy,
    Decision,
    SelectedItem,
)
from .reducers import HeadTailReducer, QueryAwareSentenceReducer, Reducer
from .tokenizers import Tokenizer


class ContextBuilder:
    def __init__(
        self,
        tokenizer: Tokenizer,
        policy: ContextPolicy | None = None,
        reducers: Sequence[Reducer] | None = None,
        compressor: Compressor | None = None,
    ):
        self.tokenizer = tokenizer
        self.policy = policy or ContextPolicy()
        self.reducers = list(
            reducers
            or [
                QueryAwareSentenceReducer(),
                HeadTailReducer(),
            ]
        )
        self.compressor = compressor
        self._validate_policy()
        if self.policy.compression_mode.lower().strip() == "extractive":
            self.compressor = compressor or ExtractiveCompressor()

    def _validate_policy(self) -> None:
        policy = self.policy
        if policy.context_window <= 0:
            raise BudgetConfigurationError("context_window must be positive")
        if policy.available_input_tokens <= 0:
            raise BudgetConfigurationError(
                "reserved output, safety margin, and overhead consume the full context window"
            )
        if policy.selection_mode not in {"score", "score_per_token"}:
            raise BudgetConfigurationError(
                "selection_mode must be 'score' or 'score_per_token'"
            )

        deduplication_mode = policy.deduplication_mode.lower().strip()
        if deduplication_mode not in SUPPORTED_DEDUPLICATION_MODES:
            supported = ", ".join(sorted(SUPPORTED_DEDUPLICATION_MODES))
            raise BudgetConfigurationError(
                "deduplication_mode must be one of: " + supported
            )
        if not 0.0 <= policy.deduplication_similarity_threshold <= 1.0:
            raise BudgetConfigurationError(
                "deduplication_similarity_threshold must be between 0.0 and 1.0"
            )

        compression_mode = policy.compression_mode.lower().strip()
        if compression_mode not in SUPPORTED_COMPRESSION_MODES:
            supported = ", ".join(sorted(SUPPORTED_COMPRESSION_MODES))
            raise BudgetConfigurationError(
                "compression_mode must be one of: " + supported
            )
        if not 0.0 < policy.compression_target_ratio < 1.0:
            raise BudgetConfigurationError(
                "compression_target_ratio must be greater than 0.0 and less than 1.0"
            )
        if policy.compression_min_tokens < 1:
            raise BudgetConfigurationError(
                "compression_min_tokens must be at least 1"
            )
        if (
            policy.compression_max_tokens is not None
            and policy.compression_max_tokens < 1
        ):
            raise BudgetConfigurationError(
                "compression_max_tokens must be at least 1 when provided"
            )
        if policy.compression_failure_mode not in {"keep_original", "raise"}:
            raise BudgetConfigurationError(
                "compression_failure_mode must be 'keep_original' or 'raise'"
            )
        if compression_mode in {"llm", "custom"} and self.compressor is None:
            raise BudgetConfigurationError(
                f"compression_mode={compression_mode!r} requires a compressor"
            )

    def _score(self, item: ContextItem, now: datetime) -> float:
        policy = self.policy
        priority = min(1.0, max(0.0, item.priority))
        relevance = min(1.0, max(0.0, item.relevance))

        recency = 0.5
        if item.created_at is not None:
            created = item.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_hours = max(
                0.0,
                (now - created.astimezone(timezone.utc)).total_seconds() / 3600,
            )
            half_life = max(0.001, policy.recency_half_life_hours)
            recency = math.pow(0.5, age_hours / half_life)

        score = (
            priority * policy.priority_weight
            + relevance * policy.relevance_weight
            + recency * policy.recency_weight
        )
        if policy.custom_score:
            score += float(policy.custom_score(item))
        return score

    def _fixed_tokens(self, fixed_texts: Iterable[str]) -> int:
        return sum(self.tokenizer.count(text) for text in fixed_texts if text)

    def _category_limit(self, category: str, item_budget: int) -> int | None:
        raw = self.policy.category_limits.get(category)
        if raw is None:
            return None
        if isinstance(raw, float) and 0 <= raw <= 1:
            return int(item_budget * raw)
        return max(0, int(raw))

    def _try_reduce(
        self,
        item: ContextItem,
        target_tokens: int,
        query: str,
    ) -> tuple[str, int] | None:
        if target_tokens <= 0:
            return None

        current = item
        for reducer in self.reducers:
            reduced = reducer.reduce(
                item=current,
                target_tokens=target_tokens,
                tokenizer=self.tokenizer,
                query=query,
            )
            if not reduced:
                continue
            tokens = self.tokenizer.count(reduced)
            if tokens <= target_tokens:
                return reduced, tokens
            current = current.with_content(reduced, token_count=tokens)
        return None

    def _compression_target(self, original_tokens: int) -> int:
        target = max(1, int(original_tokens * self.policy.compression_target_ratio))
        if self.policy.compression_max_tokens is not None:
            target = min(target, self.policy.compression_max_tokens)
        return min(target, max(1, original_tokens - 1))

    def _should_compress(self, item: ContextItem, token_count: int) -> bool:
        mode = self.policy.compression_mode.lower().strip()
        if mode == "none":
            return False
        if token_count < self.policy.compression_min_tokens:
            return False
        if item.pinned and not self.policy.compress_pinned:
            return False
        categories = self.policy.compression_categories
        if categories is not None and item.category not in set(categories):
            return False
        return True

    def _compress_item(
        self,
        item: ContextItem,
        rendered: str,
        token_count: int,
        query: str,
    ) -> tuple[ContextItem, str, int, CompressionEntry | None]:
        if not self._should_compress(item, token_count):
            return item, rendered, token_count, None

        if self.compressor is None:
            raise BudgetConfigurationError(
                "compression is enabled but no compressor is available"
            )

        target_tokens = self._compression_target(token_count)
        mode = self.policy.compression_mode.lower().strip()

        try:
            compressed = self.compressor.compress(
                item=item,
                target_tokens=target_tokens,
                tokenizer=self.tokenizer,
                query=query,
            )
        except Exception as exc:
            if self.policy.compression_failure_mode == "raise":
                raise
            return (
                item,
                rendered,
                token_count,
                CompressionEntry(
                    item_id=item.id,
                    decision=CompressionDecision.FAILED,
                    mode=mode,
                    reason=f"compressor failed; original content retained: {type(exc).__name__}: {exc}",
                    original_tokens=token_count,
                    final_tokens=token_count,
                    category=item.category,
                ),
            )

        if not compressed:
            return (
                item,
                rendered,
                token_count,
                CompressionEntry(
                    item_id=item.id,
                    decision=CompressionDecision.SKIPPED,
                    mode=mode,
                    reason="compressor returned no content; original content retained",
                    original_tokens=token_count,
                    final_tokens=token_count,
                    category=item.category,
                ),
            )

        compressed_tokens = self.tokenizer.count(compressed)
        if compressed_tokens > target_tokens:
            reduced = self._try_reduce(
                item.with_content(compressed, token_count=compressed_tokens),
                target_tokens,
                query,
            )
            if reduced is not None:
                compressed, compressed_tokens = reduced
            else:
                compressed = self.tokenizer.truncate(compressed, target_tokens)
                compressed_tokens = self.tokenizer.count(compressed)

        if compressed_tokens >= token_count:
            return (
                item,
                rendered,
                token_count,
                CompressionEntry(
                    item_id=item.id,
                    decision=CompressionDecision.SKIPPED,
                    mode=mode,
                    reason="compressed output did not reduce token usage; original content retained",
                    original_tokens=token_count,
                    final_tokens=token_count,
                    category=item.category,
                ),
            )

        compressed_item = item.with_content(compressed, token_count=compressed_tokens)
        return (
            compressed_item,
            compressed,
            compressed_tokens,
            CompressionEntry(
                item_id=item.id,
                decision=CompressionDecision.COMPRESSED,
                mode=mode,
                reason=f"compressed toward a target of {target_tokens} tokens",
                original_tokens=token_count,
                final_tokens=compressed_tokens,
                category=item.category,
            ),
        )

    def build(
        self,
        *,
        items: Iterable[ContextItem],
        query: str = "",
        fixed_texts: Iterable[str] = (),
        now: datetime | None = None,
    ) -> ContextBundle:
        now = now or datetime.now(timezone.utc)
        fixed_tokens = self._fixed_tokens(fixed_texts)
        input_budget = self.policy.available_input_tokens

        if fixed_tokens > input_budget:
            raise BudgetConfigurationError(
                f"fixed content requires {fixed_tokens} tokens, "
                f"but only {input_budget} input tokens are available"
            )

        item_budget = input_budget - fixed_tokens
        audit: list[AuditEntry] = []
        compression_audit: list[CompressionEntry] = []
        candidates: list[tuple[ContextItem, str, int, float]] = []
        deduplicator = Deduplicator(
            mode=self.policy.deduplication_mode,
            similarity_threshold=self.policy.deduplication_similarity_threshold,
            custom_normalizer=self.policy.deduplication_normalizer,
        )

        for item in items:
            rendered = item.render()
            token_count = item.token_count
            if token_count is None:
                token_count = self.tokenizer.count(rendered)
            score = self._score(item, now)

            if score < self.policy.minimum_score and not item.pinned:
                audit.append(
                    AuditEntry(
                        item_id=item.id,
                        decision=Decision.DROPPED,
                        reason="score below minimum_score",
                        original_tokens=token_count,
                        final_tokens=0,
                        score=score,
                        category=item.category,
                    )
                )
                continue

            duplicate = deduplicator.find_duplicate(rendered)
            should_deduplicate = duplicate is not None and (
                not item.pinned or self.policy.deduplicate_pinned
            )
            if should_deduplicate and duplicate is not None:
                if duplicate.mode == "similarity":
                    similarity = duplicate.similarity or 0.0
                    reason = (
                        f"similar content already provided by {duplicate.item_id} "
                        f"(similarity={similarity:.3f}, "
                        f"threshold={self.policy.deduplication_similarity_threshold:.3f})"
                    )
                elif duplicate.mode == "normalized":
                    reason = (
                        "normalized content already provided by "
                        f"{duplicate.item_id}"
                    )
                else:
                    reason = (
                        "identical content already provided by "
                        f"{duplicate.item_id}"
                    )

                audit.append(
                    AuditEntry(
                        item_id=item.id,
                        decision=Decision.DEDUPLICATED,
                        reason=reason,
                        original_tokens=token_count,
                        final_tokens=0,
                        score=score,
                        category=item.category,
                    )
                )
                continue

            deduplicator.remember(item.id, rendered)
            item, rendered, token_count, compression_entry = self._compress_item(
                item,
                rendered,
                token_count,
                query,
            )
            if compression_entry is not None:
                compression_audit.append(compression_entry)
            candidates.append((item, rendered, token_count, score))

        pinned = [entry for entry in candidates if entry[0].pinned]
        optional = [entry for entry in candidates if not entry[0].pinned]

        if self.policy.selection_mode == "score_per_token":
            optional.sort(
                key=lambda entry: (
                    entry[3] / max(1, entry[2]),
                    entry[3],
                    -entry[2],
                ),
                reverse=True,
            )
        else:
            optional.sort(
                key=lambda entry: (entry[3], -entry[2]),
                reverse=True,
            )

        pinned.sort(key=lambda entry: (entry[3], -entry[2]), reverse=True)
        ordered = pinned + optional

        selected: list[SelectedItem] = []
        consumed = 0
        category_consumed: dict[str, int] = defaultdict(int)

        for item, rendered, original_tokens, score in ordered:
            remaining = item_budget - consumed
            category_limit = self._category_limit(item.category, item_budget)
            category_remaining = (
                remaining
                if category_limit is None
                else max(0, category_limit - category_consumed[item.category])
            )
            allowed = min(remaining, category_remaining)

            if original_tokens <= allowed:
                selected.append(
                    SelectedItem(
                        item=item.with_content(item.content, token_count=original_tokens),
                        rendered=rendered,
                        tokens=original_tokens,
                        score=score,
                    )
                )
                consumed += original_tokens
                category_consumed[item.category] += original_tokens
                audit.append(
                    AuditEntry(
                        item_id=item.id,
                        decision=Decision.INCLUDED,
                        reason="fits token and category budgets",
                        original_tokens=original_tokens,
                        final_tokens=original_tokens,
                        score=score,
                        category=item.category,
                    )
                )
                continue

            can_reduce = not item.pinned or self.policy.allow_reduce_pinned
            reduced = (
                self._try_reduce(item, allowed, query)
                if can_reduce and allowed > 0
                else None
            )

            if reduced:
                reduced_text, reduced_tokens = reduced
                selected.append(
                    SelectedItem(
                        item=item.with_content(reduced_text, token_count=reduced_tokens),
                        rendered=reduced_text,
                        tokens=reduced_tokens,
                        score=score,
                    )
                )
                consumed += reduced_tokens
                category_consumed[item.category] += reduced_tokens
                audit.append(
                    AuditEntry(
                        item_id=item.id,
                        decision=Decision.REDUCED,
                        reason="reduced to fit remaining token/category budget",
                        original_tokens=original_tokens,
                        final_tokens=reduced_tokens,
                        score=score,
                        category=item.category,
                    )
                )
                continue

            if item.pinned:
                raise PinnedItemOverflowError(
                    f"pinned item {item.id!r} requires {original_tokens} tokens "
                    f"and cannot fit into the remaining {allowed} tokens"
                )

            reason = (
                "category token limit reached"
                if category_limit is not None and category_remaining <= 0
                else "insufficient remaining token budget"
            )
            audit.append(
                AuditEntry(
                    item_id=item.id,
                    decision=Decision.DROPPED,
                    reason=reason,
                    original_tokens=original_tokens,
                    final_tokens=0,
                    score=score,
                    category=item.category,
                )
            )

        return ContextBundle(
            selected=selected,
            audit=audit,
            input_budget=input_budget,
            fixed_tokens=fixed_tokens,
            context_tokens=consumed,
            total_input_tokens=fixed_tokens + consumed,
            query=query,
            compression_mode=self.policy.compression_mode.lower().strip(),
            compression_audit=compression_audit,
        )
