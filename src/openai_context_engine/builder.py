from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

from .deduplication import Deduplicator, SUPPORTED_DEDUPLICATION_MODES
from .exceptions import BudgetConfigurationError, PinnedItemOverflowError
from .models import (
    AuditEntry,
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
        self._validate_policy()

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
        )
