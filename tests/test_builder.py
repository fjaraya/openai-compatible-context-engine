from datetime import datetime, timezone
import unittest

from openai_context_engine import (
    ApproximateTokenizer,
    ContextBuilder,
    ContextItem,
    ContextPolicy,
    Decision,
    PinnedItemOverflowError,
)


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ApproximateTokenizer(characters_per_token=1)

    def test_selects_high_value_items(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=100,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                selection_mode="score",
            ),
        )
        bundle = builder.build(
            query="alpha",
            items=[
                ContextItem("low", "x" * 60, priority=0.1, relevance=0.1),
                ContextItem("high", "y" * 60, priority=1.0, relevance=1.0),
            ],
        )
        self.assertIn("high", bundle.selected_ids)

    def test_deduplicates_identical_content(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "same"),
                ContextItem("b", "same"),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a"])
        self.assertTrue(
            any(
                a.item_id == "b" and a.decision == Decision.DEDUPLICATED
                for a in bundle.audit
            )
        )

    def test_none_mode_keeps_exact_duplicates(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="none",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "same"),
                ContextItem("b", "same"),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a", "b"])

    def test_exact_mode_does_not_normalize_content(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="exact",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "Service availability is 99.95%."),
                ContextItem("b", "  service   availability is 99.95%.  "),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a", "b"])

    def test_normalized_mode_ignores_case_and_whitespace(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="normalized",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "Service availability is 99.95%."),
                ContextItem("b", "  service   availability is 99.95%.  "),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a"])
        entry = next(a for a in bundle.audit if a.item_id == "b")
        self.assertEqual(entry.decision, Decision.DEDUPLICATED)
        self.assertIn("normalized content", entry.reason)

    def test_similarity_mode_detects_near_duplicates(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=400,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="similarity",
                deduplication_similarity_threshold=0.85,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem(
                    "a",
                    "The monthly report includes usage, cost, availability, and performance.",
                ),
                ContextItem(
                    "b",
                    "The monthly report includes usage, costs, availability, and performance.",
                ),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a"])
        entry = next(a for a in bundle.audit if a.item_id == "b")
        self.assertEqual(entry.decision, Decision.DEDUPLICATED)
        self.assertIn("similarity=", entry.reason)
        self.assertIn("threshold=0.850", entry.reason)

    def test_similarity_mode_keeps_different_content(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=400,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="similarity",
                deduplication_similarity_threshold=0.95,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "The report includes usage and cost."),
                ContextItem("b", "The office parking policy assigns spaces by arrival time."),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a", "b"])

    def test_custom_normalizer_can_remove_volatile_values(self):
        import re

        def remove_timestamps(text: str) -> str:
            return re.sub(r"2026-07-21T\d{2}:\d{2}:\d{2}Z", "<timestamp>", text)

        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=300,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="normalized",
                deduplication_normalizer=remove_timestamps,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "2026-07-21T10:10:10Z service unavailable"),
                ContextItem("b", "2026-07-21T11:11:11Z service unavailable"),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a"])

    def test_pinned_duplicates_are_retained_by_default(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="exact",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "same"),
                ContextItem("b", "same", pinned=True),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["b", "a"])

    def test_pinned_duplicates_can_be_deduplicated(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                deduplication_mode="exact",
                deduplicate_pinned=True,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("a", "same"),
                ContextItem("b", "same", pinned=True),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["a"])
        entry = next(a for a in bundle.audit if a.item_id == "b")
        self.assertEqual(entry.decision, Decision.DEDUPLICATED)

    def test_dropped_low_score_item_does_not_suppress_later_duplicate(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                minimum_score=0.5,
                deduplication_mode="exact",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("low", "same", priority=0.0, relevance=0.0),
                ContextItem("high", "same", priority=1.0, relevance=1.0),
            ]
        )
        self.assertEqual(bundle.selected_ids, ["high"])
        low_entry = next(a for a in bundle.audit if a.item_id == "low")
        self.assertEqual(low_entry.decision, Decision.DROPPED)

    def test_invalid_deduplication_configuration_raises(self):
        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(deduplication_mode="semantic"),
            )

        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(
                    deduplication_mode="similarity",
                    deduplication_similarity_threshold=1.1,
                ),
            )

    def test_pinned_item_is_retained(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=100,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                selection_mode="score",
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem(
                    "pinned",
                    "p" * 70,
                    pinned=True,
                    priority=0,
                    relevance=0,
                ),
                ContextItem(
                    "optional",
                    "o" * 70,
                    priority=1,
                    relevance=1,
                ),
            ]
        )
        self.assertIn("pinned", bundle.selected_ids)

    def test_reduces_large_item(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=120,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
            ),
        )
        item = ContextItem(
            "large",
            (
                "Irrelevant sentence with many words. "
                "The alpha value is the important answer. "
                "Another irrelevant sentence with many words. "
            ) * 5,
            relevance=1,
        )
        bundle = builder.build(items=[item], query="alpha value")
        selected = bundle.selected[0]
        self.assertLessEqual(selected.tokens, 100)
        self.assertIn("alpha", selected.rendered.lower())
        self.assertTrue(
            any(a.decision == Decision.REDUCED for a in bundle.audit)
        )

    def test_category_limit(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                category_limits={"history": 0.25},
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem(
                    "history",
                    "h" * 100,
                    category="history",
                    priority=1,
                    relevance=1,
                ),
                ContextItem(
                    "document",
                    "d" * 100,
                    category="documents",
                    priority=0.9,
                    relevance=0.9,
                ),
            ]
        )
        history = next(s for s in bundle.selected if s.item.id == "history")
        self.assertLessEqual(history.tokens, 45)

    def test_pinned_overflow_raises(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=30,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                allow_reduce_pinned=False,
            ),
        )
        with self.assertRaises(PinnedItemOverflowError):
            builder.build(
                items=[ContextItem("pinned", "x" * 20, pinned=True)]
            )

    def test_openai_messages(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=200,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
            ),
        )
        bundle = builder.build(items=[ContextItem("a", "content")])
        messages = bundle.to_openai_messages("system", "question")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["content"], "question")
        self.assertIn("context_item", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
