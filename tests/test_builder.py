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
