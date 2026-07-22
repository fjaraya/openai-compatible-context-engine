from __future__ import annotations

from types import SimpleNamespace
import unittest

from openai_context_engine import (
    ApproximateTokenizer,
    CallableCompressor,
    CompressionDecision,
    ContextBuilder,
    ContextItem,
    ContextPolicy,
    OpenAIChatCompressor,
)


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ApproximateTokenizer(characters_per_token=1)

    def test_none_mode_does_not_compress(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="none",
                compression_min_tokens=10,
            ),
        )
        bundle = builder.build(
            items=[ContextItem("a", "alpha " * 50)],
            query="alpha",
        )
        self.assertEqual(bundle.compressed_ids, [])
        self.assertEqual(bundle.report()["compression"]["audit"], [])

    def test_extractively_compresses_before_selection(self):
        item = ContextItem(
            "large",
            (
                "Routine message without useful detail. " * 20
                + "The required alpha value is 42. "
                + "Routine message without useful detail. " * 20
            ),
            relevance=1.0,
        )
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=2_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="extractive",
                compression_target_ratio=0.20,
                compression_min_tokens=100,
                compression_max_tokens=250,
            ),
        )
        bundle = builder.build(items=[item], query="alpha value")
        self.assertEqual(bundle.compressed_ids, ["large"])
        selected = bundle.selected[0]
        self.assertLess(selected.tokens, self.tokenizer.count(item.render()))
        self.assertIn("alpha", selected.rendered.lower())
        compression = bundle.report()["compression"]
        self.assertGreater(compression["saved_tokens"], 0)

    def test_pinned_items_are_not_compressed_by_default(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="extractive",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
            ),
        )
        bundle = builder.build(
            items=[ContextItem("pinned", "important " * 30, pinned=True)],
            query="important",
        )
        self.assertEqual(bundle.compressed_ids, [])

    def test_pinned_items_can_be_compressed(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="extractive",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
                compress_pinned=True,
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem(
                    "pinned",
                    "Important result is 42. Routine detail. " * 30,
                    pinned=True,
                )
            ],
            query="important result",
        )
        self.assertEqual(bundle.compressed_ids, ["pinned"])

    def test_compression_category_filter(self):
        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            policy=ContextPolicy(
                context_window=2_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="extractive",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
                compression_categories=("documents",),
            ),
        )
        bundle = builder.build(
            items=[
                ContextItem("doc", "Document result. " * 30, category="documents"),
                ContextItem("state", "State result. " * 30, category="state"),
            ],
            query="result",
        )
        self.assertEqual(bundle.compressed_ids, ["doc"])

    def test_custom_compressor_is_supported(self):
        calls: list[str] = []

        def compress(item, target_tokens, tokenizer, query):
            calls.append(item.id)
            return "compressed custom result"

        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            compressor=CallableCompressor(compress),
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="custom",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
            ),
        )
        bundle = builder.build(
            items=[ContextItem("custom", "original " * 30)],
            query="result",
        )
        self.assertEqual(calls, ["custom"])
        self.assertEqual(bundle.compressed_ids, ["custom"])
        self.assertEqual(bundle.selected[0].rendered, "compressed custom result")

    def test_compression_failure_keeps_original_by_default(self):
        def fail(*args, **kwargs):
            raise RuntimeError("backend unavailable")

        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            compressor=CallableCompressor(fail),
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="custom",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
            ),
        )
        original = "original context " * 20
        bundle = builder.build(
            items=[ContextItem("a", original)],
            query="context",
        )
        self.assertEqual(bundle.selected[0].rendered, original)
        entry = bundle.compression_audit[0]
        self.assertEqual(entry.decision, CompressionDecision.FAILED)

    def test_compression_failure_can_raise(self):
        def fail(*args, **kwargs):
            raise RuntimeError("backend unavailable")

        builder = ContextBuilder(
            tokenizer=self.tokenizer,
            compressor=CallableCompressor(fail),
            policy=ContextPolicy(
                context_window=1_000,
                reserved_output_tokens=10,
                safety_margin_tokens=10,
                compression_mode="custom",
                compression_target_ratio=0.25,
                compression_min_tokens=10,
                compression_failure_mode="raise",
            ),
        )
        with self.assertRaises(RuntimeError):
            builder.build(
                items=[ContextItem("a", "original context " * 20)],
                query="context",
            )

    def test_llm_mode_requires_compressor(self):
        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(compression_mode="llm"),
            )

    def test_openai_chat_compressor_calls_compatible_client(self):
        calls: list[dict] = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Availability 99.95%. Follow up by July 31."
                            )
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        compressor = OpenAIChatCompressor(
            client=client,
            model="test-model",
        )
        result = compressor.compress(
            item=ContextItem("a", "long context " * 50),
            target_tokens=80,
            tokenizer=self.tokenizer,
            query="availability and follow up",
        )
        self.assertIn("99.95", result or "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertEqual(calls[0]["max_tokens"], 80)

    def test_invalid_compression_configuration_raises(self):
        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(compression_mode="semantic"),
            )
        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(compression_target_ratio=1.0),
            )
        with self.assertRaises(Exception):
            ContextBuilder(
                tokenizer=self.tokenizer,
                policy=ContextPolicy(compression_failure_mode="ignore"),
            )


if __name__ == "__main__":
    unittest.main()
