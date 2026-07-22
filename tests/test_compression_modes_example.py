from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "compare_compression_modes.py"
)


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "compare_compression_modes_example",
        EXAMPLE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load compression comparison example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompressionModesExampleTests(unittest.TestCase):
    def test_extractively_compressed_context_is_smaller(self):
        example = load_example_module()
        tokenizer = example.ApproximateTokenizer()
        items = example.sample_context_items()

        _, no_compression = example.build_for_mode(
            mode="none",
            items=items,
            tokenizer=tokenizer,
        )
        _, extractive = example.build_for_mode(
            mode="extractive",
            items=items,
            tokenizer=tokenizer,
        )

        self.assertGreater(
            no_compression["estimated_message_tokens"],
            extractive["estimated_message_tokens"],
        )
        self.assertEqual(extractive["compressed_ids"], ["monthly-observations"])

    def test_llm_mode_uses_openai_compatible_client(self):
        example = load_example_module()
        calls: list[dict] = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    "Availability was 99.95 percent. "
                                    "Investigate latency and review capacity."
                                )
                            )
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        tokenizer = example.ApproximateTokenizer()
        _, report = example.build_for_mode(
            mode="llm",
            items=example.sample_context_items(),
            tokenizer=tokenizer,
            client=client,
            model="test-model",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(report["compressed_ids"], ["monthly-observations"])

    def test_call_api_path_invokes_all_modes_and_compression(self):
        example = load_example_module()
        compression_calls: list[dict] = []
        final_modes: list[str] = []

        class FakeCompletions:
            def create(self, **kwargs):
                compression_calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Compressed monthly results and actions."
                            )
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

        def fake_call_endpoint(*, mode, client, model, messages, temperature):
            final_modes.append(mode)
            return {
                "mode": mode,
                "ok": True,
                "content": f"response for {mode}",
                "error": None,
                "elapsed_seconds": 0.01,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "finish_reason": "stop",
            }

        argv = [
            str(EXAMPLE_PATH),
            "--call-api",
            "--modes",
            "none",
            "extractive",
            "llm",
        ]
        output = io.StringIO()
        with patch.object(example, "create_client", return_value=(client, "test-model")):
            with patch.object(example, "call_endpoint", side_effect=fake_call_endpoint):
                with patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(output):
                        example.main()

        self.assertEqual(final_modes, ["none", "extractive", "llm"])
        self.assertEqual(len(compression_calls), 1)
        self.assertIn("IMPORTANT API COMPARISON NOTE", output.getvalue())
        self.assertIn("MODEL RESPONSE — COMPRESSION MODE: llm", output.getvalue())


if __name__ == "__main__":
    unittest.main()
