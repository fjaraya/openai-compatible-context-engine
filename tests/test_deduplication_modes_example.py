from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "compare_deduplication_modes.py"
)


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "compare_deduplication_modes_example",
        EXAMPLE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load comparison example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeduplicationModesExampleTests(unittest.TestCase):
    def test_modes_remove_increasing_levels_of_duplication(self):
        example = load_example_module()
        tokenizer = example.ApproximateTokenizer()
        items = example.sample_context_items()

        selected_counts = {}
        deduplicated_counts = {}
        for mode in example.DEFAULT_MODES:
            _, report = example.build_for_mode(
                mode=mode,
                items=items,
                tokenizer=tokenizer,
                similarity_threshold=0.84,
            )
            selected_counts[mode] = len(report["selected_ids"])
            deduplicated_counts[mode] = report["decisions"].get(
                "deduplicated",
                0,
            )

        self.assertEqual(
            selected_counts,
            {
                "none": 5,
                "exact": 4,
                "normalized": 3,
                "similarity": 2,
            },
        )
        self.assertEqual(
            deduplicated_counts,
            {
                "none": 0,
                "exact": 1,
                "normalized": 2,
                "similarity": 3,
            },
        )

    def test_call_api_path_invokes_every_selected_mode(self):
        example = load_example_module()
        calls: list[str] = []

        def fake_call_endpoint(*, mode, messages, max_tokens, temperature):
            calls.append(mode)
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
            "exact",
            "normalized",
            "similarity",
        ]

        output = io.StringIO()
        with patch.object(example, "call_endpoint", side_effect=fake_call_endpoint):
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(output):
                    example.main()

        self.assertIn("IMPORTANT API COMPARISON NOTE", output.getvalue())

        self.assertEqual(calls, list(example.DEFAULT_MODES))


if __name__ == "__main__":
    unittest.main()
