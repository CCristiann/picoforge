"""Check benchmark aggregation with deliberately uneven timings and draft counts."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "tools/eval/bench_speculate.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SummaryTests(unittest.TestCase):
    def test_inconsistent_timers_are_rejected(self):
        metrics = {"prefill_s": 6.0, "draft_prefill_s": .1, "decode_s": 3.0,
                   "draft_s": .3, "verify_s": 2.5, "generation_s": 9.1, "process_s": 9.3}
        self.assertTrue(bench.timing_consistent(metrics))
        self.assertFalse(bench.timing_consistent({**metrics, "generation_s": 346.1, "process_s": 16.6}))
        self.assertFalse(bench.timing_consistent({**metrics, "verify_s": 4.0}))
        for bad in (-1.0, math.inf, math.nan):
            self.assertFalse(bench.timing_consistent({**metrics, "decode_s": bad}))

    def test_fingerprints_include_engine_tokenizer_inputs(self):
        names = ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            for name in names:
                (model / name).write_text("test")
            files = bench.fingerprint(model)
            self.assertEqual(set(files), set(names))
            self.assertTrue(all(v["bytes"] == 4 for v in files.values()))

    def test_weighted_acceptance_and_ratio_of_medians(self):
        records = []
        for name, rates in (("plain", [10, 20, 100]), ("model-2", [40, 10, 60])):
            for i, rate in enumerate(rates):
                records.append({"prompt": "one", "config": name, "tok_s": rate,
                                "generated": 11, "prompt_tokens": 7, "passes": 5,
                                "accepted": [0, 2, 10][i] if name != "plain" else 0,
                                "drafted": [1, 2, 100][i] if name != "plain" else 0,
                                "prefill_s": .2, "draft_prefill_s": .1,
                                "draft_s": .05, "verify_s": .1,
                                "generation_s": .3 + 10 / rate,
                                "process_s": 1 + i, "peak_rss_bytes": 100 + i,
                                "identical_to_greedy": True})
        plain, model = bench.summarize(records, ["plain", "model-2"])
        self.assertEqual(plain["median_tok_s"], 20)
        self.assertEqual(model["median_tok_s"], 40)
        self.assertEqual(model["speedup"], 2)
        self.assertAlmostEqual(model["acceptance"], 12 / 103)
        self.assertEqual(plain["acceptance"], 0)
        self.assertEqual(model["tokens_per_pass"], 2)
        self.assertEqual(model["peak_rss_bytes"], 102)
        self.assertEqual(model["median_process_s"], 2)
        self.assertEqual(model["p90_process_s"], 3)
        self.assertEqual(model["reps"], 3)
        self.assertEqual(model["expert_predictions"], 0)
        self.assertEqual(model["median_expert_cost_s"], 0.0)
        records[-1].update(expert_predictions=2, expert_cost_s=.001,
                           expert_abs_error_s=.006, baseline_abs_error_s=.010, route_abs_error=4.0)
        expert = bench.summarize(records, ["plain", "model-2"])[1]
        self.assertEqual(expert["expert_mae_ms"], 3.0)
        self.assertEqual(expert["baseline_mae_ms"], 5.0)
        self.assertAlmostEqual(expert["route_mae"], 4.0 / 15)
        records[-1]["identical_to_greedy"] = False
        self.assertFalse(bench.summarize(records, ["plain", "model-2"])[1]["identical_to_greedy"])

    def test_percentiles_are_samples(self):
        self.assertEqual(bench.percentile([5, 1, 4, 2, 3], .1), 1)
        self.assertEqual(bench.percentile([5, 1, 4, 2, 3], .9), 5)

    def test_draft_limits(self):
        for name in ("model-0", "lookup-32", "model--1", "model-2junk", "unknown"):
            with self.assertRaises(ValueError):
                bench.configuration(name, Path("draft"), "prompt", 128)
        self.assertEqual(bench.configuration("model-31", Path("draft"), "prompt", 128)[-1], "31")
        self.assertEqual(bench.configuration("model-auto-experts", Path("draft"), "prompt", 128),
                         ["--spec-model", "draft", "prompt", "128", "auto-experts"])


if __name__ == "__main__":
    unittest.main()
