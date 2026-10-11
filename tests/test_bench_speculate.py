"""Check benchmark aggregation with deliberately uneven timings and draft counts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "tools/eval/bench_speculate.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
