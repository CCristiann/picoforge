"""CPU-only tests, run directly with the existing tools/venv Python.

First cache the oracle with tools/oracle/qwen38_ngram.py --fetch-source.
Subsequent runs are offline and never open the model checkpoint.
"""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
from qwen38_ngram import IdOracle, SOURCE, fixture, verified_source, validate

GOLDEN = json.loads((ROOT / "tests/fixtures/qwen38/ngram.json").read_text())


class OracleTests(unittest.TestCase):
    def test_frozen_provenance_and_golden(self):
        actual = fixture()
        # Library version is recorded, but source and all outputs must match.
        actual.pop("torch_version")
        expected = copy.deepcopy(GOLDEN)
        expected.pop("torch_version")
        self.assertEqual(actual, expected)

    def test_chunks_and_interleaved_sessions(self):
        rng = random.Random(3811)
        for case in GOLDEN["cases"]:
            cfg = case["config"]
            oracle = IdOracle(cfg)
            tokens = [rng.randrange(cfg["vocab"]) for _ in range(1100)]
            for i in (0, 1, 7, 8, 255, 256, 1023, 1024):
                tokens[i] = cfg["eos"]
            expected = oracle.ids(tokens)
            for chunk in (1, 2, 3, 7, 9, 10, 255, 256, 1023, 1024):
                got = []
                for i in range(0, len(tokens), chunk):
                    # An unrelated invocation must never alter this history.
                    oracle.ids([cfg["eos"], 0])
                    got += oracle.ids(tokens[i:i+chunk], tokens[max(0, i-2):i])
                self.assertEqual(got, expected, (case["name"], chunk))
            self.assertEqual(oracle.ids([]), [])

    def test_input_and_metadata_errors(self):
        cfg = GOLDEN["cases"][0]["config"]
        oracle = IdOracle(cfg)
        for tokens in ([-1], [100], [True], [1.5], [0]*32769):
            with self.assertRaises(ValueError):
                oracle.ids(tokens)
        with self.assertRaises(ValueError):
            oracle.ids([1], [1, 2, 3])
        for key, value in (("multipliers", [2**63, 5, 7]), ("multipliers", [3, 5]),
                           ("offsets", [0, 12, 24, 41]), ("sizes", [11, 0, 17, 19]),
                           ("rows", 59), ("eos", 100), ("ngram_size", 1)):
            bad = {**cfg, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(bad)

    def test_unverified_source_is_not_executed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            path = Path(directory) / "bad.py"
            path.write_text("raise RuntimeError('must not execute')")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                IdOracle(GOLDEN["cases"][0]["config"], source=path)
        self.assertTrue(verified_source(SOURCE))


if __name__ == "__main__":
    unittest.main()
