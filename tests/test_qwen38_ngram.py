"""CPU-only tests, run directly with the existing tools/venv Python.

First cache the oracle with tools/oracle/qwen38_ngram.py --fetch-source.
Subsequent runs are offline and never open the model checkpoint.
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import struct
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
from qwen38_ngram import IdOracle, SOURCE, fixture, verified_source, validate

GOLDEN = json.loads((ROOT / "tests/fixtures/qwen38/ngram.json").read_text())
BINARY = ROOT / "build/qwen38-e/test_ngram"


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["clang", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic",
                        "-Wshadow", "-Wconversion", "-Werror", "-fsanitize=address,undefined",
                        "-ffp-contract=off", "-fno-omit-frame-pointer", "-Isrc", "src/qwen38_ngram.c",
                        "tests/test_qwen38_ngram.c", "-o", str(BINARY)], cwd=ROOT, check=True)

    def test_native_errors(self):
        result = subprocess.run([str(BINARY)], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("PASS", result.stdout)

    def native_ids(self, cfg, tokens, chunk):
        fields = [cfg[k] for k in ("ngram_size", "heads_per_ngram", "vocab", "eos", "rows")]
        fields += cfg["multipliers"] + cfg["offsets"] + cfg["sizes"]
        fields += [len(tokens), min(chunk, len(tokens))] + tokens
        result = subprocess.run([str(BINARY), "--hash"], input=" ".join(map(str, fields)),
                                capture_output=True, text=True, check=True)
        values, prev = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        h = (cfg["ngram_size"]-1)*cfg["heads_per_ngram"]
        expected_prev = ([cfg["eos"]]*(cfg["ngram_size"]-1) + tokens)[-(cfg["ngram_size"]-1):]
        self.assertEqual(prev, expected_prev[::-1])
        return [values[i:i+h] for i in range(0, len(values), h)]

    def test_native_ids_against_independent_upstream(self):
        rng = random.Random(2837)
        for case in GOLDEN["cases"]:
            cfg = case["config"]
            oracle = IdOracle(cfg)
            self.assertEqual(self.native_ids(cfg, case["tokens"], 1), case["ids"])
            tokens = [rng.randrange(cfg["vocab"]) for _ in range(1100)]
            for i in (0, 1, 7, 255, 256, 1023, 1024):
                tokens[i] = cfg["eos"]
            tokens[3:5] = [0, cfg["vocab"]-1]
            expected = oracle.ids(tokens)
            for chunk in (1, 2, 3, 7, 9, 10, 255, 256, 1023, 1024, len(tokens)):
                self.assertEqual(self.native_ids(cfg, tokens, chunk), expected)

    def test_other_geometries_and_integer_boundaries(self):
        configs = []
        for n, per in ((2, 1), (4, 3), (16, 64)):
            heads = (n-1)*per
            configs.append(dict(ngram_size=n, heads_per_ngram=per, vocab=100, eos=99,
                                multipliers=list(range(1, 2*n, 2)), sizes=[11]*heads,
                                offsets=[11*i for i in range(heads)], rows=11*heads))
        configs.append(dict(ngram_size=3, heads_per_ngram=1, vocab=2**31, eos=2**31-1,
                            multipliers=[(2**63-1)//(2**31-1)]*3,
                            sizes=[2**32-18, 17], offsets=[0, 2**32-18], rows=2**32-1))
        for cfg in configs:
            tokens = [0, 1, cfg["eos"], 2, cfg["vocab"]-1, 3, 4]*3
            expected = IdOracle(cfg).ids(tokens)
            for chunk in (1, 2, 7, len(tokens)):
                self.assertEqual(self.native_ids(cfg, tokens, chunk), expected)

    def lookup(self, words, width, ids, *, base=0, capacity=None, max_read=65536):
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen38-e") as directory:
            path = Path(directory) / "tiny.bf16"
            payload = struct.pack("<" + "H"*len(words), *words)
            path.write_bytes(bytes(base) + payload)
            fields = [base, len(words)//width, width, len(ids),
                      capacity or max(1, len(ids)), max_read, 4 << 20] + ids
            result = subprocess.run([str(BINARY), "--lookup", str(path)],
                                    input=" ".join(map(str, fields)), capture_output=True,
                                    text=True, check=True, cwd=ROOT)
        lines = result.stdout.splitlines()
        status = int(lines[0].split()[1])
        values = [int(x, 16) for x in lines[1].split()] if len(lines)>1 else []
        return status, values

    def test_all_finite_bf16_patterns_against_torch(self):
        # 128 KiB synthetic file; PyTorch conversion is independent of C shifts.
        words = list(range(65536))
        ids = [b for b in words if b & 0x7f80 != 0x7f80]
        status, got = self.lookup(words, 1, ids)
        raw = bytearray(struct.pack("<" + "H"*len(ids), *ids))
        converted = torch.frombuffer(raw, dtype=torch.bfloat16).float().numpy().tobytes()
        expected = list(struct.unpack("<" + "I"*len(ids), converted))
        self.assertEqual(status, 0)
        self.assertEqual(got, expected)  # includes signed zero and subnormals

    def test_bf16_duplicates_order_page_boundary_and_last_row(self):
        words = [0x3000 + ((r*17+c) % 4096) for r in range(300) for c in range(160)]
        ids = [299, 0, 9, 5, 6, 5, 9, 7] + list(range(300))
        status, got = self.lookup(words, 160, ids, base=4094)
        self.assertEqual(status, 0)
        self.assertEqual(got, [words[r*160+c] << 16 for r in ids for c in range(160)])

    def test_nonfinite_invalid_row_and_staging_errors_publish_nothing(self):
        for bad in GOLDEN["nonfinite_bf16_bits"]:
            status, values = self.lookup([0x3f80, 0x4000, bad], 1, [0, 1, 2], max_read=2)
            self.assertEqual(status, 16)  # Q38_NONFINITE, after two successful intervals
            self.assertEqual(values, [])
        for ids, capacity, code in (([0, 2], 2, 4), ([0, 1], 1, 10)):
            status, values = self.lookup([0, 0], 1, ids, capacity=capacity)
            self.assertEqual(status, code)
            self.assertEqual(values, [])

    def test_hash_to_lookup_composition(self):
        case = GOLDEN["cases"][0]
        cfg, width = case["config"], 13
        ids = self.native_ids(cfg, case["tokens"], 2)
        self.assertEqual(ids, case["ids"])
        words = [0x3000+((r*17+c) % 4096) for r in range(cfg["rows"]) for c in range(width)]
        flat = [r for token in ids for r in token]
        status, got = self.lookup(words, width, flat, max_read=78)
        self.assertEqual(status, 0)
        self.assertEqual(got, [words[r*width+c] << 16 for token in case["ids"]
                              for r in token for c in range(width)])


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
