"""CPU-only tests, run directly with the existing tools/venv Python.

First cache the oracle with tools/oracle/qwen38_ngram.py --fetch-source.
Subsequent runs are offline and never open the model checkpoint.
"""
import copy
import json
import math
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
from qwen38_ngram import IdOracle, GateOracle, PleOracle, SOURCE, fixture, verified_source, validate

GOLDEN = json.loads((ROOT / "tests/fixtures/qwen38/ngram.json").read_text())
BINARY = ROOT / "build/qwen38-e/test_ngram"


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["clang", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic",
                        "-Wshadow", "-Wconversion", "-Werror", "-fsanitize=address,undefined",
                        "-ffp-contract=off", "-fno-fast-math", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-Isrc", "src/qwen38_ngram.c",
                        "src/qwen38_ple.c", "src/qwen38_conv.c", "src/qwen38_base.c",
                        "src/qwen38_gdn.c", "src/gguf_plan.c",
                        "src/qwen38_packed.c", "src/qwen38_config.c", "src/qwen38_weights.c", "src/gguf.c", "src/gguf_reader.c", "src/util.c",
                        "tests/test_qwen38_ngram.c", "-o", str(BINARY)], cwd=ROOT, check=True)

    def test_native_errors(self):
        result = subprocess.run([str(BINARY)], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("PASS", result.stdout)

    def test_complete_ple_f32_q8_abort_reset_and_history(self):
        self.check_complete_ple(23, (1, 2, 7, 9, 23), (0, 8))

    def test_complete_ple_32k_history(self):
        self.check_complete_ple(32768, (255, 1024), (8,))

    def check_complete_ple(self, t, chunks, formats):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(3877)
        cfg = GOLDEN["cases"][0]["config"]
        h, d, row_width, kernel = 2, 8, 8, 4
        channels, width = h*d, 4*row_width
        tokens = torch.randint(0, 99, (t,), generator=rng, dtype=torch.int32)
        tokens[[0, 8, 9, 16]] = cfg["eos"]
        table = (torch.randn(cfg["rows"], row_width, generator=rng)*0.1).to(torch.bfloat16)
        residual = torch.randn(t, h, d, generator=rng)
        gains = [torch.randint(8, 25, (channels,), generator=rng).float()/16 for _ in range(3)]
        taps = torch.randn(channels, kernel, generator=rng)*0.2
        worst = 0.0
        for kind in formats:
            key = torch.randint(-9, 10, (channels, width), generator=rng).float()/64
            value = torch.randint(-9, 10, (d, width), generator=rng).float()/64
            oracle = PleOracle(cfg, table, key, value, gains, taps)
            expected, _, trace = oracle.run(tokens.tolist(), residual)
            _, gate_ref, _ = GateOracle().gate(trace["norm_key"].reshape(t,h,d),
                trace["norm_query"].reshape(t,h,d), trace["value_proj"])
            with tempfile.TemporaryDirectory(dir=BINARY.parent) as directory:
                path = Path(directory)
                fields = [cfg[k] for k in ("ngram_size", "heads_per_ngram", "vocab", "eos", "rows")]
                fields += [row_width, d, h, kernel, 1e-6, t, kind]
                fields += cfg["multipliers"] + cfg["offsets"] + cfg["sizes"]
                (path / "config.txt").write_text(" ".join(map(str, fields)))
                for name, tensor in (("tokens.bin", tokens), ("residual.bin", residual),
                                     ("table.bin", table.view(torch.uint16)), ("taps.bin", taps),
                                     *zip(("gain_key.bin", "gain_query.bin", "gain_conv.bin"), gains)):
                    (path / name).write_bytes(tensor.numpy().tobytes())
                for name, tensor in (("key.bin", key), ("value.bin", value)):
                    # Construct exact dyadic Q8 fixtures from known weights; no
                    # native decoder participates in the expected values.
                    payload = tensor.numpy().tobytes() if kind==0 else b"".join(
                        struct.pack("<e", 1/64)+bytes((int(v*64)&255 for v in row))
                        for row in tensor.tolist())
                    (path / name).write_bytes(payload)
                for chunk in chunks:
                    run = subprocess.run([str(BINARY), "--ple", str(path), str(chunk)],
                                         capture_output=True, text=True, cwd=ROOT)
                    self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                    def read(name, shape):
                        return torch.frombuffer(bytearray((path/name).read_bytes()), dtype=torch.float32).reshape(shape)
                    actual = read("delta.out", expected.shape)
                    worst = max(worst, (actual-expected).abs().max().item())
                    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                    self.assertLessEqual((torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)).item(), 1e-4)
                    self.assertTrue(torch.equal(actual, read("replay.out", expected.shape)))
                    for name, reference in (("z.out", trace["norm_conv"]), ("key_norm.out", trace["norm_key"]),
                                            ("query_norm.out", trace["norm_query"]), ("value.out", trace["value_proj"])):
                        torch.testing.assert_close(read(name, reference.shape), reference, atol=2e-4, rtol=2e-4)
                    self.assertLessEqual((read("gate.out", gate_ref.shape)-gate_ref).abs().max().item(),1e-4)
                    length=(kernel-1)*cfg["ngram_size"]
                    histories=[]
                    for stop in range(chunk, t+chunk, chunk):
                        prefix=trace["norm_conv"][:min(stop,t)]
                        histories.append(torch.nn.functional.pad(prefix,(0,0,max(0,length-len(prefix)),0))[-length:])
                    history_ref=torch.stack(histories)
                    history=read("history.out",history_ref.shape)
                    torch.testing.assert_close(history, history_ref, atol=2e-4, rtol=2e-4)
                    if torch.linalg.vector_norm(history_ref)>=1:
                        self.assertLessEqual((torch.linalg.vector_norm(history-history_ref)/torch.linalg.vector_norm(history_ref)).item(),1e-4)
                    self.assertTrue(torch.equal(history,read("replay_history.out",history.shape)))
        print(f"PLE composed T={t} max |Delta-ref|={worst:.9g}; formats={formats}, chunks={chunks}")

    def native_gate(self, key, query, value, pad=0):
        fields = list(key.shape) + [pad]
        for x in (key, query, value):
            fields += x.flatten().tolist()
        run = subprocess.run([str(BINARY), "--gate"], input=" ".join(map(str, fields)),
                             cwd=ROOT, capture_output=True, text=True, check=True)
        lines = run.stdout.splitlines()
        self.assertEqual(lines[0], "status 0")
        tensors = [torch.tensor([float(v) for v in line.split()], dtype=torch.float32)
                   for line in lines[1:]]
        return tensors[0].reshape(key.shape[:2]), tensors[1].reshape(key.shape[:2]), tensors[2].reshape(key.shape)

    def test_native_gate_signed_root_and_cancellation(self):
        oracle = GateOracle()
        for case in GOLDEN["gate_cases"]:
            inputs = [torch.tensor(case[k], dtype=torch.float32) for k in ("key", "query", "value")]
            ref_dot, ref_g, ref_u = oracle.gate(*inputs)
            for pad in (0, 1, 7):
                dot, gate, output = self.native_gate(*inputs, pad=pad)
                self.assertTrue(torch.equal(dot, ref_dot), (case["name"], dot, ref_dot))
                self.assertTrue(torch.equal(torch.sign(dot), torch.sign(ref_dot)))
                self.assertTrue(torch.all(gate[ref_dot == 0] == 0.5))
                self.assertLessEqual((gate-ref_g).abs().max().item(), 1e-4)
                torch.testing.assert_close(output, ref_u, atol=2e-4, rtol=2e-4)

    def test_native_gate_normalized_inputs_strides_and_chunks(self):
        generator = torch.Generator(device="cpu").manual_seed(3812)
        oracle = GateOracle()
        worst_gate, worst_dot = 0.0, 0.0
        for t, s, d in ((1, 1, 1), (3, 2, 7), (17, 4, 160), (3, 4, 2560)):
            key, query = [torch.randn(t, s, d, generator=generator) for _ in range(2)]
            key = key / torch.sqrt(key.square().mean(-1, keepdim=True)+1e-6)
            query = query / torch.sqrt(query.square().mean(-1, keepdim=True)+1e-6)
            value = torch.randn(t, d, generator=generator)*3
            ref_dot, ref_g, ref_u = oracle.gate(key, query, value)
            reference64 = (key.double()*query.double()).sum(-1)/math.sqrt(d)
            # Independent FP64 sum and a conservative FP32 reduction bound,
            # fixed analytically rather than fitted to measured differences.
            eps = 2**-24
            gamma = (2*d+4)*eps/(1-(2*d+4)*eps)
            bound = gamma*(key.double()*query.double()).abs().sum(-1)/math.sqrt(d)
            for pad in (0, 1, 7):
                dot, gate, output = self.native_gate(key, query, value, pad)
                self.assertTrue(torch.all((dot.double()-reference64).abs() <= bound+1e-45))
                self.assertTrue(torch.equal(torch.sign(dot), torch.sign(ref_dot)))
                worst_dot = max(worst_dot, (dot-ref_dot).abs().max().item())
                worst_gate = max(worst_gate, (gate-ref_g).abs().max().item())
                self.assertLessEqual((gate-ref_g).abs().max().item(), 1e-4)
                torch.testing.assert_close(output, ref_u, atol=2e-4, rtol=2e-4)
                relative_l2 = torch.linalg.vector_norm(output-ref_u)/torch.linalg.vector_norm(ref_u)
                self.assertLessEqual(relative_l2.item(), 1e-4)
                streamed = [self.native_gate(key[i:i+1], query[i:i+1], value[i:i+1], pad) for i in range(t)]
                for idx, full in enumerate((dot, gate, output)):
                    self.assertTrue(torch.equal(torch.cat([x[idx] for x in streamed]), full))
        print(f"PLE measured max |gate-ref|={worst_gate:.9g}, |dot-ref|={worst_dot:.9g}")

    def test_native_gate_empty_and_saturated(self):
        key = torch.empty(0, 2, 3)
        dot, gate, output = self.native_gate(key, key.clone(), torch.empty(0, 3), pad=7)
        self.assertEqual(output.shape, (0, 2, 3))
        self.assertEqual(dot.numel()+gate.numel(), 0)
        key = torch.tensor([[[100.], [-100.]]])
        dot, gate, output = self.native_gate(key, torch.full_like(key, 100.), torch.tensor([[1.]]))
        self.assertEqual(gate[0, 0], 1.)
        self.assertLess(gate[0, 1], 1e-40)
        self.assertTrue(torch.isfinite(output).all())

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
    def test_gate_upstream_analytical_cases(self):
        oracle = GateOracle()
        for case in GOLDEN["gate_cases"]:
            inputs = [torch.tensor(case[k], dtype=torch.float32) for k in ("key", "query", "value")]
            dot, gate, output = oracle.gate(*inputs)
            self.assertEqual(dot.tolist(), case["dot"])
            self.assertEqual(gate.tolist(), case["gate"])
            self.assertEqual(output.tolist(), case["output"])
        # The signed root has sign(0)=0, unlike copysign(sqrt(eps), 0).
        first = GOLDEN["gate_cases"][0]
        self.assertEqual(first["gate"][0][0], 0.5)
        self.assertGreater(first["gate"][0][1], 0.5)
        self.assertLess(first["gate"][0][2], 0.5)
        self.assertAlmostEqual(first["gate"][0][3], 1/(1+math.exp(-2)), places=7)

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
