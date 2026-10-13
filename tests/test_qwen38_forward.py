"""G1 gates: the CPU graph against the composed upstream oracle.

Gates are frozen here BEFORE the first multi-layer measurement (integration
§6 forbids widening them after a failure). All errors are err_inf scaled by
max(1, maxabs(reference)) over each traced boundary of the whole prefix.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_forward as oracle

GATE_HC = 2e-5      # C §6: HC post-projection, on the SAME native input
GATE_LOCAL = 5e-4   # §6: composed layer, T <= 1025, native layer input replayed
GATE_E2E = 1e-3     # whole stack, oracle fed only with the tokens
GATE_PLE = 2e-4     # E §6: Delta atol = rtol
# 24 tokens: two EOS (511) cut the PLE n-gram history; >= 12 positions make
# QSA select a strict subset of the 4-token blocks (budget 8 = 2 blocks).
TOKENS = [5, 17, 300, 42, 7, 99, 511, 3, 3, 3, 250, 128, 64, 1, 2, 400, 77, 33, 12, 511, 9, 8, 200, 150]
STAGES = ("ple_delta", "attn_mixed", "attn_inject", "attn_contribution", "attn_residual",
          "ffn_mixed", "ffn_inject", "moe", "output")
SLICES = (1,)       # prefix depths exercised so far
NATIVE = GGUF = None
REFS = None


def run_native(layers, chunk, tokens=TOKENS):
    out = Path(tempfile.mkdtemp(prefix="g1-", dir=ROOT / "build/qwen38-g1"))
    done = subprocess.run([str(NATIVE), str(GGUF), str(out), str(layers), str(chunk), *map(str, tokens)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    if done.returncode:
        raise AssertionError(f"native graph failed: {done.stderr.strip()}")
    info = json.loads(done.stdout.strip().splitlines()[-1])
    trace = {p.name[:-4]: np.fromfile(p, "<f4").reshape(len(tokens), -1) for p in out.glob("*.f32")}
    return info, trace


def scaled(actual, expected):
    a, r = np.asarray(actual, np.float64).reshape(len(actual), -1), np.asarray(expected, np.float64)
    r = r.reshape(len(r), -1)
    if a.shape != r.shape:
        raise AssertionError(f"shape {a.shape} != {r.shape}")
    return float(np.abs(a - r).max() / max(1.0, np.abs(r).max()))


class GraphTests(unittest.TestCase):
    def check(self, actual, expected, gate, label):
        error = scaled(actual, expected)
        print(f"  {label:34s} {error:.3e} (gate {gate:g})")
        self.assertLessEqual(error, gate, label)
        return error

    def compare(self, native, ref, layers, gate, tag):
        for i in range(layers):
            for stage in STAGES:
                key = f"layer{i}.{stage}"
                if key in ref:
                    self.check(native[key], ref[key], gate, f"{tag} {key}")
        self.check(native["global.final"], ref["final"], gate, f"{tag} final")
        self.check(native["logits"], ref["logits"], gate, f"{tag} logits")

    def test_prefix_slices_end_to_end(self):
        """Oracle sees only tokens; every boundary of each prefix stack."""
        for layers in SLICES:
            _, native = run_native(layers, len(TOKENS))
            ref = oracle.flatten(oracle.forward(GGUF, TOKENS, REFS, layers=layers))
            self.compare(native, ref, layers, GATE_E2E, f"L={layers}")

    def test_embedding_mixer_and_head_slice(self):
        """layers=0: embedding repeat, output-HC mixer and untied head only."""
        info, native = run_native(0, len(TOKENS))
        self.assertFalse(info["valid"])  # a prefix slice must leave the session invalid
        ref = oracle.flatten(oracle.forward(GGUF, TOKENS, REFS, layers=0))
        np.testing.assert_array_equal(native["global.embedding"], ref["embedding"])
        self.check(native["global.final"], ref["final"], GATE_HC, "final mixer")
        self.check(native["logits"], ref["logits"], GATE_LOCAL, "logits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    NATIVE, GGUF = args.native.resolve(), args.gguf.resolve()
    with torch.no_grad():
        REFS = oracle.load_refs()
        unittest.main(argv=[sys.argv[0], *remaining])
