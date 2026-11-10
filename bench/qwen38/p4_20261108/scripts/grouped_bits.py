"""Old vs new FP32 grouped MoE: act (gate+up) and down outputs byte-equal, every routed format.
usage: grouped_bits.py OLD_NATIVE OLD_LIB NEW_NATIVE NEW_LIB   (under the GPU lock)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
import test_qwen38_fast_kernels as t
old_native, old_lib, new_native, new_lib = sys.argv[1:5]
import os
EXTRA = {"PICOFORGE_PREFILL_PRECISION": os.environ.get("PICOFORGE_PREFILL_PRECISION", "fp32")}
def both(fn, *args):
    out = []
    for native, lib in ((old_native, old_lib), (new_native, new_lib)):
        t.NATIVE = native
        out.append(fn(*args, env={"Q38_PACKED_LIBRARY": lib} | EXTRA))
    return out
cases = 0
for name, kind in sorted(t.INDEXED_KINDS.items()):
    rng, block = np.random.default_rng(7000 + kind), t.oracle.FORMATS[kind][0]
    for n, physical, logical, experts, tokens in ((70, 768, 640, 24, 40), (33, 2560 if block == 256 else 96, 2560 if block == 256 else 72, 24, 40),
                                                  (32, 2560, 2560, 16, 300)):
        slots = 10
        ids = np.concatenate([rng.permutation(experts)[:slots] for _ in range(tokens)])
        x = rng.uniform(-2, 2, (tokens, logical)).astype(np.float32)
        gate, up = (t.quant_rows(kind, physical, experts * n, rng).tobytes() for _ in range(2))
        a, b = both(t.run_gate_up, "grouped", kind, physical, logical, n, experts, gate, up, ids, x, slots)
        assert a[0] == b[0] == 0, (name, a[0], b[0], a[4], b[4])
        assert np.array_equal(a[1].view("<u4"), b[1].view("<u4")), (name, n, "gate_up")
        rows = rng.uniform(-2, 2, (len(ids), logical)).astype(np.float32)
        a, b = both(t.run_indexed, "grouped", kind, physical, logical, n, experts, gate, ids, rows, 1)
        assert a[0] == b[0] == 0, (name, a[0], b[0])
        assert np.array_equal(a[1].view("<u4"), b[1].view("<u4")), (name, n, "down")
        cases += 2
print(f"grouped {EXTRA['PICOFORGE_PREFILL_PRECISION']} old vs new: {cases} outputs byte-equal")
