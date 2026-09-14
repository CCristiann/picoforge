"""test_ops.py — judge src/ops.c against the Python oracle.

The C binary prints what it computed; this script recomputes the same thing
with qwen3_forward's own functions and compares. The reference is the oracle,
never a second implementation written here — otherwise a misunderstanding
about, say, RoPE's pairing would simply be repeated twice and agree.

Sizes, seeds and the order in which random values are drawn are mirrored from
tests/test_ops.c. Inputs are multiples of 1/1024, exactly representable, so C
and Python genuinely start from the same bits.

Run:  make test
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "oracle"))
from qwen3_forward import apply_rope, rms_norm, rope_tables, silu, softmax  # noqa: E402

# Calibrated on the first run, not guessed (docs/DESIGN.md, principle 4). Worst
# case across the five primitives was 1.3e-9 relative — one or two fp32 ulps
# on the smallest elements, which is the floor: reductions here are a few
# dozen terms long, so summation order and libm-vs-numpy transcendentals are
# the only divergence available. The budget sits ~8x above that.
#
# It is calibrated for THESE sizes. Longer reductions accumulate more, so a
# larger test case must come with a re-measured budget, never a relaxed one.
MAX_REL_ERROR = 1e-8


class Rng:
    """The LCG from test_ops.c, digit for digit."""

    def __init__(self, seed: int) -> None:
        self.s = seed

    def f(self) -> np.float32:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return np.float32(((self.s >> 16) % 2048) / 1024.0 - 1.0)

    def arr(self, n: int) -> np.ndarray:
        return np.array([self.f() for _ in range(n)], dtype=np.float32)

    def bf16(self, n: int) -> np.ndarray:
        """fp32 truncated to its top 16 bits, i.e. what the checkpoint holds."""
        v = self.arr(n)
        return ((v.view(np.uint32) >> 16) << 16).view(np.float32)


def reference() -> dict[str, np.ndarray]:
    r = Rng(1)
    x, w = r.arr(16), r.bf16(16)
    ref = {"rmsnorm": rms_norm(x, w, 1e-6)}

    r = Rng(2)
    x, w = r.arr(8), r.bf16(40).reshape(5, 8)
    ref["matmul"] = x @ w.T

    r = Rng(3)
    ref["softmax"] = softmax(r.arr(12) * np.float32(8.0))

    r = Rng(4)
    cos, sin = rope_tables(np.array([7]), 16, 1e6)
    ref["rope"] = apply_rope(r.arr(16).reshape(1, 16), cos, sin)[0]

    r = Rng(5)
    ref["silu"] = silu(r.arr(8) * np.float32(4.0))
    return ref


def main() -> None:
    proc = subprocess.run(["tests/test_ops"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"tests/test_ops failed:\n{proc.stderr}")

    got = {}
    for line in proc.stdout.strip().splitlines():
        name, *vals = line.split()
        got[name] = np.array([float(v) for v in vals], dtype=np.float64)

    ref = reference()
    all_ok = True
    print(f"{'op':<10}{'max |Δ|':>12}{'relative':>12}   verdict")
    for name, expect in ref.items():
        if name not in got:
            sys.exit(f"C produced no output for {name}")
        expect = np.asarray(expect, dtype=np.float64).ravel()
        if got[name].shape != expect.shape:
            sys.exit(f"{name}: C gave {got[name].shape}, oracle {expect.shape}")

        abs_err = float(np.max(np.abs(got[name] - expect)))
        scale = float(np.max(np.abs(expect)))
        rel = abs_err / scale if scale > 0 else abs_err
        ok = rel < MAX_REL_ERROR
        all_ok &= ok
        print(f"{name:<10}{abs_err:>12.3e}{rel:>12.3e}   {'PASS' if ok else 'FAIL'}")

    print("\n" + ("all primitives agree with the oracle"
                  if all_ok else "PRIMITIVE MISMATCH — do not compose these yet"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
