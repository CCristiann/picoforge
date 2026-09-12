"""test_quant_formats.py — the quantisers keep the promises formats.py makes.

Checked on real weights (layer 0's q_proj and layer 27's down_proj, the
widest K), not on random matrices: real weights have outliers, and outliers
are exactly what breaks a quantiser that only works on nice data.

1. Codes stay in range: [-127, 127] for Q8, [-8, 7] for Q4.
2. Rounding error is at most half a step: |w - d*q| <= |d|/2 wherever the
   value was not clipped, and at most 1 + 2^-5 steps where it was. Both are
   provable bounds, not budgets. Q4_0 scales so |w| <= 8|d_exact|, and the
   values it clips sit on the side opposite the extreme and land on 7. But d
   is stored as bf16, whose 7 fraction bits round it by up to 2^-8 relative,
   so |w|/|d| can reach 8 * (1 + 2^-8): one step plus 8 * 2^-8 = 2^-5.
   (Measured worst case on these tensors: 1.0089 steps, in q4_rowcol.)
3. d*q is EXACT in fp32 — the claim the whole reference path rests on.
4. For _mse formats the clipped bound in (2) does not apply -- they clip on
   purpose -- and the provable property is different: the search starts from
   the RTN scale, so every block's squared error is <= RTN's, never worse.

"Clipped" means rint(w/d) fell outside the range. The first version of this
test counted every code AT the range limit instead, and failed Q4_G32 at
5.6%: but every block's extreme lands on the limit by construction, which is
1/32 = 3.1% before a single value has been clipped. The instrument was wrong,
and so was the 5% budget it was checked against — nobody had measured it.

Run:  make test-quant-formats
"""

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
from formats import dequantize, quantize  # noqa: E402

FORMATS = ["q8_row", "q8_g32", "q4_row", "q4_rowcol", "q4_g128", "q4_g32", "q4_g32_mse"]
TENSORS = ["model.layers.0.self_attn.q_proj.weight", "model.layers.27.mlp.down_proj.weight"]


def main() -> None:
    import torch
    ok = True
    with safe_open(ROOT / "models/Qwen3-0.6B/model.safetensors", framework="pt") as f:
        for name in TENSORS:
            w = f.get_tensor(name).to(torch.float32).numpy()
            for fmt in FORMATS:
                p = quantize(w, fmt)
                q, d = p["q"], p["d"]
                lo, hi = (-127, 127) if fmt.startswith("q8") else (-8, 7)
                in_range = q.min() >= lo and q.max() <= hi

                g = q.shape[1] // d.shape[1]
                step = np.repeat(np.abs(d), g, axis=1)
                if "c" in p:
                    step = step * p["c"][None, :]
                err = np.abs(w - dequantize(p))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ideal = np.rint(np.where(step > 0, np.abs(w) / step, 0.0))
                # |w|/step, not w/d: the limit is by magnitude, and d's sign
                # (Q4_0 makes it negative for a positive extreme) is irrelevant.
                clipped = ideal > np.where(q < 0, -lo, hi)
                tol = 1 + 1e-5
                half_step = bool(np.all(err[~clipped] <= step[~clipped] / 2 * tol))
                if fmt.endswith("_mse"):
                    rtn = quantize(w, fmt[:-4])
                    g_rtn = err.reshape(err.shape[0], -1, g) ** 2
                    e_rtn = (np.abs(w - dequantize(rtn)).reshape(g_rtn.shape) ** 2)
                    half_step &= bool(np.all(g_rtn.sum(-1) <= e_rtn.sum(-1) * tol))
                else:
                    half_step &= bool(np.all(err[clipped] <= step[clipped] * (1 + 2**-5) * tol))
                clip_frac = float(clipped.mean())

                exact = np.array_equal(
                    dequantize(p).astype(np.float64),
                    (q.reshape(q.shape[0], d.shape[1], g).astype(np.float64)
                     * d[..., None].astype(np.float64)).reshape(q.shape)
                    * (p["c"][None, :].astype(np.float64) if "c" in p else 1.0))

                good = in_range and half_step and exact
                ok &= good
                print(f"{name.split('.weight')[0][6:]:32s} {fmt:10s} range {in_range!s:5} "
                      f"bounds {half_step!s:5} exact {exact!s:5} clipped {100*clip_frac:5.2f}%"
                      f"  {'PASS' if good else 'FAIL'}")
    print("\n" + ("quantisers keep their promises" if ok else "QUANTISER BROKEN"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
