"""test_forward_quant.py — the quantised engine judged by the NumPy oracle.

The chain of oracles, extended to Phase 3. A quantised model is a different
function from the original, so it cannot be compared with the original for
parity. What it CAN be compared with is the original forward pass run on the
dequantised weights: W = d * q, computed in fp32, where it is exact. That is
the fp32 path acting as the oracle for the quantised one.

The dequantisation here reads the FILE, independently of the converter: it
unpacks the nibbles and applies the scales with its own code. A converter
that wrote the right numbers in the wrong layout fails here, not later as
mysteriously bad perplexity.

Budgets are the ones test_forward.py calibrated, and they apply unchanged for
a reason, not by hope: the CPU computes x * (d*q), and d*q is an exact fp32
weight, so this is the same arithmetic as the unquantised comparison. The Q8
GPU path holds to them too.

The Q4 GPU path cannot, and why is the most instructive result of Phase 3.
TensorOps has no float x int4 overload, so Q4 inputs are narrowed to bf16.
Narrowing is a step function: two fp32 activations that differ by 1e-6 land
on the same bf16 value almost always, and one whole bf16 step (~0.8%) apart
when they straddle a rounding boundary. So the GPU's normal fp32 noise, which
is harmless everywhere else, turns into occasional full-step jumps, and 28
layers compound them. Measured, not argued: the CPU alone, narrowing with its
inputs scaled by (1 + 1e-6), diverges from itself by relative 1.05e-2 and KL
5.0e-4 at 9 tokens, the same numbers the GPU shows against the narrowed CPU
(1.05e-2, 4.4e-4). No implementation of this arithmetic can reproduce another
more tightly than that. Q4 budgets are therefore 10x that self-sensitivity,
and an argmax may differ only where the oracle itself had a near-tie within
the observed logit noise. A real kernel bug -- a wrong block, a swapped
nibble -- moves KL by orders of magnitude more, and the kernels are in any
case checked exactly, one matmul at a time, by --quant-check.

Run:  make test-forward-quant   (builds the quantised checkpoints it needs)
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "tests"))
from qwen3_forward import Qwen3Config, forward  # noqa: E402
from test_forward import MAX_KL_NATS, MAX_REL_ERROR, PROMPTS, c_logits  # noqa: E402
from verify import log_softmax  # noqa: E402

# Q4 on the GPU: 10x the measured self-sensitivity of narrowed arithmetic to a
# 1e-6 input perturbation (KL 5.0e-4, relative 1.05e-2 at 9 tokens).
Q4_MAX_KL_NATS = 5e-3
Q4_MAX_REL_ERROR = 1e-1


def compare(ours: np.ndarray, ref: np.ndarray, label: str, max_rel: float, max_kl: float) -> bool:
    abs_err = np.abs(ours - ref)
    rel = float(abs_err.max() / np.abs(ref).max())
    lp_o, lp_r = log_softmax(ours.astype(np.float64)), log_softmax(ref.astype(np.float64))
    kl = float(np.max(np.sum(np.exp(lp_r) * (lp_r - lp_o), axis=-1)))
    # An argmax may flip only where the oracle's own top two were closer than
    # the noise this comparison shows. With the unquantised budgets the noise
    # is ~1e-4 logits and no real prediction is that close, so this is the
    # old strict rule there.
    top2 = np.sort(ref, axis=-1)[:, -2:]
    near_tie = (top2[:, 1] - top2[:, 0]) < 2 * abs_err.max(-1)
    flips = ours.argmax(-1) != ref.argmax(-1)
    bad_flips = np.flatnonzero(flips & ~near_tie)
    ok = rel < max_rel and kl < max_kl and len(bad_flips) == 0
    print(f"  {label:44s} rel {rel:.2e} (<{max_rel:.0e})  KL {kl:.2e} (<{max_kl:.0e})  "
          f"argmax {len(ref) - flips.sum()}/{len(ref)}"
          + (f" [{flips.sum()} near-tie]" if flips.any() and not len(bad_flips) else "")
          + f"  {'PASS' if ok else 'FAIL'}")
    return ok


def unpack_int4(packed: np.ndarray, k: int) -> np.ndarray:
    """Low nibble first, two's complement: the layout TensorOps reads."""
    lo, hi = packed & 0x0F, packed >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], k).astype(np.int8)
    return np.where(codes >= 8, codes - 16, codes).astype(np.int8)


def load_dequantized(model_dir: Path) -> dict[str, np.ndarray]:
    weights = {}
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        names = list(f.keys())
        for name in names:
            if name.endswith(".scales"):
                continue
            t = f.get_tensor(name)
            if not name.endswith(".qweight"):
                weights[name] = t.to(torch.float32).numpy()
                continue
            base = name[: -len(".qweight")]
            d = f.get_tensor(base + ".scales").to(torch.float32).numpy()
            if t.dtype == torch.int8:
                q = t.numpy()
            else:
                q = unpack_int4(t.numpy(), t.shape[1] * 2)
            n, k = q.shape
            g = k // d.shape[1]
            w = (q.reshape(n, d.shape[1], g).astype(np.float32) * d[..., None]).reshape(n, k)
            weights[base + ".weight"] = w
    return weights


def main() -> None:
    from transformers import AutoTokenizer

    dirs = [Path(p) for p in sys.argv[1:]] or [ROOT / "models/Qwen3-0.6B-q8_row-embed",
                                               ROOT / "models/Qwen3-0.6B-q4_g32",
                                               ROOT / "models/Qwen3-0.6B-q4_g32-embed"]
    all_ok = True
    for model_dir in dirs:
        cfg = Qwen3Config.from_json(model_dir)
        weights = load_dequantized(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            q4 = any(f.get_slice(k).get_dtype() == "U8" for k in f.keys() if k.endswith(".qweight"))
        strict, loose = (MAX_REL_ERROR, MAX_KL_NATS), (Q4_MAX_REL_ERROR, Q4_MAX_KL_NATS)
        print(f"\n##### {model_dir.name} #####")
        for prompt in PROMPTS:
            ids = tok(prompt)["input_ids"]
            ref = forward(ids, weights, cfg)
            print(f"--- {len(ids)} tokens: {prompt!r}")
            modes = [("--forward", "CPU batch", strict), ("--forward-incr", "CPU incremental", strict),
                     ("--gpu-forward", "GPU batch", loose if q4 else strict),
                     ("--gpu-forward-incr", "GPU incremental", loose if q4 else strict)]
            if q4:
                modes.append(("--forward-narrow", "CPU, Q4 inputs narrowed to bf16", loose))
            for mode, label, (max_rel, max_kl) in modes:
                ours = c_logits(model_dir, ids, cfg.vocab_size, mode)
                all_ok &= compare(ours, ref, label, max_rel, max_kl)
            print(f"  oracle's next token: {tok.decode([int(ref[-1].argmax())])!r}")

    print(f"\nbudgets: relative {MAX_REL_ERROR:.0e}, KL {MAX_KL_NATS:.0e} (test_forward.py's); "
          f"Q4 on bf16 activations {Q4_MAX_REL_ERROR:.0e}, {Q4_MAX_KL_NATS:.0e}")
    print("\n" + ("quantised engine, CPU and GPU, matches the oracle on dequantised weights"
                  if all_ok else "QUANTISED ENGINE DIVERGES FROM THE ORACLE"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
