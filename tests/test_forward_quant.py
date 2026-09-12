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
weight, so this is the same arithmetic as the unquantised comparison. (The
GPU is a different matter -- Q4 narrows activations to bf16 -- and has its
own measured budget.)

Run:  make test-forward-quant   (models/Qwen3-0.6B-q8_row and -q4_g32 must exist)
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
from test_forward import MAX_KL_NATS, MAX_REL_ERROR, PROMPTS, c_logits, compare  # noqa: E402


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

    dirs = [Path(p) for p in sys.argv[1:]] or [ROOT / "models/Qwen3-0.6B-q8_row",
                                               ROOT / "models/Qwen3-0.6B-q4_g32"]
    all_ok = True
    for model_dir in dirs:
        cfg = Qwen3Config.from_json(model_dir)
        weights = load_dequantized(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        print(f"\n##### {model_dir.name} #####")
        for prompt in PROMPTS:
            ids = tok(prompt)["input_ids"]
            ref = forward(ids, weights, cfg)
            for mode, label in (("--forward", "CPU batch"), ("--forward-incr", "CPU incremental")):
                ours = c_logits(model_dir, ids, cfg.vocab_size, mode)
                all_ok &= compare(ours, ref, f"{model_dir.name}, {len(ids)} tokens, {label}")
            print(f"  oracle's next token   : {tok.decode([int(ref[-1].argmax())])!r}")

    print(f"\nbudgets: relative {MAX_REL_ERROR:.0e}, KL {MAX_KL_NATS:.0e} (test_forward.py's)")
    print("\n" + ("quantised CPU engine matches the oracle on dequantised weights"
                  if all_ok else "QUANTISED ENGINE DIVERGES FROM THE ORACLE"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
