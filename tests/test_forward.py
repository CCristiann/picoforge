"""test_forward.py — the C engine's logits judged by the NumPy oracle.

This is the second link in the chain of oracles. Phase 0 established that the
NumPy forward matches transformers; this establishes that the C forward
matches NumPy. Only once both hold is the C engine a reference the Metal
kernels can be measured against in Phase 2.

The two implementations are structurally identical on purpose — same order of
operations, same conventions — so any disagreement here is arithmetic, not a
different reading of the architecture. The available sources of difference
are: summation order inside matmul (naive sequential in C, blocked BLAS in
NumPy) and libm-vs-NumPy transcendentals in exp, cos and sin.

Run:  make test-forward
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
from qwen3_forward import Qwen3Config, forward, load_weights  # noqa: E402
from verify import log_softmax  # noqa: E402

# Calibrated on the first run (CLAUDE.md principle #4). Worst case over the
# three prompts: relative error 1.1e-5 and KL 1.8e-9, both at 9 tokens.
#
# Worth noting what those numbers say: C-vs-NumPy here is the same order as
# NumPy-vs-transformers in verify.py (8.4e-6). The chain of oracles is not
# degrading — each link adds comparable fp32 noise and none introduces drift.
#
# KL grows fast with length (28x from 1 to 9 tokens), so its budget carries
# more headroom than the logit one. Both are calibrated at <= 9 tokens;
# testing longer prompts means re-measuring, not relaxing.
MAX_REL_ERROR = 1e-4
MAX_KL_NATS = 1e-7

PROMPTS = ["Hello", "The capital of France is", "Roses are red and violets are"]


def c_logits(model_dir: Path, ids: list[int], vocab: int) -> np.ndarray:
    """Run the C engine on these ids and read back its raw fp32 logits."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        cmd = [str(ROOT / "picoforge"), str(model_dir), "--forward", tmp.name]
        cmd += [str(i) for i in ids]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            sys.exit(f"picoforge failed:\n{proc.stdout}\n{proc.stderr}")
        for line in proc.stdout.splitlines():
            if line.startswith(("prefill", "top-5")):
                print("  " + line.strip())
        return np.fromfile(tmp.name, dtype=np.float32).reshape(len(ids), vocab)


def compare(ours: np.ndarray, ref: np.ndarray, label: str) -> bool:
    """The same ladder verify.py uses: distance, then decision, then model."""
    print(f"\n--- {label} ---")
    abs_err = float(np.max(np.abs(ours - ref)))
    scale = float(np.max(np.abs(ref)))
    rel = abs_err / scale
    mismatched = np.flatnonzero(ours.argmax(-1) != ref.argmax(-1))

    lp_ours = log_softmax(ours.astype(np.float64))
    lp_ref = log_softmax(ref.astype(np.float64))
    p_ref = np.exp(lp_ref)
    kl = float(np.max(np.sum(p_ref * (lp_ref - lp_ours), axis=-1)))

    print(f"  max |Δlogit|          : {abs_err:.3e}   (logit scale {scale:.1f})")
    print(f"  relative error        : {rel:.3e}   (budget {MAX_REL_ERROR:.0e})")
    print(f"  argmax agreement      : {len(ours) - len(mismatched)}/{len(ours)}"
          + (f"   MISMATCH at {mismatched.tolist()}" if len(mismatched) else ""))
    print(f"  max KL(oracle || C)   : {kl:.3e} nats   (budget {MAX_KL_NATS:.0e})")

    ok = rel < MAX_REL_ERROR and len(mismatched) == 0 and kl < MAX_KL_NATS
    print(f"  verdict               : {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"
    cfg = Qwen3Config.from_json(model_dir)
    weights = load_weights(model_dir)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)

    all_ok = True
    for prompt in PROMPTS:
        ids = tok(prompt)["input_ids"]
        print(f"\n=== {len(ids)} tokens: {prompt!r} ===")
        ours = c_logits(model_dir, ids, cfg.vocab_size)
        ref = forward(ids, weights, cfg)
        all_ok &= compare(ours, ref, f"{len(ids)} tokens")
        print(f"  oracle's next token   : {tok.decode([int(ref[-1].argmax())])!r}")

    print("\n" + ("C engine matches the oracle — chain of oracles intact"
                  if all_ok else "C DIVERGES FROM THE ORACLE — do not proceed"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
