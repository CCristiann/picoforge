"""verify.py — Phase 0's closing argument: is the oracle actually an oracle?

CLAUDE.md, "chain of oracles": Python/transformers is the reference for our
NumPy forward pass, which will in turn be the reference for the C engine.
This script closes the first link. Until it is green, every self-check in
qwen3_forward.py proves only that the code is self-consistent — not correct.

Run:  ./tools/venv/bin/python tools/oracle/verify.py models/Qwen3-0.6B
"""

import sys
from pathlib import Path

import numpy as np

from qwen3_forward import Qwen3Config, forward, load_weights

# What counts as a pass. Two different KINDS of criterion, deliberately:
#
#   * DECISION criteria are strict, because they are what inference does.
#     If the argmax disagrees at one position, the C engine built on this
#     oracle emits a different sentence. No tolerance is honest here
#     other than "identical".
#   * NUMERIC criteria are loose, because fp32 arithmetic performed in a
#     different order genuinely produces different bits. They are guards
#     against catastrophe ("a layer is missing", "the norm is misplaced"),
#     not claims about precision.
# These two numbers are MEASURED, not guessed (CLAUDE.md principle #4).
# First run on Qwen3-0.6B, worst case over the three prompts below:
#
#     relative error   8.4e-06   (at 40 tokens)
#     KL(ref || ours)  2.9e-10   (at 40 tokens)
#
# The budgets sit ~10x and ~30x above those. Both quantities grow with
# sequence length — sub-linearly: 1 -> 40 tokens multiplied the relative
# error by 3.3x and the KL by 22x — so the headroom covers a few thousand
# tokens, not an unbounded context. Extending this harness to long prompts
# means re-measuring and re-justifying these two lines, not relaxing them.
MAX_REL_ERROR = 1e-4    # max|Δlogit| / max|logit|
MAX_KL_NATS = 1e-8      # KL(reference || ours), worst position


def load_reference(model_dir: Path):
    """The HuggingFace model we are judged against.

    Two arguments matter far more than they look:

    dtype=float32   config.json says bfloat16. Left alone, transformers
                    would COMPUTE in bf16 (~3 decimal digits) while we
                    compute in fp32 (~7). Every difference we measured
                    would then be the reference's rounding, not our bugs:
                    we would be grading the oracle. Forcing fp32 makes
                    both sides start from the same exactly-upcast weights
                    (bf16 -> fp32 appends 16 zero bits; nothing is lost).

    attn_implementation="eager"
                    the default SDPA path fuses and reorders the attention
                    math, and may dispatch a flash-attention kernel whose
                    summation order is deliberately different. "eager" is
                    the plain textbook formulation — the same one our
                    attention() spells out. Fewer gratuitous differences
                    means a divergence is more likely to be a real bug.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    return model.eval()


def reference_logits(model, token_ids: list[int]) -> np.ndarray:
    """Reference logits, (seq, vocab) fp32. no_grad: we want values, not a graph."""
    import torch

    with torch.no_grad():
        return model(torch.tensor([token_ids])).logits[0].numpy()


def log_softmax(x: np.ndarray) -> np.ndarray:
    """log(softmax(x)), computed without ever forming softmax(x).

    Subtracting the row max is the usual overflow guard (softmax is
    invariant to it). Taking the log analytically, instead of log(softmax),
    avoids log(0) when a probability underflows to exactly zero — which it
    will, on a 151936-wide vocabulary where most tokens are hopeless.
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare(ours: np.ndarray, ref: np.ndarray, label: str) -> bool:
    """The metric ladder, from crudest to most meaningful."""
    print(f"\n--- {label} ---")
    assert ours.shape == ref.shape, f"shape {ours.shape} != reference {ref.shape}"
    seq = ours.shape[0]

    # Rung 1: raw distance. Meaningless alone — 0.01 is huge for a quantity
    # of size 0.1 and invisible for one of size 1000 — so always print it
    # next to the scale it must be read against.
    abs_err = float(np.max(np.abs(ours - ref)))
    scale = float(np.max(np.abs(ref)))
    rel_err = abs_err / scale
    print(f"max |Δlogit|          : {abs_err:.3e}   (logit scale {scale:.1f})")
    print(f"relative error        : {rel_err:.3e}   (budget {MAX_REL_ERROR:.0e})")

    # Rung 2: the decision. Logits exist to be argmaxed, and this is the
    # property the C engine inherits. Exact agreement or nothing.
    mismatched = np.flatnonzero(ours.argmax(-1) != ref.argmax(-1))
    print(f"argmax agreement      : {seq - len(mismatched)}/{seq} positions"
          + (f"   MISMATCH at {mismatched.tolist()}" if len(mismatched) else ""))

    # Rung 3: the whole distribution. softmax is shift-invariant, so a
    # constant offset across the vocabulary would inflate max|Δlogit| while
    # changing not a single probability. KL ignores exactly that and
    # measures what is left: how different these two models are, as models.
    # float64 on purpose, and it is not paranoia. KL sums 151936 terms of
    # p*(log p - log q) where log p and log q are both around -10 and differ
    # by ~1e-5: catastrophic cancellation. In fp32 the rounding error on each
    # logarithm (~1.2e-6) is the same order as the quantity being measured,
    # and the signed sum can come out NEGATIVE — impossible for a KL, which
    # is >= 0 by Gibbs' inequality. A negative reading was exactly what the
    # first run produced. The instrument must outrank what it measures.
    lp_ours = log_softmax(ours.astype(np.float64))
    lp_ref = log_softmax(ref.astype(np.float64))
    p_ref = np.exp(lp_ref)
    max_dp = float(np.max(np.abs(np.exp(lp_ours) - p_ref)))
    kl = float(np.max(np.sum(p_ref * (lp_ref - lp_ours), axis=-1)))
    print(f"max |Δprobability|    : {max_dp:.3e}")
    print(f"max KL(ref || ours)   : {kl:.3e} nats   (budget {MAX_KL_NATS:.0e})")

    ok = rel_err < MAX_REL_ERROR and len(mismatched) == 0 and kl < MAX_KL_NATS
    print(f"verdict               : {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/Qwen3-0.6B")
    cfg = Qwen3Config.from_json(model_dir)
    weights = load_weights(model_dir)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load_reference(model_dir)

    # Three lengths, each probing a different failure mode:
    #   1 token   the shape the C engine's decode loop runs every step. Also
    #             the case where a broken RoPE hides: at position 0 the angle
    #             is 0, so cos=1 and sin=0 and the rotation is the identity.
    #   ~6 tokens short, ordinary.
    #   ~40 tokens positions far from 0, where RoPE actually rotates and
    #             error has 28 layers x 40 positions to accumulate through.
    prompts = [
        "Hello",
        "The capital of France is",
        "Machine learning systems are usually evaluated on benchmarks, but the "
        "numbers that matter in production are latency, memory footprint, and "
        "the cost of a single request; everything else is a proxy for one of those.",
    ]

    all_ok = True
    for prompt in prompts:
        ids = tok(prompt)["input_ids"]
        ours = forward(ids, weights, cfg)
        ref = reference_logits(model, ids)
        all_ok &= compare(ours, ref, f"{len(ids)} tokens: {prompt[:44]!r}")

    print("\n" + (f"oracle VERIFIED on {model_dir.name} — the C engine may be built on it."
                  if all_ok else
                  "ORACLE DIVERGES — do not build anything on it until explained."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
