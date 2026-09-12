"""test_generate.py — token-level parity: the C engine against transformers.

The last link of Phase 1. verify.py compared logits for a fixed prompt;
this compares what the two actually GENERATE, which exercises the KV cache,
the sampler's argmax path, the stop tokens and the tokenizer all at once.

Greedy on both sides, necessarily. With sampling enabled, matching output
would only prove the two random number generators agree.

A caveat this test is built to expose rather than hide: greedy decoding is
chaotic. Logits that differ by 1e-5 are irrelevant until two candidates are
nearly tied, at which point the argmax flips and every token after it
diverges for a reason that is arithmetic, not a bug. So when sequences
differ, the gap between the top two candidates at that position is printed:
a gap of 1e-4 is a coin flip, a gap of 3.0 is a defect.

Run:  make test-generate
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW = 16

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Water boils at",
]


def c_greedy(model_dir: Path, prompt: str) -> list[int]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        proc = subprocess.run(
            [str(ROOT / "picoforge"), str(model_dir), "--greedy", prompt,
             str(MAX_NEW), tmp.name],
            capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            sys.exit(f"picoforge --greedy failed:\n{proc.stdout}\n{proc.stderr}")
        return [int(x) for x in Path(tmp.name).read_text().split()]


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()

    failures = 0
    for prompt in PROMPTS:
        ours = c_greedy(model_dir, prompt)
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
        theirs = out[0][ids.shape[1]:].tolist()
        theirs = theirs[:len(theirs)]

        common = 0
        while common < min(len(ours), len(theirs)) and ours[common] == theirs[common]:
            common += 1

        print(f"\n=== {prompt!r} ===")
        print(f"  C  : {tok.decode(ours)!r}")
        print(f"  HF : {tok.decode(theirs)!r}")

        if ours == theirs:
            print(f"  identical: {len(ours)}/{len(ours)} tokens")
            continue

        failures += 1
        print(f"  DIVERGE after {common} identical tokens")
        # how close was the call? that decides bug vs coin flip
        with torch.no_grad():
            ctx = torch.tensor([ids[0].tolist() + theirs[:common]])
            logits = model(ctx).logits[0, -1].numpy()
        top = np.argsort(logits)[::-1][:2]
        print(f"  top-2 at that position: {top[0]} ({logits[top[0]]:.4f}) vs "
              f"{top[1]} ({logits[top[1]]:.4f}), gap {logits[top[0]] - logits[top[1]]:.2e}")

    print(f"\n{len(PROMPTS) - failures}/{len(PROMPTS)} prompts generate identically "
          f"to transformers, greedy, {MAX_NEW} tokens")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
