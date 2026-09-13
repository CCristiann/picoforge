"""test_speculate.py — speculative decoding must not change a single token.

Greedy speculative decoding is lossless by construction: every emitted token
is the model's own argmax, and the drafts only decide how many arrive per
pass. This holds the construction to account. For each prompt, plain greedy
(generate(), one token per pass) and the speculative loop at several draft
lengths run on the GPU, and the id sequences must be IDENTICAL.

Why identical and not "close": the verify pass runs the same kernels over
k+1 rows that decode runs over 1, and each row of every kernel is computed
independently of the others (the 8x32 and 32x32 TensorOps tiles were measured
bit-identical in step 4.1). So there is no arithmetic excuse for a flipped
argmax. A difference here is a bug in the accept loop or the cache rollback.

The prompts mix text that repeats its context -- where the prompt-lookup
drafter should win -- with text that does not, where it should cost nothing.
Timings are printed for orientation only: one run each, not the protocol.

Run:  make test-speculate
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW = 128
DRAFTS = [2, 4, 8, 16]

CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = [
    "def fibonacci(n):",
    "The capital of France is",
    CHAT.format("Repeat this list back to me exactly, one item per line: "
                "apple, banana, cherry, dragonfruit, elderberry, fig, grape, "
                "honeydew, kiwi, lemon, mango, nectarine, orange, papaya."),
    CHAT.format("Rewrite this function with type hints and a docstring:\n\n"
                "def area(width, height):\n    return width * height\n\n"
                "def perimeter(width, height):\n    return 2 * (width + height)\n"),
    CHAT.format("Write a short poem about the sea."),
]


def run(args: list[str]) -> tuple[list[int], str]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        proc = subprocess.run([str(ROOT / "picoforge"), str(MODEL), *args, tmp.name],
                              capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            sys.exit(f"picoforge {args[0]} failed:\n{proc.stdout}\n{proc.stderr}")
        ids = [int(x) for x in Path(tmp.name).read_text().split()]
        stats = [l for l in proc.stdout.splitlines() if l.startswith("spec:")]
        return ids, stats[0] if stats else ""


MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"


def main() -> None:
    failures = 0
    for prompt in PROMPTS:
        ref, _ = run(["--gpu-greedy", prompt, str(MAX_NEW)])
        print(f"\n=== {prompt[-60:]!r} ({len(ref)} tokens) ===")
        _, base = run(["--spec-greedy", prompt, str(MAX_NEW), "0"])
        print(f"  draft  0 : {base.removeprefix('spec: ')}")
        for k in DRAFTS:
            ids, stats = run(["--spec-greedy", prompt, str(MAX_NEW), str(k)])
            same = ids == ref
            if not same:
                failures += 1
                first = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b),
                             min(len(ids), len(ref)))
                print(f"  draft {k:2d} : DIFFERS from greedy at token {first} "
                      f"({len(ids)} vs {len(ref)} tokens)")
            else:
                print(f"  draft {k:2d} : identical   {stats.removeprefix('spec: ')}")

    print()
    if failures:
        sys.exit(f"FAIL: {failures} speculative runs changed the output")
    print(f"PASS: {len(PROMPTS)} prompts x {len(DRAFTS)} draft lengths, "
          f"every sequence identical to plain greedy")


if __name__ == "__main__":
    main()
