"""bench_speculate.py — speculative decoding on a real target, end to end.

    ./tools/venv/bin/python tools/eval/bench_speculate.py TARGET DRAFT OUT.csv [REPS]

For each prompt: plain greedy through generate() once, as the reference the
output must equal; then the speculative loop with no drafts (its own baseline
speed), with prompt lookup, and with the draft model at several lengths, each
REPS times. Rows carry the median decode tok/s (wall time after prefill, as a
user waits for it), tokens per pass, draft acceptance, the split of time
between drafting and verifying, and whether every run's ids equal the
reference. A run that changes the output is a bug, and the script exits
non-zero.
"""

import csv
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW = 128
DRAFTS = [2, 3, 4, 6, 8]
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = {
    "explain": CHAT.format("Explain in a short paragraph why the sky is blue."),
    "code": CHAT.format("Write a Python function that returns the n-th Fibonacci number "
                        "iteratively, with a docstring."),
    "list": CHAT.format("List ten European capitals and their countries, one per line."),
    "rewrite": CHAT.format("Rewrite this function with type hints and a docstring:\n\n"
                           "def area(width, height):\n    return width * height\n"),
    "story": CHAT.format("Write the opening three sentences of a mystery novel."),
}
STATS = re.compile(r"spec: (\d+) tokens in (\d+) passes \(([\d.]+) per pass\), (\d+)/(\d+) drafts "
                   r"accepted, decode ([\d.]+) s \(([\d.]+) tok/s; drafting ([\d.]+) s, "
                   r"verifying ([\d.]+) s\)")


def engine(model: Path, args: list[str]) -> tuple[list[int], dict]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        p = subprocess.run([str(ROOT / "picoforge"), str(model), *args, tmp.name],
                           capture_output=True, text=True, cwd=ROOT)
        if p.returncode:
            sys.exit(f"picoforge {' '.join(args[:1])} failed:\n{p.stdout[-2000:]}\n{p.stderr}")
        ids = [int(x) for x in Path(tmp.name).read_text().splitlines()[0].split()]
    m = STATS.search(p.stdout)
    if not m:
        return ids, {}
    g = m.groups()
    return ids, {"tokens": int(g[0]), "passes": int(g[1]), "per_pass": float(g[2]),
                 "accepted": int(g[3]), "drafted": int(g[4]), "decode_s": float(g[5]),
                 "tok_s": float(g[6]), "draft_s": float(g[7]), "verify_s": float(g[8])}


def main() -> None:
    target, draft, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    configs = [("plain", ["--spec-greedy", None, str(MAX_NEW), "0"])]
    configs.append(("lookup-4", ["--spec-greedy", None, str(MAX_NEW), "4"]))
    configs.append(("lookup-auto", ["--spec-greedy", None, str(MAX_NEW), "auto"]))
    configs += [(f"model-{k}", ["--spec-model", str(draft), None, str(MAX_NEW), str(k)]) for k in DRAFTS]
    configs.append(("model-auto", ["--spec-model", str(draft), None, str(MAX_NEW), "auto"]))

    rows, bad = [], 0
    for name, prompt in PROMPTS.items():
        ref, _ = engine(target, ["--gpu-greedy", prompt, str(MAX_NEW)])
        base = None
        for cfg, template in configs:
            runs = []
            for _ in range(reps):
                args = [prompt if a is None else a for a in template]
                ids, st = engine(target, args)
                bad += ids != ref
                runs.append((ids == ref, st))
            same = all(r[0] for r in runs)
            med = lambda k: statistics.median(r[1][k] for r in runs)  # noqa: E731
            tok_s = med("tok_s")
            base = tok_s if cfg == "plain" else base
            row = {"prompt": name, "config": cfg, "reps": reps, "tokens": runs[0][1]["tokens"],
                   "median_tok_s": f"{tok_s:.2f}", "speedup": f"{tok_s / base:.3f}",
                   "tokens_per_pass": f"{med('per_pass'):.2f}",
                   "acceptance": f"{runs[0][1]['accepted'] / max(1, runs[0][1]['drafted']):.3f}",
                   "median_draft_s": f"{med('draft_s'):.3f}", "median_verify_s": f"{med('verify_s'):.3f}",
                   "identical_to_greedy": same}
            rows.append(row)
            print(f"  {name:8s} {cfg:11s} {tok_s:7.2f} tok/s  x{tok_s / base:4.2f}  "
                  f"{row['tokens_per_pass']} tok/pass  accept {row['acceptance']}  "
                  f"draft {row['median_draft_s']} s  verify {row['median_verify_s']} s  "
                  f"{'identical' if same else 'DIFFERS'}", flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")
    if bad:
        sys.exit(f"FAIL: {bad} runs changed the output")


if __name__ == "__main__":
    main()
