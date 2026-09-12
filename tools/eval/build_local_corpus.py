"""build_local_corpus.py — a perplexity corpus that needs no download.

Every perplexity number is only as honest as its text. A public benchmark is
comparable with the literature but was very probably in the training set, so
its absolute perplexity flatters the model. This corpus is the opposite trade:
picoforge's own sources and docs, written in 2026, a year after Qwen3 shipped.
The model cannot have seen them. It is small and not comparable with anybody
else's numbers, which is why it is the PILOT corpus — the one that picks the
quantisation format — and not the only one.

It is frozen to a commit, never to the working tree: the corpus must not
change because a comment was edited. The SHA256 printed at the end goes into
docs/corpus.lock, and a mismatch there means the numbers are not comparable.

Run:  tools/venv/bin/python tools/eval/build_local_corpus.py [COMMIT] [OUT]
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "8697213"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "build/corpus_local.txt"

# Prose and code, both. unicode_tables.c is excluded: 4000 lines of generated
# hex ranges would dominate the corpus and measure nothing a human wrote.
KEEP = (".md", ".c", ".h", ".m", ".metal", ".py", ".html")
SKIP = ("src/unicode_tables.c",)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> None:
    files = sorted(p for p in git("ls-tree", "-r", "--name-only", COMMIT).decode().split("\n")
                   if p.endswith(KEEP) and p not in SKIP)
    parts = []
    for path in files:
        parts.append(git("show", f"{COMMIT}:{path}").decode("utf-8"))
    # Documents separated by a blank line: no synthetic headers, which would be
    # 60 copies of the same easy-to-predict tokens inflating the score.
    text = "\n\n".join(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"{len(files)} files at {COMMIT}, {len(text.encode())} bytes -> {OUT}")
    print(f"{digest}  corpus_local.txt")


if __name__ == "__main__":
    main()
