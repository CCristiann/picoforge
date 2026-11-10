"""qwen38_r7_corpus.py — the frozen text for R7 (quality of Q2 against Q4).

R7 compares two quantisations of the same model on the same tokens. The text
must be (1) unseen by the model, or both look better than they are, and
(2) frozen, or two runs on different days measure different things.

Unseen: picoforge's own files ADDED after 2026-10-12, the day the Qwen3.8
work started, which is after Qwen3.8-Flash-Next shipped. Frozen: read from a
pinned commit, never from the working tree, and the token IDs get a SHA256
that goes into docs/corpus.lock.

Two streams, scored separately because quantisation can hurt them
differently: prose (.md, mostly Italian and English) and code (C, ObjC,
Metal, Python). Each file is tokenized on its own by the engine's tokenizer
(`--encode-only`, the same IDs Q2 and Q4 see), files are joined in path order,
and WINDOWS windows of LENGTH tokens are taken evenly spaced from each stream.

Run: tools/venv/bin/python tools/eval/qwen38_r7_corpus.py [COMMIT] [OUT.json]
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "33f6d86"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "build/r7/corpus.json"
SINCE = "2026-10-12"
LENGTH, WINDOWS = 2048, 4            # per stream: 2 x 4 x 2047 = 16376 scored positions
CODE = (".c", ".h", ".m", ".metal", ".py")
PROSE = (".md",)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def tokenize(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        out = subprocess.run([str(ROOT / "picoforge"), "-m", "q2", "--complete-file", f.name,
                              "--encode-only", "-c", "32768"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)["prompt_ids"]


def stream(paths: list[str]) -> tuple[list[int], list[str]]:
    ids, used = [], []
    sep = tokenize("\n\n")           # documents separated by a blank line, no headers
    for path in paths:
        text = git("show", f"{COMMIT}:{path}")
        if not text.strip():
            continue
        ids += tokenize(text) + sep
        used.append(path)
    return ids, used


def windows(ids: list[int]) -> list[dict]:
    assert len(ids) >= LENGTH * WINDOWS, f"stream too short: {len(ids)} tokens"
    step = (len(ids) - LENGTH) // (WINDOWS - 1)
    return [{"start": i * step, "ids": ids[i * step:i * step + LENGTH]} for i in range(WINDOWS)]


def main() -> None:
    added = git("log", "--diff-filter=A", f"--since={SINCE}", "--name-only", "--format=", COMMIT,
                "--", "src", "docs", "tools").split()
    present = set(git("ls-tree", "-r", "--name-only", COMMIT).split())
    files = sorted(p for p in set(added) if p in present)
    corpus = {"commit": git("rev-parse", "--short", COMMIT).strip(), "since": SINCE,
              "length": LENGTH, "streams": {}}
    for name, ext in (("prose", PROSE), ("code", CODE)):
        ids, used = stream([p for p in files if p.endswith(ext)])
        corpus["streams"][name] = {"files": used, "tokens": len(ids), "windows": windows(ids)}
        print(f"{name}: {len(used)} files, {len(ids)} tokens, {WINDOWS} windows of {LENGTH}")
    every = [w["ids"] for s in corpus["streams"].values() for w in s["windows"]]
    corpus["sha256"] = hashlib.sha256(json.dumps(every).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(corpus) + "\n")
    print(f"{corpus['sha256']}  r7 windows (sha256 of the JSON list of window IDs) -> {OUT}")


if __name__ == "__main__":
    main()
