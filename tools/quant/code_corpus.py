"""code_corpus.py — frozen multi-language code windows for calibration and evaluation.

Six permissively licensed repositories, cloned shallow at their default branch
(the commit is recorded), one language each. Calibration and evaluation use
different repositories and different languages, so a ranking that only fits
the calibration code shows up as a gap. These projects are public and probably
in the model's training data: fine for routing statistics and for comparing
two quantizations of the same model, not for measuring the model itself.

  tools/venv/bin/python tools/quant/code_corpus.py [CLONE_DIR] [OUT.json]
  tools/venv/bin/python tools/quant/code_corpus.py --rank [CLONE_DIR] [OUT.json]

--rank builds the expert-ranking corpus (M4c): calibration repositories only, eight
languages, one long window each. The evaluation repositories above and picoforge's
own code (the R7 corpus) stay out of it, so both remain held out.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RANK = "--rank" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if a != "--rank"]
CLONES = Path(ARGS[0]) if ARGS else ROOT / "models/code-corpus"
OUT = Path(ARGS[1]) if len(ARGS) > 1 else ROOT / ("build/quant/rank-corpus.json" if RANK else "build/quant/code-corpus.json")
LENGTH, WINDOWS = (12288, 1) if RANK else (2048, 2)
# name, repository, license, split, directory, extensions
REPOS = [
    ("llama.cpp", "https://github.com/ggml-org/llama.cpp", "MIT", "calib", "src", (".cpp", ".h")),
    ("flask", "https://github.com/pallets/flask", "BSD-3-Clause", "calib", "src/flask", (".py",)),
    ("react", "https://github.com/facebook/react", "MIT", "calib", "packages/react-reconciler/src", (".js",)),
    ("tokio", "https://github.com/tokio-rs/tokio", "MIT", "eval", "tokio/src/sync", (".rs",)),
    ("cobra", "https://github.com/spf13/cobra", "Apache-2.0", "eval", ".", (".go",)),
    ("guava", "https://github.com/google/guava", "Apache-2.0", "eval", "guava/src/com/google/common/collect", (".java",)),
]
RANK_REPOS = [r for r in REPOS if r[3] == "calib"] + [
    ("ripgrep", "https://github.com/BurntSushi/ripgrep", "MIT OR Unlicense", "calib", "crates", (".rs",)),
    ("gin", "https://github.com/gin-gonic/gin", "MIT", "calib", ".", (".go",)),
    ("gson", "https://github.com/google/gson", "Apache-2.0", "calib", "gson/src/main/java", (".java",)),
    ("zod", "https://github.com/colinhacks/zod", "MIT", "calib", "packages/zod/src", (".ts",)),
    ("curl", "https://github.com/curl/curl", "curl", "calib", "lib", (".c", ".h")),
]
MAX_FILE = 200_000      # bytes; skips generated or vendored blobs
PER_FILE = 3_000 if RANK else 6_000   # characters taken from each file, so a window spans many files


def run(*cmd, cwd=None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def tokenize(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        out = run(str(ROOT / "pf"), "-m", "q2", "--complete-file", f.name, "--encode-only", "-c", "65536")
    return json.loads(out)["prompt_ids"]


def repo_text(path: Path, directory: str, exts: tuple, budget: int) -> tuple[str, list[str]]:
    files = sorted(p for p in (path / directory).rglob("*")
                   if p.suffix in exts and p.is_file() and "test" not in p.name.lower() and p.stat().st_size <= MAX_FILE)
    parts, used, size = [], [], 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")[:PER_FILE]
        if not text.strip():
            continue
        parts.append(text); used.append(str(p.relative_to(path))); size += len(text)
        if size >= budget:
            break
    return "\n\n".join(parts), used


def main() -> None:
    CLONES.mkdir(parents=True, exist_ok=True)
    corpus = {"length": LENGTH, "streams": {}}
    for name, url, lic, split, directory, exts in (RANK_REPOS if RANK else REPOS):
        dest = CLONES / name
        if not dest.exists():
            run("git", "clone", "--depth", "1", "--quiet", url, str(dest))
        commit = run("git", "rev-parse", "HEAD", cwd=dest).strip()
        text, used = repo_text(dest, directory, exts, budget=LENGTH * WINDOWS * 6)
        ids = tokenize(text)
        assert len(ids) >= LENGTH * WINDOWS, f"{name}: only {len(ids)} tokens"
        step = (len(ids) - LENGTH) // max(WINDOWS - 1, 1)
        corpus["streams"][name] = {"url": url, "commit": commit, "license": lic, "split": split,
                                   "files": used, "tokens": len(ids),
                                   "windows": [{"start": i * step, "ids": ids[i * step:i * step + LENGTH]} for i in range(WINDOWS)]}
        print(f"{name:10s} {split:5s} {commit[:10]} {len(used):3d} files {len(ids):6d} tokens")
    every = [w["ids"] for s in corpus["streams"].values() for w in s["windows"]]
    corpus["sha256"] = hashlib.sha256(json.dumps(every).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(corpus) + "\n")
    print(f"{corpus['sha256']}  -> {OUT}")


if __name__ == "__main__":
    main()
