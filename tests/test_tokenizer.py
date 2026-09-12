"""test_tokenizer.py — every vocabulary entry and every merge rule, checked.

Not a spot check. The C engine dumps both tables it built and this script
rebuilds them independently in Python, then compares all 151643 tokens and
all 151387 rules. Spot-checking a tokenizer is close to useless: a wrong
entry buried at id 90000 is invisible until one prompt happens to hit it,
and then the symptom is a subtly wrong generation, not an error.

The merge dump is walked out of the C hash table rather than re-read from
merges.txt, so a bug in insertion or in linear probing shows up here instead
of being hidden by reading the file twice.

Run:  make test-tokenizer
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def byte_alphabet() -> dict[str, int]:
    """GPT-2's bytes_to_unicode, inverted: character -> the byte it stands for."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def run_dump(model_dir: Path, flag: str) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        path = Path(tmp.name)
    proc = subprocess.run([str(ROOT / "picoforge"), str(model_dir), flag, str(path)],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"picoforge {flag} failed:\n{proc.stdout}\n{proc.stderr}")
    lines = path.read_text().splitlines()
    path.unlink()
    return lines


def check_vocab(model_dir: Path, inv: dict[str, int]) -> bool:
    vocab = json.load(open(model_dir / "vocab.json"))
    want = {tid: bytes(inv[ch] for ch in text).hex() for text, tid in vocab.items()}

    got = {}
    for line in run_dump(model_dir, "--dump-vocab"):
        tid, hexbytes = line.split("\t")
        got[int(tid)] = hexbytes

    print(f"vocabulary            : C {len(got)} tokens, Python {len(want)}")
    if len(got) != len(want):
        print("  FAIL: different token counts")
        return False

    bad = [tid for tid in want if got.get(tid) != want[tid]]
    if bad:
        print(f"  FAIL: {len(bad)} tokens differ, e.g. id {bad[0]}: "
              f"C {got.get(bad[0])!r} vs Python {want[bad[0]]!r}")
        return False
    print(f"  all {len(want)} token byte-strings identical")
    return True


def check_merges(model_dir: Path) -> bool:
    vocab = json.load(open(model_dir / "vocab.json"))
    lines = (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    rules = lines[1:] if lines and lines[0].startswith("#version") else lines

    want = {}
    for rank, rule in enumerate(rules):
        left, right = rule.split(" ", 1)
        want[rank] = (vocab[left], vocab[right])

    got = {}
    for line in run_dump(model_dir, "--dump-merges"):
        rank, left, right = line.split("\t")
        got[int(rank)] = (int(left), int(right))

    print(f"merge rules           : C {len(got)}, Python {len(want)}")
    if len(got) != len(want):
        print("  FAIL: different rule counts — 96 of these start with '#' and "
              "are data, not comments")
        return False

    bad = [r for r in want if got.get(r) != want[r]]
    if bad:
        print(f"  FAIL: {len(bad)} rules differ, e.g. rank {bad[0]}: "
              f"C {got.get(bad[0])} vs Python {want[bad[0]]}")
        return False
    print(f"  all {len(want)} rules map to the same (left_id, right_id)")
    return True


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"
    inv = byte_alphabet()
    ok = check_vocab(model_dir, inv)
    ok &= check_merges(model_dir)
    print("\n" + ("tokenizer tables match Python exactly"
                  if ok else "TOKENIZER TABLES DIVERGE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
