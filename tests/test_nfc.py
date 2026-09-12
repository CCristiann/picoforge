"""test_nfc.py — fuzz src/normalize.c against Python's unicodedata.

test_encode.py shows NFC works on cases chosen to exercise it. That is worth
little on its own: hand-picked cases test what the author already thought of.
This one generates sequences the author did not think of — random starters
followed by random runs of combining marks, Hangul jamo, and codepoints
pulled straight from the decomposition tables — and compares byte for byte.

Combining-mark runs in random order are the interesting part: they are where
canonical ordering has to be both correct and stable, and where a > instead
of a >= silently reorders marks that must not move.

Run:  make test-nfc
"""

import random
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_RANDOM = 4000


def corpus() -> list[str]:
    rng = random.Random(20260912)

    starters = [chr(c) for c in list(range(0x41, 0x5B)) + [0x61, 0x65, 0x69, 0x6F, 0x75]]
    starters += ["Α", "А", "あ", "ᄀ", "가", "א"]
    # every codepoint that carries a nonzero combining class in the BMP
    marks = [chr(c) for c in range(0x300, 0x370)] + [chr(c) for c in range(0x483, 0x48A)]
    marks += ["ٔ", "़", "ா", "゙", "゚"]
    # things with canonical decompositions, so composition has work to do
    composed = ["é", "ñ", "ü", "Å", "ǽ", "ự", "ḝ", "Ω", "가", "각", "יִ"]

    out = ["", "a", "é", "é", "ḍ̇", "ḍ̇"]
    out += composed
    for _ in range(N_RANDOM):
        s = []
        for _ in range(rng.randint(1, 5)):
            s.append(rng.choice(starters + composed))
            for _ in range(rng.randint(0, 4)):
                s.append(rng.choice(marks))
        out.append("".join(s))
    return out


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"
    texts = corpus()

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(b"\0".join(t.encode() for t in texts) + b"\0")
        proc = subprocess.run(
            [str(ROOT / "picoforge"), str(model_dir), "--nfc-file", str(src), str(dst)],
            capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            sys.exit(f"picoforge --nfc-file failed:\n{proc.stdout}\n{proc.stderr}")
        got = dst.read_bytes().split(b"\0")[:-1]

    if len(got) != len(texts):
        sys.exit(f"C normalised {len(got)} texts, expected {len(texts)}")

    failures = 0
    for text, ours in zip(texts, got):
        theirs = unicodedata.normalize("NFC", text).encode()
        if ours == theirs:
            continue
        failures += 1
        if failures <= 3:
            print(f"\nFAIL input  {text!r}")
            print(f"     C      {ours.decode(errors='replace')!r}")
            print(f"     Python {theirs.decode(errors='replace')!r}")

    print(f"\n{len(texts) - failures}/{len(texts)} strings normalise identically "
          f"to Python's unicodedata")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
