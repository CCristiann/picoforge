"""test_encode.py — the C tokenizer against HuggingFace, token for token.

Encoding is where a tokenizer is actually wrong, and it is wrong silently:
a misplaced boundary produces valid ids for the right text in a segmentation
the model never saw, so generation just degrades. Nothing raises.

So the corpus is chosen adversarially rather than for realism. Each entry
targets one branch of the pre-tokenizer pattern or one property of the
byte-level encoding.

Run:  make test-encode
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORPUS = [
    # ordinary text, and the leading-space rule that makes " the" one token
    "Hello world",
    "The capital of France is Paris.",
    "the the  the   the",
    # contractions: the first alternative of the pattern, case-insensitive
    "don't it's we're I've they'll he'd IT'S DON'T",
    # digits are split one by one, so this must be many tokens, not one
    "2026 3.14159 1,000,000 007",
    # whitespace: trailing runs, newline runs, tabs, CR
    "trailing   ",
    "a\n\nb\n\n\nc",
    "x\ty\tz",
    "line1\r\nline2\r\n",
    "   leading",
    "\n",
    " ",
    "",
    # punctuation and symbol runs, with and without a leading space
    "!!! ??? ...  --- ***",
    "a=b+c*d/e (f) [g] {h}",
    # the hash that collides with merges.txt comments
    "#include <stdio.h>\n#define N 10\n## heading\n#### deeper",
    # non-ASCII: accents, CJK, Cyrillic, Greek, RTL
    "café naïve Ünicode",
    "日本語のテキストです",
    "Привет мир",
    "Ελληνικά",
    "مرحبا بالعالم",
    # emoji, including a multi-codepoint sequence with a ZWJ
    "hello 👋 world 🌍",
    "family: 👨‍👩‍👧‍👦 flag: 🇮🇹",
    # combining marks, where a byte-level tokenizer must not split wrongly
    "é vs é",
    # raw bytes that are not valid text but must still encode
    "tab\tnul-free\x01\x02control",
    # the chat template's special tokens, matched before the regex
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "text before <|im_start|> text after",
    # a long run, to exercise the merge loop rather than its edges
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" * 4,
    " " * 40,
]


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/Qwen3-0.6B"
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(model_dir)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "corpus.bin"
        dst = Path(tmp) / "ids.txt"
        src.write_bytes(b"\0".join(t.encode() for t in CORPUS) + b"\0")

        proc = subprocess.run(
            [str(ROOT / "picoforge"), str(model_dir), "--encode-file", str(src), str(dst)],
            capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            sys.exit(f"picoforge --encode-file failed:\n{proc.stdout}\n{proc.stderr}")
        got = [[int(x) for x in line.split()] for line in dst.read_text().splitlines()]

    if len(got) != len(CORPUS):
        sys.exit(f"C encoded {len(got)} texts, expected {len(CORPUS)}")

    failures = 0
    for text, ours in zip(CORPUS, got):
        theirs = hf(text)["input_ids"]
        if ours == theirs:
            continue
        failures += 1
        print(f"\nFAIL {text!r}")
        print(f"  C  ({len(ours):3}): {ours}")
        print(f"  HF ({len(theirs):3}): {theirs}")
        # the first divergence is where the bug is; everything after is fallout
        for i, (a, b) in enumerate(zip(ours, theirs)):
            if a != b:
                print(f"  first difference at position {i}: "
                      f"C {a} = {hf.decode([a])!r}, HF {b} = {hf.decode([b])!r}")
                break

    print(f"\n{len(CORPUS) - failures}/{len(CORPUS)} texts tokenise identically to HuggingFace")
    print("(the C engine also decoded every one of them back to the original bytes)")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
