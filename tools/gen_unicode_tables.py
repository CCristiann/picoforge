"""gen_unicode_tables.py — emit src/unicode_tables.c from Python's unicodedata.

The pre-tokenizer regex Qwen3 uses refers to \\p{L} and \\p{N}. Those are
Unicode general categories, not ASCII ranges, and the engine links no ICU.
The honest options were to approximate them (wrong for CJK punctuation,
emoji and every non-Latin script) or to carry the real tables. They compress
to 821 sorted ranges, 6.4 KB, so we carry them.

Generated data is committed so the build stays dependency-free. Regenerate
only when the Unicode version changes:

    ./tools/venv/bin/python tools/gen_unicode_tables.py
"""

import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "unicode_tables.c"


def ranges(prefix: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for cp in range(0x110000):
        if unicodedata.category(chr(cp)).startswith(prefix):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def emit(name: str, rs: list[tuple[int, int]]) -> str:
    body = "".join(
        f"    {{0x{a:05X}, 0x{b:05X}}},\n" if i % 4 == 3 else f"    {{0x{a:05X}, 0x{b:05X}}},"
        for i, (a, b) in enumerate(rs)
    )
    return (f"const CodepointRange {name}[] = {{\n{body}\n}};\n"
            f"const int {name}_count = {len(rs)};\n\n")


HANGUL_S, HANGUL_N = 0xAC00, 11172


def nfc_tables():
    """Canonical decompositions, composition pairs and combining classes.

    Hangul syllables are skipped: their decomposition and composition are
    arithmetic, so tabulating 11172 of them would be 11172 rows of nothing.

    Composition exclusions are not read from a file, they are derived. A
    canonical pair (a, b) composes to cp only if Python agrees that NFC of
    a+b is cp; every exclusion — singleton, non-starter, script-specific —
    falls out of that one test.
    """
    decomp, compose, ccc = {}, {}, {}
    for cp in range(0x110000):
        if HANGUL_S <= cp < HANGUL_S + HANGUL_N:
            continue
        ch = chr(cp)
        c = unicodedata.combining(ch)
        if c:
            ccc[cp] = c
        d = unicodedata.decomposition(ch)
        if not d or d.startswith("<"):      # compatibility mapping, not canonical
            continue
        parts = [int(x, 16) for x in d.split()]
        decomp[cp] = parts
        if len(parts) == 2 and unicodedata.normalize(
                "NFC", chr(parts[0]) + chr(parts[1])) == ch:
            compose[(parts[0], parts[1])] = cp
    return decomp, compose, ccc


def main() -> None:
    letters, numbers = ranges("L"), ranges("N")
    decomp, compose, ccc = nfc_tables()
    text = f'''/* unicode_tables.c — GENERATED, do not edit.
 *
 * Source: tools/gen_unicode_tables.py, Unicode {unicodedata.unidata_version}.
 *
 * Qwen3's pre-tokenizer regex splits on \\\\p{{L}} and \\\\p{{N}}, which are Unicode
 * general categories. The engine links no ICU, and approximating them (say,
 * "anything above 0x7F is a letter") is wrong for CJK punctuation, emoji and
 * every non-Latin script — and wrong in the silent way, producing a different
 * segmentation rather than an error. The real tables are {len(letters) + len(numbers)} sorted ranges,
 * {(len(letters) + len(numbers)) * 8 / 1024:.1f} KB, so we carry them and binary-search.
 */
#include "picoforge.h"

'''
    text += emit("unicode_letters", letters)
    text += emit("unicode_numbers", numbers)

    rows = "".join(f"    {{0x{cp:05X}, 0x{v[0]:05X}, 0x{v[1]:05X}}},\n"
                   for cp, v in sorted(decomp.items())
                   for v in [v if len(v) == 2 else [v[0], 0]])
    text += (f"const Decomposition unicode_decomp[] = {{\n{rows}}};\n"
             f"const int unicode_decomp_count = {len(decomp)};\n\n")

    rows = "".join(f"    {{0x{a:05X}, 0x{b:05X}, 0x{cp:05X}}},\n"
                   for (a, b), cp in sorted(compose.items()))
    text += (f"const Composition unicode_compose[] = {{\n{rows}}};\n"
             f"const int unicode_compose_count = {len(compose)};\n\n")

    rows = "".join(f"    {{0x{cp:05X}, {v}}},\n" for cp, v in sorted(ccc.items()))
    text += (f"const CombiningClass unicode_ccc[] = {{\n{rows}}};\n"
             f"const int unicode_ccc_count = {len(ccc)};\n")

    OUT.write_text(text)
    print(f"wrote {OUT} — {len(letters)} letter ranges, {len(numbers)} number ranges, "
          f"{len(decomp)} decompositions, {len(compose)} compositions, {len(ccc)} classes")


if __name__ == "__main__":
    main()
