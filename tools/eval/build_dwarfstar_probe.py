"""Build observation copies of pinned DwarfStar, leaving its checkout unchanged.

Instrumentation counts n-gram pread calls/bytes and summed service time across
threads (NOT critical-path latency or physical SSD traffic). Metal reports the
device's actual currentAllocatedSize at the existing memory-report boundary.
All changes are archived as a patch; stock DwarfStar remains the speed control.
"""
import argparse
import difflib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "fc80bd695da76ee14bcb86820bf900a5cdfda806"
COUNTERS = '''#include <stdatomic.h>
static _Atomic uint64_t pf_ngram_calls, pf_ngram_bytes, pf_ngram_ns, pf_ngram_errors;
void picoforge_ngram_report(const char *phase) {
    fprintf(stderr, "ngram_io: {\\\"phase\\\":\\\"%s\\\",\\\"calls\\\":%llu,"
        "\\\"bytes\\\":%llu,\\\"summed_service_ns\\\":%llu,\\\"errors\\\":%llu}\\n", phase,
        (unsigned long long)atomic_load(&pf_ngram_calls),
        (unsigned long long)atomic_load(&pf_ngram_bytes),
        (unsigned long long)atomic_load(&pf_ngram_ns),
        (unsigned long long)atomic_load(&pf_ngram_errors));
}
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("upstream instrumentation anchor changed")
    return text.replace(old, new)


def build(upstream, output):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != REVISION:
        raise ValueError("DwarfStar revision differs from the reviewed baseline")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "ds4.c", "ds4_metal.m"],
                   cwd=upstream, check=True)
    output.mkdir(parents=True, exist_ok=False)
    patch = []
    for name in ("ds4.c", "ds4_metal.m"):
        original = (upstream / name).read_text()
        if name == "ds4.c":
            anchor = "static bool qwen4_ngram_row(const ds4_model *m, uint32_t row, float *out) {"
            modified = replace_once(original, anchor, COUNTERS + "\n" + anchor)
            read = "ssize_t n = pread(m->ngram_fd, raw + done, bytes - done, (off_t)(offset + done));"
            modified = replace_once(modified, read, "double pf_start = now_sec();\n        " + read + '''
        int pf_errno = errno;
        double pf_elapsed = now_sec() - pf_start;
        atomic_fetch_add_explicit(&pf_ngram_calls, 1, memory_order_relaxed);
        if (n > 0) atomic_fetch_add_explicit(&pf_ngram_bytes, (uint64_t)n, memory_order_relaxed);
        if (n <= 0) atomic_fetch_add_explicit(&pf_ngram_errors, 1, memory_order_relaxed);
        if (pf_elapsed >= 0.0)
            atomic_fetch_add_explicit(&pf_ngram_ns, (uint64_t)(pf_elapsed * 1e9), memory_order_relaxed);
        errno = pf_errno;''')
        else:
            anchor = "    ds4_gpu_print_task_memory_report();"
            modified = replace_once(original, anchor, anchor + '''
    fprintf(stderr, "metal_allocated_bytes: %llu\\n",
            (unsigned long long)[g_device currentAllocatedSize]);''')
        (output / name).write_text(modified)
        patch.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                        fromfile="a/" + name, tofile="b/" + name))
    (output / "observability.patch").write_text("".join(patch))
    for name in ("ds4.c", "ds4_metal.m"):
        subprocess.run(["clang", "-O3", "-ffast-math", "-g", "-mcpu=native", "-Wall", "-Wextra",
                        *( ["-fobjc-arc"] if name.endswith(".m") else ["-std=c11"] ),
                        "-I", str(upstream), "-c", str(output / name),
                        "-o", str((output / name).with_suffix(".o"))], check=True)
    objects = [output / "ds4.o", output / "ds4_metal.o"]
    objects += [upstream / (name + ".o") for name in
                ("ds4_image", "ds4_distributed", "ds4_tp", "ds4_ssd", "ds4_layer_pack", "ds4_engram")]
    subprocess.run(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wconversion",
                    "-DPICOFORGE_DWARFSTAR_OBSERVE", "-I", str(upstream),
                    str(ROOT / "tools/eval/dwarfstar_probe.c"), *map(str, objects),
                    "-lm", "-pthread", "-framework", "Foundation", "-framework", "Metal",
                    "-o", str(output / "dwarfstar_probe")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.upstream.resolve(), args.output.resolve())
