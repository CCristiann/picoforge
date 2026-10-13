"""Replay CPU/Metal C compositions and fresh bounded D/E real-weight cases."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    args = parser.parse_args()
    fd = int(os.environ["PICOFORGE_GPU_LOCK_FD"])
    parent = ROOT / "build/qwen38-integration"
    parent.mkdir(exist_ok=True)
    report = Path(tempfile.mkdtemp(prefix="compositions.", dir=parent))
    print(f"Composition reports: {report}", flush=True)

    def run(*command):
        subprocess.run([sys.executable, "-B", *map(str, command)], cwd=ROOT,
                       pass_fds=(fd,), check=True)

    common = ["--source", "build/qwen38-f/upstream/modeling_qwen4_exp.py",
              "--packed-reference", "build/qwen38-b/reference/reference"]
    run("tools/qwen38_test.py", "tests/test_qwen38_recurrent.py", *common,
        "--cpu-lib", "build/qwen38-c/libcpu.dylib", "--report", report / "c-controls.json",
        "--controls", *[f"build/qwen38-c/controls-{n}" for n in (0, 1, 46)])
    for case in ("synthetic", "real-0"):
        for backend in ("cpu", "gpu"):
            gpu = (["--composed-gpu", "build/qwen38-c/test_gpu", "--metallib", args.metallib]
                   if backend == "gpu" else [])
            run("tools/qwen38_test.py", "tests/test_qwen38_recurrent.py", *common,
                "--composition", f"build/qwen38-c/composed-{case}",
                "--composed-native", "build/qwen38-c/test_cpu", *gpu,
                "--report", report / f"c-{case}-{backend}.json")
    run("tools/oracle/qwen38_attention.py", *common, "--slice-checkpoint", args.checkpoint,
        "--native", "build/qwen38-d/test_attention", "--output", report / "d-real")
    run("tools/oracle/qwen38_ngram.py", "--real-checkpoint", args.checkpoint,
        "--native", "build/qwen38-e/test_ngram", "--packed-reference",
        "build/qwen38-b/reference/reference", "--output", report / "e-real")


if __name__ == "__main__":
    main()
