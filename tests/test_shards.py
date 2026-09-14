"""test_shards.py — a sharded checkpoint is the same model, to the bit.

Qwen3-30B-A3B ships as 16 files and an index. Before any of it is read, every
reader is held to the tiny MoE split into 3 shards the Hub's way
(tools/synth/shard_checkpoint.py), against the same model in one file:

  - the C engine's CPU logits: identical bits (same bytes, same arithmetic);
  - tools/quant/quantize.py: an identical output file, byte for byte;
  - the NumPy oracle's lazy loader, with a cache far smaller than the model:
    identical logits to the eager single-file load;
  - the GPU: refuses a sharded checkpoint by name (one buffer, 41.75 GB cap).

Run:  make test-shards
"""

import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
from qwen3_forward import LazyWeights, Qwen3Config, forward, load_weights  # noqa: E402

SINGLE, SHARDED = ROOT / "build/tiny-qwen3-moe", ROOT / "build/tiny-qwen3-moe-sharded"
IDS = [785, 6722, 315, 9625, 374]


def c_logits(model: Path, mode: str) -> tuple[bytes, subprocess.CompletedProcess]:
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        p = subprocess.run([str(ROOT / "picoforge"), str(model), mode, tmp.name, *map(str, IDS)],
                           capture_output=True, text=True, cwd=ROOT)
        return Path(tmp.name).read_bytes(), p


def check(label: str, ok: bool) -> bool:
    print(f"  {label:62s} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ok = True
    a, _ = c_logits(SINGLE, "--forward")
    b, _ = c_logits(SHARDED, "--forward")
    ok &= check("C engine, CPU logits: 3 shards == 1 file, bit for bit", len(a) > 0 and a == b)

    _, p = c_logits(SHARDED, "--gpu-forward")
    ok &= check("GPU refuses a sharded checkpoint", p.returncode != 0 and "sharded" in p.stderr + p.stdout)

    for src in (SINGLE, SHARDED):
        subprocess.run([str(ROOT / "tools/venv/bin/python"), str(ROOT / "tools/quant/quantize.py"),
                        str(src), "q8_row"], check=True, capture_output=True, cwd=ROOT)
    same = filecmp.cmp(ROOT / "build/tiny-qwen3-moe-q8_row/model.safetensors",
                       ROOT / "build/tiny-qwen3-moe-sharded-q8_row/model.safetensors", shallow=False)
    ok &= check("quantize.py: 3 shards in == 1 file in, byte for byte", same)

    cfg = Qwen3Config.from_json(SINGLE)
    lazy = LazyWeights(SHARDED, budget_bytes=40 << 20)
    same = np.array_equal(forward(IDS, load_weights(SINGLE), cfg), forward(IDS, lazy, cfg))
    ok &= check("oracle: lazy sharded weights (40 MB cache) == eager load", same)

    print("\n" + ("PASS: sharded checkpoints read as the same model" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
