"""shard_checkpoint.py — split a checkpoint the way the Hub ships large ones.

    ./tools/venv/bin/python tools/synth/shard_checkpoint.py SRC_DIR DST_DIR N

Qwen3-30B-A3B arrives as 16 files and a model.safetensors.index.json whose
weight_map names the file each tensor lives in. Every tool that will read it
-- the C loader, the quantiser, the oracle -- is first held to a small
checkpoint split the same way: same tensors, same bytes, N files. Tensors are
dealt out in name order, cutting at roughly equal byte counts, and the index
carries metadata.total_size as the Hub's does.
"""

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    with safe_open(src / "model.safetensors", framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in sorted(f.keys())}
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    shards, cur, size = [], {}, 0
    for name, t in tensors.items():
        cur[name] = t
        size += t.numel() * t.element_size()
        if size >= total / n and len(shards) < n - 1:
            shards.append(cur)
            cur, size = {}, 0
    shards.append(cur)

    weight_map = {}
    for i, shard in enumerate(shards):
        file = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, dst / file)
        weight_map |= {k: file for k in shard}
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model"):
            shutil.copy2(f, dst / f.name)
    print(f"{len(tensors)} tensors in {len(shards)} shards -> {dst}")


if __name__ == "__main__":
    main()
