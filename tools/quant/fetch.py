"""fetch.py — download one tensor of a Hugging Face safetensors checkpoint.

The BF16 checkpoint is 360 GB; one expert tensor is 1.7-3.4 GB. HTTP range
requests read the shard's header, then only the tensor's bytes. The data goes
to <name>.bin.part and is renamed only when complete and hashed, so a
<name>.bin on disk is always whole. A sidecar <name>.json records where it
came from and its sha256.

  fetch.py REPO REVISION INDEX_JSON TENSOR OUT_DIR
"""

import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

_UA = {"User-Agent": "picoforge-quant"}


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={**_UA, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise IOError(f"{url}: short read {len(data)} of {end - start + 1} bytes")
    return data


def fetch_tensor(repo: str, revision: str, index_path: Path, name: str, out_dir: Path,
                 base_url: str = "https://huggingface.co") -> Path:
    shard = json.loads(Path(index_path).read_text())["weight_map"][name]
    url = f"{base_url}/{repo}/resolve/{revision}/{shard}"
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    info = json.loads(_get(url, 8, 8 + n - 1))[name]
    begin, end = info["data_offsets"]
    total = end - begin
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final, part = out_dir / f"{name}.bin", out_dir / f"{name}.bin.part"
    h, done = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={**_UA, "Range": f"bytes={8 + n + begin}-{8 + n + end - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            while chunk := r.read(16 << 20):
                f.write(chunk); h.update(chunk); done += len(chunk)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if done != total:
        part.unlink(missing_ok=True)
        raise IOError(f"{name}: downloaded {done} of {total} bytes")
    part.replace(final)
    final.with_suffix(".json").write_text(json.dumps({
        "repo": repo, "revision": revision, "shard": shard, "dtype": info["dtype"],
        "shape": info["shape"], "bytes": total, "sha256": h.hexdigest()}, indent=2) + "\n")
    return final


if __name__ == "__main__":
    print(fetch_tensor(sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])))
