"""Explicit bounded fixture preparation, under the common repository lock.

An existing fixture is accepted only against its coordinator hash receipt.
No download, full checkpoint mapping, or full checkpoint checksum occurs here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_attention as attention
import qwen38_moe as moe
import qwen38_recurrent as recurrent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory, inputs, generate):
    receipt = directory / "integration-receipt.json"
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    if directory.exists():
        saved = json.loads(receipt.read_text())  # Partial preparations fail closed.
        if saved["inputs"] != hashes:
            raise ValueError(f"preparation inputs changed: {directory}")
        for name, expected in saved["files"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"fixture changed: {directory / name}")
    else:
        generate()
        files = {str(p.relative_to(directory)): digest(p)
                 for p in sorted(directory.rglob("*")) if p.is_file()}
        receipt.write_text(json.dumps(dict(inputs=hashes, files=files), indent=2) + "\n")
    print(f"Verified composition fixture: {directory.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PICOFORGE_GPU_LOCK_FD"):
        raise RuntimeError("preparation requires the shared GPU/heavy-work lock")
    reference = ROOT / "build/qwen38-b/reference/reference"
    source = ROOT / "build/qwen38-f/upstream/modeling_qwen4_exp.py"
    native = ROOT / "build/qwen38-c/test_cpu"
    inputs = [native, reference, source, Path(recurrent.__file__)]
    directory = ROOT / "build/qwen38-c/composed-synthetic"
    prepare(directory, inputs, lambda: recurrent.prepare_composition(native, reference, directory))
    for layer in (0, 1, 46):
        directory = ROOT / f"build/qwen38-c/controls-{layer}"
        prepare(directory, inputs, lambda: recurrent.prepare_composition(
            native, reference, directory, checkpoint=args.checkpoint,
            layer=layer, tokens=5, controls_only=True))
    directory = ROOT / "build/qwen38-c/composed-real-0"
    prepare(directory, inputs, lambda: recurrent.prepare_composition(
        native, reference, directory, checkpoint=args.checkpoint, tokens=9))
    native = ROOT / "build/qwen38-d/test_attention"
    directory = ROOT / "build/qwen38-d/slice-01"
    prepare(directory, [native, reference, source, Path(attention.__file__)],
            lambda: attention.verify_real_slice(attention.load_reference(source),
                                                native, reference, args.checkpoint, directory))
    native = ROOT / "build/qwen38-f/test_moe"
    for layer in (0, 3, 47):
        directory = ROOT / f"build/qwen38-f/slice-{layer}"
        prepare(directory, [native, source, Path(moe.__file__)],
                lambda: moe.prepare_real_slice(native, args.checkpoint, directory, layer))


if __name__ == "__main__":
    main()
