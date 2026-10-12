"""Explicit preparation; test targets themselves never download assets.

Use --source-dir for an offline, hash-checked copy of the three upstream files.
Without it, download only those pinned sources and their license. Packed slices
read bounded rows from the already authenticated checkpoint, never the PLE table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_moe as moe
import qwen38_packed as packed


def prepare(args):
    if args.source_dir:
        moe.CACHE.mkdir(parents=True, exist_ok=True)
        for name in moe.HASHES:
            data = moe.checked_source(args.source_dir, name)
            destination = moe.CACHE / name
            if destination.exists():
                moe.checked_source(moe.CACHE, name)
            else:
                with destination.open("xb") as output:
                    output.write(data)
    else:
        moe.prepare_upstream()
    source = moe.checked_source(moe.CACHE, "modeling_qwen4_exp.py")
    destination = ROOT / "build/qwen38-e/modeling_qwen4_exp.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source:
            raise ValueError("existing E oracle differs from the pinned source")
    else:
        with destination.open("xb") as output:
            output.write(source)
    directory = ROOT / "build/qwen38-b/reference"
    if not directory.exists():
        reference = packed.build_reference(args.dwarfstar, directory, sanitize=True)
    else:
        manifest = json.loads((directory / "manifest.json").read_text())
        reference = directory / "reference"
        if manifest["upstream_revision"] != packed.REVISION or not manifest["sanitizer"]:
            raise ValueError("incompatible existing reference")
        hashes = {**manifest["source_sha256"], str(reference): manifest["binary_sha256"]}
        for path, expected in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"reference input/binary changed: {path}")
    for kind in packed.FORMATS:
        directory = ROOT / f"build/qwen38-b/slices-{kind}"
        if not directory.exists():
            packed.extract_slices(reference, args.checkpoint, directory, kind)
        count = packed.replay_slices(reference, directory)
        print(f"Verified packed format {kind}: {count} real tensor slices", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--dwarfstar", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    prepare(parser.parse_args())
