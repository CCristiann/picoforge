"""Independent packed-weight oracle; no picoforge decoder or GPU is used.

Build the pinned, unmodified DwarfStar CPU reference in this worktree's build/.
Its decoder remains external; this module provides provenance and transport.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "fc80bd695da76ee14bcb86820bf900a5cdfda806"
DS4_SHA256 = "d137a9a5936d35152d1aa7abd9631462697abd69b429f52e51663292c1d063bd"
ADAPTER = ROOT / "tools/oracle/qwen38_packed_reference.c"
FORMATS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84),
           16: (256, 66), 30: (1, 2)}
BYTE_LIMIT = 32 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command_output(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def build_reference(upstream, output, *, sanitize=False):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / "build") or output == ROOT / "build":
        raise ValueError("oracle output must be a new directory under this worktree's build/")
    revision = command_output(["git", "rev-parse", "HEAD"], cwd=upstream)
    if revision != REVISION or digest((upstream / "ds4.c").read_bytes()) != DS4_SHA256:
        raise ValueError("DwarfStar source differs from the pinned reference")
    flags = ["clang", "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off",
             "-Wall", "-Wextra", "-Wno-unused-function",
             '-DDS4_REFERENCE_SOURCE="' + str(upstream / "ds4.c") + '"',
             "-I", str(upstream)]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    dependency_text = command_output([*flags, "-MM", "-MT", "reference", str(ADAPTER)])
    dependencies = [Path(p).resolve() for p in
                    shlex.split(dependency_text.replace("\\\n", "").partition(":")[2])]
    inputs = {}
    for path in [*dependencies, upstream / "LICENSE"]:
        data = path.read_bytes()
        if path.is_relative_to(upstream):
            name = path.relative_to(upstream).as_posix()
            pinned = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=upstream)
            if data != pinned:
                raise ValueError(f"modified upstream dependency: {name}")
        elif path != ADAPTER:
            raise ValueError(f"unexpected oracle dependency: {path}")
        inputs[str(path)] = digest(data)
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "reference"
    command = [*flags, str(ADAPTER), "-Wl,-dead_strip", "-lm", "-pthread", "-o", str(binary)]
    subprocess.run(command, check=True)
    for path, expected in inputs.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError(f"oracle source changed during compilation: {path}")
    (output / "DwarfStar-LICENSE").write_bytes((upstream / "LICENSE").read_bytes())
    manifest = dict(schema_version=1, upstream_revision=revision,
                    decoder="ds4.c:qwen4_ref_row", adapter_sha256=inputs[str(ADAPTER)],
                    source_sha256=inputs, compiler=command_output(["clang", "--version"]),
                    command=command, binary_sha256=digest(binary.read_bytes()),
                    sanitizer=sanitize, byte_limit=BYTE_LIMIT, gpu=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return binary


def decode(reference, packed, kind, columns, rows):
    """Return physical rows as FP32; never infer or remove logical padding."""
    import numpy as np

    if kind not in FORMATS or type(columns) is not int or type(rows) is not int:
        raise ValueError("unsupported format or noninteger shape")
    block, size = FORMATS[kind]
    if columns <= 0 or rows <= 0 or columns % block:
        raise ValueError("invalid physical shape")
    expected = columns // block * size * rows
    if expected > BYTE_LIMIT or columns * rows * 4 > BYTE_LIMIT:
        raise ValueError("oracle request exceeds the 32 MiB transport budget")
    if len(packed) != expected:
        raise ValueError("packed byte count differs from the physical shape")
    reference = Path(reference).resolve()
    manifest = json.loads(reference.with_name("manifest.json").read_text())
    if (manifest["upstream_revision"] != REVISION or manifest["decoder"] != "ds4.c:qwen4_ref_row"
            or digest(reference.read_bytes()) != manifest["binary_sha256"]):
        raise ValueError("unverified oracle binary")
    result = subprocess.run([str(reference), str(kind), str(columns), str(rows)],
                            input=packed, capture_output=True, check=True, timeout=30)
    if len(result.stdout) != columns * rows * 4:
        raise ValueError("incomplete oracle output")
    return np.frombuffer(result.stdout, dtype="<f4").reshape(rows, columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    print(build_reference(args.upstream, args.output, sanitize=args.sanitize))
