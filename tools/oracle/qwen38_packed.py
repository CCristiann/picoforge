"""Independent packed-weight oracle; no picoforge decoder or GPU is used.

Build the pinned, unmodified DwarfStar CPU reference in this worktree's build/.
Its decoder remains external; this module provides provenance and transport.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "fc80bd695da76ee14bcb86820bf900a5cdfda806"
DS4_SHA256 = "d137a9a5936d35152d1aa7abd9631462697abd69b429f52e51663292c1d063bd"
ADAPTER = ROOT / "tools/oracle/qwen38_packed_reference.c"
FORMATS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84),
           16: (256, 66), 30: (1, 2)}
BYTE_LIMIT = 32 * 1024 * 1024
IQ2_GRID_SHA256 = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"


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
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                  "-fno-omit-frame-pointer"]
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


def iq2_grid(reference):
    """Read external format constants, checked independently against ggml's hash.

    This is not a decoder. Production constants are never imported by the oracle.
    The canonical little-endian grid hash was cross-checked with ggml at
    8bb3cf0d216a5ae48360179f2931daf284be4d01 during SPEC B.
    """
    import numpy as np

    manifest = json.loads(Path(reference).resolve().with_name("manifest.json").read_text())
    sources = [Path(p) for p, h in manifest["source_sha256"].items()
               if Path(p).name == "ds4.c" and h == DS4_SHA256]
    if len(sources) != 1:
        raise ValueError("missing pinned external codebook source")
    data = sources[0].read_bytes()
    if digest(data) != DS4_SHA256:
        raise ValueError("external codebook source changed")
    match = re.search(rb"static const uint64_t iq2xxs_grid\[256\] = \{([^}]+)\};", data)
    if match is None:
        raise ValueError("external codebook declaration changed")
    raw = b"".join(int(word, 16).to_bytes(8, "little")
                   for word in re.findall(rb"0x[0-9a-fA-F]+", match[1]))
    if len(raw) != 2048 or digest(raw) != IQ2_GRID_SHA256:
        raise ValueError("external codebook differs from canonical ggml grid")
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 8)


def pread_exact(fd, offset, size):
    """Bounded fixture I/O; EOF or any error invalidates the entire slice."""
    if offset < 0 or not 0 < size <= BYTE_LIMIT:
        raise ValueError("invalid fixture read range")
    chunks, done = [], 0
    while done < size:
        try:
            chunk = os.pread(fd, size - done, offset + done)
        except InterruptedError:
            continue
        if not chunk:
            raise EOFError(f"short fixture read at offset {offset + done}")
        chunks.append(chunk)
        done += len(chunk)
    return b"".join(chunks)


def matmul_reference(weights, x, k_logical):
    """FP64 product and sum of absolute terms for the SAME decoded Q2 weights.

    Physical padding is excluded, regardless of its values. This is neither
    BF16-quality measurement nor a production matrix multiplication.
    """
    import numpy as np

    if (weights.dtype != np.float32 or x.dtype != np.float32 or
            weights.ndim != 2 or x.ndim != 2 or
            type(k_logical) is not int or not 0 < k_logical <= weights.shape[1] or
            x.shape[1] != k_logical):
        raise ValueError("expected FP32 [N,Kphysical] and [M,Klogical]")
    if not np.isfinite(weights).all() or not np.isfinite(x).all():
        raise ValueError("nonfinite fixture inputs")
    if weights.nbytes + x.nbytes + 16 * x.shape[0] * weights.shape[0] > BYTE_LIMIT:
        raise ValueError("matrix fixture exceeds the transport budget")
    a, w = x.astype(np.float64), weights[:, :k_logical].astype(np.float64)
    return a @ w.T, np.abs(a) @ np.abs(w.T)


def mathematical_rows(packed, kind, columns, *, grid=None):
    """Declarative FP64 interpretation, cross-checked with the external decoder.

    This local implementation is NOT an independent oracle by itself. Expected
    values rely on original DwarfStar execution plus analytical fixtures and
    the independently pinned ggml codebook. No production decoder is imported.
    """
    import numpy as np

    if kind not in (10, 16) or columns <= 0 or columns % 256:
        raise ValueError("expected Q2_K or IQ2_XXS block-aligned physical rows")
    block_bytes = FORMATS[kind][1]
    row_bytes = columns // 256 * block_bytes
    if (not len(packed) or len(packed) % row_bytes or len(packed) > BYTE_LIMIT or
            len(packed) // block_bytes * 256 * 4 > BYTE_LIMIT):
        raise ValueError("invalid packed row extent")
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(-1, block_bytes)
    if kind == 10:
        d = raw[:, 80:82].copy().view("<f2").astype(np.float64)
        minimum = raw[:, 82:84].copy().view("<f2").astype(np.float64)
        scales = raw[:, :16]
        # Two 128-column halves, each with four 32-column bit planes.
        codes = raw[:, 16:80].reshape(-1, 2, 1, 32)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
        codes = ((codes >> shifts) & 3).reshape(-1, 16, 16)
        values = ((d * (scales & 15))[..., None] * codes -
                  (minimum * (scales >> 4))[..., None])
    else:
        if grid is None or grid.shape != (256, 8):
            raise ValueError("canonical external IQ2_XXS grid required")
        d = raw[:, :2].copy().view("<f2").astype(np.float64)
        words = raw[:, 2:].copy().view("<u4").reshape(-1, 8, 2)
        ids = words[:, :, 0, None] >> np.arange(0, 32, 8, dtype=np.uint32) & 255
        signs = words[:, :, 1, None] >> np.arange(0, 28, 7, dtype=np.uint32) & 127
        bits = signs[..., None] >> np.arange(7, dtype=np.uint32) & 1
        parity = bits.sum(axis=-1).astype(np.uint32) & 1
        sign_mask = signs | (parity << 7)
        negative = sign_mask[..., None] >> np.arange(8, dtype=np.uint32) & 1
        factor = d * (2 * (words[:, :, 1] >> 28) + 1) / 8
        values = factor[..., None, None] * grid[ids] * np.where(negative, -1, 1)
    return values.astype("<f4").reshape(-1, columns)


def extract_slices(reference, checkpoint, output, kind):
    """Read only selected routed-expert rows; publish a manifest after parity."""
    import numpy as np

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if kind not in (10, 16) or not output.is_relative_to(ROOT / "build"):
        raise ValueError("choose Q2_K/IQ2_XXS and an isolated build output directory")
    verification = json.loads((ROOT / "bench/qwen38_q2_20261011_phase0/checkpoint.verification.json").read_text())
    if (verification["matched"] is not True or verification["sha256"] !=
            "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9"):
        raise ValueError("missing authenticated checkpoint provenance")
    spec = importlib.util.spec_from_file_location("inventory", ROOT / "tools/eval/inspect_qwen38_gguf.py")
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(checkpoint, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        if (before.st_size, before.st_ino, before.st_mtime_ns) != (
                verification["size_bytes"], verification["file_inode"], verification["file_mtime_ns"]):
            raise ValueError("checkpoint identity differs from the recorded full-hash verification")
        layout = inventory.inspect(checkpoint)  # header only, no model mapping
        if layout["header_sha256"] != "ec931bcfabe3d85d69a0f7a07cd697b7205afd620b220d4b3099c7cced7a6e75":
            raise ValueError("checkpoint header differs from the pinned inventory")
        tensors = {t["name"]: t for t in layout["tensors"]}
        meta = layout["metadata"]
        trunk = meta["qwen4exp.block_count"] - meta["qwen4exp.nextn_predict_layers"]
        experts = meta["qwen4exp.expert_count"]
        hidden = meta["qwen4exp.embedding_length"]
        width = meta["qwen4exp.expert_feed_forward_length"]
        logical = width if kind == 10 else hidden
        physical, nrows = ((width + 255) // 256 * 256, hidden) if kind == 10 else (hidden, width)
        selections = []
        for layer in sorted({0, (trunk - 1) // 2, trunk - 1}):
            for projection in (("down",) if kind == 10 else ("gate", "up")):
                tensor = tensors[f"blk.{layer}.ffn_{projection}_exps.weight"]
                if tensor["type_id"] != kind or tensor["shape"] != [physical, nrows, experts]:
                    raise ValueError("routed-expert layout differs from metadata")
                row_bytes = physical // 256 * FORMATS[kind][1]
                for expert in sorted({0, (experts - 1) // 2, experts - 1}):
                    for row in sorted({0, nrows // 2, nrows - 1}):
                        offset = tensor["file_offset"] + (expert * nrows + row) * row_bytes
                        if offset + row_bytes > tensor["file_offset"] + tensor["size_bytes"]:
                            raise ValueError("slice crosses tensor boundary")
                        selections.append((tensor, expert, row, offset, row_bytes))
        total_bytes = sum(s[4] for s in selections)
        if total_bytes > BYTE_LIMIT:
            raise ValueError("real slice read budget exceeded")
        output.mkdir(parents=True, exist_ok=False)
        grid = iq2_grid(reference) if kind == 16 else None
        rng = np.random.default_rng(3811)
        records = []
        for tensor, expert, row, offset, size in selections:
            raw = pread_exact(fd, offset, size)
            values = decode(reference, raw, kind, physical, 1)
            mathematical = mathematical_rows(raw, kind, physical, grid=grid)
            np.testing.assert_array_equal(values.view("<u4"), mathematical.view("<u4"))
            x = np.zeros((3, logical), dtype="<f4")
            x[0, 0], x[1, -1] = 1, 1
            x[2] = rng.integers(-16, 17, logical).astype(np.float32) / 16
            y, magnitude = matmul_reference(values, x, logical)
            if kind == 10:
                changed = values.copy()
                changed[:, logical:] = 65504
                changed_y, _ = matmul_reference(changed, x, logical)
                np.testing.assert_array_equal(y, changed_y)
            blobs = {"packed": raw, "decoded.f32": values.tobytes(), "input.f32": x.tobytes(),
                     "product.f64": y.astype("<f8").tobytes(),
                     "sum_abs.f64": magnitude.astype("<f8").tobytes()}
            artifacts = {}
            for suffix, data in blobs.items():
                name = f"{len(records):03d}.{suffix}"
                (output / name).write_bytes(data)
                artifacts[suffix] = {"file": name, "bytes": len(data), "sha256": digest(data)}
            records.append(dict(tensor=tensor["name"], shape=tensor["shape"], format=kind,
                                expert=expert, row=row, file_offset=offset, packed_bytes=size,
                                k_logical=logical, k_physical=physical, input_rows=3,
                                bit_exact=True, padding_nonzero=int(np.count_nonzero(values[:, logical:])),
                                artifacts=artifacts))
        if identity(os.fstat(fd)) != identity(before) or identity(checkpoint.stat()) != identity(before):
            raise ValueError("checkpoint changed during fixture extraction")
        manifest = dict(schema_version=1, complete=True, seed=3811, kind=kind,
                        checkpoint_sha256=verification["sha256"], checkpoint_revision=verification["revision"],
                        checkpoint_identity=identity(before), header_sha256=layout["header_sha256"],
                        bytes_read=total_bytes, header_bytes_read=layout["data_offset"],
                        oracle_manifest=json.loads(Path(reference).resolve().with_name("manifest.json").read_text()),
                        generator_sha256=digest(Path(__file__).read_bytes()), records=records,
                        scope="same-checkpoint decode and FP64 products; no native/GPU/BF16-quality claim")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        os.close(fd)


def replay_slices(reference, directory):
    """Verify saved hashes, decode parity and FP64 products without checkpoint I/O."""
    import numpy as np

    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("incomplete or unsupported slice manifest")
    kind = manifest["kind"]
    if kind not in (10, 16) or not manifest["records"]:
        raise ValueError("invalid slice suite")
    grid = iq2_grid(reference) if kind == 16 else None
    read_bytes = 0
    for record in manifest["records"]:
        columns, logical, rows = record["k_physical"], record["k_logical"], record["input_rows"]
        sizes = {"packed": columns // 256 * FORMATS[kind][1], "decoded.f32": columns * 4,
                 "input.f32": rows * logical * 4, "product.f64": rows * 8, "sum_abs.f64": rows * 8}
        if set(record["artifacts"]) != set(sizes):
            raise ValueError("incomplete slice artifacts")
        blobs = {}
        for name, expected_size in sizes.items():
            info = record["artifacts"][name]
            file = directory / info["file"]
            if (file.resolve().parent != directory or not 0 < expected_size <= BYTE_LIMIT or
                    info["bytes"] != expected_size or file.stat().st_size != expected_size):
                raise ValueError("invalid slice artifact extent")
            data = file.read_bytes()
            if digest(data) != info["sha256"]:
                raise ValueError("slice artifact checksum mismatch")
            blobs[name] = data
        values = decode(reference, blobs["packed"], kind, columns, 1)
        expected = np.frombuffer(blobs["decoded.f32"], dtype="<f4").reshape(1, columns)
        mathematical = mathematical_rows(blobs["packed"], kind, columns, grid=grid)
        np.testing.assert_array_equal(values.view("<u4"), expected.view("<u4"))
        np.testing.assert_array_equal(values.view("<u4"), mathematical.view("<u4"))
        x = np.frombuffer(blobs["input.f32"], dtype="<f4").reshape(rows, logical)
        product, magnitude = matmul_reference(values, x, logical)
        np.testing.assert_array_equal(product.ravel(), np.frombuffer(blobs["product.f64"], dtype="<f8"))
        np.testing.assert_array_equal(magnitude.ravel(), np.frombuffer(blobs["sum_abs.f64"], dtype="<f8"))
        read_bytes += sizes["packed"]
    if read_bytes != manifest["bytes_read"]:
        raise ValueError("slice payload total mismatch")
    return len(manifest["records"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build", help="compile the original CPU reference")
    build.add_argument("upstream", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--sanitize", action="store_true")
    slices = commands.add_parser("slices", help="extract bounded real rows and verify parity")
    slices.add_argument("reference", type=Path)
    slices.add_argument("checkpoint", type=Path)
    slices.add_argument("output", type=Path)
    slices.add_argument("--kind", type=int, choices=[10, 16], required=True)
    replay = commands.add_parser("replay", help="verify saved fixtures without reading the checkpoint")
    replay.add_argument("reference", type=Path)
    replay.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        print(build_reference(args.upstream, args.output, sanitize=args.sanitize))
    elif args.action == "slices":
        result = extract_slices(args.reference, args.checkpoint, args.output, args.kind)
        print(f"PASS: {len(result['records'])} real rows, {result['bytes_read']} payload bytes, bit-exact decode")
    else:
        print(f"PASS: replayed {replay_slices(args.reference, args.directory)} real rows")
