"""Authenticate the complete Q2 file before publishing its runnable filename."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

EXPECTED_SIZE = 147207127040
EXPECTED_HASH = "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9"
REVISION = "d600fe1a43d2e1cdcadb85144ce3142f66f9eefe"


def verify(path, report):
    if report.exists():
        raise FileExistsError(report)
    initial = path.stat()
    if initial.st_size != EXPECTED_SIZE:
        raise ValueError(f"incomplete file: {initial.st_size} != {EXPECTED_SIZE}")
    final_path = path.with_suffix("") if path.suffix == ".part" else path
    if final_path != path and final_path.exists():
        raise FileExistsError(final_path)
    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    final = path.stat()
    stable = (initial.st_ino, initial.st_size, initial.st_mtime_ns) == (
        final.st_ino, final.st_size, final.st_mtime_ns)
    matched = stable and digest == EXPECTED_HASH
    result = dict(repository="antirez/qwen3.8-flash-next-gguf", revision=REVISION,
                  checked_utc=datetime.now(timezone.utc).isoformat(), size_bytes=final.st_size,
                  sha256=digest, expected_sha256=EXPECTED_HASH, matched=matched,
                  file_stable=stable, file_mtime_ns=final.st_mtime_ns, file_inode=final.st_ino,
                  verification_s=time.clock_gettime(time.CLOCK_MONOTONIC)-start,
                  path=str(final_path.resolve() if matched else path.resolve()),
                  cache_note="Full sequential hashing warms the file cache; subsequent load is not storage-cold.")
    if matched and final_path != path:
        # Publish without replacing another session's model. Hard links consume
        # no second payload copy; remove only our successfully verified .part.
        os.link(path, final_path)
        path.unlink()
    with report.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    if not matched:
        raise ValueError("checksum mismatch or file changed during hashing; partial file preserved")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.path, args.report), indent=2))
