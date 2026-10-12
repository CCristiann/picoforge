"""Resume the one pinned Q2 artifact; keep partial transfers after network errors.

Curl's default --retry does not retry a prematurely closed transfer (exit 18).
Each retry here restarts curl so -C - re-reads the *current* partial-file length.
The final file remains .part until verify_qwen38_download.py hashes every byte.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from verify_qwen38_download import EXPECTED_SIZE, REVISION

FILENAME = "Qwen3.8-Flash-Next-Q2.gguf"
URL = f"https://huggingface.co/antirez/qwen3.8-flash-next-gguf/resolve/{REVISION}/{FILENAME}"


def download(directory, evidence, attempts=10):
    directory.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    part = directory / (FILENAME + ".part")
    if (directory / FILENAME).exists():
        raise FileExistsError("published checkpoint exists; verify it instead of downloading again")
    with (directory / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (evidence / "download.log").open("ab") as log, (evidence / "download-attempts.jsonl").open("a") as audit:
            for attempt in range(attempts):
                before = part.stat().st_size if part.exists() else 0
                if before == EXPECTED_SIZE:
                    return 0
                if before > EXPECTED_SIZE:
                    raise ValueError("partial file exceeds expected size")
                if shutil.disk_usage(directory).free < EXPECTED_SIZE - before + 16 * 2**30:
                    raise ValueError("insufficient disk space for the remaining file plus 16 GiB reserve")
                command = ["/usr/bin/curl", "--location", "--fail", "--show-error", "--silent",
                           "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                           "--continue-at", "-", "--output", str(part), URL]
                started = datetime.now(timezone.utc).isoformat()
                begin = time.clock_gettime(time.CLOCK_MONOTONIC)
                print(f"attempt {attempt+1}: resume at {before} bytes", flush=True)
                completed = subprocess.run(command, stdout=log, stderr=log)
                after = part.stat().st_size if part.exists() else 0
                record = dict(started_utc=started, command=command, offset=before, bytes_after=after,
                              returncode=completed.returncode,
                              elapsed_s=time.clock_gettime(time.CLOCK_MONOTONIC)-begin)
                audit.write(json.dumps(record) + "\n")
                audit.flush()
                if completed.returncode == 0 and after == EXPECTED_SIZE:
                    return 0
                if completed.returncode not in (5, 6, 7, 18, 22, 28, 35, 52, 55, 56):
                    return completed.returncode or 1
                time.sleep(5)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    raise SystemExit(download(args.directory.resolve(), args.evidence.resolve()))
