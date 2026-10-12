#!/usr/bin/env python3
"""Run one foreground command under the repository's cross-worktree lock.

Only the Python standard library is needed. Never unlink the lock file.
exec preserves the locked descriptor, so wrapper death cannot orphan a child.
The command must not close that descriptor, daemonize, or launch detached work.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


def lock_path():
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True
    ).strip()
    return Path(common).resolve() / "qwen38-gpu.lock"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="seconds to wait; 0 means immediate, exit 75 if busy")
    parser.add_argument("--print-path", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("timeout must be finite and nonnegative")
    path = lock_path()
    if args.print_path:
        print(path)
        return 0
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a foreground command after -- is required")
    # Deliberately prohibit nested wrappers: one owner wraps the entire suite.
    if "PICOFORGE_GPU_LOCK_FD" in os.environ:
        parser.error("nested GPU lock wrapper; wrap the outer command only")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("lock path is not a regular file")
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(f"GPU lock busy: {path}", file=sys.stderr)
                    return 75
                time.sleep(min(0.05, remaining))
        # Human-readable information, not a PID-based stale-lock mechanism.
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} command={Path(command[0]).name}\n".encode())
        os.set_inheritable(fd, True)
        os.environ["PICOFORGE_GPU_LOCK_FD"] = str(fd)
        os.environ["PICOFORGE_GPU_LOCK_PATH"] = str(path)
        print(f"GPU lock acquired: {path}", file=sys.stderr, flush=True)
        os.execvp(command[0], command)
    finally:
        os.close(fd)  # exec success never reaches here; the command owns it.


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"GPU lock error: {exc}", file=sys.stderr)
        sys.exit(126)
