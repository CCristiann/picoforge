"""Run one command with monotonic stream events and macOS memory/I/O samples.

Output directories are exclusive. Counter samples are system-wide where named
system_*; process disk bytes are not attributed to n-grams. No shell is used.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def now():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def sample(probe, pid=None):
    result = subprocess.run([str(probe), *([str(pid)] if pid else [])],
                            capture_output=True, text=True, timeout=5)
    return json.loads(result.stdout) if result.returncode == 0 else None


# Kill the child after `samples` consecutive samples at pressure level >= `pressure`
# or swap growth > `swap_bytes`. Callers may pass a stricter guard; this is the default.
# Optional `gap_s` (not in the default): also kill the child when two consecutive sample
# attempts are more than that many seconds apart on our clock (see the loop below).
DEFAULT_GUARD = dict(pressure=4, swap_bytes=2 * 2**30, samples=3)


def run(output, cwd, command, timeout=1200, interval=1.0, env_overrides=None, guard=None):
    limits = dict(DEFAULT_GUARD, **(guard or {}))
    probe = ROOT / "build/process_memory"
    before = sample(probe)
    if before is None:
        raise RuntimeError("process_memory probe failed; compile it first")
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(command[0]).resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    env["DS4_METAL_MEMORY_REPORT"] = "1"
    env.update(env_overrides or {})
    thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=True).stdout
    with executable.open("rb") as binary:
        binary_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    metadata = dict(command=command, cwd=str(cwd), pilot=True, complete=False,
                    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    executable_sha256=binary_hash,
                    environment={"DS4_METAL_MEMORY_REPORT": "1", **(env_overrides or {})},
                    removed_environment_names=sorted(k for k in os.environ if k.startswith("DS4_")),
                    thermal_before=thermal, sample_interval_s=interval, system_before=before,
                    guard=("stop after 3 samples: critical pressure or swap growth > 2 GiB" if guard is None else
                           f"stop after {limits['samples']} samples: pressure level >= {limits['pressure']} "
                           f"or swap growth > {limits['swap_bytes'] / 2**20:g} MiB"),
                    guard_limits=limits)
    meta_path = output / "process.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    start = now()
    # Preserve a caller-owned GPU lock even if the observer dies. The child
    # may run in another checkout, so it must inherit, never reacquire it.
    lock_fd = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    pass_fds = (int(lock_fd),) if lock_fd is not None else ()
    for fd in pass_fds:
        os.fstat(fd)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True, pass_fds=pass_fds)
    selector = selectors.DefaultSelector()
    logs = {name: (output / f"{name}.log").open("wb") for name in ("stdout", "stderr")}
    for name in logs:
        selector.register(getattr(process, name), selectors.EVENT_READ, name)
    first, samples, reason, bad, next_sample = {}, [], None, 0, start
    last_attempt, max_gap = None, None
    terminated = None
    try:
        with (output / "events.jsonl").open("w") as events, (output / "memory.jsonl").open("w") as memory:
            while selector.get_map() or process.poll() is None:
                current = now()
                if current >= next_sample and process.poll() is None:
                    # now() is CLOCK_MONOTONIC, which on macOS keeps counting while the Mac sleeps,
                    # so a sleep shows up as one long gap between attempts (a failed probe counts
                    # too). A sample that spanned a sleep is not a measurement: stop the child.
                    if last_attempt is not None:
                        max_gap = max(current - last_attempt, max_gap or 0)
                        if limits.get("gap_s") and current - last_attempt > limits["gap_s"]:
                            reason = reason or "sleep gap"
                    last_attempt = current
                    row = sample(probe, process.pid)
                    next_sample = current + interval
                    if row is not None:
                        row["elapsed_s"] = row["monotonic_s"] - start
                        samples.append(row)
                        memory.write(json.dumps(row) + "\n")
                        memory.flush()
                        unsafe = (row["system_pressure_level"] >= limits["pressure"] or
                                  row["system_swap_used_bytes"] - before["system_swap_used_bytes"] > limits["swap_bytes"])
                        bad = bad + 1 if unsafe else 0
                        if bad >= limits["samples"]:
                            reason = "memory guard"
                if current - start > timeout:
                    reason = reason or "timeout"
                if reason and process.poll() is None:
                    if terminated is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        terminated = now()
                    elif now() - terminated > 5:
                        os.killpg(process.pid, signal.SIGKILL)
                for key, _ in selector.select(.1):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    elapsed = now() - start
                    first.setdefault(key.data, elapsed)
                    logs[key.data].write(data)
                    logs[key.data].flush()
                    events.write(json.dumps(dict(stream=key.data, elapsed_s=elapsed, bytes=len(data))) + "\n")
                    events.flush()
        returncode = process.wait()
    except BaseException:
        reason = "runner interrupted"
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    finally:
        elapsed = now() - start
        selector.close()
        for log in logs.values():
            log.close()
        metadata.update(pid=process.pid, start_monotonic_s=start, process_s=elapsed,
                        returncode=process.returncode, stop_reason=reason,
                        complete=process.returncode == 0 and reason is None,
                        first_stream_bytes_s=first, system_after=sample(probe),
                        samples=len(samples), max_sample_gap_s=max_gap,
                        sampled_peak_footprint_bytes=max((s["phys_footprint_bytes"] for s in samples), default=None),
                        lifetime_peak_footprint_bytes=max((s["lifetime_peak_phys_footprint_bytes"] for s in samples), default=None))
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return returncode if returncode else (1 if reason else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("a command and positive timeout are required")
    raise SystemExit(run(args.output.resolve(), args.cwd.resolve(), command, args.timeout))
