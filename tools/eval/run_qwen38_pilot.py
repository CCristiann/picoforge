"""Serial, pinned-checkpoint pilot; generate answers but do not execute them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_observed import run

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot(evidence, upstream, output):
    verification = json.loads((evidence / "checkpoint.verification.json").read_text())
    model = Path(verification["path"])
    stat = model.stat()
    if (not verification["matched"] or
        (stat.st_size, stat.st_mtime_ns, stat.st_ino) !=
        (verification["size_bytes"], verification["file_mtime_ns"], verification["file_inode"])):
        raise ValueError("checkpoint is missing verification or changed since hashing")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != "fc80bd695da76ee14bcb86820bf900a5cdfda806":
        raise ValueError("unexpected DwarfStar revision")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=upstream, check=True)
    suite_dir = evidence / "suite"
    suite = json.loads((suite_dir / "suite.json").read_text())
    cases = {case["id"]: case for case in suite["cases"]}
    for case in cases.values():
        if digest(suite_dir / case["prompt_file"]) != case["prompt_sha256"]:
            raise ValueError("prompt changed after freezing the suite")
    binary = ROOT / "build/dwarfstar-observed/dwarfstar_probe"
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "tools/eval").glob("*qwen38*"))
    sources += [ROOT / "tools/eval" / name for name in
                ("dwarfstar_probe.c", "build_dwarfstar_probe.py", "run_observed.py")]
    sources += list((ROOT / "tools/probe").glob("*memory*"))
    plan = dict(pilot=True, complete=False, suite_sha256=digest(suite_dir / "suite.json"),
                checkpoint_sha256=verification["sha256"], dwarfstar_revision=revision,
                probe_sha256=digest(binary), source_sha256={str(p.relative_to(ROOT)): digest(p)
                                                          for p in sources if p.is_file()},
                case_order=suite["process_order"], measured_repeats=suite["measured_repetitions"],
                warmup_repeats=suite["first_use_repetitions"], finished=[], quality_graded=False,
                context_tokens=8192, prefill_chunk_tokens=1024, mtp=False,
                limitations=["Pilot only; three hot repeats are not a stable performance comparison.",
                             "First-use process load after hashing is not controlled cold-storage load.",
                             "N-gram instrumentation adds overhead; stock binaries are separate controls."])
    plan_path = output / "pilot.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    # A stock CLI control runs before the measured client. No inference timing
    # overlaps with builds, checksum reads, downloads, or another GPU workload.
    case = cases["lower-bound"]
    control = [str(upstream / "ds4"), "-m", str(model), "--ctx", "8192", "--prefill-chunk", "1024",
               "--temp", "0", "--nothink", "-sys", "", "-n", str(case["max_new"]),
               "--prompt-file", str(suite_dir / case["prompt_file"])]
    rc = run(output / "stock-cli", upstream, control)
    if rc:
        return rc
    for case_id in suite["process_order"]:
        case = cases[case_id]
        request = output / case_id
        command = [str(binary), str(model), str(suite_dir / case["prompt_file"]),
                   str(request / "answer"), str(case["max_new"]), case["mode"],
                   str(suite["measured_repetitions"] + suite["first_use_repetitions"])]
        print("running", case_id, flush=True)
        rc = run(request, upstream, command)
        if rc:
            plan["failed"] = case_id
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            return rc
        plan["finished"].append(case_id)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    corpus = upstream / "speed-bench/promessi_sposi.txt"
    plan["throughput_corpus_sha256"] = digest(corpus)
    command = [str(upstream / "ds4-bench"), "-m", str(model), "--prefill-chunk", "1024",
               "--prompt-file", str(corpus), "--ctx-start", "8063", "--ctx-max", "8063",
               "--ctx-alloc", "8192", "--gen-tokens", "128", "--teacher-forced-decode",
               "--csv", str(output / "stock-throughput.csv")]
    rc = run(output / "stock-throughput", upstream, command)
    plan["complete"] = rc == 0
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(pilot(args.evidence.resolve(), args.upstream.resolve(), args.output.resolve()))
