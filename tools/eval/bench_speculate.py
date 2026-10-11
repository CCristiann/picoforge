"""Compare greedy, fixed and adaptive drafting; retain every run.

Usage: bench_speculate.py TARGET DRAFT OUT.csv [REPS]
Run --help for prompt files, context length and configuration selection.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_PROMPTS = Path(__file__).with_name("prompts.jsonl")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def command_output(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def fingerprint(model):
    # Read once before timing; this also makes the checkpoint identity explicit.
    names = ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt")
    paths = [model / name for name in names if (model / name).exists()]
    paths += sorted(model.glob("*.safetensors"))
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}


def configuration(name, draft, prompt, count):
    if name == "plain":
        return ["--spec-greedy", prompt, str(count), "0"]
    kind, sep, length = name.partition("-")
    if not sep or kind not in ("model", "lookup"):
        raise ValueError(f"unknown configuration: {name}")
    if length != "auto" and (not length.isdigit() or not 1 <= int(length) <= 31):
        raise ValueError(f"draft length must be auto or 1..31: {name}")
    return (["--spec-model", str(draft)] if kind == "model" else ["--spec-greedy"]) + [prompt, str(count), length]


def engine(model, args, context):
    with tempfile.NamedTemporaryFile(suffix=".txt") as out:
        cmd = [str(ROOT / "picoforge"), "--context", str(context), str(model), *args, out.name]
        start = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
        wall_s = time.perf_counter() - start
        if p.returncode:
            raise RuntimeError(f"engine failed: {p.stderr}\n{p.stdout[-2000:]}")
        ids = [int(t) for t in Path(out.name).read_text().splitlines()[0].split()]
    lines = [line.removeprefix("metrics: ") for line in p.stdout.splitlines() if line.startswith("metrics: ")]
    metrics = json.loads(lines[0]) if lines else None
    if metrics is not None:
        if metrics["generated"] != len(ids):
            raise RuntimeError("metrics and output disagree on the token count")
        metrics["process_s"] = wall_s
        metrics["generation_s"] = metrics["prefill_s"] + metrics["draft_prefill_s"] + metrics["decode_s"]
        metrics["tok_s"] = max(0, len(ids) - 1) / metrics["decode_s"] if metrics["decode_s"] else 0.0
    return ids, metrics


def percentile(values, p):
    values = sorted(values)
    return values[int(p * (len(values) - 1) + 0.5)]


def summarize(records, names):
    rows = []
    for prompt in dict.fromkeys(r["prompt"] for r in records):
        grouped = {name: [r for r in records if r["prompt"] == prompt and r["config"] == name]
                   for name in names}
        base = statistics.median(r["tok_s"] for r in grouped["plain"])
        for name, runs in grouped.items():
            med = lambda k: statistics.median(r[k] for r in runs)
            drafted = sum(r["drafted"] for r in runs)
            row = {"prompt": prompt, "config": name, "reps": len(runs),
                   "tokens": runs[0]["generated"], "prompt_tokens": runs[0]["prompt_tokens"],
                   "median_tok_s": med("tok_s"),
                   "p10_tok_s": percentile([r["tok_s"] for r in runs], .1),
                   "p90_tok_s": percentile([r["tok_s"] for r in runs], .9),
                   "speedup": med("tok_s") / base if base else 0.0,
                   "tokens_per_pass": statistics.median(max(0, r["generated"] - 1) / max(1, r["passes"]) for r in runs),
                   "acceptance": sum(r["accepted"] for r in runs) / drafted if drafted else 0.0,
                   "median_prefill_s": med("prefill_s"), "median_draft_prefill_s": med("draft_prefill_s"),
                   "median_draft_s": med("draft_s"), "median_verify_s": med("verify_s"),
                   "median_generation_s": med("generation_s"), "median_process_s": med("process_s"),
                   "p90_process_s": percentile([r["process_s"] for r in runs], .9),
                   "peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs),
                   "identical_to_greedy": all(r["identical_to_greedy"] for r in runs)}
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("reps", type=int, nargs="?", default=25)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--configs", nargs="+", default=["plain", "model-2", "model-auto"])
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1, help="discarded requests per prompt and configuration")
    parser.add_argument("--seed", type=int, default=20261011, help="configuration order only; generation is greedy")
    args = parser.parse_args()
    args.target = args.target.resolve()
    args.draft = args.draft.resolve()
    if args.reps < 1 or args.warmup < 1 or not 1 <= args.max_new <= args.context:
        parser.error("use positive repeats, at least one warmup, and 1 <= max-new <= context")
    if "plain" not in args.configs or len(set(args.configs)) != len(args.configs):
        parser.error("configs must contain plain and have no duplicates")
    try:
        for name in args.configs:
            configuration(name, args.draft, "test", args.max_new)
    except ValueError as e:
        parser.error(str(e))
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        parser.error("prompt ids must be nonempty and unique")
    for p in prompts:
        if not isinstance(p["id"], str) or not p["id"] or not isinstance(p["text"], str) or not p["text"]:
            parser.error("each prompt needs an id and nonempty text")
    if args.out.suffix != ".csv":
        parser.error("output must end in .csv")
    raw_path = args.out.with_suffix(".runs.jsonl")
    meta_path = args.out.with_suffix(".meta.json")
    for p in (args.out, raw_path, meta_path):
        if p.exists():
            parser.error(f"refusing to overwrite {p}; choose a new output name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Hashing checkpoints before measurement...", flush=True)
    meta = {"schema": 1, "complete": False, "started_utc": datetime.now(timezone.utc).isoformat(),
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "tracked_diff_sha256": hashlib.sha256(command_output(["git", "diff", "HEAD"]).encode()).hexdigest(),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                              sorted((ROOT / "src").glob("*"))
                              if p.suffix in (".c", ".h", ".m", ".metal")},
            "engine_sha256": sha256(ROOT / "picoforge"), "metallib_sha256": sha256(ROOT / "picoforge.metallib"),
            "runner_sha256": sha256(Path(__file__)), "runner_source": Path(__file__).read_text(),
            "platform": platform.platform(),
            "hardware": command_output(["sysctl", "-n", "hw.model", "hw.memsize", "machdep.cpu.brand_string"]),
            "thermal_before": command_output(["pmset", "-g", "therm"]),
            "target": args.target.name, "target_files": fingerprint(args.target),
            "draft": args.draft.name, "draft_files": fingerprint(args.draft),
            "prompts_sha256": sha256(args.prompts), "prompts": prompts,
            "configs": args.configs, "reps": args.reps, "warmup": args.warmup,
            "max_new": args.max_new, "context": args.context, "seed": args.seed,
            "timing": "decode excludes both prefills; generation includes both; process includes startup and teardown",
            "memory": "macOS process ru_maxrss in bytes; may omit GPU-resident mapped weights and driver allocations; not total unified-memory use"}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    rng = random.Random(args.seed)
    measured = []
    with open(raw_path, "w") as raw:
        for item in prompts:
            prompt = CHAT.format(item["text"]) if item.get("chat", True) else item["text"]
            ref, _ = engine(args.target, ["--gpu-greedy", prompt, str(args.max_new)], args.context)
            if len(ref) < 2:
                raise RuntimeError(f"{item['id']}: fewer than two output tokens; decode throughput is undefined")
            for rep in range(-args.warmup, args.reps):
                order = args.configs.copy()
                rng.shuffle(order)
                for name in order:
                    ids, metrics = engine(args.target, configuration(name, args.draft, prompt, args.max_new), args.context)
                    if metrics is None:
                        raise RuntimeError("engine did not emit JSON metrics; rebuild it")
                    record = {"prompt": item["id"], "config": name, "rep": rep,
                              "phase": "warmup" if rep < 0 else "measure", "ids": ids,
                              "identical_to_greedy": ids == ref, **metrics}
                    raw.write(json.dumps(record, separators=(",", ":")) + "\n")
                    raw.flush()
                    if ids != ref:
                        raise RuntimeError(f"{item['id']} {name}: output differs from greedy; run saved in {raw_path}")
                    if rep >= 0:
                        measured.append(record)
                    print(f"{item['id']:16s} {name:12s} run {rep + 1:2d}  {metrics['tok_s']:7.2f} tok/s  "
                          f"request {metrics['process_s']:.3f}s", flush=True)
    rows = summarize(measured, args.configs)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                thermal_after=command_output(["pmset", "-g", "therm"]))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for name in args.configs:
        ratios = [r["speedup"] for r in rows if r["config"] == name]
        print(f"{name}: geomean {math.exp(statistics.mean(map(math.log, ratios))):.3f}x; worst {min(ratios):.3f}x")
    print(f"Saved {args.out}, {raw_path}, {meta_path}")


if __name__ == "__main__":
    main()
