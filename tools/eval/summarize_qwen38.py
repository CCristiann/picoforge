"""Validate all pilot repetitions before writing a timing summary."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from bench_speculate import percentile


def case_summary(directory, repeats, warmups):
    process = json.loads((directory / "process.json").read_text())
    if not process["complete"] or process["returncode"] != 0:
        raise ValueError(f"incomplete process: {directory.name}")
    rows = [json.loads(line) for line in (directory / "stdout.log").read_text().splitlines()]
    if [r["repeat"] for r in rows] != list(range(warmups + repeats)):
        raise ValueError("missing, duplicated, or reordered repetitions")
    reference = None
    for row in rows:
        timers = [row[k] for k in ("load_s", "session_s", "warmup_s", "prefill_s", "decode_s", "request_s")]
        if not all(math.isfinite(t) and t >= 0 for t in timers):
            raise ValueError("invalid timer")
        if sum(timers[1:5]) > row["request_s"] + .001:
            raise ValueError("phases exceed request timer")
        ids = [int(n) for n in (directory / f"answer.{row['repeat']}.ids").read_text().split()]
        if any(n < 0 for n in ids):
            raise ValueError("negative token ID")
        if len(ids) != row["generated"] or not 0 <= row["decode_evals"] <= row["generated"]:
            raise ValueError("inconsistent token counts")
        if reference is not None and ids != reference:
            raise ValueError("ordinary greedy repetitions diverged")
        reference = ids
        row["decode_tok_s"] = max(0, row["generated"] - 1) / row["decode_s"] if row["decode_s"] else 0
        if row["generated"] and not 0 <= row["ttft_session_s"] <= row["request_s"]:
            raise ValueError("invalid first-token time")
    if rows[0]["load_s"] + sum(r["request_s"] for r in rows) > process["process_s"] + .1:
        raise ValueError("internal phases exceed enclosing process timer")
    memory = [json.loads(line) for line in (directory / "memory.jsonl").read_text().splitlines()]
    if not memory:
        raise ValueError("no physical-memory samples")
    before, after = process["system_before"], process["system_after"]
    if after is None:
        raise ValueError("missing final system counters")
    stderr = (directory / "stderr.log").read_text()
    metal = [int(n) for n in re.findall(r"^metal_allocated_bytes: (\d+)$", stderr, re.M)]
    io = [json.loads(line.removeprefix("ngram_io: ")) for line in stderr.splitlines()
          if line.startswith("ngram_io: ")]
    if len(metal) < len(rows) or len(io) != 3 * len(rows):
        raise ValueError("missing allocation or n-gram instrumentation")
    if [r["phase"] for r in io] != ["before_prefill", "after_prefill", "after_decode"] * len(rows):
        raise ValueError("unexpected n-gram phase order")
    for previous, current in zip(io, io[1:]):
        if any(current[k] < previous[k] for k in ("calls", "bytes", "summed_service_ns", "errors")):
            raise ValueError("n-gram counters decreased")
    if io[-1]["errors"]:
        raise ValueError("n-gram reads failed")
    selected = rows[warmups:]
    rates = [r["decode_tok_s"] for r in selected]
    first_request_end = rows[0]["decode_start_monotonic_s"] + rows[0]["decode_s"]
    first_token = rows[0]["engine_start_monotonic_s"] + rows[0]["first_token_from_start_s"]
    result = dict(case=directory.name, prompt_tokens=rows[0]["prompt_tokens"],
                  generated=rows[0]["generated"], repeats=repeats, greedy_repeat_parity=True,
                  load_s=rows[0]["load_s"], first_use_prefill_s=rows[0]["prefill_s"],
                  prefill_s=statistics.median(r["prefill_s"] for r in selected),
                  decode_tok_s=statistics.median(rates), decode_p10=percentile(rates, .1),
                  decode_p90=percentile(rates, .9),
                  ttft_session_s=statistics.median(r["ttft_session_s"] for r in selected),
                  request_s=statistics.median(r["request_s"] for r in selected),
                  first_request_to_output_s=first_request_end - process["start_monotonic_s"],
                  first_request_ttft_s=first_token - process["start_monotonic_s"] if rows[0]["generated"] else None,
                  process_s=process["process_s"],
                  sampled_peak_footprint_bytes=max(r["phys_footprint_bytes"] for r in memory),
                  observed_lifetime_peak_footprint_bytes=max(r["lifetime_peak_phys_footprint_bytes"] for r in memory),
                  sampled_peak_metal_allocated_bytes=max(metal),
                  system_swap_delta_bytes=after["system_swap_used_bytes"] - before["system_swap_used_bytes"],
                  system_swapouts_delta_pages=after["system_swapouts_pages"] - before["system_swapouts_pages"],
                  system_peak_pressure=max(r["system_pressure_level"] for r in memory),
                  ngram_read_calls=io[-1]["calls"] - io[0]["calls"],
                  ngram_read_bytes=io[-1]["bytes"] - io[0]["bytes"],
                  ngram_summed_service_s=(io[-1]["summed_service_ns"] - io[0]["summed_service_ns"]) / 1e9)
    return result


def summarize(pilot, output):
    plan = json.loads((pilot / "pilot.json").read_text())
    if not plan["complete"] or plan["finished"] != plan["case_order"]:
        raise ValueError("pilot incomplete")
    summaries = [case_summary(pilot / name, plan["measured_repeats"], plan["warmup_repeats"])
                 for name in plan["case_order"]]
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.pilot, args.output), indent=2))
