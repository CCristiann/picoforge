"""Exercise accepted/rejected routes, diagnostics and greedy parity on a mixed MoE."""

import json
import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "build/tiny-qwen3-moe"
DENSE = ROOT / "models/Qwen3-0.6B"
PROMPT = "The capital of France is"


def run(model, args, context=128, trace=False):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "ids.txt"
        diagnostic = Path(directory) / "trace.jsonl"
        cmd = [str(ROOT / "picoforge"), "--context", str(context), str(model), *args, str(out)]
        if trace:
            cmd.append(str(diagnostic))
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr
        lines = out.read_text().splitlines()
        ids = list(map(int, lines[0].split()))
        stats = next((json.loads(line[9:]) for line in p.stdout.splitlines()
                      if line.startswith("metrics: ")), None)
        records = [json.loads(line) for line in diagnostic.read_text().splitlines()] if trace else []
        return ids, stats, records


def main():
    config = json.loads((TARGET / "config.json").read_text())
    experts, topk = config["num_experts"], config["num_experts_per_tok"]
    ref, _, _ = run(TARGET, ["--gpu-greedy", PROMPT, "48"])
    saw_accepted = saw_rejected = False
    for draft in (TARGET, DENSE, None):
        label = draft.name if draft else "prompt lookup"
        command = ["--spec-model", str(draft)] if draft else ["--spec-greedy"]
        ids, metrics, trace = run(TARGET, [*command, PROMPT, "48", "auto-experts"], trace=True)
        assert ids == ref, f"greedy parity failed with {label}"
        assert len(trace) == metrics["passes"] == sum(metrics["k_hist"])
        assert metrics["expert_cost_s"] > 0
        assert metrics["decode_s"] >= metrics["expert_cost_s"]
        history = [[] for _ in range(config["num_hidden_layers"])]
        position = metrics["prompt_tokens"]
        for record in trace:
            n = record["k"] + 1
            assert record["position"] == position
            assert record["history_tokens"] == min(32, position - metrics["prompt_tokens"])
            actual, forecast = [], []
            for layer, routes in enumerate(record["routes"]):
                if layer in config["mlp_only_layers"]:
                    assert routes == []
                    continue
                assert len(routes) == n and all(len(set(row)) == topk for row in routes)
                assert all(0 <= e < experts for row in routes for e in row)
                actual.append(len(set(e for row in routes for e in row)))
                # Independent probability-of-absence calculation on the actual
                # accepted prefix: rejected suffixes must never enter this list.
                past = history[layer][-32:]
                probabilities = [(sum(e in row for row in past) + topk / experts) / (len(past) + 1)
                                 for e in range(experts)]
                forecast.append(sum(1 - (1 - p) ** n for p in probabilities))
                history[layer].extend(routes[:record["accepted"] + 1])
            assert math.isclose(record["actual_experts"], sum(actual) / len(actual), abs_tol=1e-8)
            assert math.isclose(record["predicted_experts"], sum(forecast) / len(forecast), abs_tol=1e-8)
            assert record["verify_s"] > 0 and record["predictor_s"] >= 0
            position += record["accepted"] + 1
            saw_accepted |= record["accepted"] > 0
            saw_rejected |= record["accepted"] < record["k"]
        assert metrics["accepted"] == sum(r["accepted"] for r in trace)
        print(f"PASS: {label} drafting, greedy parity and {len(trace)} audited passes")
    assert saw_accepted and saw_rejected

    ids, _, full_trace = run(TARGET, ["--spec-model", str(TARGET), PROMPT, "48", "31"], trace=True)
    assert ids == ref and full_trace[0]["k"] == 31
    assert full_trace[0]["accepted"] == 31
    assert full_trace[1]["history_tokens"] == 32
    print("PASS: full 32-row verify trace and history window")

    for room in (1, 2, 9, 40):
        context = metrics["prompt_tokens"] + room
        expected, _, _ = run(TARGET, ["--gpu-greedy", PROMPT, str(context)], context)
        for limit in ("auto-experts:1", "auto-experts:31"):
            ids, _, _ = run(TARGET, ["--spec-model", str(TARGET), PROMPT, str(context), limit], context)
            assert ids == expected, (room, limit)
    print("PASS: expert mode at context boundaries and draft limits 1 and 31")

    with tempfile.TemporaryDirectory() as directory:
        out, trace = Path(directory) / "ids.txt", Path(directory) / "trace.jsonl"
        trace.write_text("keep this measurement\n")
        cmd = [str(ROOT / "picoforge"), str(TARGET), "--spec-greedy", PROMPT, "4", "auto-experts", str(out), str(trace)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert p.returncode != 0 and "must not exist" in p.stderr
        assert trace.read_text() == "keep this measurement\n"
        aliased = Path(directory) / "alias.jsonl"
        p = subprocess.run([str(ROOT / "picoforge"), str(TARGET), "--spec-greedy", PROMPT,
                            "4", "auto-experts", str(aliased), directory + "/./alias.jsonl"],
                           cwd=ROOT, capture_output=True, text=True)
        assert p.returncode != 0 and "different paths" in p.stderr
        assert json.loads(aliased.read_text().splitlines()[0])["pass"] == 1
        for length in ("auto-experts:0", "auto-experts:32", "auto-experts:bad"):
            p = subprocess.run([str(ROOT / "picoforge"), str(TARGET), "--spec-greedy", PROMPT, "4", length, str(out)],
                               cwd=ROOT, capture_output=True, text=True)
            assert p.returncode != 0 and "automatic draft limit" in p.stderr
        p = subprocess.run([str(ROOT / "picoforge"), str(DENSE), "--spec-greedy", PROMPT, "4", "auto-experts", str(out)],
                           cwd=ROOT, capture_output=True, text=True)
        assert p.returncode != 0 and "MoE target" in p.stderr
    print("PASS: invalid expert mode arguments and diagnostic overwrite protection")


if __name__ == "__main__":
    main()
