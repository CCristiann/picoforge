"""Freeze prompts and settings before any model output is observed."""
import argparse
import hashlib
import json
from pathlib import Path
import random

SOURCE = Path(__file__).with_name("qwen38_cases.json")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    suite = json.loads(SOURCE.read_text())
    # Long-context correction: all facts are in the supplied text, with a later
    # replacement of the first value. Distractors are deterministic, not facts.
    lines = ["Archivio sintetico. Leggi fino alla fine; una rettifica sostituisce il valore precedente."]
    for i in range(300):
        lines.append(f"Riga {i:04d}: lotto L{i:04d}, stato registrato, reparto R{i % 17:02d}.")
        if i == 12:
            lines.append("FATTO: il codice del progetto Iris è ROMA47.")
        if i == 139:
            lines.append("FATTO: la chiave del progetto Loto è VERDE82.")
        if i == 286:
            lines.append("RETTIFICA: il codice attuale del progetto Iris è BARI29, che sostituisce ROMA47.")
    lines.append('Rispondi solo con JSON: {"Iris":"codice attuale", "Loto":"chiave"}.')
    suite["cases"].append(dict(id="archive-correction", kind="json", mode="nothink",
                               max_new=192, prompt="\n".join(lines),
                               expected={"Iris": "BARI29", "Loto": "VERDE82"},
                               grading="Exact JSON after a long prompt and a correction; actual token count recorded by the runtime."))
    for case in suite["cases"]:
        prompt = case.pop("prompt")
        case["prompt_file"] = case["id"] + ".txt"
        data = prompt.encode("utf-8")
        (output / case["prompt_file"]).write_bytes(data)
        case["prompt_sha256"] = hashlib.sha256(data).hexdigest()
    suite.update(seed=3811, context_tokens=8192, prefill_chunk_tokens=1024, mtp=False,
                 temperature=0, system_message="", first_use_repetitions=1,
                 measured_repetitions=3, performance_decisions=False,
                 source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    order = [case["id"] for case in suite["cases"]]
    random.Random(suite["seed"]).shuffle(order)
    suite["process_order"] = order
    (output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
