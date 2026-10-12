"""Render a self-contained, offline report from the recorded Qwen3.8 pilot."""
import argparse
from html import escape
import json
import os
from pathlib import Path
from string import Template
from urllib.parse import quote


TITLES = {
    "two-file-fix": ("Bug fix su due file", "Python"),
    "archive-correction": ("Correzione in un archivio lungo", "Contesto lungo"),
    "lower-bound": ("Ricerca binaria", "C11"),
    "merge-ranges": ("Unione di intervalli", "Python"),
    "crt-italian": ("Congruenze in italiano", "Ragionamento"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(baseline, storage, output):
    manifest = read_json(baseline / "manifest.json")
    summary = read_json(baseline / "summary.json")
    grading = read_json(baseline / "grading.json")
    native = read_json(storage / "manifest.json")
    process = read_json(storage / native["real_gpu_probe"] / "process.json")
    probe = read_json_line(storage / native["real_gpu_probe"] / "stdout.log")
    if manifest["status"] != "pilot_complete" or not process["complete"] or process["returncode"]:
        raise ValueError("report requires completed pilot and storage checks")
    if not probe["byte_exact"] or not probe["survived_owner_close"]:
        raise ValueError("native byte/lifetime checks failed")
    if native["full_project_build_exit"] or native["full_project_test_all_exit"]:
        raise ValueError("report requires a passing build and regression suite")
    grades = {row["case"]: row for row in grading["results"]}
    if set(grades) != {row["case"] for row in summary}:
        raise ValueError("timing and grading cases differ")
    if not all(row["greedy_repeat_parity"] for row in summary):
        raise ValueError("greedy repeat parity failed")

    def link(path):
        return escape(quote(os.path.relpath(path, output.parent)), quote=True)

    rows, answers = [], []
    for row in summary:
        case = row["case"]
        title, kind = TITLES[case]
        grade = grades[case]
        passed = grade["tests_passed"] and grade["format_followed"]
        status = "Superato" if passed else "Formato da correggere"
        color = "pass" if passed else "review"
        rows.append(f'''<tr><th scope="row"><a href="#answer-{case}">{title}</a><small>{kind}</small></th>
          <td>{row['prompt_tokens']:,}</td><td>{row['generated']:,}</td>
          <td class="rate"><span>{row['decode_tok_s']:.2f}</span><div class="spark" aria-hidden="true"><i style="width:{row['decode_tok_s'] / 40 * 100:.2f}%"></i></div></td>
          <td>{row['ttft_session_s']:.3f} s</td><td>{row['request_s']:.3f} s</td>
          <td><span class="{color}">{status}</span></td></tr>''')
        response = escape((baseline / "pilot" / case / "answer.0.txt").read_text(encoding="utf-8"))
        tests = "Superati" if grade["tests_passed"] else "Non superati (JSON con delimitatori Markdown)"
        fmt = "Rispettato" if grade["format_followed"] else "Non rispettato: delimitatori Markdown aggiunti"
        answers.append(f'''<details id="answer-{case}"><summary>{title}<span class="{color}">{status}</span></summary>
          <div class="answer"><p>Test congelati: <strong>{tests}</strong>. Formato: {fmt}.</p>
          <p>Decode p10–p90: {row['decode_p10']:.2f}–{row['decode_p90']:.2f} token/s.
          Prefill mediano: {row['prefill_s']:.3f} s. Sequenze greedy identiche nelle quattro esecuzioni.</p>
          <p><a href="{link(baseline / 'suite' / (case + '.txt'))}">Apri il prompt</a> ·
          <a href="{link(baseline / 'pilot' / case / 'answer.0.txt')}">Apri la risposta grezza</a></p>
          <pre tabindex="0" aria-label="Risposta grezza: {title}">{response}</pre></div></details>''')

    local, hardware = manifest["local_results"], manifest["hardware"]
    peak = local["peak_sampled_metal_allocated_bytes"] / 2**30
    physical = hardware["physical_memory_bytes"] / 2**30
    recommended = hardware["recommended_working_set_bytes"] / 2**30
    values = dict(
        rows="\n".join(rows), answers="\n".join(answers), peak=f"{peak:.2f}", physical=f"{physical:g}",
        recommended=f"{recommended:.2f}", peak_percent=f"{100 * peak / physical:.3f}",
        recommended_percent=f"{100 * recommended / physical:.3f}",
        low=f"{min(r['decode_tok_s'] for r in summary):.2f}",
        high=f"{max(r['decode_tok_s'] for r in summary):.2f}",
        prefill=f"{local['stock_8063_prefill_tok_s']:.2f}",
        strict=grading["strict_pass_count"], tasks=grading["independent_task_count"],
        coding=local["executable_code_tasks_passed"], swap=f"{local['global_swap_growth_across_processes_bytes'] / 2**20:.1f}",
        weights=f"{probe['weights']:,}", views=probe["views"], probes=f"{probe['byte_probes']:,}",
        date=escape(manifest["updated_utc"][:10]), revision=escape(manifest["dwarfstar"]["revision"]),
        checksum=escape(manifest["checkpoint"]["verified_sha256"]),
        summary_link=link(baseline / "summary.csv"), grading_link=link(baseline / "grading.json"),
        manifest_link=link(baseline / "manifest.json"), native_link=link(storage / "manifest.json"),
    )
    template = Template(Path(__file__).with_name("qwen38_report.html").read_text(encoding="utf-8"))
    output.write_text(template.substitute(values), encoding="utf-8")


def read_json_line(path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("storage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.baseline, args.storage, args.output)
