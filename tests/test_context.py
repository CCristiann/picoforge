"""Check CLI bounds and greedy parity when generation reaches the cache limit."""

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/tiny-qwen3-moe"


def run(context, args):
    with tempfile.NamedTemporaryFile() as out:
        p = subprocess.run([str(ROOT / "picoforge"), "--context", str(context),
                            str(MODEL), *args, out.name], cwd=ROOT,
                           capture_output=True, text=True)
        lines = Path(out.name).read_text().splitlines()
        ids = list(map(int, lines[0].split())) if lines else []
        metrics = next((json.loads(l[9:]) for l in p.stdout.splitlines()
                        if l.startswith("metrics: ")), None)
        return p, ids, metrics


def main():
    prompt = "The capital of France is"
    p, _, metrics = run(64, ["--spec-greedy", prompt, "1", "0"])
    assert p.returncode == 0, p.stderr
    prompt_tokens = metrics["prompt_tokens"]
    for room in (1, 2, 7):
        context = prompt_tokens + room
        p, ref, _ = run(context, ["--gpu-greedy", prompt, str(context)])
        assert p.returncode == 0 and len(ref) == room, p.stderr
        for draft in ("0", "2", "auto:31"):
            p, ids, stats = run(context, ["--spec-greedy", prompt, str(context), draft])
            assert p.returncode == 0, p.stderr
            assert ids == ref, (context, draft, ids, ref)
            assert stats["generated"] == room
            assert stats["prefill_s"] > 0 and stats["decode_s"] >= 0
            assert stats["peak_rss_bytes"] > 0
            assert sum(stats["k_hist"]) == stats["passes"]
    for context, prompt_text, count, draft, message in (
        ("1", prompt, "1", "0", "context"),
        ("abc", prompt, "1", "0", "context"),
        (64, prompt, "0", "0", "max new tokens"),
        (64, prompt, "1", "-1", "draft length"),
        (64, prompt, "1", "auto:32", "automatic draft limit"),
        (64, prompt, "1", "auto:0", "automatic draft limit"),
        (64, prompt, "1", "2junk", "draft length"),
        (64, "", "1", "0", "prompt"),
    ):
        p, _, _ = run(context, ["--spec-greedy", prompt_text, count, draft])
        assert p.returncode != 0 and message in p.stderr, (context, draft, p.stderr)
    print("PASS: context boundary parity, metrics and invalid arguments")


if __name__ == "__main__":
    main()
