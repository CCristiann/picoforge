"""test_gpu_vs_cpu.py — the GPU pass judged by the CPU pass, on any checkpoint.

The chain of oracles for a model too large for the NumPy oracle to hold in its
quantised-and-dequantised form: the C CPU path (itself held to the oracle on
the bf16 checkpoint) is the reference for the GPU path on the same file. Both
read the same bytes, so the budgets are test_forward.py's strict ones -- the
q8_row GPU kernels already meet them against the oracle on the tiny models.

    ./tools/venv/bin/python tests/test_gpu_vs_cpu.py models/Qwen3-30B-A3B-q8_row
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
from qwen3_forward import Qwen3Config  # noqa: E402
from test_forward import LONG_PROMPT, PROMPTS, c_logits, compare  # noqa: E402


def main() -> None:
    from transformers import AutoTokenizer

    model_dir = Path(sys.argv[1])
    cfg = Qwen3Config.from_json(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    ok = True
    for prompt in PROMPTS + ([LONG_PROMPT] if cfg.num_experts else []):
        ids = tok(prompt)["input_ids"]
        print(f"\n=== {len(ids)} tokens: {prompt[:50]!r} ===")
        ref = c_logits(model_dir, ids, cfg.vocab_size, "--forward")
        for mode, label in (("--gpu-forward", "GPU batch"), ("--gpu-forward-incr", "GPU incremental")):
            ok &= compare(c_logits(model_dir, ids, cfg.vocab_size, mode), ref, f"{label} vs CPU batch")
    print("\n" + ("GPU matches the CPU on this checkpoint" if ok else "GPU DIVERGES FROM THE CPU"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
