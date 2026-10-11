"""test_routing_trace.py — the GPU's routing trace is what the router chose.

The routing-overlap measurement (how many distinct experts k consecutive tokens
touch) will be read off the engine's trace, so the trace is held to an
independent reference first: transformers on the tiny MoE, asked for its router
logits, top-k taken in PyTorch. Every token, every MoE layer, every one of the
k experts, in order of probability, must be the same id; dense layers must be
marked as such.

Run:  make test-routing-trace
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/tiny-qwen3-moe"
TEXT = ROOT / "build/trace_text.txt"
TRACE = ROOT / "build/trace_tiny.bin"


def read_trace(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    t, layers, k, experts = np.frombuffer(raw[:16], dtype=np.uint32)
    return np.frombuffer(raw[16:], dtype=np.uint16).reshape(t, layers, k), int(experts)


def check(model, tok, repeats: int, limit: int) -> bool:
    import torch
    text = ("The capital of France is Paris. The capital of Italy is Rome. The capital "
                    "of Spain is Madrid, and the capital of Portugal is Lisbon; all four are old "
                    "cities with long histories and busy ports, museums, and universities. ") * repeats
    TEXT.write_text(text)
    subprocess.run([str(ROOT / "picoforge"), str(MODEL), "--routing-trace", str(TEXT), str(limit),
                    str(TRACE)], check=True, capture_output=True, cwd=ROOT)
    trace, _ = read_trace(TRACE)
    ids = tok(text)["input_ids"][:limit]
    refs = []
    cache = None
    k = model.config.num_experts_per_tok
    # Skip the LM head and retain only top-k routes. Full-vocabulary logits
    # for 4,096 positions would consume gigabytes without testing anything.
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            out = model.model(torch.tensor([ids[start:start + 32]]), past_key_values=cache,
                              use_cache=True, output_router_logits=True)
            cache = out.past_key_values
            refs.append([torch.topk(torch.softmax(r.float(), dim=-1), k, dim=-1).indices.numpy()
                         for r in out.router_logits])

    ok = trace.shape[0] == len(ids)
    print(f"  tokens: engine {trace.shape[0]}, transformers {len(ids)}")
    moe_layers = [l for l, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")]
    for i, l in enumerate(moe_layers):
        ref = np.concatenate([r[i] for r in refs])
        same = np.array_equal(trace[:, l, :].astype(np.int64), ref)
        print(f"  layer {l} (MoE): {int((trace[:, l, :] == ref).all(-1).sum())}/{len(ids)} tokens "
              f"route identically  {'PASS' if same else 'FAIL'}")
        ok &= same
    for l in range(trace.shape[1]):
        if l not in moe_layers:
            dense = bool((trace[:, l, :] == 0xFFFF).all())
            print(f"  layer {l} (dense): marked dense  {'PASS' if dense else 'FAIL'}")
            ok &= dense
    return ok


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    ok = check(model, tok, 1, 256)
    ok &= check(model, tok, 100, 4096)
    # Refuse contexts beyond the GPU's threadgroup-memory capacity before dispatch.
    TEXT.write_text("hello " * 10000)
    p = subprocess.run([str(ROOT / "picoforge"), str(MODEL), "--routing-trace",
                        str(TEXT), "10000", str(TRACE)], capture_output=True, text=True, cwd=ROOT)
    refused = p.returncode != 0 and "threadgroup memory" in p.stderr
    print(f"  oversized context refused: {'PASS' if refused else 'FAIL'}")
    ok &= refused
    print("\n" + ("PASS: routing matches transformers through 4,096 tokens" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
