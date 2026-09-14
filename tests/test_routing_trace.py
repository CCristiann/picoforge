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


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TEXT.write_text("The capital of France is Paris. The capital of Italy is Rome. The capital "
                    "of Spain is Madrid, and the capital of Portugal is Lisbon; all four are old "
                    "cities with long histories and busy ports, museums, and universities.")
    subprocess.run([str(ROOT / "picoforge"), str(MODEL), "--routing-trace", str(TEXT), "256",
                    str(TRACE)], check=True, capture_output=True, cwd=ROOT)
    trace, _ = read_trace(TRACE)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(TEXT.read_text())["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_router_logits=True)
    k = model.config.num_experts_per_tok

    ok = trace.shape[0] == len(ids)
    print(f"  tokens: engine {trace.shape[0]}, transformers {len(ids)}")
    moe_layers = [l for l, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")]
    for i, l in enumerate(moe_layers):
        probs = torch.softmax(out.router_logits[i].float(), dim=-1)
        ref = torch.topk(probs, k, dim=-1).indices.numpy()
        same = np.array_equal(trace[:, l, :].astype(np.int64), ref)
        print(f"  layer {l} (MoE): {int((trace[:, l, :] == ref).all(-1).sum())}/{len(ids)} tokens "
              f"route identically  {'PASS' if same else 'FAIL'}")
        ok &= same
    for l in range(trace.shape[1]):
        if l not in moe_layers:
            dense = bool((trace[:, l, :] == 0xFFFF).all())
            print(f"  layer {l} (dense): marked dense  {'PASS' if dense else 'FAIL'}")
            ok &= dense
    print("\n" + ("PASS: the GPU routing trace is the router's choice" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
