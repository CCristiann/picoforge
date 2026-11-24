"""layer_lab.py — what weight quantization does to one layer's routed experts, on real inputs.

Inputs: the layer's BF16 weights, the expert inputs captured by the engine
(QW4_ACT_LOG: x, chosen ids, gate weights g), and quantized tensors (run
directories from quantize_layer.py). Expert e computes
  E_e(x) = W_down (SiLU(W_gate x) * W_up x)
and the routed output of a token is y = sum_k g_k E_k(x). Reported per run:
  routed  sum_t |y_t - y^_t|^2  (and relative to sum |y_t|^2): the error the next layer sees
  gate/up sum over routed pairs g^2 |(W - W^) x|^2   one projection quantized, the others exact
  down    sum over routed pairs g^2 |(W - W^) a|^2   a = the exact SwiGLU output
A tensor missing from a run directory, or excluded with --only, stays BF16.

  layer_lab.py --source DIR --layer L --act LOG... --runs DIR... [--only {all,gate-up,down}] [--max-tokens N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qw4q  # noqa: E402
from act_log import layer_rows  # noqa: E402
from quantize_layer import text_config  # noqa: E402

PROJ = ("gate", "up", "down")


def bf16_rows(source: Path, layer: int, name: str, e: int) -> np.ndarray:
    """Exact weights of expert e for a GGUF tensor name (gate/up/down), as f32 [rows, logical cols]."""
    cfg = text_config(source)
    E, H, FF = cfg["experts"], cfg["hidden"], cfg["ff"]
    base = Path(source) / "tensors" / f"model.language_model.layers.{layer}.mlp.experts."
    if "ffn_down" in name:
        u = np.memmap(str(base) + "down_proj.bin", "<u2", "r", shape=(E, H, FF))[e]
    else:
        part = 0 if "ffn_gate" in name else 1
        u = np.memmap(str(base) + "gate_up_proj.bin", "<u2", "r", shape=(E, 2 * FF, H))[e, part * FF:(part + 1) * FF]
    return (np.asarray(u).astype("<u4") << 16).view("<f4")


def decode_expert(run: Path, name: str, e: int, rows: int, cols: int, qtype: int) -> np.ndarray:
    """Decoded weights of expert e from a run's tensor file, [rows, cols] (padding columns dropped)."""
    physical = -(-cols // 256) * 256
    per = rows * physical // 256 * (66 if qtype == qw4q.IQ2_XXS else 84)
    raw = np.memmap(Path(run) / f"{name}.bin", np.uint8, "r", offset=e * per, shape=(per,))
    return qw4q.dequantize(qtype, bytes(raw), rows, physical)[:, :cols]


def silu(v: np.ndarray) -> np.ndarray:
    return v / (1.0 + np.exp(-v))


def measure(source: Path, layer: int, x, ids, g, runs, only: str = "all", decode=None) -> list:
    decode = decode or decode_expert
    cfg = text_config(source)
    H, FF = cfg["hidden"], cfg["ff"]
    names = {p: f"blk.{layer}.ffn_{p}_exps.weight" for p in PROJ}
    shapes = {"gate": (FF, H, qw4q.IQ2_XXS), "up": (FF, H, qw4q.IQ2_XXS), "down": (H, FF, qw4q.Q2_K)}
    take = {"all": PROJ, "gate-up": ("gate", "up"), "down": ("down",)}[only]
    x = np.asarray(x, np.float64)
    y_ref = np.zeros((len(x), H))
    y_q = [np.zeros((len(x), H)) for _ in runs]
    proj = [dict.fromkeys(PROJ, 0.0) for _ in runs]
    for e in np.unique(ids):
        t, slot = np.nonzero(ids == e)
        xe, ge = x[t], np.asarray(g, np.float64)[t, slot][:, None]
        W = {p: bf16_rows(source, layer, names[p], int(e)).astype(np.float64) for p in PROJ}
        hg, hu = xe @ W["gate"].T, xe @ W["up"].T
        a = silu(hg) * hu
        np.add.at(y_ref, t, ge * (a @ W["down"].T))
        for i, run in enumerate(runs):
            Q = {}
            for p in PROJ:
                rows, cols, qtype = shapes[p]
                if p in take and (Path(run) / f"{names[p]}.bin").exists():
                    Q[p] = decode(run, names[p], int(e), rows, cols, qtype).astype(np.float64)
                else:
                    Q[p] = W[p]
            qg, qu = xe @ Q["gate"].T, xe @ Q["up"].T
            np.add.at(y_q[i], t, ge * ((silu(qg) * qu) @ Q["down"].T))
            proj[i]["gate"] += float((ge ** 2 * (hg - qg) ** 2).sum())
            proj[i]["up"] += float((ge ** 2 * (hu - qu) ** 2).sum())
            proj[i]["down"] += float((ge ** 2 * (a @ (W["down"] - Q["down"]).T) ** 2).sum())
    ref2 = float((y_ref ** 2).sum())
    out = []
    for i in range(len(runs)):
        routed = float(((y_ref - y_q[i]) ** 2).sum())
        out.append({"routed": routed, "routed_rel": routed / ref2 if ref2 else 0.0, **proj[i]})
    return out


def main(argv) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--act", type=Path, nargs="+", required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--only", choices=("all", "gate-up", "down"), default="all")
    p.add_argument("--max-tokens", type=int, default=0)
    a = p.parse_args(argv)
    parts = [layer_rows(f, a.layer) for f in a.act]
    x, ids, g = (np.concatenate([p[i] for p in parts]) for i in range(3))
    if a.max_tokens:
        x, ids, g = x[:a.max_tokens], ids[:a.max_tokens], g[:a.max_tokens]
    res = measure(a.source, a.layer, x, ids, g, a.runs, a.only)
    print(f"layer {a.layer}, {len(x)} captured tokens, tensors from runs: {a.only}; relative to {a.runs[0].name}")
    print(f"{'run':24s} {'routed |dy|^2/|y|^2':>20s} {'routed':>9s} {'gate':>9s} {'up':>9s} {'down':>9s}")
    for r, m in zip(a.runs, res):
        rel = lambda k: f"{m[k] / res[0][k] - 1:+8.2%}" if res[0][k] else "     n/a"
        print(f"{r.name:24s} {m['routed_rel']:20.4e} {rel('routed')} {rel('gate')} {rel('up')} {rel('down')}")


if __name__ == "__main__":
    main(sys.argv[1:])
