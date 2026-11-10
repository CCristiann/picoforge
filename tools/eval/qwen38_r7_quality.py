"""qwen38_r7_quality.py — how much Q2 loses against Q4 on the R7 corpus.

Both runs are teacher-forced on the same window: row i of a logits file is the
distribution after tokens[0..i], so it predicts tokens[i+1]. With P = softmax
of the Q4 row (the reference: more bits, closer to the original weights) and
Q = softmax of the Q2 row, per position:

  KL(P||Q) = sum_v P_v (log P_v - log Q_v)   information lost using Q2 for Q4
  top-1    = argmax P == argmax Q            what greedy decoding sees
  NLL      = -log P(true next token)         per model; PPL = exp(mean NLL)

KL is 0 only if the distributions are equal. NLL tells if Q2 is WORSE, not
only different: a near-tie can flip the argmax at almost no NLL cost. The Q4
top-1/top-2 log-prob margin is kept so flips can be checked against
near-ties. Q4 is a proxy: the BF16 original does not fit in 64 GiB.

  reduce CORPUS.json STREAM WINDOW Q4.f32 Q2.f32 OUT.npz   one window, FP64
  report CORPUS.json DIR [OUT.json]                       aggregate DIR/*.npz
"""

import json
import sys
from pathlib import Path

import numpy as np

CHUNK = 128          # rows per FP64 block: 128 x 248k x 8 B = 254 MB


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def reduce(corpus: str, stream: str, window: str, q4: str, q2: str, out: str) -> None:
    ids = np.array(json.loads(Path(corpus).read_text())["streams"][stream]["windows"][int(window)]["ids"])
    n = len(ids) - 1                                     # rows with a next token
    a, b = (np.memmap(p, "<f4", "r") for p in (q4, q2))
    assert a.size == b.size and a.size % len(ids) == 0, "logits files do not match the window"
    vocab = a.size // len(ids)
    a, b = a.reshape(len(ids), vocab), b.reshape(len(ids), vocab)
    cols = {k: np.empty(n) for k in ("kl", "nll4", "nll2", "margin4")}
    cols.update(am4=np.empty(n, np.int64), am2=np.empty(n, np.int64))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p, q = log_softmax(a[s:e].astype(np.float64)), log_softmax(b[s:e].astype(np.float64))
        assert np.isfinite(p).all() and np.isfinite(q).all(), f"non-finite logits in rows {s}..{e}"
        rows, nxt = np.arange(e - s), ids[s + 1:e + 1]
        cols["kl"][s:e] = (np.exp(p) * (p - q)).sum(axis=1)
        cols["nll4"][s:e], cols["nll2"][s:e] = -p[rows, nxt], -q[rows, nxt]
        cols["am4"][s:e], cols["am2"][s:e] = p.argmax(axis=1), q.argmax(axis=1)
        top2 = np.partition(p, -2, axis=1)[:, -2:]
        cols["margin4"][s:e] = top2[:, 1] - top2[:, 0]
    np.savez(out, vocab=vocab, **cols)
    print(f"{stream}[{window}]: {n} rows, vocab {vocab}, KL mean {cols['kl'].mean():.4f}, "
          f"top-1 {np.mean(cols['am4'] == cols['am2']):.4f}")


def block_ci(x: np.ndarray, block: int = 128, draws: int = 2000) -> list[float]:
    """95% CI of the mean, resampling 128-position blocks: neighbours are correlated."""
    blocks = np.array([x[i:i + block].mean() for i in range(0, len(x) - block + 1, block)])
    rng = np.random.default_rng(0)
    means = blocks[rng.integers(0, len(blocks), (draws, len(blocks)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(d: dict) -> dict:
    kl, agree = d["kl"], (d["am4"] == d["am2"]).astype(np.float64)
    flips = ~agree.astype(bool)
    return {"positions": int(kl.size),
            "kl_mean": float(kl.mean()), "kl_mean_ci95": block_ci(kl),
            **{f"kl_p{q}": float(np.percentile(kl, q)) for q in (50, 90, 99)}, "kl_max": float(kl.max()),
            "top1": float(agree.mean()), "top1_ci95": block_ci(agree),
            "ppl_q4": float(np.exp(d["nll4"].mean())), "ppl_q2": float(np.exp(d["nll2"].mean())),
            "dnll_mean": float((d["nll2"] - d["nll4"]).mean()),
            "dnll_ci95": block_ci(d["nll2"] - d["nll4"]),
            "flips_margin4_median": float(np.median(d["margin4"][flips])) if flips.any() else None,
            "flips_margin4_lt_0p5": float(np.mean(d["margin4"][flips] < 0.5)) if flips.any() else None}


def report(corpus: str, folder: str, out: str | None = None) -> None:
    c = json.loads(Path(corpus).read_text())
    keys = ("kl", "nll4", "nll2", "margin4", "am4", "am2")
    parts, res = {}, {"corpus_sha256": c["sha256"], "commit": c["commit"]}
    for stream, s in c["streams"].items():
        files = [Path(folder) / f"{stream}-{i}.npz" for i in range(len(s["windows"]))]
        missing = [f.name for f in files if not f.exists()]
        assert not missing, f"missing windows: {missing}"
        loaded = [np.load(f) for f in files]
        parts[stream] = {k: np.concatenate([z[k] for z in loaded]) for k in keys}
        res[stream] = summarize(parts[stream])
    res["all"] = summarize({k: np.concatenate([p[k] for p in parts.values()]) for k in keys})
    text = json.dumps(res, indent=1)
    print(text)
    if out:
        Path(out).write_text(text + "\n")


if __name__ == "__main__":
    {"reduce": reduce, "report": report}[sys.argv[1]](*sys.argv[2:])
