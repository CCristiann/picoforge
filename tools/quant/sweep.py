"""sweep.py — what does each quantisation format cost in quality?

This decides the format before any kernel exists. Weights are quantised with
tools/quant/formats.py, dequantised back to fp32 and loaded into transformers
("fake quantisation"): the arithmetic is then exact fp32, so everything this
measures is the rounding of the weights and nothing else. Activation precision
(Q4 forces bf16 activations into the matmul) is a separate error, measured
later in the engine where it actually happens.

Per format, against the unquantised model on the same windows:
  ppl        exp(mean NLL) over the scored tokens
  dNLL       mean paired NLL difference, with a standard error taken across
             WINDOWS, not tokens: neighbouring tokens are correlated, and a
             per-token SE would claim more certainty than the data holds
  KL         mean KL(fp32 || quantised) per token, in nats
  top-1      how often both models pick the same next token

Windows are 1024 tokens, non-overlapping; the second half of each is scored
(llama.cpp's convention), so every scored token has >= 512 tokens of context.

Run:  tools/venv/bin/python tools/quant/sweep.py CORPUS OUT.csv [fmt ...]
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formats import dequantize, parse, quantize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/Qwen3-0.6B"
CTX = 1024
DEFAULT = ["q8_row", "q8_g32", "q4_row", "q4_rowcol", "q4_g128", "q4_g64", "q4_g32",
           "q8_g32+embed", "q4_g32+embed"]


def load(device: str):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return m.to(device).eval()


def apply_format(work, base, fmt: str) -> float:
    """Overwrite work's projection weights (and the tied embedding, with
    '+embed') with quantise->dequantise of base's. Returns bits per weight
    over the tensors touched, counting the scales."""
    name, _, extra = fmt.partition("+")
    bits = parse(name)[0]
    total_bits = total_w = 0
    for (pname, p), (_, pb) in zip(work.named_parameters(), base.named_parameters()):
        is_proj = pname.endswith("proj.weight")
        is_embed = pname.endswith("embed_tokens.weight")
        w = pb.detach().cpu().numpy()
        if is_proj or (is_embed and extra == "embed"):
            parts = quantize(w, name)
            p.data.copy_(torch.from_numpy(dequantize(parts)))
            total_bits += w.size * bits + 16 * (parts["d"].size + parts.get("c", np.empty(0)).size)
        else:
            p.data.copy_(pb.data)
            if is_proj or is_embed:
                total_bits += w.size * 16
        if is_proj or is_embed:
            total_w += w.size
    return total_bits / total_w


@torch.no_grad()
def score(work, base, windows: torch.Tensor):
    nll_q, nll_b, kls, agree, win_d = [], [], [], [], []
    # Logits leave the GPU before widening: MPS has no fp64, and KL summed in
    # fp32 once read NEGATIVE in this project (devlog, Phase 0).
    for w in windows:
        x = w[None]
        lq = torch.log_softmax(work(x).logits[0, CTX // 2 - 1:-1].cpu().double(), -1)
        lb = torch.log_softmax(base(x).logits[0, CTX // 2 - 1:-1].cpu().double(), -1)
        tgt = w[CTX // 2:].cpu()
        a = -lq.gather(-1, tgt[:, None])[:, 0]
        b = -lb.gather(-1, tgt[:, None])[:, 0]
        nll_q.append(a.cpu()); nll_b.append(b.cpu())
        kls.append((lb.exp() * (lb - lq)).sum(-1).cpu())
        agree.append((lq.argmax(-1) == lb.argmax(-1)).cpu())
        win_d.append(float((a - b).mean()))
    nq, nb = torch.cat(nll_q), torch.cat(nll_b)
    wd = np.array(win_d)
    return dict(ppl=float(nq.mean().exp()), ppl_fp32=float(nb.mean().exp()),
                dnll=float((nq - nb).mean()), dnll_se=float(wd.std(ddof=1) / np.sqrt(len(wd))),
                kl=float(torch.cat(kls).mean()), top1=float(torch.cat(agree).float().mean()),
                tokens=int(nq.numel()))


def main() -> None:
    corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
    fmts = sys.argv[3:] or DEFAULT
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = AutoTokenizer.from_pretrained(MODEL)(corpus.read_text())["input_ids"]
    n_win = len(ids) // CTX
    windows = torch.tensor(ids[:n_win * CTX]).view(n_win, CTX).to(device)
    print(f"{len(ids)} tokens, {n_win} windows of {CTX}, device {device}")

    base, work = load(device), load(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["format", "bpw", "ppl", "ppl_fp32", "dppl_pct", "dnll", "dnll_se",
                     "kl", "top1", "tokens"])
        for fmt in fmts:
            t0 = time.time()
            bpw = apply_format(work, base, fmt)
            r = score(work, base, windows)
            dppl = 100 * (r["ppl"] / r["ppl_fp32"] - 1)
            wr.writerow([fmt, f"{bpw:.3f}", f"{r['ppl']:.4f}", f"{r['ppl_fp32']:.4f}",
                         f"{dppl:.3f}", f"{r['dnll']:.5f}", f"{r['dnll_se']:.5f}",
                         f"{r['kl']:.5f}", f"{r['top1']:.4f}", r["tokens"]])
            f.flush()
            print(f"{fmt:14s} {bpw:5.2f} bpw  ppl {r['ppl']:.3f} (fp32 {r['ppl_fp32']:.3f}, "
                  f"{dppl:+.2f}%)  dNLL {r['dnll']:+.4f}±{r['dnll_se']:.4f}  "
                  f"KL {r['kl']:.4f}  top-1 {100 * r['top1']:.1f}%  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
