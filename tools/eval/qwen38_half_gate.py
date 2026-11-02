"""R3.5 gate of the reduced-precision prefill: one row per (half run, FP32 run) pair.

usage: qwen38_half_gate.py SUITE_JSON OUT_CSV LABEL:NP:HALF_DIR:FP32_DIR ...

Each directory holds logits.f32 of a teacher-forced run (rows = forced IDs + 1, the
first row the last prefill position) over the first NP IDs of the 8K prompt (NP 8063:
the 8K frontier and its forced IDs). Per pair: max |half - fp32| scaled by max |fp32|,
argmax agreement, KL(fp32 || half) per row (max, mean), mean NLL of the true next IDs
in both runs and its difference, and the FP32 top-1/top-2 logit margin of the rows
whose argmax differs (a small margin: an output near-tie, not a lost token).
"""
import csv
import json
import sys

import numpy as np


def log_softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main(suite, out, pairs):
    frontier = json.load(open(suite))['frontiers']['8k']
    prompt, forced = frontier['prompt_ids'], frontier['forced_ids']
    rows = []
    for spec in pairs:
        label, np_, half_dir, fp32_dir = spec.split(':')
        n = int(np_)
        h, f = (np.fromfile(f'{d}/logits.f32', '<f4') for d in (half_dir, fp32_dir))
        width = 248320 if h.size % 248320 == 0 else None
        if width is None or h.size != f.size:
            raise SystemExit(f'{label}: unexpected logits size {h.size} / {f.size}')
        h, f = h.reshape(-1, width), f.reshape(-1, width)
        nxt = (prompt + forced)[n:n + len(f)] if n < len(prompt) else forced[:len(f)]
        lh, lf = log_softmax(h), log_softmax(f)
        kl = (np.exp(lf) * (lf - lh)).sum(1)
        r = np.arange(len(nxt))
        nll_h, nll_f = -lh[r, nxt], -lf[r, nxt]
        top = np.sort(f, 1)
        flip = h.argmax(1) != f.argmax(1)
        rows.append(dict(label=label, prompt_ids=n, rows=len(f),
                         scaled=float(np.abs(h.astype(np.float64) - f).max() / max(1, np.abs(f).max())),
                         argmax_equal=int((~flip).sum()), kl_max=float(kl.max()), kl_mean=float(kl.mean()),
                         nll_fp32=float(nll_f.mean()), nll_half=float(nll_h.mean()),
                         nll_delta=float((nll_h - nll_f).mean()), positions=len(nxt),
                         flip_margins=' '.join(f'{m:.3f}' for m in (top[:, -1] - top[:, -2])[flip])))
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r['positions'] for r in rows)
    delta = sum(r['nll_delta'] * r['positions'] for r in rows) / total
    for r in rows:
        print(f"{r['label']:>8} scaled {r['scaled']:.3f} argmax {r['argmax_equal']}/{r['rows']} "
              f"KL max {r['kl_max']:.3f} mean {r['kl_mean']:.4f} dNLL {r['nll_delta']:+.4f}")
    print(f'positions {total}, weighted mean dNLL {delta:+.4f} nat')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
