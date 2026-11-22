# Quant M2 — captured expert inputs and the layer lab (2026-11-22)

## Capture

`QW4_ACT_LOG` (engine, opt-in) recorded the routed experts' input x, the chosen
ids and the gate weights of layers 4, 28 and 44 on the four R7 code windows
(`build/r7/corpus.json`, 4 x 2048 tokens of picoforge code added after the
model shipped, so unseen by it). Q4 file (closest to BF16 that runs here),
teacher-forced, one guarded run per window (GPU lock, launcher with release
wait, AC power, Mac awake): 132 / 113 / 117 / 121 s, 63.5 MB per window.
Run conditions: `capture-code-*-conditions.json`.

## O1 on real inputs

`tools/quant/layer_lab.py`: routed output of a layer, y = sum_k g_k E_k(x), with
gate/up from each run and down exact (BF16), 8192 captured tokens per layer.
Raw output: `layer_lab_o1.txt`. Error relative to compat (lower is better):

| Layer | Proxy (imatrix-weighted) | gate proj. | up proj. | routed output |
|---:|---:|---:|---:|---:|
| 4 | −7.89% | −3.17% | −4.32% | **+23.29%** |
| 28 | −6.95% | −2.45% | −4.79% | −4.91% |
| 44 | −7.23% | −3.84% | −5.09% | −5.95% |

Relative routed error |dy|^2/|y|^2 of compat: 0.053 (layer 4), 0.137 (28), 0.153 (44).

### Why layer 4 gets worse: shrinkage

A least-squares fit to a few levels is a projection, and a projection is
shorter than the vector: <q,w> = |q|^2 < |w|^2. The compat encoder models the
IQ2_XXS magnitudes as 1, 3, 5 while the decoder uses 1, 3.125, 5.375, so its
weights come out a few percent larger and the mismatch partly cancels the
shrinkage. O1 removes the mismatch and keeps the shrinkage; gate and up
multiply, so the routed output shrinks twice:

| Layer 4 | <q,w>/|w|^2 (gate+up) | gate pre-activation gain | output gain <y^,y>/|y|^2 |
|---|---:|---:|---:|
| compat | 0.974 | 0.992 | 0.951 |
| o1 | 0.916 | 0.936 | 0.843 |

(layer 28: compat 0.975 / 0.985 / 0.957, o1 0.918 / 0.929 / 0.856.)

Scaling decoded gate and up by a constant c (routed error relative to compat at c = 1):

| Layer | run | c=1.00 | 1.02 | 1.04 | 1.06 | 1.08 | 1.10 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | compat | 0 | +2.9% | +12.0% | +27.7% | +50.4% | +80.6% |
| 4 | o1 | +23.3% | +10.7% | +2.7% | −0.4% | +1.7% | +9.3% |
| 28 | compat | 0 | +6.3% | +14.9% | +26.1% | +39.9% | +56.4% |
| 28 | o1 | −4.9% | −4.9% | −3.2% | +0.5% | +6.3% | +14.1% |
| 44 | compat | 0 | +5.9% | +14.0% | +24.2% | +36.8% | +51.8% |
| 44 | o1 | −6.0% | −5.6% | −3.6% | −0.0% | +5.3% | +12.5% |

The worst 1% of tokens carry only 4-6% of the error at layer 4, so it is not a
few outliers.

## Conclusion

O1 is not a robust improvement: −5% to −6% on two layers, +23% on layer 4,
and a gain correction only brings layer 4 back to parity. The per-weight
objective (diagonal imatrix weights) does not predict the output error well:
it ignores the correlation between input columns, the shrinkage, and how much
each intermediate neuron matters after SiLU(gate)·up and down. O1 stays
opt-in and is not used for a published file. The next encoder objective must
be output-aware, measured with this lab. Analysis scripts (private):
`notes/quant/{shrink,gain,l4_dbg}.py`.
