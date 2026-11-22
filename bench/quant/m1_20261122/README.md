# Quant M1 — per-layer expert quantization, layer 28 (2026-11-22)

CPU only (M5 Pro, 12 threads), no model run. Sources:
`Qwen/Qwen3.8-Flash-Next` BF16 at `de4b8e4d43b917e7706784d8bb445c9af86a3540`
(single tensors fetched with HTTP ranges), Unsloth imatrix
`unsloth/Qwen3.8-Flash-Next-GGUF` at `c8b5954` (sha256 `a5863123…`), template
`antirez/qwen3.8-flash-next-gguf` Q2 at `d600fe1` (`docs/weights.lock`).

## Compat: byte identity with the published Q2

```
tools/venv/bin/python -I tools/quant/quantize_layer.py --source models/Qwen3.8-Flash-Next-BF16 \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --imatrix models/unsloth-imatrix/imatrix_unsloth.gguf_file \
  --layer 28 --projection {gate-up,down} --mode compat --out build/quant/m1-l28-compat \
  --check-template models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
```

| Tensor | Format | Byte-identical experts | Wall time |
|---|---|---:|---:|
| blk.28.ffn_gate_exps | IQ2_XXS | 512/512 | 28.6 s (gate+up) |
| blk.28.ffn_up_exps | IQ2_XXS | 512/512 | |
| blk.28.ffn_down_exps | Q2_K, 640 → 768 | 512/512 | 44.6 s (incl. 1.68 GB fetch) |

A first run with picoforge's own rewrite of the IQ2_XXS compat encoder gave
507/512 and 508/512, although it matched all 64 synthetic golden rows: clang's
may-fuse FMAs are decided per call site after inlining, so only DwarfStar's
unchanged translation unit reproduces the bytes (`tools/quant/third_party/README.md`).

## O1: refinement against the decoder's real levels

The compat search models the IQ2_XXS magnitudes as 1, 3, 5; the decoder's grid
bytes 8, 25, 43 mean 1, 3.125, 5.375. O1 re-searches patterns, signs and
scales against the real levels and keeps a block only if its error goes down.

```
tools/venv/bin/python -I tools/quant/quantize_layer.py ... --layer 28 --projection gate-up --mode o1 --out build/quant/m1-l28-o1
tools/venv/bin/python -I tools/quant/measure_layer.py --source models/Qwen3.8-Flash-Next-BF16 \
  --imatrix models/unsloth-imatrix/imatrix_unsloth.gguf_file --layer 28 --runs build/quant/m1-l28-compat build/quant/m1-l28-o1
```

All 512 experts, gate + up (1024 matrices), error relative to compat:

| Run | imatrix-weighted | encoder-weighted | unweighted | median per expert |
|---|---:|---:|---:|---:|
| compat | 0 | 0 | 0 | 0 |
| o1 | −6.95% | −6.60% | −3.53% | −6.90% |

Wall time of the O1 encode: 132.8 s for gate + up (12 threads).

Limits: the imatrix-weighted error is a proxy of the output error (it ignores
correlations between input columns and the router's gate weights); one layer;
IQ2_XXS only. The real test is the per-layer lab with captured activations
(M2) and R7 on a full build (M4).
