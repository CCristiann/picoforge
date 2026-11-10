# Qwen3 engine results (archived)

These files come from the Qwen3-0.6B / Qwen3-30B-A3B engine, which was removed
from the tree on 2026-11-05. To rerun one, check out the tag
`archive/pre-qwen38-focus`. The protocols are summarised in
[`docs/history.md`](../../docs/history.md).

| File | What it is |
|---|---|
| `e2e_quant_m5pro.csv`, `e2e_tile_m5pro.csv` | Whole GPU forward passes of the 0.6B per format: prefill 512, decode and verify 32 at position 512 |
| `quant_quality_local.csv`, `quant_quality_engine.csv` | Q8/Q4 quality against the unquantised 0.6B: dNLL, KL, top-1 |
| `quant_tradeoffs.png` | Speed, memory and quality trade-off plot |
| `attention_ab_m5pro.csv`, `profile_m5pro.csv` | 0.6B attention A/B and per-op profile |
| `moe_verify_cost_m5pro.csv`, `moe_verify_cost.png` | Verify time of one synthetic 30B MoE layer against the number of distinct experts |
| `moe_e2e_synth30b_m5pro.csv`, `moe_e2e_qwen3_30b_q8_m5pro.csv` | End-to-end MoE passes, synthetic and real 30B |
| `profile_qwen3_30b_20261011.csv` | Per-op GPU timings of the 30B. Use `TOTAL, uncut` as the baseline |
| `routing_overlap_qwen3_30b_20261011.csv` | Corrected 4,096-token routing trace. The `est_*` columns are estimates |
| `corrected-routing-20261011.meta.json` | Hashes and commands for the two files above |
| `spec_qwen3_30b_20261011_pilot.*` | Adaptive speculative decoding pilot: 10 prompts × 5 requests, raw runs and metadata |
| `spec_experts_qwen3_30b_20261011_pilot.*` | Expert-cost predictor pilot (negative result), with its selection recipe |

Two files are kept only so that their corrections can be audited:

- `routing_overlap_qwen3_30b_corpus.csv` is **invalid** beyond 2,048 positions,
  because of the old attention buffer overrun.
- `spec_qwen3_30b_m5pro.csv` is the first five-prompt, three-repetition pilot.
  It has no raw timings and does not follow the later protocol.
