# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 160585.8 µs (median 160584.6, p10 160226.6, p90 161006.1)
- Weight bytes: 6104.78 MB, 38.0 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2578 (floor x 1.4 µs = 3609.2 µs); barriers: 2541
- Normal-mode decode gpu_s: 162401.4 µs; sum of groups / normal: 0.9888

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| moe | MIXED | 48 | 70714.0 | 44.0% | 1217.987 | 17.22 |
| gdn | MIXED | 36 | 54373.0 | 33.9% | 2716.140 | 49.95 |
| hc_ffn.rms | NONE | 48 | 13935.8 | 8.7% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 3482.9 | 2.2% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3107.0 | 1.9% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 2485.3 | 1.5% | 675.430 | 271.77 |
| hc_ffn.up | F16 | 48 | 2204.9 | 1.4% | 314.573 | 142.67 |
| hc_ffn.down | F16 | 48 | 1917.5 | 1.2% | 314.573 | 164.05 |
| qsa.proj | Q8_0 | 12 | 1878.7 | 1.2% | 455.393 | 242.40 |
| ple | MIXED | 1 | 1335.6 | 0.8% | 34.826 | 26.08 |
| qsa.out | Q8_0 | 12 | 862.3 | 0.5% | 200.540 | 232.57 |
| hc_ffn.hc_read | NONE | 48 | 704.2 | 0.4% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 568.4 | 0.4% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 550.0 | 0.3% | 78.643 | 142.98 |
| hc_attn.down | F16 | 12 | 464.2 | 0.3% | 78.643 | 169.43 |
| hc_ffn.inject | F16 | 48 | 437.4 | 0.3% | 3.932 | 8.99 |
| hc_ffn.hc_low | NONE | 48 | 385.5 | 0.2% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 315.7 | 0.2% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 290.5 | 0.2% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 173.9 | 0.1% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 108.7 | 0.1% | 0.983 | 9.04 |
| hc_attn.hc_low | NONE | 12 | 91.0 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 78.0 | 0.0% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.9 | 0.0% | 6.554 | 142.84 |
| final.down | F16 | 1 | 39.0 | 0.0% | 6.554 | 168.15 |
| final.hc_read | NONE | 1 | 14.6 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 14.4 | 0.0% | 0.005 | 0.36 |
| final.hc_low | NONE | 1 | 7.5 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
