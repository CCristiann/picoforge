# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 248583.4 µs (median 248446.5, p10 248381.3, p90 249036.8)
- Weight bytes: 6104.78 MB, 24.6 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2986 (floor x 1.4 µs = 4180.4 µs); barriers: 3033
- Normal-mode decode gpu_s: 250959.5 µs; sum of groups / normal: 0.9905

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| moe | MIXED | 48 | 159320.5 | 64.1% | 1217.987 | 7.64 |
| gdn | MIXED | 36 | 53916.5 | 21.7% | 2716.140 | 50.38 |
| hc_ffn.rms | NONE | 48 | 13923.4 | 5.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 3482.6 | 1.4% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3125.5 | 1.3% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 2418.5 | 1.0% | 675.430 | 279.27 |
| hc_ffn.up | F16 | 48 | 2197.3 | 0.9% | 314.573 | 143.16 |
| hc_ffn.down | F16 | 48 | 1865.4 | 0.8% | 314.573 | 168.64 |
| qsa.qg | Q8_0 | 12 | 1563.3 | 0.6% | 401.080 | 256.56 |
| ple | MIXED | 1 | 1330.3 | 0.5% | 34.826 | 26.18 |
| qsa.out | Q8_0 | 12 | 773.9 | 0.3% | 200.540 | 259.12 |
| hc_ffn.hc_read | NONE | 48 | 704.4 | 0.3% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 564.0 | 0.2% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 545.6 | 0.2% | 78.643 | 144.15 |
| hc_attn.down | F16 | 12 | 455.9 | 0.2% | 78.643 | 172.50 |
| hc_ffn.inject | F16 | 48 | 431.4 | 0.2% | 3.932 | 9.11 |
| hc_ffn.hc_low | NONE | 48 | 373.8 | 0.2% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 320.3 | 0.1% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 290.0 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 174.3 | 0.1% | 0.000 | 0.00 |
| qsa.k | Q8_0 | 12 | 133.0 | 0.1% | 16.712 | 125.68 |
| qsa.iq | Q8_0 | 12 | 108.8 | 0.0% | 16.712 | 153.58 |
| qsa.v | Q8_0 | 12 | 107.7 | 0.0% | 16.712 | 155.20 |
| hc_attn.inject | F16 | 12 | 107.0 | 0.0% | 0.983 | 9.19 |
| hc_attn.hc_low | NONE | 12 | 88.9 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 78.7 | 0.0% | 0.000 | 0.00 |
| qsa.ik | Q8_0 | 12 | 69.4 | 0.0% | 4.178 | 60.18 |
| final.up | F16 | 1 | 45.6 | 0.0% | 6.554 | 143.79 |
| final.down | F16 | 1 | 38.3 | 0.0% | 6.554 | 171.01 |
| final.hc_read | NONE | 1 | 14.3 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 7.5 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.4 | 0.0% | 0.005 | 0.69 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
