# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 248634.6 µs (median 248601.6, p10 248366.2, p90 249004.5)
- Weight bytes: 6104.78 MB, 24.6 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2770 (floor x 1.4 µs = 3878.0 µs); barriers: 2733
- Normal-mode decode gpu_s: 250799.6 µs; sum of groups / normal: 0.9914

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| moe | MIXED | 48 | 159357.7 | 64.1% | 1217.987 | 7.64 |
| gdn | MIXED | 36 | 54051.9 | 21.7% | 2716.140 | 50.25 |
| hc_ffn.rms | NONE | 48 | 13930.6 | 5.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 3483.1 | 1.4% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3113.0 | 1.3% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 2406.1 | 1.0% | 675.430 | 280.71 |
| hc_ffn.up | F16 | 48 | 2195.8 | 0.9% | 314.573 | 143.26 |
| hc_ffn.down | F16 | 48 | 1878.8 | 0.8% | 314.573 | 167.44 |
| qsa.proj | Q8_0 | 12 | 1817.5 | 0.7% | 455.393 | 250.55 |
| ple | MIXED | 1 | 1328.1 | 0.5% | 34.826 | 26.22 |
| qsa.out | Q8_0 | 12 | 823.7 | 0.3% | 200.540 | 243.48 |
| hc_ffn.hc_read | NONE | 48 | 705.1 | 0.3% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 575.2 | 0.2% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 546.3 | 0.2% | 78.643 | 143.97 |
| hc_attn.down | F16 | 12 | 455.1 | 0.2% | 78.643 | 172.82 |
| hc_ffn.inject | F16 | 48 | 430.6 | 0.2% | 3.932 | 9.13 |
| hc_ffn.hc_low | NONE | 48 | 371.3 | 0.1% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 316.7 | 0.1% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 290.1 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 175.3 | 0.1% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 106.8 | 0.0% | 0.983 | 9.20 |
| hc_attn.hc_low | NONE | 12 | 86.2 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 76.7 | 0.0% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.4 | 0.0% | 6.554 | 144.20 |
| final.down | F16 | 1 | 38.4 | 0.0% | 6.554 | 170.85 |
| final.hc_read | NONE | 1 | 14.5 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.3 | 0.0% | 0.005 | 0.70 |
| final.hc_low | NONE | 1 | 7.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
