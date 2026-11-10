# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 722250.8 µs (median 722380.2, p10 720151.8, p90 723691.3)
- Weight bytes: 6104.78 MB, 8.5 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2986 (floor x 1.4 µs = 4180.4 µs); barriers: 3033
- Normal-mode decode gpu_s: 724796.6 µs; sum of groups / normal: 0.9965

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 236510.9 | 32.7% | 2716.140 | 11.48 |
| moe | MIXED | 48 | 200220.5 | 27.7% | 1217.987 | 6.08 |
| hc_ffn.down | F16 | 48 | 102694.9 | 14.2% | 314.573 | 3.06 |
| hc_ffn.inject | F16 | 48 | 96265.8 | 13.3% | 3.932 | 0.04 |
| hc_attn.down | F16 | 12 | 25621.4 | 3.5% | 78.643 | 3.07 |
| hc_attn.inject | F16 | 12 | 24047.7 | 3.3% | 0.983 | 0.04 |
| hc_ffn.rms | NONE | 48 | 13937.0 | 1.9% | 0.000 | 0.00 |
| hc_ffn.up | F16 | 48 | 3796.7 | 0.5% | 314.573 | 82.85 |
| hc_attn.rms | NONE | 12 | 3483.1 | 0.5% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3146.2 | 0.4% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 2440.1 | 0.3% | 675.430 | 276.81 |
| final.down | F16 | 1 | 2141.8 | 0.3% | 6.554 | 3.06 |
| qsa.qg | Q8_0 | 12 | 1680.1 | 0.2% | 401.080 | 238.72 |
| ple | MIXED | 1 | 1333.0 | 0.2% | 34.826 | 26.13 |
| hc_attn.up | F16 | 12 | 945.7 | 0.1% | 78.643 | 83.16 |
| qsa.out | Q8_0 | 12 | 798.1 | 0.1% | 200.540 | 251.28 |
| hc_ffn.hc_read | NONE | 48 | 700.5 | 0.1% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 570.6 | 0.1% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 397.7 | 0.1% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 340.7 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 290.0 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 171.7 | 0.0% | 0.000 | 0.00 |
| qsa.k | Q8_0 | 12 | 139.0 | 0.0% | 16.712 | 120.25 |
| qsa.iq | Q8_0 | 12 | 114.9 | 0.0% | 16.712 | 145.41 |
| qsa.v | Q8_0 | 12 | 113.2 | 0.0% | 16.712 | 147.60 |
| hc_attn.hc_low | NONE | 12 | 91.5 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 78.9 | 0.0% | 0.000 | 0.00 |
| final.up | F16 | 1 | 78.2 | 0.0% | 6.554 | 83.80 |
| qsa.ik | Q8_0 | 12 | 70.3 | 0.0% | 4.178 | 59.40 |
| final.hc_read | NONE | 1 | 14.0 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 8.9 | 0.0% | 0.005 | 0.58 |
| final.hc_low | NONE | 1 | 7.8 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
