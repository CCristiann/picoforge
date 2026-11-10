# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 935452.0 µs (median 935623.2, p10 934420.0, p90 936474.6)
- Weight bytes: 6104.78 MB, 6.5 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2986 (floor x 1.4 µs = 4180.4 µs); barriers: 3033
- Normal-mode decode gpu_s: 934528.4 µs; sum of groups / normal: 1.0010

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 323960.3 | 34.6% | 2716.140 | 8.38 |
| moe | MIXED | 48 | 262743.3 | 28.1% | 1217.987 | 4.64 |
| hc_ffn.down | F16 | 48 | 102575.8 | 11.0% | 314.573 | 3.07 |
| hc_ffn.inject | F16 | 48 | 96115.2 | 10.3% | 3.932 | 0.04 |
| hc_attn.down | F16 | 12 | 25574.1 | 2.7% | 78.643 | 3.08 |
| hc_attn.inject | F16 | 12 | 23996.2 | 2.6% | 0.983 | 0.04 |
| qsa.out | Q8_0 | 12 | 16871.5 | 1.8% | 200.540 | 11.89 |
| hc_ffn.rms | NONE | 48 | 13954.4 | 1.5% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 12985.2 | 1.4% | 675.430 | 52.02 |
| qsa.qg | Q8_0 | 12 | 9328.5 | 1.0% | 401.080 | 43.00 |
| qsa.k | Q8_0 | 12 | 7212.4 | 0.8% | 16.712 | 2.32 |
| qsa.v | Q8_0 | 12 | 7176.4 | 0.8% | 16.712 | 2.33 |
| qsa.ik | Q8_0 | 12 | 7164.4 | 0.8% | 4.178 | 0.58 |
| qsa.iq | Q8_0 | 12 | 7138.0 | 0.8% | 16.712 | 2.34 |
| hc_ffn.up | F16 | 48 | 3814.6 | 0.4% | 314.573 | 82.47 |
| hc_attn.rms | NONE | 12 | 3478.2 | 0.4% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3102.7 | 0.3% | 0.000 | 0.00 |
| ple | MIXED | 1 | 2442.8 | 0.3% | 34.826 | 14.26 |
| final.down | F16 | 1 | 2127.0 | 0.2% | 6.554 | 3.08 |
| hc_attn.up | F16 | 12 | 953.5 | 0.1% | 78.643 | 82.48 |
| hc_ffn.hc_read | NONE | 48 | 703.1 | 0.1% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 571.2 | 0.1% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 391.5 | 0.0% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 330.0 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 289.9 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 171.0 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_low | NONE | 12 | 92.4 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 80.6 | 0.0% | 0.000 | 0.00 |
| final.up | F16 | 1 | 78.5 | 0.0% | 6.554 | 83.51 |
| final.hc_read | NONE | 1 | 14.2 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 7.7 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.4 | 0.0% | 0.005 | 0.69 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
