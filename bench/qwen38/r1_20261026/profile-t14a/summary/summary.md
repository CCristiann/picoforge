# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 68845.3 µs (median 68822.1, p10 68683.5, p90 69032.5)
- Weight bytes: 6104.78 MB, 88.7 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2307 (floor x 1.4 µs = 3229.8 µs); barriers: 1788
- Normal-mode decode gpu_s: 62326.9 µs; sum of groups / normal: 1.1046

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 41147.1 | 59.8% | 2716.140 | 66.01 |
| moe | MIXED | 48 | 11840.5 | 17.2% | 1217.987 | 102.87 |
| lm_head | Q8_0 | 1 | 2438.0 | 3.5% | 675.430 | 277.04 |
| hc_ffn.up | F16 | 48 | 2196.9 | 3.2% | 314.573 | 143.19 |
| qsa.core | MIXED | 12 | 2179.6 | 3.2% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 1881.2 | 2.7% | 314.573 | 167.22 |
| qsa.proj | Q8_0 | 12 | 1807.3 | 2.6% | 455.393 | 251.98 |
| qsa.out | Q8_0 | 12 | 819.4 | 1.2% | 200.540 | 244.73 |
| hc_add | NONE | 60 | 598.6 | 0.9% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 546.5 | 0.8% | 78.643 | 143.91 |
| hc_ffn.hc_read | NONE | 48 | 516.2 | 0.7% | 0.000 | 0.00 |
| ple | MIXED | 1 | 490.4 | 0.7% | 34.826 | 71.02 |
| hc_ffn.rms | NONE | 48 | 490.1 | 0.7% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 457.2 | 0.7% | 78.643 | 172.02 |
| hc_ffn.inject | F16 | 48 | 431.9 | 0.6% | 3.932 | 9.10 |
| hc_ffn.hc_low | NONE | 48 | 217.1 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 204.7 | 0.3% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 129.4 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 127.8 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 107.9 | 0.2% | 0.983 | 9.11 |
| hc_attn.hc_low | NONE | 12 | 52.0 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 49.0 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.4 | 0.1% | 6.554 | 144.45 |
| final.down | F16 | 1 | 38.3 | 0.1% | 6.554 | 170.94 |
| final.hc_read | NONE | 1 | 10.8 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.4 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.1 | 0.0% | 0.005 | 0.72 |
| final.hc_low | NONE | 1 | 4.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
