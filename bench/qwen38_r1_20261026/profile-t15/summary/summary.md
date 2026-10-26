# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 69188.7 µs (median 69211.5, p10 68932.9, p90 69400.4)
- Weight bytes: 6104.78 MB, 88.2 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2308 (floor x 1.4 µs = 3231.2 µs); barriers: 1789
- Normal-mode decode gpu_s: 71558.6 µs; sum of groups / normal: 0.9669

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 41341.2 | 59.8% | 2716.140 | 65.70 |
| moe | MIXED | 48 | 11856.2 | 17.1% | 1217.987 | 102.73 |
| lm_head | Q8_0 | 1 | 2505.9 | 3.6% | 675.430 | 269.54 |
| hc_ffn.up | F16 | 48 | 2200.3 | 3.2% | 314.573 | 142.97 |
| qsa.core | MIXED | 12 | 2179.4 | 3.1% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 1897.1 | 2.7% | 314.573 | 165.82 |
| qsa.proj | Q8_0 | 12 | 1812.9 | 2.6% | 455.393 | 251.20 |
| qsa.out | Q8_0 | 12 | 822.0 | 1.2% | 200.540 | 243.96 |
| hc_add | NONE | 60 | 640.4 | 0.9% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 546.6 | 0.8% | 78.643 | 143.88 |
| hc_ffn.hc_read | NONE | 48 | 515.7 | 0.7% | 0.000 | 0.00 |
| hc_ffn.rms | NONE | 48 | 492.6 | 0.7% | 0.000 | 0.00 |
| ple | MIXED | 1 | 487.3 | 0.7% | 34.826 | 71.46 |
| hc_attn.down | F16 | 12 | 457.8 | 0.7% | 78.643 | 171.80 |
| hc_ffn.inject | F16 | 48 | 432.3 | 0.6% | 3.932 | 9.10 |
| hc_ffn.hc_low | NONE | 48 | 218.2 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 202.5 | 0.3% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 129.0 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 126.0 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 108.2 | 0.2% | 0.983 | 9.09 |
| hc_attn.hc_low | NONE | 12 | 51.4 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 49.4 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.5 | 0.1% | 6.554 | 144.18 |
| final.down | F16 | 1 | 38.3 | 0.1% | 6.554 | 171.24 |
| final.hc_read | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.4 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.2 | 0.0% | 0.005 | 0.71 |
| final.hc_low | NONE | 1 | 4.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
