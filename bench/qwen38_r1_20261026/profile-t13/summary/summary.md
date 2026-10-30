# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 105662.0 µs (median 104555.9, p10 101611.4, p90 110180.4)
- Weight bytes: 6104.78 MB, 57.8 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2482 (floor x 1.4 µs = 3474.8 µs); barriers: 2445
- Normal-mode decode gpu_s: 107641.3 µs; sum of groups / normal: 0.9816

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 56268.2 | 53.3% | 2716.140 | 48.27 |
| hc_ffn.rms | NONE | 48 | 14063.5 | 13.3% | 0.000 | 0.00 |
| moe | MIXED | 48 | 12872.8 | 12.2% | 1217.987 | 94.62 |
| hc_attn.rms | NONE | 12 | 3528.9 | 3.3% | 0.000 | 0.00 |
| qsa.core | MIXED | 12 | 3223.5 | 3.1% | 0.000 | 0.00 |
| hc_ffn.up | F16 | 48 | 2504.5 | 2.4% | 314.573 | 125.61 |
| lm_head | Q8_0 | 1 | 2450.0 | 2.3% | 675.430 | 275.68 |
| hc_ffn.down | F16 | 48 | 1975.9 | 1.9% | 314.573 | 159.21 |
| qsa.proj | Q8_0 | 12 | 1919.6 | 1.8% | 455.393 | 237.23 |
| ple | MIXED | 1 | 1354.5 | 1.3% | 34.826 | 25.71 |
| qsa.out | Q8_0 | 12 | 853.0 | 0.8% | 200.540 | 235.09 |
| hc_add | NONE | 60 | 748.9 | 0.7% | 0.000 | 0.00 |
| hc_ffn.hc_read | NONE | 48 | 716.3 | 0.7% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 555.2 | 0.5% | 78.643 | 141.65 |
| hc_attn.down | F16 | 12 | 465.3 | 0.4% | 78.643 | 169.02 |
| hc_ffn.inject | F16 | 48 | 458.2 | 0.4% | 3.932 | 8.58 |
| hc_ffn.hc_low | NONE | 48 | 428.9 | 0.4% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 351.6 | 0.3% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 302.9 | 0.3% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 196.6 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 109.3 | 0.1% | 0.983 | 8.99 |
| hc_attn.hc_inject | NONE | 12 | 94.2 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_low | NONE | 12 | 92.8 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.7 | 0.0% | 6.554 | 143.46 |
| final.down | F16 | 1 | 40.4 | 0.0% | 6.554 | 162.17 |
| embed | BF16 | 1 | 18.0 | 0.0% | 0.005 | 0.28 |
| final.hc_read | NONE | 1 | 14.8 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 8.5 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
