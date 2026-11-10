# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 75617.7 µs (median 75669.0, p10 75176.2, p90 75902.6)
- Weight bytes: 6104.78 MB, 80.7 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2307 (floor x 1.4 µs = 3229.8 µs); barriers: 2270
- Normal-mode decode gpu_s: 80753.7 µs; sum of groups / normal: 0.9364

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 44429.0 | 58.8% | 2716.140 | 61.13 |
| moe | MIXED | 48 | 13024.9 | 17.2% | 1217.987 | 93.51 |
| lm_head | Q8_0 | 1 | 3001.4 | 4.0% | 675.430 | 225.04 |
| hc_ffn.up | F16 | 48 | 2245.0 | 3.0% | 314.573 | 140.12 |
| qsa.proj | Q8_0 | 12 | 2214.1 | 2.9% | 455.393 | 205.67 |
| qsa.core | MIXED | 12 | 2202.3 | 2.9% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 2156.6 | 2.9% | 314.573 | 145.86 |
| qsa.out | Q8_0 | 12 | 1059.2 | 1.4% | 200.540 | 189.32 |
| hc_ffn.hc_read | NONE | 48 | 708.6 | 0.9% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 624.1 | 0.8% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 556.6 | 0.7% | 78.643 | 141.28 |
| ple | MIXED | 1 | 530.2 | 0.7% | 34.826 | 65.69 |
| hc_attn.down | F16 | 12 | 516.6 | 0.7% | 78.643 | 152.24 |
| hc_ffn.rms | NONE | 48 | 495.2 | 0.7% | 0.000 | 0.00 |
| hc_ffn.inject | F16 | 48 | 439.7 | 0.6% | 3.932 | 8.94 |
| hc_ffn.hc_low | NONE | 48 | 386.4 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 316.1 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 175.3 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 128.8 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 110.1 | 0.1% | 0.983 | 8.93 |
| hc_attn.hc_low | NONE | 12 | 91.1 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 76.9 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 46.2 | 0.1% | 6.554 | 141.78 |
| final.down | F16 | 1 | 43.0 | 0.1% | 6.554 | 152.48 |
| final.hc_read | NONE | 1 | 14.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 7.5 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.4 | 0.0% | 0.005 | 0.69 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
