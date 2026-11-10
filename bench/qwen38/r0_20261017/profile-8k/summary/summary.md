# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 4 decode appends.

- GPU time (sum of groups): 973955.7 µs (median 974227.8, p10 972688.4, p90 975005.4)
- Weight bytes: 6104.78 MB, 6.3 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2986 (floor x 1.4 µs = 4180.4 µs); barriers: 3033

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn | MIXED | 36 | 324480.3 | 33.3% | 2716.140 | 8.37 |
| moe | MIXED | 48 | 262975.3 | 27.0% | 1217.987 | 4.63 |
| hc_ffn.down | F16 | 48 | 102692.8 | 10.5% | 314.573 | 3.06 |
| hc_ffn.inject | F16 | 48 | 96200.8 | 9.9% | 3.932 | 0.04 |
| qsa.core | MIXED | 12 | 40531.8 | 4.2% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 25622.4 | 2.6% | 78.643 | 3.07 |
| hc_attn.inject | F16 | 12 | 24032.2 | 2.5% | 0.983 | 0.04 |
| qsa.out | Q8_0 | 12 | 16890.3 | 1.7% | 200.540 | 11.87 |
| hc_ffn.rms | NONE | 48 | 13961.6 | 1.4% | 0.000 | 0.00 |
| lm_head | Q8_0 | 1 | 12928.8 | 1.3% | 675.430 | 52.24 |
| qsa.qg | Q8_0 | 12 | 9378.9 | 1.0% | 401.080 | 42.76 |
| qsa.k | Q8_0 | 12 | 7202.1 | 0.7% | 16.712 | 2.32 |
| qsa.v | Q8_0 | 12 | 7175.0 | 0.7% | 16.712 | 2.33 |
| qsa.ik | Q8_0 | 12 | 7174.4 | 0.7% | 4.178 | 0.58 |
| qsa.iq | Q8_0 | 12 | 7158.1 | 0.7% | 16.712 | 2.33 |
| hc_ffn.up | F16 | 48 | 3805.5 | 0.4% | 314.573 | 82.66 |
| hc_attn.rms | NONE | 12 | 3480.4 | 0.4% | 0.000 | 0.00 |
| ple | MIXED | 1 | 2432.6 | 0.2% | 34.826 | 14.32 |
| final.down | F16 | 1 | 2136.2 | 0.2% | 6.554 | 3.07 |
| hc_attn.up | F16 | 12 | 954.3 | 0.1% | 78.643 | 82.41 |
| hc_ffn.hc_read | NONE | 48 | 706.5 | 0.1% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 580.8 | 0.1% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 380.7 | 0.0% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 329.8 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 290.3 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 174.1 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_low | NONE | 12 | 91.5 | 0.0% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 80.0 | 0.0% | 0.000 | 0.00 |
| final.up | F16 | 1 | 78.4 | 0.0% | 6.554 | 83.55 |
| final.hc_read | NONE | 1 | 14.6 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 7.7 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.5 | 0.0% | 0.005 | 0.68 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
