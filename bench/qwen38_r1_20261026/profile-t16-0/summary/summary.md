# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 70647.7 µs (median 70493.3, p10 70231.5, p90 71246.6)
- Weight bytes: 6104.78 MB, 86.4 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2308 (floor x 1.4 µs = 3231.2 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 42212.6 µs, 59.8%, 2716.14 MB
- Normal-mode decode gpu_s: 73667.2 µs; sum of groups / normal: 0.9590

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.prep | NONE | 36 | 23229.7 | 32.9% | 0.000 | 0.00 |
| moe | MIXED | 48 | 12038.9 | 17.0% | 1217.987 | 101.17 |
| gdn.proj | MIXED | 36 | 6786.5 | 9.6% | 1639.711 | 241.61 |
| gdn.hc_pre | MIXED | 36 | 4189.4 | 5.9% | 474.808 | 113.33 |
| gdn.scan | NONE | 36 | 3676.7 | 5.2% | 0.000 | 0.00 |
| gdn.out | Q8_0 | 36 | 2969.9 | 4.2% | 601.620 | 202.57 |
| lm_head | Q8_0 | 1 | 2608.3 | 3.7% | 675.430 | 258.96 |
| hc_ffn.up | F16 | 48 | 2227.6 | 3.2% | 314.573 | 141.22 |
| qsa.core | MIXED | 12 | 2174.4 | 3.1% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 1975.6 | 2.8% | 314.573 | 159.23 |
| qsa.proj | Q8_0 | 12 | 1920.0 | 2.7% | 455.393 | 237.18 |
| qsa.out | Q8_0 | 12 | 883.6 | 1.3% | 200.540 | 226.95 |
| gdn.conv | NONE | 36 | 734.9 | 1.0% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 638.4 | 0.9% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 549.7 | 0.8% | 78.643 | 143.07 |
| hc_ffn.hc_read | NONE | 48 | 519.1 | 0.7% | 0.000 | 0.00 |
| ple | MIXED | 1 | 496.2 | 0.7% | 34.826 | 70.18 |
| hc_ffn.rms | NONE | 48 | 488.2 | 0.7% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 470.0 | 0.7% | 78.643 | 167.32 |
| hc_ffn.inject | F16 | 48 | 431.4 | 0.6% | 3.932 | 9.11 |
| gdn.add | NONE | 36 | 393.3 | 0.6% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 232.2 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 219.8 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 208.7 | 0.3% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 128.8 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 126.6 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 107.2 | 0.2% | 0.983 | 9.17 |
| hc_attn.hc_low | NONE | 12 | 51.1 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 49.2 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.6 | 0.1% | 6.554 | 143.87 |
| final.down | F16 | 1 | 39.5 | 0.1% | 6.554 | 165.98 |
| final.hc_read | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 10.4 | 0.0% | 0.005 | 0.49 |
| final.rms | NONE | 1 | 10.3 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 6.0 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
