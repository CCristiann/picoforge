# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 40554.2 µs (median 40449.1, p10 40147.0, p90 40977.5)
- Weight bytes: 6104.78 MB, 150.5 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 1368 (floor x 1.4 µs = 1915.2 µs); barriers: 1283
- gdn rollup (sum of gdn.*): 15694.5 µs, 38.7%, 2716.14 MB
- moe rollup (sum of moe.*): 10963.1 µs, 27.0%, 1217.99 MB
- Normal-mode decode gpu_s: 49105.3 µs; sum of groups / normal: 0.8259

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6487.4 | 16.0% | 1639.711 | 252.75 |
| moe.gate_up | MIXED | 48 | 3305.2 | 8.2% | 405.504 | 122.69 |
| moe.down | MIXED | 48 | 3186.1 | 7.9% | 309.658 | 97.19 |
| gdn.hc_pre | MIXED | 36 | 3092.6 | 7.6% | 474.808 | 153.53 |
| gdn.out | Q8_0 | 36 | 2713.3 | 6.7% | 601.620 | 221.73 |
| lm_head | Q8_0 | 1 | 2499.3 | 6.2% | 675.430 | 270.24 |
| qsa.core | MIXED | 12 | 2406.6 | 5.9% | 0.000 | 0.00 |
| hc_ffn.up | F16 | 48 | 2065.1 | 5.1% | 314.573 | 152.33 |
| moe.shared_in | MIXED | 48 | 2052.0 | 5.1% | 419.267 | 204.32 |
| gdn.scan | NONE | 36 | 1858.9 | 4.6% | 0.000 | 0.00 |
| qsa.proj | Q8_0 | 12 | 1812.8 | 4.5% | 455.393 | 251.20 |
| hc_ffn.down | F16 | 48 | 1797.5 | 4.4% | 318.505 | 177.20 |
| moe.route | NONE | 48 | 1294.9 | 3.2% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1125.0 | 2.8% | 83.558 | 74.28 |
| qsa.out | Q8_0 | 12 | 846.9 | 2.1% | 200.540 | 236.79 |
| gdn.conv | NONE | 36 | 728.4 | 1.8% | 0.000 | 0.00 |
| ple | MIXED | 1 | 506.9 | 1.3% | 34.826 | 68.70 |
| hc_ffn.rms | NONE | 48 | 488.9 | 1.2% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 483.8 | 1.2% | 78.643 | 162.56 |
| hc_attn.down | F16 | 12 | 421.3 | 1.0% | 79.626 | 189.02 |
| hc_add | NONE | 60 | 353.4 | 0.9% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 327.5 | 0.8% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 252.3 | 0.6% | 0.000 | 0.00 |
| gdn.add | NONE | 36 | 234.1 | 0.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 123.1 | 0.3% | 0.000 | 0.00 |
| final.up | F16 | 1 | 39.7 | 0.1% | 6.554 | 164.89 |
| final.down | F16 | 1 | 33.2 | 0.1% | 6.554 | 197.26 |
| final.rms | NONE | 1 | 10.4 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.6 | 0.0% | 0.005 | 0.68 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
