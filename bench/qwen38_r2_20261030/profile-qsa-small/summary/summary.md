# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 40155.7 µs (median 40037.9, p10 39506.7, p90 40865.1)
- Weight bytes: 6104.78 MB, 152.0 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 1560 (floor x 1.4 µs = 2184.0 µs); barriers: 1367
- gdn rollup (sum of gdn.*): 16155.7 µs, 40.2%, 2716.14 MB
- moe rollup (sum of moe.*): 11068.8 µs, 27.6%, 1217.99 MB
- Normal-mode decode gpu_s: 43457.5 µs; sum of groups / normal: 0.9240

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6668.0 | 16.6% | 1639.711 | 245.91 |
| moe.gate_up | MIXED | 48 | 3361.7 | 8.4% | 405.504 | 120.62 |
| moe.down | MIXED | 48 | 3182.6 | 7.9% | 309.658 | 97.30 |
| gdn.hc_pre | MIXED | 36 | 3135.2 | 7.8% | 474.808 | 151.45 |
| gdn.out | Q8_0 | 36 | 2791.8 | 7.0% | 601.620 | 215.50 |
| lm_head | Q8_0 | 1 | 2627.9 | 6.5% | 675.430 | 257.02 |
| moe.shared_in | MIXED | 48 | 2089.2 | 5.2% | 419.267 | 200.68 |
| hc_ffn.up | F16 | 48 | 2057.7 | 5.1% | 314.573 | 152.88 |
| qsa.proj | Q8_0 | 12 | 1879.0 | 4.7% | 455.393 | 242.36 |
| gdn.scan | NONE | 36 | 1859.4 | 4.6% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 1820.1 | 4.5% | 318.505 | 174.99 |
| moe.route | NONE | 48 | 1295.3 | 3.2% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1139.9 | 2.8% | 83.558 | 73.30 |
| qsa.core | MIXED | 12 | 950.7 | 2.4% | 0.000 | 0.00 |
| qsa.out | Q8_0 | 12 | 848.1 | 2.1% | 200.540 | 236.45 |
| gdn.conv | NONE | 36 | 741.2 | 1.8% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 605.0 | 1.5% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 509.6 | 1.3% | 78.643 | 154.32 |
| hc_ffn.rms | NONE | 48 | 497.3 | 1.2% | 0.000 | 0.00 |
| ple | MIXED | 1 | 489.8 | 1.2% | 34.826 | 71.10 |
| hc_attn.down | F16 | 12 | 430.8 | 1.1% | 79.626 | 184.82 |
| gdn.add | NONE | 36 | 383.3 | 1.0% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 327.1 | 0.8% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 249.8 | 0.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 123.5 | 0.3% | 0.000 | 0.00 |
| final.up | F16 | 1 | 40.1 | 0.1% | 6.554 | 163.41 |
| final.down | F16 | 1 | 33.6 | 0.1% | 6.554 | 195.30 |
| final.rms | NONE | 1 | 10.4 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.5 | 0.0% | 0.005 | 0.68 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
