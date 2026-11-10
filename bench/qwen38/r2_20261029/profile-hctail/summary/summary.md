# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 40662.0 µs (median 40572.0, p10 40432.5, p90 41001.8)
- Weight bytes: 6104.78 MB, 150.1 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 1560 (floor x 1.4 µs = 2184.0 µs); barriers: 1367
- gdn rollup (sum of gdn.*): 15819.3 µs, 38.9%, 2716.14 MB
- moe rollup (sum of moe.*): 10887.6 µs, 26.8%, 1217.99 MB
- Normal-mode decode gpu_s: 43955.8 µs; sum of groups / normal: 0.9251

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6480.7 | 15.9% | 1639.711 | 253.01 |
| moe.gate_up | MIXED | 48 | 3282.7 | 8.1% | 405.504 | 123.53 |
| moe.down | MIXED | 48 | 3176.1 | 7.8% | 309.658 | 97.49 |
| gdn.hc_pre | MIXED | 36 | 3080.0 | 7.6% | 474.808 | 154.16 |
| gdn.out | Q8_0 | 36 | 2736.1 | 6.7% | 601.620 | 219.88 |
| lm_head | Q8_0 | 1 | 2507.9 | 6.2% | 675.430 | 269.32 |
| qsa.core | MIXED | 12 | 2327.7 | 5.7% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 2032.3 | 5.0% | 419.267 | 206.30 |
| hc_ffn.up | F16 | 48 | 2022.7 | 5.0% | 314.573 | 155.52 |
| gdn.scan | NONE | 36 | 1852.0 | 4.6% | 0.000 | 0.00 |
| qsa.proj | Q8_0 | 12 | 1827.2 | 4.5% | 455.393 | 249.23 |
| hc_ffn.down | F16 | 48 | 1763.3 | 4.3% | 318.505 | 180.63 |
| moe.route | NONE | 48 | 1296.1 | 3.2% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1100.3 | 2.7% | 83.558 | 75.94 |
| qsa.out | Q8_0 | 12 | 828.6 | 2.0% | 200.540 | 242.02 |
| gdn.conv | NONE | 36 | 730.1 | 1.8% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 574.9 | 1.4% | 0.000 | 0.00 |
| ple | MIXED | 1 | 492.1 | 1.2% | 34.826 | 70.78 |
| hc_ffn.rms | NONE | 48 | 487.4 | 1.2% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 485.1 | 1.2% | 78.643 | 162.12 |
| hc_attn.down | F16 | 12 | 421.0 | 1.0% | 79.626 | 189.15 |
| gdn.add | NONE | 36 | 374.0 | 0.9% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 321.6 | 0.8% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 244.6 | 0.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 123.3 | 0.3% | 0.000 | 0.00 |
| final.up | F16 | 1 | 39.3 | 0.1% | 6.554 | 166.88 |
| final.down | F16 | 1 | 32.6 | 0.1% | 6.554 | 200.84 |
| embed | BF16 | 1 | 11.6 | 0.0% | 0.005 | 0.44 |
| final.rms | NONE | 1 | 10.4 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
