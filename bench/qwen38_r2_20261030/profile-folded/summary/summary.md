# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 38627.1 µs (median 38665.0, p10 38510.6, p90 38723.8)
- Weight bytes: 6104.78 MB, 158.0 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 1356 (floor x 1.4 µs = 1898.4 µs); barriers: 1271
- gdn rollup (sum of gdn.*): 15457.8 µs, 40.0%, 2716.14 MB
- moe rollup (sum of moe.*): 10884.2 µs, 28.2%, 1217.99 MB
- Normal-mode decode gpu_s: 42017.2 µs; sum of groups / normal: 0.9193

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6409.0 | 16.6% | 1639.711 | 255.84 |
| moe.gate_up | MIXED | 48 | 3281.4 | 8.5% | 405.504 | 123.58 |
| moe.down | MIXED | 48 | 3171.2 | 8.2% | 309.658 | 97.65 |
| gdn.hc_pre | MIXED | 36 | 3049.3 | 7.9% | 474.808 | 155.71 |
| gdn.out | Q8_0 | 36 | 2649.6 | 6.9% | 601.620 | 227.06 |
| lm_head | Q8_0 | 1 | 2493.1 | 6.5% | 675.430 | 270.92 |
| moe.shared_in | MIXED | 48 | 2030.9 | 5.3% | 419.267 | 206.44 |
| hc_ffn.up | F16 | 48 | 2004.2 | 5.2% | 314.573 | 156.96 |
| gdn.scan | NONE | 36 | 1847.2 | 4.8% | 0.000 | 0.00 |
| qsa.proj | Q8_0 | 12 | 1804.7 | 4.7% | 455.393 | 252.34 |
| hc_ffn.down | F16 | 48 | 1773.3 | 4.6% | 318.505 | 179.61 |
| moe.route | NONE | 48 | 1306.1 | 3.4% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1094.6 | 2.8% | 83.558 | 76.33 |
| qsa.core | MIXED | 12 | 942.1 | 2.4% | 0.000 | 0.00 |
| qsa.out | Q8_0 | 12 | 818.0 | 2.1% | 200.540 | 245.16 |
| gdn.conv | NONE | 36 | 724.0 | 1.9% | 0.000 | 0.00 |
| hc_ffn.rms | NONE | 48 | 494.9 | 1.3% | 0.000 | 0.00 |
| ple | MIXED | 1 | 491.4 | 1.3% | 34.826 | 70.88 |
| hc_attn.up | F16 | 12 | 480.1 | 1.2% | 78.643 | 163.80 |
| hc_attn.down | F16 | 12 | 418.5 | 1.1% | 79.626 | 190.28 |
| hc_add | NONE | 60 | 352.5 | 0.9% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 323.3 | 0.8% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 241.3 | 0.6% | 0.000 | 0.00 |
| gdn.add | NONE | 36 | 214.2 | 0.6% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 122.9 | 0.3% | 0.000 | 0.00 |
| final.up | F16 | 1 | 39.6 | 0.1% | 6.554 | 165.30 |
| final.down | F16 | 1 | 32.3 | 0.1% | 6.554 | 203.08 |
| final.rms | NONE | 1 | 10.2 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.3 | 0.0% | 0.005 | 0.71 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
