# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 45215.5 µs (median 45170.4, p10 45054.0, p90 45496.3)
- Weight bytes: 6104.78 MB, 135.0 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18354.7 µs, 40.6%, 2716.14 MB
- moe rollup (sum of moe.*): 10929.4 µs, 24.2%, 1217.99 MB
- Normal-mode decode gpu_s: 49542.1 µs; sum of groups / normal: 0.9127

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6288.3 | 13.9% | 1639.711 | 260.76 |
| gdn.hc_pre | MIXED | 36 | 4070.0 | 9.0% | 474.808 | 116.66 |
| gdn.scan | NONE | 36 | 3703.5 | 8.2% | 0.000 | 0.00 |
| moe.down | MIXED | 48 | 3352.9 | 7.4% | 309.658 | 92.36 |
| moe.gate_up | MIXED | 48 | 3288.6 | 7.3% | 405.504 | 123.30 |
| gdn.out | Q8_0 | 36 | 2645.8 | 5.9% | 601.620 | 227.39 |
| lm_head | Q8_0 | 1 | 2508.2 | 5.5% | 675.430 | 269.29 |
| qsa.core | MIXED | 12 | 2337.9 | 5.2% | 0.000 | 0.00 |
| hc_ffn.up | F16 | 48 | 2167.5 | 4.8% | 314.573 | 145.13 |
| moe.shared_in | MIXED | 48 | 1932.0 | 4.3% | 419.267 | 217.01 |
| hc_ffn.down | F16 | 48 | 1873.8 | 4.1% | 314.573 | 167.88 |
| qsa.proj | Q8_0 | 12 | 1764.3 | 3.9% | 455.393 | 258.12 |
| moe.route | NONE | 48 | 1280.7 | 2.8% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1075.2 | 2.4% | 83.558 | 77.71 |
| qsa.out | Q8_0 | 12 | 816.2 | 1.8% | 200.540 | 245.70 |
| gdn.conv | NONE | 36 | 724.7 | 1.6% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 591.9 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 539.6 | 1.2% | 78.643 | 145.74 |
| hc_ffn.hc_read | NONE | 48 | 510.2 | 1.1% | 0.000 | 0.00 |
| ple | MIXED | 1 | 483.6 | 1.1% | 34.826 | 72.02 |
| hc_ffn.rms | NONE | 48 | 475.4 | 1.1% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 458.0 | 1.0% | 78.643 | 171.73 |
| hc_ffn.inject | F16 | 48 | 424.6 | 0.9% | 3.932 | 9.26 |
| gdn.add | NONE | 36 | 372.3 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 332.3 | 0.7% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 218.3 | 0.5% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 217.7 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 198.2 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.1 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 120.5 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 104.8 | 0.2% | 0.983 | 9.38 |
| hc_attn.hc_low | NONE | 12 | 49.8 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 46.1 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 44.9 | 0.1% | 6.554 | 146.02 |
| final.down | F16 | 1 | 38.1 | 0.1% | 6.554 | 172.02 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.1 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.4 | 0.0% | 0.005 | 0.69 |
| final.hc_low | NONE | 1 | 4.3 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
