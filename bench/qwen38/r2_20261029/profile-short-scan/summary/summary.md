# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 41682.7 µs (median 41639.8, p10 41557.3, p90 41884.9)
- Weight bytes: 6104.78 MB, 146.5 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2236 (floor x 1.4 µs = 3130.4 µs); barriers: 1753
- gdn rollup (sum of gdn.*): 15933.5 µs, 38.2%, 2716.14 MB
- moe rollup (sum of moe.*): 10976.2 µs, 26.3%, 1217.99 MB
- Normal-mode decode gpu_s: 46070.2 µs; sum of groups / normal: 0.9048

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6399.5 | 15.4% | 1639.711 | 256.23 |
| gdn.hc_pre | MIXED | 36 | 3364.8 | 8.1% | 474.808 | 141.11 |
| moe.down | MIXED | 48 | 3334.7 | 8.0% | 309.658 | 92.86 |
| moe.gate_up | MIXED | 48 | 3255.6 | 7.8% | 405.504 | 124.56 |
| gdn.out | Q8_0 | 36 | 2644.0 | 6.3% | 601.620 | 227.55 |
| lm_head | Q8_0 | 1 | 2478.6 | 5.9% | 675.430 | 272.50 |
| qsa.core | MIXED | 12 | 2325.0 | 5.6% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 1998.5 | 4.8% | 419.267 | 209.79 |
| gdn.scan | NONE | 36 | 1825.3 | 4.4% | 0.000 | 0.00 |
| qsa.proj | Q8_0 | 12 | 1799.0 | 4.3% | 455.393 | 253.13 |
| hc_ffn.down | F16 | 48 | 1655.3 | 4.0% | 314.573 | 190.04 |
| hc_ffn.up | F16 | 48 | 1501.8 | 3.6% | 314.573 | 209.47 |
| moe.route | NONE | 48 | 1299.4 | 3.1% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1088.0 | 2.6% | 83.558 | 76.80 |
| qsa.out | Q8_0 | 12 | 815.4 | 2.0% | 200.540 | 245.94 |
| gdn.conv | NONE | 36 | 736.1 | 1.8% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 607.0 | 1.5% | 0.000 | 0.00 |
| hc_ffn.hc_read | NONE | 48 | 510.4 | 1.2% | 0.000 | 0.00 |
| ple | MIXED | 1 | 488.8 | 1.2% | 34.826 | 71.25 |
| hc_ffn.rms | NONE | 48 | 483.9 | 1.2% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 389.3 | 0.9% | 78.643 | 202.03 |
| gdn.add | NONE | 36 | 387.4 | 0.9% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 368.1 | 0.9% | 78.643 | 213.67 |
| hc_ffn.inject | F16 | 48 | 366.8 | 0.9% | 3.932 | 10.72 |
| gdn.prep | NONE | 36 | 351.7 | 0.8% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 232.2 | 0.6% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 224.8 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 210.6 | 0.5% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.4 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 122.3 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 92.3 | 0.2% | 0.983 | 10.66 |
| hc_attn.hc_low | NONE | 12 | 52.9 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 50.7 | 0.1% | 0.000 | 0.00 |
| final.down | F16 | 1 | 32.3 | 0.1% | 6.554 | 202.76 |
| final.up | F16 | 1 | 30.4 | 0.1% | 6.554 | 215.76 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.1 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.3 | 0.0% | 0.005 | 0.70 |
| final.hc_low | NONE | 1 | 4.6 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
