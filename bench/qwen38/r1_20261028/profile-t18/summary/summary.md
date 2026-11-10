# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 44994.3 µs (median 44909.6, p10 44767.5, p90 45318.4)
- Weight bytes: 6104.78 MB, 135.7 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18266.5 µs, 40.6%, 2716.14 MB
- moe rollup (sum of moe.*): 10936.5 µs, 24.3%, 1217.99 MB
- Normal-mode decode gpu_s: 49304.6 µs; sum of groups / normal: 0.9126

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6233.7 | 13.9% | 1639.711 | 263.04 |
| gdn.hc_pre | MIXED | 36 | 4074.4 | 9.1% | 474.808 | 116.53 |
| gdn.scan | NONE | 36 | 3697.5 | 8.2% | 0.000 | 0.00 |
| moe.down | MIXED | 48 | 3351.8 | 7.4% | 309.658 | 92.38 |
| moe.gate_up | MIXED | 48 | 3277.5 | 7.3% | 405.504 | 123.72 |
| gdn.out | Q8_0 | 36 | 2625.8 | 5.8% | 601.620 | 229.12 |
| lm_head | Q8_0 | 1 | 2476.0 | 5.5% | 675.430 | 272.79 |
| hc_ffn.up | F16 | 48 | 2200.0 | 4.9% | 314.573 | 142.98 |
| qsa.core | MIXED | 12 | 2173.9 | 4.8% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 1930.7 | 4.3% | 419.267 | 217.16 |
| hc_ffn.down | F16 | 48 | 1890.1 | 4.2% | 314.573 | 166.44 |
| qsa.proj | Q8_0 | 12 | 1754.1 | 3.9% | 455.393 | 259.62 |
| moe.route | NONE | 48 | 1294.5 | 2.9% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1081.9 | 2.4% | 83.558 | 77.23 |
| qsa.out | Q8_0 | 12 | 812.0 | 1.8% | 200.540 | 246.97 |
| gdn.conv | NONE | 36 | 721.4 | 1.6% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 604.9 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 538.8 | 1.2% | 78.643 | 145.96 |
| hc_ffn.hc_read | NONE | 48 | 511.6 | 1.1% | 0.000 | 0.00 |
| ple | MIXED | 1 | 485.7 | 1.1% | 34.826 | 71.71 |
| hc_ffn.rms | NONE | 48 | 479.9 | 1.1% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 462.3 | 1.0% | 78.643 | 170.12 |
| hc_ffn.inject | F16 | 48 | 416.5 | 0.9% | 3.932 | 9.44 |
| gdn.add | NONE | 36 | 375.4 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 320.7 | 0.7% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 224.6 | 0.5% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 217.6 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 195.2 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.2 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 122.0 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 104.7 | 0.2% | 0.983 | 9.39 |
| hc_attn.hc_low | NONE | 12 | 50.2 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 46.1 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 44.9 | 0.1% | 6.554 | 145.91 |
| final.down | F16 | 1 | 38.1 | 0.1% | 6.554 | 171.83 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.1 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.5 | 0.0% | 0.005 | 0.69 |
| final.hc_low | NONE | 1 | 4.1 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
