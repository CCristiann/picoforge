# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 46854.2 µs (median 46816.8, p10 46684.8, p90 47060.0)
- Weight bytes: 6104.78 MB, 130.3 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18802.2 µs, 40.1%, 2716.14 MB
- Normal-mode decode gpu_s: 50199.7 µs; sum of groups / normal: 0.9334

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| moe | MIXED | 48 | 11930.2 | 25.5% | 1217.987 | 102.09 |
| gdn.proj | MIXED | 36 | 6589.4 | 14.1% | 1639.711 | 248.84 |
| gdn.hc_pre | MIXED | 36 | 4132.6 | 8.8% | 474.808 | 114.89 |
| gdn.scan | NONE | 36 | 3679.4 | 7.9% | 0.000 | 0.00 |
| gdn.out | Q8_0 | 36 | 2716.0 | 5.8% | 601.620 | 221.51 |
| lm_head | Q8_0 | 1 | 2553.5 | 5.4% | 675.430 | 264.52 |
| hc_ffn.up | F16 | 48 | 2212.6 | 4.7% | 314.573 | 142.17 |
| qsa.core | MIXED | 12 | 2203.1 | 4.7% | 0.000 | 0.00 |
| hc_ffn.down | F16 | 48 | 1911.4 | 4.1% | 314.573 | 164.58 |
| qsa.proj | Q8_0 | 12 | 1833.6 | 3.9% | 455.393 | 248.37 |
| qsa.out | Q8_0 | 12 | 834.3 | 1.8% | 200.540 | 240.38 |
| gdn.conv | NONE | 36 | 737.2 | 1.6% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 623.7 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 548.6 | 1.2% | 78.643 | 143.36 |
| hc_ffn.hc_read | NONE | 48 | 518.6 | 1.1% | 0.000 | 0.00 |
| ple | MIXED | 1 | 493.2 | 1.1% | 34.826 | 70.61 |
| hc_ffn.rms | NONE | 48 | 484.3 | 1.0% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 462.1 | 1.0% | 78.643 | 170.18 |
| hc_ffn.inject | F16 | 48 | 432.4 | 0.9% | 3.932 | 9.09 |
| gdn.add | NONE | 36 | 389.8 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 326.3 | 0.7% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 231.5 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 222.7 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 201.3 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 130.0 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 127.2 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 107.7 | 0.2% | 0.983 | 9.13 |
| hc_attn.hc_low | NONE | 12 | 51.6 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 47.7 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.4 | 0.1% | 6.554 | 144.32 |
| final.down | F16 | 1 | 38.7 | 0.1% | 6.554 | 169.49 |
| embed | BF16 | 1 | 13.1 | 0.0% | 0.005 | 0.39 |
| final.hc_read | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.2 | 0.0% | 0.000 | 0.00 |
| final.hc_low | NONE | 1 | 4.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
