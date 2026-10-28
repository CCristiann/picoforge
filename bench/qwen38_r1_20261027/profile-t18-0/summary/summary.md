# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 46404.7 µs (median 46347.3, p10 46286.4, p90 46611.5)
- Weight bytes: 6104.78 MB, 131.6 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18500.0 µs, 39.9%, 2716.14 MB
- moe rollup (sum of moe.*): 11953.5 µs, 25.8%, 1217.99 MB
- Normal-mode decode gpu_s: 50704.1 µs; sum of groups / normal: 0.9152

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6405.3 | 13.8% | 1639.711 | 255.99 |
| gdn.hc_pre | MIXED | 36 | 4101.7 | 8.8% | 474.808 | 115.76 |
| moe.down | MIXED | 48 | 3826.0 | 8.2% | 309.658 | 80.94 |
| moe.gate_up | MIXED | 48 | 3747.4 | 8.1% | 405.504 | 108.21 |
| gdn.scan | NONE | 36 | 3672.6 | 7.9% | 0.000 | 0.00 |
| gdn.out | Q8_0 | 36 | 2652.2 | 5.7% | 601.620 | 226.84 |
| lm_head | Q8_0 | 1 | 2481.8 | 5.3% | 675.430 | 272.15 |
| hc_ffn.up | F16 | 48 | 2199.5 | 4.7% | 314.573 | 143.02 |
| qsa.core | MIXED | 12 | 2196.0 | 4.7% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 2010.5 | 4.3% | 419.267 | 208.53 |
| hc_ffn.down | F16 | 48 | 1894.6 | 4.1% | 314.573 | 166.04 |
| qsa.proj | Q8_0 | 12 | 1806.8 | 3.9% | 455.393 | 252.04 |
| moe.route | NONE | 48 | 1277.4 | 2.8% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1092.2 | 2.4% | 83.558 | 76.51 |
| qsa.out | Q8_0 | 12 | 824.0 | 1.8% | 200.540 | 243.37 |
| gdn.conv | NONE | 36 | 734.4 | 1.6% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 613.8 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 545.5 | 1.2% | 78.643 | 144.16 |
| hc_ffn.hc_read | NONE | 48 | 515.2 | 1.1% | 0.000 | 0.00 |
| hc_ffn.rms | NONE | 48 | 489.9 | 1.1% | 0.000 | 0.00 |
| ple | MIXED | 1 | 489.7 | 1.1% | 34.826 | 71.11 |
| hc_attn.down | F16 | 12 | 456.6 | 1.0% | 78.643 | 172.23 |
| hc_ffn.inject | F16 | 48 | 433.8 | 0.9% | 3.932 | 9.06 |
| gdn.add | NONE | 36 | 381.5 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 322.0 | 0.7% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 230.3 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 217.0 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 206.0 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 128.7 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 126.0 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 110.0 | 0.2% | 0.983 | 8.94 |
| hc_attn.hc_low | NONE | 12 | 51.8 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 48.5 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.3 | 0.1% | 6.554 | 144.61 |
| final.down | F16 | 1 | 38.3 | 0.1% | 6.554 | 171.22 |
| final.hc_read | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.2 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.3 | 0.0% | 0.005 | 0.70 |
| final.hc_low | NONE | 1 | 4.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
