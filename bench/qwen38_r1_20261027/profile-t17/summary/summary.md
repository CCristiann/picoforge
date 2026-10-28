# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 46350.3 µs (median 46286.2, p10 46084.3, p90 46766.1)
- Weight bytes: 6104.78 MB, 131.7 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18530.0 µs, 40.0%, 2716.14 MB
- Normal-mode decode gpu_s: 49671.3 µs; sum of groups / normal: 0.9331

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| moe | MIXED | 48 | 11829.3 | 25.5% | 1217.987 | 102.96 |
| gdn.proj | MIXED | 36 | 6395.9 | 13.8% | 1639.711 | 256.37 |
| gdn.hc_pre | MIXED | 36 | 4142.4 | 8.9% | 474.808 | 114.62 |
| gdn.scan | NONE | 36 | 3677.6 | 7.9% | 0.000 | 0.00 |
| gdn.out | Q8_0 | 36 | 2646.2 | 5.7% | 601.620 | 227.36 |
| lm_head | Q8_0 | 1 | 2483.7 | 5.4% | 675.430 | 271.95 |
| qsa.core | MIXED | 12 | 2239.3 | 4.8% | 0.000 | 0.00 |
| hc_ffn.up | F16 | 48 | 2207.6 | 4.8% | 314.573 | 142.49 |
| hc_ffn.down | F16 | 48 | 1904.6 | 4.1% | 314.573 | 165.17 |
| qsa.proj | Q8_0 | 12 | 1805.4 | 3.9% | 455.393 | 252.23 |
| qsa.out | Q8_0 | 12 | 823.3 | 1.8% | 200.540 | 243.59 |
| gdn.conv | NONE | 36 | 729.1 | 1.6% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 605.5 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 545.6 | 1.2% | 78.643 | 144.15 |
| hc_ffn.hc_read | NONE | 48 | 516.8 | 1.1% | 0.000 | 0.00 |
| ple | MIXED | 1 | 486.2 | 1.0% | 34.826 | 71.63 |
| hc_ffn.rms | NONE | 48 | 483.2 | 1.0% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 457.1 | 1.0% | 78.643 | 172.06 |
| hc_ffn.inject | F16 | 48 | 432.1 | 0.9% | 3.932 | 9.10 |
| gdn.add | NONE | 36 | 381.1 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 327.5 | 0.7% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 230.2 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 217.2 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 205.6 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 128.9 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 126.1 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 107.3 | 0.2% | 0.983 | 9.16 |
| hc_attn.hc_low | NONE | 12 | 51.7 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 48.1 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.3 | 0.1% | 6.554 | 144.72 |
| final.down | F16 | 1 | 37.9 | 0.1% | 6.554 | 172.70 |
| final.hc_read | NONE | 1 | 10.8 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.3 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.2 | 0.0% | 0.005 | 0.71 |
| final.hc_low | NONE | 1 | 4.3 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
