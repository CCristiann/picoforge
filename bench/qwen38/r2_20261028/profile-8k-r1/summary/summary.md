# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 82694.9 µs (median 82680.8, p10 82536.1, p90 82888.3)
- Weight bytes: 6104.78 MB, 73.8 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 18440.2 µs, 22.3%, 2716.14 MB
- moe rollup (sum of moe.*): 10975.5 µs, 13.3%, 1217.99 MB
- Normal-mode decode gpu_s: 87596.2 µs; sum of groups / normal: 0.9440

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| qsa.core | MIXED | 12 | 39589.3 | 47.9% | 0.000 | 0.00 |
| gdn.proj | MIXED | 36 | 6318.7 | 7.6% | 1639.711 | 259.50 |
| gdn.hc_pre | MIXED | 36 | 4095.4 | 5.0% | 474.808 | 115.94 |
| gdn.scan | NONE | 36 | 3711.8 | 4.5% | 0.000 | 0.00 |
| moe.down | MIXED | 48 | 3353.9 | 4.1% | 309.658 | 92.33 |
| moe.gate_up | MIXED | 48 | 3282.9 | 4.0% | 405.504 | 123.52 |
| gdn.out | Q8_0 | 36 | 2661.5 | 3.2% | 601.620 | 226.04 |
| lm_head | Q8_0 | 1 | 2491.2 | 3.0% | 675.430 | 271.13 |
| hc_ffn.up | F16 | 48 | 2170.2 | 2.6% | 314.573 | 144.95 |
| moe.shared_in | MIXED | 48 | 1966.5 | 2.4% | 419.267 | 213.20 |
| hc_ffn.down | F16 | 48 | 1886.8 | 2.3% | 314.573 | 166.72 |
| qsa.proj | Q8_0 | 12 | 1781.5 | 2.2% | 455.393 | 255.62 |
| moe.route | NONE | 48 | 1288.4 | 1.6% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1083.8 | 1.3% | 83.558 | 77.10 |
| qsa.out | Q8_0 | 12 | 866.9 | 1.0% | 200.540 | 231.32 |
| gdn.conv | NONE | 36 | 726.6 | 0.9% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 599.7 | 0.7% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 540.0 | 0.7% | 78.643 | 145.64 |
| hc_ffn.hc_read | NONE | 48 | 518.3 | 0.6% | 0.000 | 0.00 |
| ple | MIXED | 1 | 489.7 | 0.6% | 34.826 | 71.11 |
| hc_ffn.rms | NONE | 48 | 487.2 | 0.6% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 458.4 | 0.6% | 78.643 | 171.55 |
| hc_ffn.inject | F16 | 48 | 418.4 | 0.5% | 3.932 | 9.40 |
| gdn.add | NONE | 36 | 375.2 | 0.5% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 325.2 | 0.4% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 225.8 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 220.5 | 0.3% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 193.3 | 0.2% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 128.4 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 122.9 | 0.1% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 105.2 | 0.1% | 0.983 | 9.34 |
| hc_attn.hc_low | NONE | 12 | 49.9 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 45.5 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.0 | 0.1% | 6.554 | 145.52 |
| final.down | F16 | 1 | 37.3 | 0.0% | 6.554 | 175.54 |
| final.rms | NONE | 1 | 10.9 | 0.0% | 0.000 | 0.00 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.5 | 0.0% | 0.005 | 0.69 |
| final.hc_low | NONE | 1 | 4.3 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
