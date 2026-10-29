# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 41699.1 µs (median 41468.1, p10 41292.0, p90 42337.8)
- Weight bytes: 6104.78 MB, 146.4 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2140 (floor x 1.4 µs = 2996.0 µs); barriers: 1657
- gdn rollup (sum of gdn.*): 16081.4 µs, 38.6%, 2716.14 MB
- moe rollup (sum of moe.*): 11020.0 µs, 26.4%, 1217.99 MB
- Normal-mode decode gpu_s: 45626.8 µs; sum of groups / normal: 0.9139

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6487.0 | 15.6% | 1639.711 | 252.77 |
| moe.down | MIXED | 48 | 3340.5 | 8.0% | 309.658 | 92.70 |
| gdn.hc_pre | MIXED | 36 | 3323.9 | 8.0% | 474.808 | 142.85 |
| moe.gate_up | MIXED | 48 | 3264.6 | 7.8% | 405.504 | 124.21 |
| gdn.out | Q8_0 | 36 | 2677.4 | 6.4% | 601.620 | 224.70 |
| lm_head | Q8_0 | 1 | 2518.1 | 6.0% | 675.430 | 268.23 |
| qsa.core | MIXED | 12 | 2331.5 | 5.6% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 2024.3 | 4.9% | 419.267 | 207.12 |
| gdn.scan | NONE | 36 | 1838.3 | 4.4% | 0.000 | 0.00 |
| qsa.proj | Q8_0 | 12 | 1829.2 | 4.4% | 455.393 | 248.96 |
| hc_ffn.down | F16 | 48 | 1770.8 | 4.2% | 318.505 | 179.86 |
| hc_ffn.up | F16 | 48 | 1545.0 | 3.7% | 314.573 | 203.61 |
| moe.route | NONE | 48 | 1307.2 | 3.1% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1083.4 | 2.6% | 83.558 | 77.12 |
| qsa.out | Q8_0 | 12 | 831.0 | 2.0% | 200.540 | 241.32 |
| gdn.conv | NONE | 36 | 778.2 | 1.9% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 618.3 | 1.5% | 0.000 | 0.00 |
| hc_ffn.hc_read | NONE | 48 | 510.8 | 1.2% | 0.000 | 0.00 |
| ple | MIXED | 1 | 489.6 | 1.2% | 34.826 | 71.14 |
| hc_ffn.rms | NONE | 48 | 473.7 | 1.1% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 420.6 | 1.0% | 79.626 | 189.33 |
| gdn.add | NONE | 36 | 387.9 | 0.9% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 380.7 | 0.9% | 78.643 | 206.57 |
| gdn.prep | NONE | 36 | 355.8 | 0.9% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 233.0 | 0.6% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 222.1 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 207.9 | 0.5% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.6 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 123.4 | 0.3% | 0.000 | 0.00 |
| hc_attn.hc_low | NONE | 12 | 50.8 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 50.8 | 0.1% | 0.000 | 0.00 |
| final.down | F16 | 1 | 33.0 | 0.1% | 6.554 | 198.78 |
| final.up | F16 | 1 | 30.8 | 0.1% | 6.554 | 212.87 |
| final.hc_read | NONE | 1 | 10.7 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.1 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.2 | 0.0% | 0.005 | 0.71 |
| final.hc_low | NONE | 1 | 4.1 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
