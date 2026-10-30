# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 51076.4 µs (median 51031.8, p10 50899.6, p90 51345.4)
- Weight bytes: 6104.78 MB, 119.5 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2284 (floor x 1.4 µs = 3197.6 µs); barriers: 1801
- gdn rollup (sum of gdn.*): 18351.0 µs, 35.9%, 2716.14 MB
- moe rollup (sum of moe.*): 10946.4 µs, 21.4%, 1217.99 MB
- Normal-mode decode gpu_s: 56069.9 µs; sum of groups / normal: 0.9109

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| qsa.core | MIXED | 12 | 8144.4 | 15.9% | 0.000 | 0.00 |
| gdn.proj | MIXED | 36 | 6254.4 | 12.2% | 1639.711 | 262.17 |
| gdn.hc_pre | MIXED | 36 | 4075.8 | 8.0% | 474.808 | 116.50 |
| gdn.scan | NONE | 36 | 3732.6 | 7.3% | 0.000 | 0.00 |
| moe.down | MIXED | 48 | 3351.7 | 6.6% | 309.658 | 92.39 |
| moe.gate_up | MIXED | 48 | 3280.3 | 6.4% | 405.504 | 123.62 |
| gdn.out | Q8_0 | 36 | 2648.2 | 5.2% | 601.620 | 227.18 |
| lm_head | Q8_0 | 1 | 2480.8 | 4.9% | 675.430 | 272.27 |
| hc_ffn.up | F16 | 48 | 2172.3 | 4.3% | 314.573 | 144.81 |
| moe.shared_in | MIXED | 48 | 1960.4 | 3.8% | 419.267 | 213.87 |
| hc_ffn.down | F16 | 48 | 1899.5 | 3.7% | 314.573 | 165.61 |
| qsa.proj | Q8_0 | 12 | 1758.9 | 3.4% | 455.393 | 258.90 |
| moe.route | NONE | 48 | 1272.6 | 2.5% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1081.3 | 2.1% | 83.558 | 77.27 |
| qsa.out | Q8_0 | 12 | 835.8 | 1.6% | 200.540 | 239.93 |
| gdn.conv | NONE | 36 | 727.0 | 1.4% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 594.0 | 1.2% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 541.1 | 1.1% | 78.643 | 145.33 |
| hc_ffn.hc_read | NONE | 48 | 510.8 | 1.0% | 0.000 | 0.00 |
| hc_ffn.rms | NONE | 48 | 485.2 | 0.9% | 0.000 | 0.00 |
| ple | MIXED | 1 | 484.5 | 0.9% | 34.826 | 71.89 |
| hc_attn.down | F16 | 12 | 456.4 | 0.9% | 78.643 | 172.31 |
| hc_ffn.inject | F16 | 48 | 419.6 | 0.8% | 3.932 | 9.37 |
| gdn.add | NONE | 36 | 373.0 | 0.7% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 322.9 | 0.6% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 226.3 | 0.4% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 217.1 | 0.4% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 199.6 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.3 | 0.2% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 121.9 | 0.2% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 104.8 | 0.2% | 0.983 | 9.38 |
| hc_attn.hc_low | NONE | 12 | 51.8 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 48.2 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.0 | 0.1% | 6.554 | 145.64 |
| final.down | F16 | 1 | 38.3 | 0.1% | 6.554 | 170.96 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.2 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.2 | 0.0% | 0.005 | 0.71 |
| final.hc_low | NONE | 1 | 4.3 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
