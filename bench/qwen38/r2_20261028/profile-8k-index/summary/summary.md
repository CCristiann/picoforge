# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 47096.4 µs (median 47013.7, p10 46928.9, p90 47371.9)
- Weight bytes: 6104.78 MB, 129.6 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2284 (floor x 1.4 µs = 3197.6 µs); barriers: 1801
- gdn rollup (sum of gdn.*): 18360.2 µs, 39.0%, 2716.14 MB
- moe rollup (sum of moe.*): 10970.7 µs, 23.3%, 1217.99 MB
- Normal-mode decode gpu_s: 51841.6 µs; sum of groups / normal: 0.9085

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6271.9 | 13.3% | 1639.711 | 261.44 |
| qsa.core | MIXED | 12 | 4092.3 | 8.7% | 0.000 | 0.00 |
| gdn.hc_pre | MIXED | 36 | 4086.7 | 8.7% | 474.808 | 116.18 |
| gdn.scan | NONE | 36 | 3712.7 | 7.9% | 0.000 | 0.00 |
| moe.down | MIXED | 48 | 3349.5 | 7.1% | 309.658 | 92.45 |
| moe.gate_up | MIXED | 48 | 3287.8 | 7.0% | 405.504 | 123.34 |
| gdn.out | Q8_0 | 36 | 2640.3 | 5.6% | 601.620 | 227.86 |
| lm_head | Q8_0 | 1 | 2472.7 | 5.3% | 675.430 | 273.15 |
| hc_ffn.up | F16 | 48 | 2175.5 | 4.6% | 314.573 | 144.60 |
| moe.shared_in | MIXED | 48 | 1958.0 | 4.2% | 419.267 | 214.13 |
| hc_ffn.down | F16 | 48 | 1878.6 | 4.0% | 314.573 | 167.45 |
| qsa.proj | Q8_0 | 12 | 1790.6 | 3.8% | 455.393 | 254.33 |
| moe.route | NONE | 48 | 1283.1 | 2.7% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1092.3 | 2.3% | 83.558 | 76.50 |
| qsa.out | Q8_0 | 12 | 842.2 | 1.8% | 200.540 | 238.10 |
| gdn.conv | NONE | 36 | 728.4 | 1.5% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 603.9 | 1.3% | 0.000 | 0.00 |
| hc_attn.up | F16 | 12 | 541.4 | 1.1% | 78.643 | 145.26 |
| hc_ffn.hc_read | NONE | 48 | 512.8 | 1.1% | 0.000 | 0.00 |
| hc_ffn.rms | NONE | 48 | 490.1 | 1.0% | 0.000 | 0.00 |
| ple | MIXED | 1 | 488.5 | 1.0% | 34.826 | 71.30 |
| hc_attn.down | F16 | 12 | 458.8 | 1.0% | 78.643 | 171.43 |
| hc_ffn.inject | F16 | 48 | 421.7 | 0.9% | 3.932 | 9.32 |
| gdn.add | NONE | 36 | 370.4 | 0.8% | 0.000 | 0.00 |
| gdn.prep | NONE | 36 | 330.5 | 0.7% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 224.2 | 0.5% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 219.3 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 201.1 | 0.4% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 128.2 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 122.9 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 105.4 | 0.2% | 0.983 | 9.32 |
| hc_attn.hc_low | NONE | 12 | 51.3 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 48.1 | 0.1% | 0.000 | 0.00 |
| final.up | F16 | 1 | 45.0 | 0.1% | 6.554 | 145.75 |
| final.down | F16 | 1 | 38.2 | 0.1% | 6.554 | 171.38 |
| final.hc_read | NONE | 1 | 10.6 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.3 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.1 | 0.0% | 0.005 | 0.72 |
| final.hc_low | NONE | 1 | 4.1 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
