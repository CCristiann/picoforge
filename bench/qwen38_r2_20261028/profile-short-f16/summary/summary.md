# Qwen3.8 decode profile, per decode token

Sources: profile.csv; 8 decode appends.

- GPU time (sum of groups): 43601.5 µs (median 43544.2, p10 43488.5, p90 43771.4)
- Weight bytes: 6104.78 MB, 140.0 GB/s; floor at 307 GB/s: 19885.3 µs
- Dispatches: 2272 (floor x 1.4 µs = 3180.8 µs); barriers: 1789
- gdn rollup (sum of gdn.*): 17821.3 µs, 40.9%, 2716.14 MB
- moe rollup (sum of moe.*): 10969.8 µs, 25.2%, 1217.99 MB
- Normal-mode decode gpu_s: 42053.2 µs; sum of groups / normal: 1.0368

| op | format | calls/token | µs/token | share | weight MB/token | GB/s |
|---|---|---:|---:|---:|---:|---:|
| gdn.proj | MIXED | 36 | 6402.4 | 14.7% | 1639.711 | 256.11 |
| gdn.scan | NONE | 36 | 3710.5 | 8.5% | 0.000 | 0.00 |
| gdn.hc_pre | MIXED | 36 | 3354.9 | 7.7% | 474.808 | 141.53 |
| moe.down | MIXED | 48 | 3330.5 | 7.6% | 309.658 | 92.98 |
| moe.gate_up | MIXED | 48 | 3259.3 | 7.5% | 405.504 | 124.41 |
| gdn.out | Q8_0 | 36 | 2651.0 | 6.1% | 601.620 | 226.94 |
| lm_head | Q8_0 | 1 | 2499.8 | 5.7% | 675.430 | 270.20 |
| qsa.core | MIXED | 12 | 2357.2 | 5.4% | 0.000 | 0.00 |
| moe.shared_in | MIXED | 48 | 1994.5 | 4.6% | 419.267 | 210.21 |
| qsa.proj | Q8_0 | 12 | 1804.5 | 4.1% | 455.393 | 252.36 |
| hc_ffn.down | F16 | 48 | 1662.1 | 3.8% | 314.573 | 189.26 |
| hc_ffn.up | F16 | 48 | 1492.5 | 3.4% | 314.573 | 210.77 |
| moe.route | NONE | 48 | 1303.4 | 3.0% | 0.000 | 0.00 |
| moe.shared | MIXED | 48 | 1082.1 | 2.5% | 83.558 | 77.22 |
| qsa.out | Q8_0 | 12 | 817.3 | 1.9% | 200.540 | 245.36 |
| gdn.conv | NONE | 36 | 748.0 | 1.7% | 0.000 | 0.00 |
| hc_add | NONE | 60 | 615.2 | 1.4% | 0.000 | 0.00 |
| hc_ffn.hc_read | NONE | 48 | 511.0 | 1.2% | 0.000 | 0.00 |
| ple | MIXED | 1 | 486.4 | 1.1% | 34.826 | 71.59 |
| hc_ffn.rms | NONE | 48 | 476.9 | 1.1% | 0.000 | 0.00 |
| hc_attn.down | F16 | 12 | 391.9 | 0.9% | 78.643 | 200.69 |
| gdn.add | NONE | 36 | 380.9 | 0.9% | 0.000 | 0.00 |
| hc_ffn.inject | F16 | 48 | 366.8 | 0.8% | 3.932 | 10.72 |
| hc_attn.up | F16 | 12 | 366.6 | 0.8% | 78.643 | 214.55 |
| gdn.prep | NONE | 36 | 353.5 | 0.8% | 0.000 | 0.00 |
| hc_ffn.hc_low | NONE | 48 | 221.3 | 0.5% | 0.000 | 0.00 |
| gdn.norm | NONE | 36 | 220.2 | 0.5% | 0.000 | 0.00 |
| hc_ffn.hc_inject | NONE | 48 | 204.9 | 0.5% | 0.000 | 0.00 |
| hc_attn.hc_read | NONE | 12 | 127.0 | 0.3% | 0.000 | 0.00 |
| hc_attn.rms | NONE | 12 | 121.8 | 0.3% | 0.000 | 0.00 |
| hc_attn.inject | F16 | 12 | 92.8 | 0.2% | 0.983 | 10.59 |
| hc_attn.hc_low | NONE | 12 | 51.5 | 0.1% | 0.000 | 0.00 |
| hc_attn.hc_inject | NONE | 12 | 48.0 | 0.1% | 0.000 | 0.00 |
| final.down | F16 | 1 | 32.5 | 0.1% | 6.554 | 201.65 |
| final.up | F16 | 1 | 30.4 | 0.1% | 6.554 | 215.72 |
| final.hc_read | NONE | 1 | 10.5 | 0.0% | 0.000 | 0.00 |
| final.rms | NONE | 1 | 10.0 | 0.0% | 0.000 | 0.00 |
| embed | BF16 | 1 | 7.2 | 0.0% | 0.005 | 0.71 |
| final.hc_low | NONE | 1 | 4.2 | 0.0% | 0.000 | 0.00 |

qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.
