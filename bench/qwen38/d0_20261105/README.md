# D0 — ablation of the fused IQ2_XXS gate+up kernel

Question: is the routed MoE (3.28 ms/token at 124 GB/s) limited by ALU work per weight?
Bench: `gbench fused` of `tests/test_qwen38_fast_kernels.m` (synthetic weights, 512 experts,
n 640, K 2560, 20 reps per command buffer), `Q38_PACKED_LIBRARY` pointing at metallibs built
from a scratch copy of `qwen38_moe_fused.metal` / `qwen38_quant_decode.h` with `-DA_*`
switches (`ablation.patch`; numerically wrong on purpose, timing only, nothing committed in
src/). Runs in ABBA order under the GPU lock, AC power, battery 30% (`conditions.txt`).
The 40-slot rows are steady; 10-slot rows are shorter and noisier. Raw lines: `ablate-*.jsonl`.

| Variant (40 slots) | median us | GB/s | vs baseline |
|---|---:|---:|---:|
| baseline | 237.8 | 142.7 | - |
| no sign work | 216.8 | 156.6 | -9% |
| no scale multiply | 229.6 | 147.8 | -3% |
| no grid-table lookup | 237.9 | 142.7 | 0% |
| no sign, scale, table | 216.9 | 156.5 | -9% |
| x loaded as float4 (same math) | 238.2 | 142.5 | 0% |
| no x loads | 214.9 | 157.9 | -10% |
| no decode ops and no x loads | 216.8 | 156.5 | -9% |
| one 8-byte group load | 237.7 | 142.8 | 0% |
| **streaming only** (same rows, same geometry, uint4 contiguous loads, no decode) | **106.4** | **318.9** | **-55%** |

Reading: all the arithmetic together is worth at most ~10%. The same bytes with the same
dispatch geometry stream at ~319 GB/s (the data partly sits in the system cache, so this is
an upper bound of the access pattern, not a DRAM claim). The kernel's remaining 2.2x is its
structure: how loads are issued and overlapped per simdgroup (little work and few loads in
flight per lane), not the decode math. 10-slot figures agree in direction (54 vs 62 us).
