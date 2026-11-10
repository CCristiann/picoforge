# R5.2b0 — where a streamed token's time goes, 2026-11-06

Real Q2, 64 greedy tokens, `Q38_REF_KERNELS=gemm`, guarded launcher on AC, 2 alternating pairs:
resident against a 50% pool (`PICOFORGE_STREAM_EXPERTS=12288`, 8 readers, no eviction). Same logits.

Resident decode: 39.2 ms/token. Streamed: 61.3-61.9 ms/token. Total time in `q38_pool_acquire` is
0.79-0.81 s, prefill included (the misses are cold first reads). That is under 13 ms/token of the
~22 ms gap. The rest comes from the 48 per-layer syncs: commit, wait, then encode the rest of the
layer while the GPU idles. Conclusion for R5.2b: overlapping only the reads cannot close the gap.
The sync has to stop serialising the GPU (pre-encoded work released by an event, or a GPU-side
expert->slot table). 2 pairs; not the 3+20 protocol.
