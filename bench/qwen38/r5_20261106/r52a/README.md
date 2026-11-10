# R5.2a — parallel miss reads, 2026-11-06

Real Q2, `PICOFORGE_STREAM_EXPERTS=2458` (10% pool, 3.4 GiB), 64 greedy tokens of one chat prompt,
`Q38_REF_KERNELS=gemm`, guarded launcher on AC, 3 alternating pairs of `PICOFORGE_STREAM_READERS=1`
(serial, the R5.1 code path) and `=8` (default). Script `run.sh`, numbers `runs.tsv`.

| readers | stall in acquire (s) | per miss (us) | decode s / 63 tok | ms/token |
|---|---|---|---|---|
| 1 | 6.168-6.201 | ~470 | 8.57-8.78 | ~138 |
| 8 | 2.011-2.013 | ~153 | 4.83-5.11 | ~77 |

Same misses (13140), evictions (10682) and logits (one sha256 across all six runs) in both modes.
An expert is ~1.49 MB in the Q2, so a miss reads 3.2 GB/s serial and 9.7 GB/s with 8 readers
(R5.0 peak 13.1). About 13.7 of the 18.2 GiB came from the SSD; the rest was still in the page
cache from earlier runs (no `purge` without sudo). 3 pairs, not the 3+20 protocol; resident decode on
this binary was ~41.5 ms/token (R5.1c). The 10% pool misses 33% of lookups on this prompt, so it is
the stress case, not a usable configuration.
