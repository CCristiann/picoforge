# Measurements

Every file here was measured on the reference machine (MacBook Pro, M5 Pro,
64 GB) under the protocol in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).
These are local results. No other machine or independent reproduction has
been run.

| Folder | Contents |
|---|---|
| [`qwen38/`](qwen38/) | Qwen3.8-Flash-Next: one folder per run, named `<stage>_<YYYYMMDD>` |
| [`microbench/`](microbench/) | Model-agnostic matmul, quantised-matmul and dispatch-cost sweeps (`tools/microbench`) |
| [`archive-qwen3/`](archive-qwen3/) | Results of the removed Qwen3-0.6B / 30B-A3B engine, described in [`docs/history.md`](../docs/history.md) |

## Qwen3.8 runs by stage

The folder that closes a stage holds its summary README. The other dates hold
that stage's intermediate steps, described in [`docs/devlog.md`](../docs/devlog.md).

| Stage | Closing evidence | Intermediate runs |
|---|---|---|
| Phase 0: DwarfStar baseline, checkpoint verification | [`q2_20261011_phase0`](qwen38/q2_20261011_phase0/) | — |
| Native GGUF storage | [`q2_20261012_storage`](qwen38/q2_20261012_storage/) | `gpu_lifecycle_20261012` |
| Native components and integration (A–F) | [`integration_v2_20261012`](qwen38/integration_v2_20261012/README.md) | `integration_20261012` |
| G1 CPU graph, G2/G3 GPU graph | [`g2_g3_20261013`](qwen38/g2_g3_20261013/README.md) | `g1_20261013`, `g2_d_20261013` |
| Extended native suite, first MTP | [`native_20261013`](qwen38/native_20261013/README.md), [`mtp_20261013`](qwen38/mtp_20261013/README.md) | — |
| R0 Measure | [`r0_20261017`](qwen38/r0_20261017/README.md) | — |
| R1 Decode | [`r1_20261028`](qwen38/r1_20261028/README.md) | `r1_20261019` … `r1_20261027` |
| R2 Decode gap | [`r2_20261030`](qwen38/r2_20261030/README.md) | `r2_20261028`, `r2_20261029` |
| R2 Terminal | [`terminal_20261101`](qwen38/terminal_20261101/README.md) | — |
| R3 Prefill | [`r3_20261102`](qwen38/r3_20261102/README.md) | — |
| R4 MTP | [`r4_20261103`](qwen38/r4_20261103/README.md) | — |
| Decode II | [`d1_20261105`](qwen38/d1_20261105/README.md) | `d0_20261105` |
| R5 SSD streaming | [`r5_20261105`](qwen38/r5_20261105/README.md) | `r5_20261106`, `r5_20261108` |
| P3 Chat path | [`p3_20261108`](qwen38/p3_20261108/README.md) | — |
| P4 GDN scan, MoE tiles | [`p4_20261108`](qwen38/p4_20261108/README.md) | — |
| P5 Long-prefill attention | [`p5_20261109`](qwen38/p5_20261109/README.md) | — |
| R7 Quality | [`r7_20261110`](qwen38/r7_20261110/README.md) | — |

## What is kept, and what is in the archive tag

Each run keeps what is needed to check its claim: CSVs, `runs.jsonl`, summary
and report JSON, token IDs, conditions, scripts and READMEs. The routing traces
in `r5_20261105/route-*` stay too, because `tools/eval/qwen38_expert_cache_sim.py`
reads them.

On 2026-11-10 the per-process raw evidence was removed from the tree: stdout and
stderr logs, memory and event samples, release-wait records, logit dumps
(`*.f32`) and route dumps (`*.u32`), about 95 MB. All of it is still in git at
the tag **`archive/bench-raw-20261110`**, under the old flat paths
(`bench/qwen38_<stage>_<date>/`):

```sh
git checkout archive/bench-raw-20261110 -- bench/qwen38_d1_20261105   # restore one run
```

Some older READMEs link files that are now only in that tag, or the
development specs, which are not published.
