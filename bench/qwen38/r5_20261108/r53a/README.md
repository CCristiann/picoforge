# R5.3a — the Q4 file runs, streamed, and matches the FP32 reference, 2026-11-08

Checkpoint: `antirez/qwen3.8-flash-next-gguf` revision d600fe1, `Qwen3.8-Flash-Next-Q4.gguf`,
177280286720 bytes, full sha256 `680944…0f25a8a` (equal to the Hugging Face LFS oid; `docs/weights.lock`).
Only the routed experts differ from the Q2: Q4_K gate/up and MXFP4 down (2.71 MB per expert, 62.1 GiB in the
trunk); every other tensor has the Q2's type. Those are the formats of the MTP layer that already ran.

Method (`run.sh`, helpers `launch-any.py`, `compare-logits.py`): the G3 tokens (12), DwarfStar fc80bd6
`ds4 --cpu --first-token-test` as the FP32 reference on the same file, picoforge `--tokens ... --teacher-forced 11`
(12 logits rows). Every process went through the guarded launcher under the GPU lock, on AC.

| model | run | scaled error vs FP32 | argmax | top-5 order | peak wired GiB |
|---|---|---|---|---|---|
| Q2 | resident | 2.08e-7 | 12/12 | 12/12 | 45.3 |
| Q2 | pool 2458 | 2.08e-7 | 12/12 | 12/12 | 14.6 |
| Q4 | pool 2458 (6.2 GiB) | 2.31e-7 | 12/12 | 12/12 | 16.9 |
| Q4 | pool 14244 (36 GiB, 58%) | 2.31e-7 | 12/12 | 12/12 | 46.7 |
| Q4 | pool 14244, `PICOFORGE_STREAM_SYNC=wait` | 2.31e-7 | 12/12 | 12/12 | 46.8 |

Gate 1e-3 (G3), never widened. The three Q4 runs give the same logits bit for bit (sha256 in `runs.tsv`), so do
the two Q2 runs; the Q2 run repeated after the loader change gives the same sha256 (30dcdbf6…). Q4 Metal
allocation with the 36 GiB pool: 43.8 GiB of 51.8 recommended. Memory pressure stayed at level 1 everywhere.
Swap grew from 0.25 to 11.4 MiB only during the DwarfStar CPU reference (it maps the whole 165 GiB file),
and not at all during the picoforge runs.
