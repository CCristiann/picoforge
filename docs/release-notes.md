# Draft release notes

## Safer attention and repeatable generation measurements

picoforge is a small Qwen3 inference engine in C11 and Metal, tested on an
M5 Pro. It runs dense and MoE models, with a CPU reference path and greedy
speculative decoding.

This update fixes an attention buffer overrun when the context exceeded
2,048 tokens. Attention now allocates score storage for the active context
and rejects requests that exceed GPU capacity. Tests cover 2,048, 2,049 and
4,096 positions. The CLI accepts `--context N` before the model directory.

The speculative benchmark now keeps every run and checks its tokens against
ordinary greedy. It reports prefill, decode, total process time and peak
process RSS separately, with checkpoint and source hashes. No dependency was
added to the engine or benchmark runner.

On ten local prompts with five repetitions each, adaptive drafting measured
1.427x decode speedup and 1.131x full-process speedup over no draft. Fixed
`k=2` reached 1.412x and 1.099x. All 180 measured and warmup outputs matched
ordinary greedy. The adaptive margin over fixed drafting is small; this pilot
does not establish a stable advantage.

The corrected 30B routing trace touches 29.8 distinct experts per layer for
eight consecutive tokens, compared with 51.6 under uniform routing. The old
trace crossed the attention buffer limit and its locality estimate is withdrawn.

The current scheduler adapts draft length from measured timings and acceptance.
Expert prediction is still planned. This update makes no speed claim against
MLX, llama.cpp or other runtimes.

Build instructions are in the README. Measurement commands and limits are in
`docs/BENCHMARKS.md`. GPU MoE experts support bf16 and q8_row; Q4 MoE, GPU
checkpoint shards and a resident inference server are outside this release.
