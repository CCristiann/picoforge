# Roadmap stage R2 "Terminal" — closure evidence (2026-11-01/02)

Branch `feat/qwen38-terminal` (from `main` 55cc4ef). Tasks T-R2.1..T-R2.6 in
`docs/devlog.md`. The decode path is unchanged: only host orchestration.

## Gates (at 05d88fd, battery, `ALLOW_BATTERY=1`, swap 0 before and after)

| Gate | Result |
|---|---|
| `make test-qwen38` (incl. new `test-qwen38-repl`, 10 tests then) | PASS |
| `test-qwen38-mtp`, `-speculative`, `-fast-kernels`, `-compare` | PASS |
| `Q38_REF_KERNELS=1 make test-qwen38-fast-kernels` | PASS |
| G3 (`make qwen38-g3`) | PASS, scaled 2.08e-7 (R2 close: 2.08e-7) |
| Oracle, `Q38_REF_KERNELS=1` logits vs R0 oracle (360acf0) | BIT-IDENTICAL, sha256 36e67ce6… |
| Oracle, default path | max abs 4.95e-5, scaled 1.96e-6, argmax 9/9 (R2 close: same) |

3fa96fb (after the gates) only adds an `fflush(stdout)` after each chat
command plus its test (`test-qwen38-repl` 11/11, `test-qwen38-cli` 7/7); it
cannot change any numeric path.

## Recorded chat session (at 3fa96fb, AC charging)

`transcript.md`: one real session on Flash-Next Q2, stdin piped by the
controller helper, through the same guarded launcher as every real run
(release wait before/after, memory guard; raw logs in `run/`). It shows
`/help`, two answered turns, `/read note.txt`, a long poem stopped with
Ctrl+C (SIGINT) mid-reply, a follow-up that knows what was cut, `/think`
with the reasoning then the answer, `/nothink`, `/ctx 2048` (new graph,
new conversation) and `/quit`. Generation 23.4–24.4 tok/s per turn.

A first recording (at 05d88fd) showed command output one turn late: a piped
stdout is fully buffered. Fixed in 3fa96fb and recorded again; the first
run is not kept.

## Short bench (NOT the protocol: battery, 1 warm-up + 3 runs)

`picoforge bench -m <gguf> --warmup 1 --repeat 3 --weight-mb 6104.78`:
decode 24.37 tok/s median (R2 close, paired AC protocol: 24.15), 148.8 GB/s
= 48.5% of 307 (6104.78 MB/token from the R2 profile summaries). With
`--mtp` on the story prompt: 18.27 tok/s, slower than plain — low draft
acceptance on free text (adaptive MTP is roadmap stage R4).

Personal paths are redacted to `/path/to/`.
