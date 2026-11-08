# R5.0 — what streaming experts from the SSD would cost (no new weights downloaded)

Date 2026-11-05, M5 Pro 64 GiB, AC, battery 90%, no thermal warning, one process at a time
(a `ds4 --mtp` of the user was stopped before the SSD run). Code: `tools/probe/ssd_bench.c`,
`tools/eval/qwen38_expert_cache_sim.py`, `tools/eval/qwen38_streaming_predict.py`.

## 1. The SSD (`ssd_bench.txt`, `ssd_bench.csv`)

Random reads at random 16 KiB-aligned offsets inside the Q2 file's per-layer n-gram table (95.4 GiB,
cold), `pread` + `F_NOCACHE`, 3 warm-up + 20 measured repetitions of ~512 MiB per configuration.
Median GB/s: 256 KB 1.9 / 3.4 / 5.5 / 8.5 / 10.7 at 1 / 2 / 4 / 8 / 16 threads; 1.4 MB (a Q2 expert)
5.3 / 9.8 / 12.0 / 12.8 / 12.8; **2.6 MB (a Q4K expert) 7.0 / 12.3 / 13.1 / 13.1 / 13.1**; sequential 8 MB
10.3 (1 thread), 13.1 (4). Spread p10-p90 within 3%. Latency p50 of a 2.6 MB read: 384 us at one
thread, 433 us at two. Two threads already saturate the device at ~13 GB/s.

## 2. Routing traces (`route-*`)

Five real greedy generations of the Q2 model with `Q38_ROUTE_LOG` (code, explanation, story,
arithmetic, English), 600 new tokens each, chat template, context 8192: `route.bin` (per decode
token, 48 layers x 10 experts), `ids.json`, `runs.jsonl`.

## 3. Cache simulation (`cache_sim.csv`)

Replay of the decode tokens of each trace; the first half warms the cache, the second half is measured.
Cache size in GiB of Q4K experts (2.58 MB each, 24 576 experts, 62 GiB in all). Mean miss rate over the
five traces:

| cache GiB | LRU global | LRU per layer | static (profile from the other traces) | Belady (offline optimum) |
|---:|---:|---:|---:|---:|
| 8 | 0.271 | 0.279 | 0.752 | 0.143 |
| 16 | 0.141 | 0.151 | 0.601 | 0.061 |
| 24 | 0.065 | 0.082 | 0.463 | 0.028 |
| 32 | 0.032 | 0.044 | 0.333 | 0.025 |
| 36 | 0.027 | 0.034 | 0.272 | 0.025 |
| 48 | 0.025 | 0.025 | 0.125 | 0.025 |

From 32 GiB up LRU is within 0.007 of the offline optimum: capacity matters, the policy hardly does. The
floor of 0.025 is the experts a trace had not used before (about 12 new experts per token, even late
in a 600-token generation). A static profile built from other texts is poor (0.27 at 36 GiB): the
experts a text needs depend on the text.

## 4. Predicted decode (`predicted_toks.txt`) — a model, not a measurement

Token time = compute + misses x 2.58 MB / 13 GB/s, 480 expert picks per token. Compute is not measured
for Q4K: 38.5 ms (the Q2 graph), 43 ms (central guess) or 49 ms (conservative). With a 36 GiB cache:
**22 tok/s serial, 23 tok/s with perfect overlap at 43 ms**; 19-24 tok/s across the range. A 16 GiB
cache still predicts 16-19 tok/s, 8 GiB 14-16, 4 GiB 11-13.

## Caveats

- The traces are single-topic and short; topic changes in a long session raise the miss rate, and the
  first tokens of a session pay the cold fill (36 GiB at 13 GB/s is ~3 s, about 300 tokens to warm).
- The routing is the Q2 model's; the Q4K model routes slightly differently.
- A prefill chunk touches nearly every expert of a layer, so prefill streams the whole 62 GiB per chunk
  (~5 s at 13 GB/s per chunk): larger chunks and overlap are an R5.2 topic, not covered here.
- The Q4K compute time and its memory budget (non-expert weights ~8 GiB + cache + graph) are R5.3 facts.

**Gate (predicted best case under 3 tok/s stops the stage): passed with a wide margin.**
