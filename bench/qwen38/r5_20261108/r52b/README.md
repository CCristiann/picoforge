# R5.2b — event gates instead of commit-and-wait, 2026-11-08

Real Q2, one chat prompt, 64 greedy tokens, guarded launcher on AC, while the Q4 GGUF was downloading
(~48 MB/s of writes; the Q2 was partly out of the page cache, so cold runs read more from the SSD).

**Steady state** (`run-warm.sh`, `warm-runs.tsv`): one process per mode, `--warmup 2 --repeat 5`, pool at
100% (`PICOFORGE_STREAM_EXPERTS=24576`), so after the warm-up nothing misses and only the sync costs.
Decode ms/token over the 5 measured repetitions:

| set | mode | min | median | max |
|---|---|---|---|---|
| run1 | resident | 38.5 | 38.7 | 39.7 |
| run1 | wait (R5.1b commit+wait) | 49.3 | 49.5 | 49.9 |
| run1 | event, MTLSharedEventListener | 46.3 | 46.4 | 46.5 |
| run2 | event, polling queue (default) | 44.4 | 44.4 | 44.5 |
| run2 | event, listener (`PICOFORGE_STREAM_NOTIFY=1`) | 46.4 | 46.5 | 46.5 |

The streaming overhead drops from 10.8 to 5.7 ms/token, ~120 us per gate (48 per token): command-buffer
boundary and event propagation to the GPU, not CPU work.

**Cold** (`run-cold.sh`, `cold-runs.tsv`): 50% pool, 2 alternating pairs wait/event (listener version),
decode 4.67-4.77 s against 4.23-4.51 s for 63 tokens; dominated by cold reads (1.7 s in acquire).
All runs give the logits of the resident run (sha256 `b07822bc...`, the same as R5.2a/R5.2b0).
Not the 3+20 protocol.
