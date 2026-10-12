# Qwen3.8 Q2: local DwarfStar baseline

Started 2026-10-11; pilot completed on 2026-10-12. This is phase 0 of the
[handoff](qwen38-q2-session-prompt.md), with Qwen3.8-Flash-Next Q2 unchanged as
the selected target. It is a pilot, not a performance comparison or proof of
compression quality. Native Qwen3.8 inference is not implemented in picoforge.

## Recorded inputs

Evidence lives in [`bench/qwen38_q2_20261011_phase0/`](../bench/qwen38_q2_20261011_phase0/).
`manifest.json` records the initial picoforge revision and hashes of the five
pre-existing modified/untracked handoff documents. Those files are preserved.

The [local results page](../bench/qwen38_q2_20261011_phase0/report.html) embeds
the measured summary, grading and original answers. It works offline and
does not start inference. Regenerate it from the recorded evidence with:

```sh
python3 tools/eval/render_qwen38_report.py \
  bench/qwen38_q2_20261011_phase0 bench/qwen38_q2_20261012_storage \
  bench/qwen38_q2_20261011_phase0/report.html
```

| Input | Pinned value |
|---|---|
| picoforge initial HEAD | `7e3156a93503eb0785004993ece4d32278a7167d` |
| DwarfStar | `fc80bd695da76ee14bcb86820bf900a5cdfda806` |
| Original Qwen tokenizer/config | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Q2 artifact revision | `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe` |
| Artifact | `antirez/qwen3.8-flash-next-gguf/Qwen3.8-Flash-Next-Q2.gguf` |
| Published bytes | 147,207,127,040 |
| Published SHA-256 | `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9` |

The runtime checkout is `/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline`,
outside picoforge. Its upstream `main` matched the pinned revision at inspection.
The checkpoint downloads to the existing ignored `models/` tree, with no second
full Hugging Face cache copy. Do not treat `.part` as a runnable checkpoint.

Hardware was probed again: Apple M5 Pro, 68,719,476,736 physical bytes,
55,662,788,608 recommended Metal working-set bytes, 41,747,087,360 maximum
buffer bytes, and 32,768 threadgroup bytes. The machine uses macOS 27.0.1
(26A434), SDK 26.5 and Metal compiler 32023.883. AC power was present. No
system memory limit, SDK, dependency or model selection was changed.

## Reproduce acquisition and preflight

From picoforge, the exact resumable transfer is in `download-command.txt`.
It uses system curl and caffeinate. The first transfer ended early at
71,263,952,385 bytes (curl exit 18, which its default retry does not retry).
The recorded recovery uses the helper below; each retry derives its resume
offset from the updated partial file. Keep `.part` to resume; no other model
needs to be deleted:

```sh
caffeinate -i tools/venv/bin/python tools/eval/download_qwen38_q2.py \
  models/Qwen3.8-Flash-Next-Q2 bench/qwen38_q2_20261011_phase0
```

The interrupted session subsequently stopped at 125,166,878,209 bytes. On
2026-10-12 no transfer process remained; the same helper resumed that partial.
`resumed-preflight.json` records the new hardware, power and idle-runtime check.

After the transfer finishes:

```sh
tools/venv/bin/python tools/eval/verify_qwen38_download.py \
  models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf.part \
  bench/qwen38_q2_20261011_phase0/checkpoint.verification.json
```

This hashes every byte and publishes the `.gguf` name only on an exact match.
The report records local file identity, hash and verification duration. Hashing
reads the entire file: a later fresh-process load is not a controlled
storage-cold experiment. We do not purge caches or change system settings.

`inspect_qwen38_gguf.py` reads at most 128 MiB of header data and no tensor
payload. It validates rank, quantization block stride, tensor bounds, alignment,
duplicate names and overlaps. The early `inventory-prefix.json` explicitly
labels its missing payload as unverified; it is an inventory, not a checksum.
It finds 1,256 tensors, including mixed IQ2_XXS, Q2_K, Q4_K, MXFP4, Q8_0, F16,
BF16 and F32. The BF16 `per_layer_token_embd.weight` table starts at byte
44,806,635,520 and occupies 102,400,491,520 bytes (95.37 GiB).

```sh
make
make test-all
make build/process_memory build/memory_budget
make -C ../ds4-qwen38-baseline -j6
make -C ../ds4-qwen38-baseline test-qwen4-kernels test-qwen4-q2 test-qwen4-ngrams
tools/venv/bin/python tools/eval/build_dwarfstar_probe.py \
  ../ds4-qwen38-baseline build/dwarfstar-observed
```

Use a new build directory if the observation build already exists. The builder
requires the pinned, unchanged upstream sources. It leaves that checkout and
its stock binaries untouched; source copies and objects are under `build/`.
The small observation patch is retained alongside the evidence. Upstream code
remains under [DwarfStar's MIT license](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/LICENSE);
the model weights have the separate Qwen Community License.

## Frozen pilot and execution

`suite/suite.json` freezes original local tasks, exact prompts, hashes, output
budgets and seed 3811. They are not calibration data or a standard leaderboard.
Tasks cover interval merging, C binary search, a two-file Python bug fix,
Italian modular arithmetic, and a long archive with a correction. The archive
has 7,614 tokens after the official template; its earlier over-budget draft is
retained separately and was shortened before observing any model output.

The settings are one session, total context 8,192, prefill chunks 1,024, greedy
generation, empty system message, and no MTP or expert streaming. The reasoning
case uses the model's normal thinking template; the other tasks disable it.
Each task creates one model process, then four independent sessions in sequence.
The first request is recorded as first-use/warmup; three hot requests contribute
to pilot medians and nearest-sample p10/p90. Task order is shuffled and recorded.
The public `ds4_session_gpu_warmup` call is a no-op for Qwen at this revision;
the full first request is therefore the warmup excluded from hot statistics.

After the download, checksum and tests have finished, with other inference
workloads idle, run into a **new** output directory:

```sh
caffeinate -i tools/venv/bin/python tools/eval/run_qwen38_pilot.py \
  bench/qwen38_q2_20261011_phase0 ../ds4-qwen38-baseline \
  bench/qwen38_q2_20261011_phase0/pilot
```

The runner first calls the stock CLI as a generation control, then the measured
API client. It finishes with a stock `ds4-bench` control at 8,063 prompt tokens
plus 128 teacher-forced decode tokens within an 8,192-token allocation. That
single throughput control is separate from the task measurements.

The independent tokenizer check uses the original pinned `tokenizer.json` and
`tokenizer_config.json`, fetched into `build/`; the existing Transformers
environment renders the official template. `hf-prompt-tokens.json` holds the
expected IDs before generation. Compare every ID with the DwarfStar outputs:

```sh
tools/venv/bin/python tools/eval/check_qwen38_tokens.py \
  bench/qwen38_q2_20261011_phase0/suite build/qwen38-tokenizer.json \
  build/qwen38-tokenizer-config.json --pilot bench/qwen38_q2_20261011_phase0/pilot
tools/venv/bin/python tools/eval/summarize_qwen38.py \
  bench/qwen38_q2_20261011_phase0/pilot bench/qwen38_q2_20261011_phase0/summary.csv
```

Review generated source before running `grade_qwen38.py SUITE CASE RESPONSE`.
It runs executable Python/C tests, including C sanitizers, in temporary
directories. This is a correctness scaffold, not a security sandbox. Score
each task once; timing repetitions do not add independent quality examples.

## Measurement boundaries and limits

- The API client measures engine open, session creation, GPU warmup, prompt
  prefill, time to the first emitted token, decode and complete session request.
  First-request TTFT and response completion are also measured from process
  launch, including loading. The enclosing process timer covers all four
  sessions plus initialization and teardown. All timers
  use `CLOCK_MONOTONIC`; inconsistent/nonfinite timings invalidate the summary.
- Decode throughput is `(generated - 1) / decode_s`; actual evaluation counts
  and stop conditions are also retained. Greedy IDs must match across all
  repetitions, including the first-use request.
- `process_memory.c` samples physical footprint, lifetime peak, RSS and process
  disk I/O every second, together with system pressure and swap counters.
  These are distinct metrics. Short peaks after the final sample can be missed.
- Observation builds report actual Metal `currentAllocatedSize` at session
  boundaries as well as the runtime's own allocation accounting.
- N-gram counters record successful pread bytes, calls, errors and summed
  service time across threads. This is neither SSD physical traffic nor the
  critical-path I/O latency. Process disk counters include all model I/O.
- Instrumentation has overhead. Stock controls are retained; a three-repeat
  pilot cannot justify a performance optimization or establish BF16 quality.
- The monitor terminates its own child after three critical-pressure samples
  or swap growth above 2 GiB. It retains evidence and does not close other apps.

## Local pilot results

Completed on 2026-10-12. The complete local checkpoint matches the published
SHA-256; verification took 54.25 s. All seven inference processes completed:
the stock CLI control, five task processes with four sessions each, and the
stock throughput control. The full generated token sequences match across
all four repetitions of each task. Every prompt ID also matches the original
Hugging Face tokenizer and template. The stock CLI and observed API client
produce the same complete C answer, ignoring only a trailing newline.

These are the **three-hot-request medians**; the first request is excluded:

| Task | Prompt tokens | Output tokens | Prefill s | Decode tok/s | TTFT s | Full request s |
|---|---:|---:|---:|---:|---:|---:|
| Two-file fix | 173 | 172 | 0.670 | 32.34 | 0.722 | 6.001 |
| Archive correction | 7,614 | 23 | 12.595 | 29.99 | 12.647 | 13.379 |
| C lower bound | 126 | 122 | 0.532 | 31.96 | 0.587 | 4.373 |
| Merge intervals | 96 | 253 | 0.406 | 31.72 | 0.456 | 8.400 |
| Italian arithmetic | 142 | 850 | 0.609 | 32.09 | 0.661 | 27.120 |

The stock `ds4-bench` control reports **600.68 prefill tokens/s for 8,063
tokens** (about 13.42 s), and **30.88 tokens/s** for 128 teacher-forced decode
tokens. Its `kvcache_bytes=0` CSV field is not a measured absence of cache:
the runtime separately reports 1,504.29 MiB of context buffers. Both initial
speed targets are met in this pilot. Three repeats do not support optimization
decisions; the recorded p10/p90 are only descriptive pilot samples.

The first stock process emits its first stdout bytes at 9.896 s and finishes
in 13.73 s. Observed API engine-open times range from 0.400 to 2.878 s after
that control has run. Per-task first-request TTFT and completion times, which
include process startup/loading, are retained in `summary.csv`. None of these
loads is a controlled cold-storage measurement.

Peak sampled Metal allocation is **49,940,365,312 bytes (46.511 GiB)**. Task
physical-footprint lifetime peaks are 3.715–3.766 GiB; the initial stock CLI
reaches 5.873 GiB. Footprint/RSS alone clearly do not describe all checkpoint
residency here. These accounting categories overlap and must not simply be
added. System-wide wired memory peaks at 51.079 GiB during the initial control,
then below 47.756 GiB in the task processes; those totals also include macOS
and other applications. Memory pressure remains normal. Swap grows by
**5.5 MiB globally**, in two isolated samples (0.5625 MiB in the stock CLI,
4.9375 MiB in the arithmetic process), with no sustained growth observed.
System counters cannot identify the process responsible for those writes.

N-gram pread counters report zero read errors. Across all four archive
requests they record 162,368 calls, 49.55 MiB of successful reads and 18.343 s
of summed service time. This includes concurrent threads and cached reads;
it is not 18.343 s of request latency or measured physical SSD traffic.

After reviewing generated source, all three coding answers pass their
executable tests, and the arithmetic answer is correct. The long archive
answer contains both correct facts but is fenced Markdown, so the unchanged
strict JSON grader rejects it. Merge intervals and C lower bound also use
fences despite the instructions. Thus **4/5 pass the frozen task tests and
2/5 satisfy both tests and output format**. Do not turn these five original
examples into a leaderboard score or a Q2-versus-BF16 quality claim.

Raw answers, commands, streams and memory samples are under `pilot/`;
`summary.csv`, `summary.json`, `grading.json`, `tokenizer-parity.json` and
`stock-control-parity.json` retain the checks. Small CPU-only parser builds
and checks occurred during this pilot; no other GPU workload ran. A dedicated
25-repeat performance comparison remains future work.

The next verified increment is [native GGUF storage](qwen38-storage.md).
Native hybrid operators, logits/state references and the complete Metal
generation graph are still required; the DwarfStar baseline is not that engine.
