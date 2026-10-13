# picoforge

A small Qwen3 inference engine in C11 and Metal, built and tested on an Apple
M5 Pro. It includes a tokenizer, CPU reference path, GPU forward pass, weight
quantisation, MoE routing and greedy speculative decoding. The engine has no
Python or ML framework dependency.

This is a working inference engine and a place to study its costs. The code
keeps one GPU backend. Performance claims refer to the
local measurements linked below; there is no measured lead over other runtimes.

The development target is **Qwen3.8-Flash-Next Q2** on the M5 Pro with 64 GiB.
Its [local DwarfStar baseline](docs/qwen38-q2-baseline.md) is measured. Native
[GGUF storage and multiple no-copy Metal views](docs/qwen38-storage.md) are
implemented and tested. Native Qwen3.8 greedy generation now runs the full
48-layer FP32 graph on Metal, including SSD n-grams and device-side QSA/MoE
selection. [Implementation and correctness evidence](docs/specs/qwen38/G2.md).
The existing Qwen3 models remain the inference and regression reference.

## Build and run

You need an M5-family Mac, macOS 26.2 or later, Xcode and the Metal 4 toolchain.
The default GPU kernels use Metal TensorOps. Python 3.12 or later is needed
for the pinned tooling dependencies; tests here use Python 3.14.2. Python is
used for model download, conversion, tests and the optional local UI.

```sh
make
python3 -m venv tools/venv
tools/venv/bin/pip install -r tools/requirements.txt
tools/venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

./picoforge models/Qwen3-0.6B --gpu-chat "Why does a cache help?"
```

Run commands from the repository root. Weights stay in `models/` and are not
included in the repository. To choose a larger context, put the option before
the model directory:

```sh
./picoforge --context 4096 models/Qwen3-0.6B --gpu-chat "Explain memory bandwidth." 128
make test-all
make ui                       # http://127.0.0.1:8000
```

Qwen3.8 uses its GGUF directly (the n-gram table stays on disk):

```sh
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  ./picoforge --qwen38 models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --chat "Rispondi in italiano: quanto fa 2 + 2?" --max-new 32
```

This path defaults to context 8192, prefill chunk 1024, greedy decoding and
thinking disabled. `--complete TEXT` supplies raw text; `--thinking` enables
the chat reasoning prefix. `--logits FILE.f32` saves one FP32 vocabulary row
per sampling step; `--token-ids FILE.json` saves prompt/output IDs. This is a
correctness baseline with no native performance claim or MTP execution.

For the Qwen3 path, context includes prompt and output and defaults to 1,024
tokens. Its attention is tested through 4,096 tokens; larger requests are
checked against the model limit and GPU threadgroup-memory capacity.

## Supported paths

| Area | Current support |
|---|---|
| Native inference tested | Qwen3-0.6B dense, Qwen3-30B-A3B MoE; synthetic MoE for regression tests |
| Dense GPU weights | bf16, q8_row/q8_g32, q4_row/q4_g32/q4_g64 |
| MoE GPU experts | bf16 or q8_row; Q4 experts are not implemented |
| Qwen3 checkpoints | Hugging Face safetensors; GPU needs a single file, CPU accepts shards |
| Qwen3.8 storage | Bounded GGUF v3 inventory; mixed packed formats; multiple Metal views excluding disk-only n-grams |
| Qwen3.8 inference | Full FP32 Metal graph; PLE, HC/GDN, QSA, routed/shared MoE; bounded prefill, greedy chat/completion, reset/replay |
| Generation | Single-turn chat, sampling, greedy, prompt-lookup or model-based speculation |
| Adaptive drafting | Default uses acceptance and timings; optional `auto-experts` forecasts cost from accepted routes |
| Local UI | Python standard-library server; loads a new engine process for each request |

For the 30B, the local q8_row checkpoint occupies about 31 GB. Conversion from
its original checkpoint is handled by `tools/quant/quantize.py`. A 64 GB machine
was used for the measurements; smaller memory configurations have not been tested.

## Measurements

The [Qwen3.8 results page](bench/qwen38_q2_20261011_phase0/report.html) contains
the memory plot, per-task timings, raw answers and native storage status.
Open the self-contained HTML file directly, or serve the recorded evidence:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory bench
# http://127.0.0.1:8765/qwen38_q2_20261011_phase0/report.html
```

This page displays a completed pilot; it does not launch model generation.

The [October 12 Qwen3.8 Q2 pilot](docs/qwen38-q2-baseline.md#local-pilot-results)
uses DwarfStar at a pinned revision, 8K total context, chunks of 1,024 and no
MTP. Five task medians range from 29.99 to 32.34 decode tokens/s; a separate
stock run prefills 8,063 tokens at 600.68 tokens/s. This is an external runtime
baseline, not picoforge inference performance. Only two of five tasks satisfy
both the frozen correctness and output-format criteria; all three coding
answers pass their executable tests after extracting the code.

The October 11 pilot used ten prompts, five repetitions per configuration and
up to 128 output tokens. All 180 measured and warmup outputs matched ordinary
greedy generation. Geometric-mean speedups over the same loop with no draft:

| Drafting | Decode | Full process |
|---|---:|---:|
| Fixed k=2 | 1.412x | 1.099x |
| Adaptive | 1.427x | 1.131x |

Adaptive is only **1.1% ahead of k=2 in decode** and wins on five of ten prompts.
That margin needs more evidence. The first target forward pass accounts for
much of the difference between decode and request time. See [the CSV](bench/spec_qwen3_30b_20261011_pilot.csv)
and [run notes](docs/BENCHMARKS.md#october-11-pilot).

The [older five-prompt result](bench/spec_qwen3_30b_m5pro.csv) used three
repetitions and retained no individual timings. It remains available as a
historical pilot.

The separate [expert-cost experiment](docs/phase4-expert-experiment.md) adds
an opt-in `auto-experts` mode. Its six-prompt pilot reached only 1.009x decode
speed and 0.988x full-process speed versus `auto`, below the 10% adoption
threshold. It remains experimental; the report retains per-prompt regressions
and the timing-recovery audit trail.

The current runner saves every request, checks token identity against plain
greedy, shuffles configuration order and reports prefill, decode, process time
and peak process RSS separately. Run a comparison with existing checkpoints:

```sh
tools/venv/bin/python tools/eval/bench_speculate.py \
  models/Qwen3-30B-A3B-q8_row models/Qwen3-0.6B \
  bench/my-run.csv 25
```

It writes a summary CSV, raw `.runs.jsonl` and `.meta.json` containing hashes
and run settings. Existing results are never overwritten. Details, timing
boundaries and limitations are in [the benchmark protocol](docs/BENCHMARKS.md).

A 4,096-token routing measurement published earlier crossed a fixed-size
attention buffer. It has been replaced by a tested run. For eight consecutive
tokens the corrected trace touches **29.8 distinct experts per layer**, against
51.6 under uniform routing. See [the corrected CSV](bench/routing_overlap_qwen3_30b_20261011.csv).

## Correctness and next steps

Tests compare Transformers with a NumPy reference, then the C path with NumPy,
then Metal with the C path. Speculative tests compare complete token sequences
with ordinary greedy generation. Long-context tests also compare attention
with an fp64 reference and all router choices with Transformers on the tiny MoE.

Qwen3.8 now has independent operator references, a CPU graph and the ordinary
Metal generation path. The full-checkpoint gate compares all 48 layer outputs
and logits against pinned DwarfStar FP32; its stock Metal FP16 cache comparison
is recorded separately, including a near-tie routing divergence. See the
[G2/G3 report](docs/specs/qwen38/G2.md). Performance tuning and MTP remain outside
this correctness baseline.
The 30B [expert-cost experiment](docs/phase4-expert-experiment.md) remains
available; its pilot does not justify replacing the default scheduler.
The [project review](docs/project-review-2026-10-11.md) records the scope and
related work. [DESIGN.md](docs/DESIGN.md) explains the implementation.

## Layout and license

`src/` contains the engine; `tools/` the reference, conversion and UI tools;
`tests/` the checks; `bench/` the measured data; `docs/` the design and run notes.

MIT; see [LICENSE](LICENSE). Qwen3 weights have their own Apache-2.0 license.
The selected Qwen3.8 weights use the separate Qwen Community License; provenance
and the locally verified artifact hash are in the baseline report.
The small-runtime approach was inspired by [DwarfStar](https://github.com/antirez/ds4).
