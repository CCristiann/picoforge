# picoforge

A small, readable LLM inference engine in C11 and Metal for Apple Silicon.
Tokenizer, CPU reference path, GPU forward pass, quantisation, MoE routing and
speculative decoding, with no ML framework in the engine. Python appears only
in `tools/` for reference implementations, conversion and tests.

The goal is to understand where inference time and memory go, so correctness
comes first and every number is measured. All results come from one machine
(M5 Pro, 64 GiB) and make no claim of a lead over other runtimes.

## What it runs

| Model | Status |
|---|---|
| Qwen3-0.6B (dense) | CPU and Metal; bf16, q8 and q4 weights; regression reference |
| Qwen3-30B-A3B (MoE) | Metal with bf16/q8_row experts (~31 GB), greedy speculative decoding |
| Qwen3.8-Flash-Next Q2 | Full 48-layer FP32 graph on Metal, read directly from GGUF; optional MTP |

## Build and run

Requires an M5-family Mac, macOS 26.2+, Xcode with the Metal 4 toolchain and
Python 3.12+ for the tooling. Weights live in `models/` and are never committed.

```sh
make
python3 -m venv tools/venv
tools/venv/bin/pip install -r tools/requirements.txt
tools/venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

./picoforge models/Qwen3-0.6B --gpu-chat "Why does a cache help?"
./picoforge --context 4096 models/Qwen3-0.6B --gpu-chat "Explain memory bandwidth." 128
make test-all
make ui                       # local UI at http://127.0.0.1:8000
```

Qwen3.8 runs from its GGUF file (8K context, greedy, thinking off by default):

```sh
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  ./picoforge --qwen38 models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --chat "What is 2 + 2?" --max-new 32
make ui-qwen38                # resident chat at http://127.0.0.1:8001
```

Useful flags: `--mtp` (multi-token prediction with exact greedy verification),
`--thinking`, `--complete TEXT`, `--logits FILE.f32`, `--token-ids FILE.json`.

## How correctness is checked

A chain of oracles: Transformers → NumPy reference → C CPU path → Metal.
Speculative and MTP paths must produce exactly the same tokens as plain greedy
decoding. Attention is checked against an fp64 reference at long context, and
the Qwen3.8 gate compares all 48 layer outputs and logits against an
independent FP32 reference ([G2 spec](docs/specs/qwen38/G2.md)).

## Selected measurements

| Experiment | Result | Details |
|---|---|---|
| Speculative decoding, Qwen3-30B-A3B + 0.6B draft | 1.41× (k=2) / 1.43× (adaptive) decode; outputs identical to greedy | [protocol](docs/BENCHMARKS.md), [CSV](bench/spec_qwen3_30b_20261011_pilot.csv) |
| Expert-cost-aware drafting (`auto-experts`) | 1.009× decode vs `auto`, below the adoption threshold | [writeup](docs/phase4-expert-experiment.md) |
| Expert reuse across 8 tokens (30B) | 29.8 distinct experts/layer vs 51.6 uniform | [CSV](bench/routing_overlap_qwen3_30b_20261011.csv) |
| Qwen3.8 Q2 decode on GPU (R1) | 20.7 tok/s short (0.62× DwarfStar), 11.7 tok/s at 8K (0.37×); 19.5× over R0 | [R1 close](bench/qwen38_r1_20261028/README.md), [ladder](bench/qwen38_r1_20261019/ladder.csv) |
| Qwen3.8 native FP32 decode | ~1.07 tok/s ordinary; MTP 1.19–1.57× on two short pilots | [MTP results](bench/qwen38_mtp_20261013/README.md) |

These are small pilots, not general claims. Raw CSVs, logs, hashes and the
plotting scripts are committed under `bench/`; the
[native report](bench/qwen38_native_20261013/report-mtp-final.html) is a
self-contained HTML page (`python3 -m http.server --directory bench`).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Engine sources and `.metal` kernels |
| `tools/` | Python oracles, converters, benchmark runners, local UI |
| `tests/` | Correctness harnesses and fixtures |
| `bench/` | Measured data, run evidence and plots |
| `docs/` | [Design](docs/DESIGN.md), [benchmark protocol](docs/BENCHMARKS.md), [devlog](docs/devlog.md), specs |

## License

MIT; see [LICENSE](LICENSE). Qwen3 weights are Apache-2.0; Qwen3.8 weights use
the Qwen Community License (provenance in
[the baseline report](docs/qwen38-q2-baseline.md)). The small-runtime approach
was inspired by [DwarfStar](https://github.com/antirez/ds4).
