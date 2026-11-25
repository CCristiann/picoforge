# picoforge

An LLM inference engine for Apple Silicon in C11 and Metal, small enough to read
end to end. It runs one architecture, **Qwen4** (Hugging Face `qwen4_exp`, GGUF
`qwen4exp`: Gated DeltaNet, sparse attention with an indexer, 512-expert MoE,
hyper-connections, n-gram embeddings), from a quantised GGUF file, entirely on
the GPU and with no ML framework in the engine. Today's checkpoint is
**Qwen3.8-Flash-Next** (48 layers), the public preview of that architecture.
Correctness comes first and every number is measured; the approach follows
[DwarfStar](https://github.com/antirez/ds4).

The loader also takes the upstream variants Flash-Next does not use (no n-gram
table, the SiLU gate in Gated DeltaNet, any order of attention layers), checked
against the reference on synthetic checkpoints (`make test-qwen4-variants`)
until open-weight Qwen4 models exist. A dense model (the announced 27B) is not
supported yet.

## Requirements

- An M5-family Mac with macOS 26.2 or later (prefill uses the GPU Neural
  Accelerators through Metal 4 TensorOps) and 64 GB of unified memory.
- Xcode with the Metal toolchain. Python 3.12+ only for tests and tools.

The Q2 file (137 GiB) keeps ~42 GiB of weights resident and reads its n-gram
tables from the SSD; the Q4 file (165 GiB) streams its routed experts from the SSD.

## Usage

Download a GGUF from
[antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
into `models/` and check it against the pinned hashes:

```sh
cd models && shasum -a 256 -c --ignore-missing ../docs/weights.lock && cd ..
make                                   # builds pf, pf-bench, pf-agent and pf-server
./pf                                   # interactive chat (/help for commands)
./pf -m q2 -p "What is 2 + 2?" -n 32   # one answer, then exit
./pf-bench -m q2                       # 3 warm-up + 20 runs, median and p10/p90
./pf models                            # models found and the name -m takes
./pf-agent -m q2                       # coding agent in the current directory (/help)
./pf-server -m q2                      # OpenAI/Anthropic API on 127.0.0.1:8080 (docs/SERVER.md)
```

The chat samples (temperature 0.6, top-k 20, top-p 0.95) and drafts with
multi-token prediction, which never changes the output; benchmarks and oracles
decode greedily. Context is 8K tokens, thinking is off (`--think`, `/think`). On a terminal the
chat opens with a summary of the model, memory and settings (`/info` shows it again) and closes each
reply with its speeds and a bar of the context used. A pasted block of several lines is one message.
The three commands share their options: `-m`, `-c`, `-n`, `--think` / `--no-think`, `--no-mtp`,
`--temp`, `--power` (thinking is on by default in `pf-agent` and `pf-server`).

| Option | Effect |
|---|---|
| `--prefill-precision half` | Prefill GEMMs with half operands: faster, slightly different logits. Default `fp32`. |
| `--chunk 2048` | Larger prefill batches: ~5% faster on long prompts, +1.2 GB. Default 1024. |
| `--power N`, `/power N` | GPU duty cycle 1–100 for a cooler machine; output unchanged. Default 100. |

Run one process at a time: wired GPU memory takes ~1.5 s to be released after exit.

### Coding agent

`pf-agent` runs the same engine as a coding agent in the terminal: it reads, searches and edits the
files of the directory it starts in, runs commands and looks things up on the web, asking before
anything that changes files or runs a command. A status line shows the context used and the speeds;
sessions can be saved and resumed.

### Server for other agents

`pf-server` serves the model over the OpenAI Chat Completions and Anthropic Messages APIs, with
streaming, tool calls and reasoning, so opencode, Claude Code, Pi and any OpenAI SDK can use it. It
keeps the model's own tokens across stateless requests and checkpoints the graph state on disk, so an
agent's resent conversation is not prefilled again. Setup for each client: [docs/SERVER.md](docs/SERVER.md).

## Code-tuned weights

picoforge also loads its own variant of the Q2: in each of the 48 layers, the experts that code
routes to most are kept at 4 bits (their bytes copied unchanged from the published Q4), and the
rest stay at 2 bits. The ranking was measured on eight languages, and the evaluation uses text it
never saw. Against the Q4 as the reference:

| | Q2 | Q2-Code5 (5% of experts at Q4) | Q2-Code10 (10%) |
|---|---|---|---|
| KL to Q4, unseen picoforge code | 0.288 | 0.217 | 0.167 |
| KL to Q4, unseen Rust/Go/Java repositories | 0.083 | 0.034 | 0.024 |
| KL to Q4, English prose | 0.513 | 0.485 | 0.455 |
| decode speed | 36–37 tok/s | about −4% | −5 to −8% |
| largest context with MTP | 100K | 100K | ~74K |

[ccristiann/Qwen3.8-Flash-Next-picoforge-gguf](https://huggingface.co/ccristiann/Qwen3.8-Flash-Next-picoforge-gguf)
holds the 4-bit experts and the ranking (6.6 GB). `tools/quant/build_mixed.py` rebuilds either
file from the Q2 in about two minutes, byte for byte (`--expect` checks the sha256; the hashes are
in `docs/weights.lock`). Then `./pf -m code5` or `./pf-agent -m code5`. DwarfStar and llama.cpp do
not read this layout. Method and data: [`bench/quant/m4_20261123`](bench/quant/m4_20261123/README.md).

## Results

Qwen3.8-Flash-Next Q2 on a MacBook Pro 14" (M5 Pro, 64 GB, 307 GB/s), against
DwarfStar `fc80bd6` on the same file and machine. Paired runs, 3 warm-up + 20
measured per engine, alternating, on AC power ([`bench/qwen38/pg_20261115`](bench/qwen38/pg_20261115/README.md)).

| | picoforge | DwarfStar | ratio |
|---|---:|---:|---:|
| Decode, short prompt | 32.2 tok/s | 33.2 tok/s | 0.97x |
| Decode, 8K context | 29.0 tok/s | 31.1 tok/s | 0.94x |
| Prefill, 8K prompt (FP32) | 26.9 s | ~14 s¹ | — |

¹ DwarfStar prefills with f16 activations; its figure is the earlier R3 measurement.

- The FP32 prefill runs its GEMMs and attention on the Neural Accelerators with
  FP32 operands; the GEMMs are bit-identical to the simdgroup path.
- Multi-token prediction: 1.09–1.65x over plain decode, token-identical.
- Q4 with experts streamed from the SSD: ~18–19 tok/s (it does not fit in 64 GB otherwise).
- Q2 against Q4 on an unseen corpus: KL 0.40, top-1 agreement 77%, perplexity +8%.

Protocol: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Raw data: [`bench/`](bench/README.md).

## Correctness

Every kernel is checked against a chain of oracles: an independent FP32
reference, the C CPU path, then Metal. The end-to-end gate requires logits
within 1e-3 of the FP32 reference and identical greedy output on a frozen suite;
speculative decoding must emit exactly the tokens of plain greedy decoding.

```sh
make test-all      # offline checks, no checkpoint
make test-qwen4    # full gate (needs the reference assets)
```

## Documentation

[Design and stages](docs/DESIGN.md) · [Benchmark protocol](docs/BENCHMARKS.md) ·
[Devlog](docs/devlog.md) · [History](docs/history.md)

## License

MIT, see [LICENSE](LICENSE). Model weights: Qwen Community License.
