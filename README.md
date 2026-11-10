# picoforge

An LLM inference engine for Apple Silicon, written in C11 and Metal, small
enough to read end to end. It runs one model, **Qwen3.8-Flash-Next** (a
48-layer mixture-of-experts), from its quantised GGUF file, entirely on the
GPU, with no ML framework in the engine. It is an educational project:
correctness comes first, and every number is measured. Its philosophy follows
[DwarfStar](https://github.com/antirez/ds4).

## Hardware

- An **M5-family Mac** with macOS 26.2 or later. Prefill uses the GPU Neural
  Accelerators through Metal 4 TensorOps.
- **64 GB of unified memory.** The Q2 file (137 GiB) keeps ~42 GiB of weights
  resident and reads its n-gram tables from the SSD. The Q4 file (165 GiB)
  streams its routed experts from the SSD.
- Xcode with the Metal 4 toolchain. Python 3.12+ is needed only for the tests
  and tools.

All results come from one machine: MacBook Pro 14", M5 Pro, 64 GB, 307 GB/s.

## Usage

Download a GGUF from
[antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
into `models/`, then check it against the pinned hashes:

```sh
cd models && shasum -a 256 -c --ignore-missing ../docs/weights.lock && cd ..
```

```sh
make
./picoforge                                   # interactive chat
./picoforge -m q2 -p "What is 2 + 2?" -n 32   # one answer, then exit
./picoforge -m q2 -p "..." --mtp              # multi-token prediction, same output
./picoforge bench -m q2                       # 3 warm-up + 20 runs, median and p10/p90
./picoforge models                            # models found and their short names
```

Defaults: 8K context, greedy decoding, thinking off (`--thinking` or `/think`
turns it on). In the chat, `/help` lists the commands. **Run one process at a
time**: the GPU-wired pages take about 1.5 s to be released after an exit.

## Results

Measured on the Q2 file. DwarfStar `fc80bd6` ran on the same file and machine.

| | picoforge | DwarfStar |
|---|---:|---:|
| Decode, short prompt | 26.0 tok/s | 33.2 tok/s |
| Prefill, 8K prompt | 61.0 s FP32, 37.6 s half | ~14 s |
| MTP over plain decode | 1.09–1.65x | — |
| Q4 file, experts streamed from SSD | ~18–19 tok/s | doesn't fit on 64 GB |

Q2 against Q4 on a corpus neither model was trained on: KL 0.40, 77% top-1
agreement, perplexity +8%. The protocol is in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and the raw data in
[`bench/`](bench/README.md).

## Correctness

Each kernel is checked against a chain of oracles: an independent FP32
reference, then the C CPU path, then Metal. The end-to-end gate requires
logits within 1e-3 of the FP32 reference, and the same greedy output on a
frozen suite. Speculative paths must emit exactly the tokens of plain greedy
decoding.

```sh
make test-all      # offline checks: no checkpoint, no long GPU workload
make test-qwen38   # the full gate (needs the reference assets)
```

## Documentation

[Design and stages](docs/DESIGN.md) · [Benchmark protocol](docs/BENCHMARKS.md) ·
[Devlog](docs/devlog.md) · [History](docs/history.md)

## License

MIT, see [LICENSE](LICENSE). The model weights use the Qwen Community License.
