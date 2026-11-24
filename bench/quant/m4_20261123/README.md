# M4 — mixed-precision experts: the most-used experts at Q4, the rest at Q2 (2026-11-23/24)

A picoforge-only GGUF: the published Q2 (IQ2_XXS gate/up, Q2_K down) where, in every trunk
layer, the experts that carry the most gate mass on code are replaced by their exact bytes from
the published Q4 (Q4_K gate/up, MXFP4 down). Nothing is re-encoded. Two variants: 5% (26 of 512
experts per layer, +1.4 GB) and 10% (51 per layer, +2.9 GB). DwarfStar and llama.cpp cannot load
these files.

## Method

- **Ranking** (`tools/quant/build_mixed.py rank`): gate mass per expert summed over the routes of
  every trunk layer (`QW4_ACT_LAYERS=all QW4_ACT_X=0`), Q4 prefill on the rank corpus
  (`tools/quant/code_corpus.py --rank`, sha256 `d6d6bbb3…`): 8 repositories, one language each
  (llama.cpp C++, flask Python, react JS, ripgrep Rust, gin Go, gson Java, zod TS, curl C), 12288
  tokens each, 98 304 rows. The evaluation repositories (tokio, cobra, guava) and picoforge's own
  code (the R7 corpus) are not in it. `ranking.json` is the result.
- **Pool**: the top 102 experts per layer copied from the Q4 (`pool-sha256.txt`, 0 byte
  mismatches over 14 688 experts), so the Q4 could be deleted before the builds.
- **Build** (`build_mixed.py build`): every expert of the output is checked byte for byte against
  the Q2 (cold) or the pool (hot), every other tensor against the Q2: 0 mismatches for both
  files. The 10% file's sha256 and recipe are in `hot10-*`; it is rebuilt identically in ~2 min.
- **Quality** (`notes`: m4_logits/m4_compare, reusing `tools/eval/qwen4_r7_quality.py`): FP32
  logits teacher-forced on 11 windows of 2048 tokens, against the **Q4 as reference**, all runs
  with the same binary: the 8 R7 windows (picoforge code and prose, written after the model was
  released) and window 0 of tokio, cobra and guava. The Q4 reference logits' sha256 are in
  `q4-ref-logits-sha256.txt`. The Q2 numbers equal R7's of 2026-11-10 to three digits.
- **Speed/memory**: guarded single runs, alternating Q2 and mixed, on AC. Not protocol benches
  (no 3+20); enough to see the size of the effect.

## Where the hot experts' gate mass lands (held-out, `coverage.txt`)

| top experts per layer | other-language code | picoforge code | prose |
|---|---|---|---|
| 5% (code ranking) | 33.5% | 30.2% | 9.3% |
| 10% (code ranking) | 47.8% | 50.9% | 17.6% |
| 10% (Unsloth imatrix counts) | 10.9% | 8.4% | 12.3% |
| 10% (oracle: ranked on the eval set itself) | 52.8% | 65.7% | 54.9% |

A per-layer allocation of the same budget (greedy on calibration mass) gains +0.3 points: the
files use the same count in every layer.

## Quality against Q4 (lower KL / ΔNLL = closer to Q4)

| set | file | KL mean [95% CI] | top-1 | PPL (Q4 3.347 / 12.63 / 1.230) | ΔNLL nats [95% CI] |
|---|---|---|---|---|---|
| picoforge code (R7) | Q2 | 0.288 [0.248, 0.335] | 84.1% | 3.566 | 0.063 [0.045, 0.082] |
| | 5% | 0.217 [0.187, 0.254] | 85.9% | 3.478 | 0.038 [0.020, 0.056] |
| | 10% | **0.167** [0.143, 0.197] | **88.1%** | 3.429 | **0.024** [0.009, 0.041] |
| prose (R7) | Q2 | 0.513 [0.462, 0.561] | 69.9% | 13.895 | 0.095 [0.067, 0.125] |
| | 5% | 0.485 [0.438, 0.532] | 70.7% | 13.606 | 0.074 [0.047, 0.103] |
| | 10% | **0.455** [0.412, 0.502] | **71.5%** | 13.320 | **0.053** [0.023, 0.085] |
| other languages (tokio/cobra/guava) | Q2 | 0.083 [0.064, 0.105] | 95.9% | 1.292 | 0.049 [0.030, 0.068] |
| | 5% | 0.034 [0.025, 0.044] | 97.6% | 1.244 | 0.011 [0.000, 0.023] |
| | 10% | **0.024** [0.017, 0.030] | **98.0%** | 1.242 | **0.009** [0.000, 0.018] |

The 10% file removes 42% of Q2's KL to Q4 on unseen picoforge code, 71% on the other languages
and 11% on prose; of the ΔNLL, 62%, 82% and 44%. Per-window arrays in `r7-*/`.

## Speed and memory

| | Q2 | 5% | 10% |
|---|---|---|---|
| file | 137.1 GiB | 138.5 GiB | 139.9 GiB |
| decode, plain greedy, 256 tokens (alternating runs) | 35.7–37.5 tok/s | 33.3–36.5 | 32.5–35.4 |
| decode, 400 tokens greedy / with MTP (same IDs) | 36.4 / 38.4 | 32.7 / 35.9 | — |
| prefill 8K tokens | 30.9–32.2 s | — | 31.6–32.8 s |
| Metal allocated at 8K context | 43.1 GiB | — | 46.1 GiB |
| largest context with MTP (budget 95% of 51.8 GiB) | 100K ok | 100K ok | 73.7K ok, 81.9K refused |

pf-agent's default context is 100K with MTP: the 10% file does not fit it, the 5% one does.

The first engine path (M4b: both formats on every pair, unfused down) cost -30% decode and -22%
prefill on the real file; the fast paths (sentinel skip, one mixed parallel-slot down+reduce,
grouped kernels per format) brought that to the numbers above. Q2 logits are byte-identical
before and after those kernel changes (R7 code-0 sha256 `c8843481…`).

## Published (2026-11-24)

Hugging Face `ccristiann/Qwen3.8-Flash-Next-picoforge-gguf` holds a patch, not the files: the top 51
experts per layer cut from the pool (`build_mixed.py pool`, 6 642 915 552 bytes, sha256 `92d731b5…`),
`ranking.json`, the two recipes, the Qwen license and `SHA256SUMS`. A build's recipe names the
pool's origin rather than the local file name, so both files come back byte for byte from that
patch and antirez's Q2. Checked with `build --hash-only --expect`: 5% `dfda107b…` (equal to the file
measured here, renamed Q2-Code5) and 10% `a2df6e8d…` (equal to `hot10-sha256.txt`). The full files
are not uploaded: two ~148 GB files do not fit a free account's 100 GB of private storage.

## Not done here

- G3 on the real file: the FP32 oracle on real weights is DwarfStar's, which does not read the
  mixed layout. The oracle chain for mixed files is verified on synthetic checkpoints
  (`make test-quant-engine`: CPU, Metal fast, grouped, exact, CLI).
- Coding task benchmarks (e.g. HumanEval+) and protocol benches 3+20.
