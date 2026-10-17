# R0–R1 Implementation Plan — misura e decode veloce di Flash-Next Q2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** portare il decode ordinario di Qwen3.8-Flash-Next Q2 da ~1,07 tok/s a
≥ 0,95 × DwarfStar `fc80bd6` a entrambe le frontiere, con il gate G3 invariato.

**Architecture:** prima si misura (R0: probe, profilo per op, byte e dispatch,
harness contro DwarfStar, baseline), poi si sostituiscono i kernel oracolo con
kernel cooperativi selezionati dentro `q38_packed_encode`/MoE/grafo, ognuno
verificato contro il proprio oracolo che resta selezionabile (R1).

**Tech Stack:** C11, Objective-C (solo glue Metal), Metal Shading Language
(`-std=metal4.0 -fno-fast-math -ffp-contract=off`), Python del venv solo in
`tools/` e `tests/`.

**Spec:** [R0-R1-decode.md](R0-R1-decode.md), regole comuni in
[roadmap.md](roadmap.md). Contesto: [G2.md](G2.md), [native-next.md](native-next.md),
[MTP.md](MTP.md), [integration.md](integration.md) §5–6.

## Global Constraints

- Worktree `../picoforge-perf-decode`, branch `perf/qwen38-decode`. Python:
  `PYTHON=../picoforge/tools/venv/bin/python` (il venv esiste solo nel checkout
  principale). Modelli in `models/` tramite symlink relativi (gitignored).
- Attivazioni, accumulatori, KV e stato in **FP32**. Nessun fast-math globale;
  i `.metal` nuovi si compilano con `-fno-fast-math -ffp-contract=off`. FMA o
  `fp contract(fast)` solo kernel per kernel e solo se la tolleranza misurata regge.
- Gate end-to-end **mai allargato**: `make qwen38-g3` (logits ≤ 1e-3 contro
  DwarfStar CPU FP32, argmax e margini). Near-tie di routing come in `G2.md`.
- Gli oracoli restano selezionabili: `Q38_REF_KERNELS=1` riproduce **bit per
  bit** i logits del codice di partenza (`360acf0`). `Q38_QSA_SCALAR` invariato.
- Ogni processo GPU passa da `tools/qwen38_gpu_lock.py` (i target Make lo fanno
  già: non annidare il wrapper). Mai due processi con il checkpoint caricato,
  nemmeno picoforge + DwarfStar. Memoria con `build/process_memory`: nessuna
  crescita dello swap. Se il lock risponde exit 75, aspettare e riprovare.
- Benchmark: ≥ 3 warm-up + ≥ 20 misure, mediana e p10/p90, ordine alternato
  contro DwarfStar, alimentazione AC verificata (`pmset -g batt`), annotazione
  termica, `caffeinate -i` per le run lunghe. `powermetrics` richiede sudo con
  password: si annota `pmset -g therm` + `NSProcessInfo.thermalState` e lo si
  dichiara. La baseline lenta usa le ripetizioni possibili, dichiarate.
- Evidenze in `bench/qwen38_r0_<AAAAMMGG>/` e `bench/qwen38_r1_<AAAAMMGG>/`
  (data della run). Mai sovrascrivere evidenze; ogni run in una cartella nuova.
- Nessun download, dipendenza nuova o modifica di sistema. Nessun push/PR.
  Non toccare altri worktree né `../ds4-qwen38-baseline` (solo lettura/esecuzione).
- Nei file committati niente percorsi assoluti personali.
- DwarfStar è riferimento di tecniche, non sorgente: nessun kernel copiato.
  Un'idea adattata si cita nel commento (`ds4 fc80bd6, MIT, file:funzione`).
- Ogni task ≤ ~150 righe nuove (se serve di più, si divide e si avvisa
  l'orchestratore). Test verdi prima di ogni commit, un commit per modifica
  logica, messaggio inglese all'imperativo, trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ogni task aggiunge in `docs/devlog.md`, sezione **R0/R1** in cima, una nota
  di 5–15 righe: concetto (cos'è, perché, matematica se serve), cosa è
  cambiato, numero misurato.
- Non iniziare R2.

## Review Focus

1. **Forme non multiple della tile**: `n_out` ∈ {1, 3, 33, 512, 248320},
   `K` ∈ {32, 640, 2560}, blocchi Q8_0 parziali in coda. Ci si aspetta lo stesso
   risultato dell'oracolo entro tolleranza e nessuna scrittura fuori da `y`.
   Test: `test_shapes_not_multiple_of_tile` (Task 7, riusato da 9, 11).
2. **`k_logical < k_storage`** (down 640/768): il padding finito non entra nel
   prodotto e X non viene letto oltre `k_logical`. Test: `test_logical_k_padding`
   (Task 7 e 11).
3. **Non finiti**: NaN in x o Inf nei pesi devono ancora accendere il bit 1
   della parola d'errore, anche se il controllo passa dall'elemento all'output.
   Test: `test_nonfinite_sets_error_word` (Task 7, 9, 11, 12, 13).
4. **Indipendenza dalle righe**: il risultato di una riga non dipende da quante
   righe ci sono nel dispatch né dalla posizione, quindi chunk/decode/MTP
   restano bit-identici fra loro. Test: `test_rows_bit_independent` (Task 7,
   9, 11, 12, 13); in grafo, `test-qwen38-g2-state` e `test-qwen38-speculative`.
5. **View con pitch e offset** (LM head su `slice(mixed, t-rows, rows)`, viste
   MTP): pitch > K·4 e offset non nulli. Test: `test_strided_offset_views`
   (Task 7). Indici di esperto fuori intervallo → bit 2: `test_expert_id_bounds`
   (Task 11).

## Mappa dei file

| File | Responsabilità | Task |
|---|---|---|
| `tools/probe/qwen38_capabilities.m` | Probe JSON delle capacità Metal | 1 |
| `tools/qwen38_r1.mk` | Target R0/R1 (incluso da `tools/qwen38.mk`) | 1+ |
| `src/qwen38_profile.{h,m}` | Registratore del profilo: gruppi, tempi, conteggi, CSV | 2, 3 |
| `src/qwen38_g_internal.h` | API private grafo/sessione (profilo, claim, argmax) | 2, 15 |
| `src/qwen38_forward_gpu.m` | Grafo: punti di profilo, multi-GEMV, barriere, argmax | 2, 3, 10, 14, 15 |
| `src/qwen38_cli.m` | `--profile`, `--teacher-forced`, greedy su GPU | 2, 4, 15 |
| `tools/eval/qwen38_profile_summary.py` | Tabella op → µs, GB/s, quota | 3 |
| `tools/eval/qwen38_compare.py` | Harness picoforge contro DwarfStar | 4 |
| `src/qwen38_ref.h` | Selettore `Q38_REF_KERNELS` | 6 |
| `src/qwen38_matvec.metal` | Matvec cooperativi per formato, multi, indicizzati | 7, 9, 10, 11 |
| `src/qwen38_moe_fused.metal` | Gate+up+SwiGLU e down+riduzione fusi | 12, 13 |
| `src/qwen38_packed_gpu.m` | Pipeline e scelta veloce/oracolo dentro `q38_packed_encode` | 6–11 |
| `src/qwen38_b_internal.h` | API private B: modo, encode veloce, multi | 6, 7, 10 |
| `src/qwen38_moe.m` | Scelta MoE fuso/oracolo | 12, 13 |
| `tests/test_qwen38_fast_kernels.{m,py}` | Harness kernel veloce contro oracolo e FP64 | 6+ |
| `tests/qwen38_kernel_tolerances.json` | Tolleranze per kernel, misurate e motivate | 7+ |

## Interfacce comuni (fissate qui, usate da più task)

```c
/* src/qwen38_ref.h — letto alla creazione dei contesti, mai per token. */
/* Q38_REF_KERNELS: assente/""/"0" → percorsi veloci; "1"/"all" → tutti gli
 * oracoli; altrimenti lista separata da virgole di famiglie:
 * linear, indexed, moe, fuse, barrier, claim, argmax. */
static inline bool q38_ref_kernels(const char *family);

/* src/qwen38_b_internal.h */
void q38_packed_gpu_set_reference(Q38PackedGpu *, bool linear, bool indexed);
/* Stesso contratto e stessa admission di q38_packed_encode, sempre veloce:
 * Q38_UNSUPPORTED se il formato non ha ancora un kernel veloce. */
Qwen38Status q38_packed_encode_fast(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
```

Regola numerica di tutti i kernel veloci: l'ordine di riduzione di una riga di
output dipende solo da K e dal formato, **mai** dal numero di righe, dalla
posizione o dal tiling su n_out. Questo preserva la bit-identità chunk/decode/MTP
fra percorsi veloci (non con l'oracolo). Controllo di finitezza sull'output
(bit 1), controlli di bounds come l'oracolo (bit 2).

Tolleranza di un kernel veloce: due condizioni, entrambe nel report.
(a) Il bound di B contro FP64 (`integration.md` §6:
`|y−r| ≤ γ(2K+8)·Σ|w·x| + 2·ulp32(r)`), che vale per qualsiasi ordine di
somma. (b) `|y_fast − y_oracle| / max(Σ|w·x|, 2^-126) ≤ tol[kernel]`, con
`tol` = potenza di 2 immediatamente sopra il massimo misurato sull'intero
corpus di test, scritta in `tests/qwen38_kernel_tolerances.json` con massimo,
p99, numero di campioni e commit della misura.

---

## R0 — Misura

### Task 1: Probe delle capacità (R0.1)

**Files:**
- Create: `tools/probe/qwen38_capabilities.m`, `tools/qwen38_r1.mk`,
  `tests/test_qwen38_capabilities.py`
- Modify: `tools/qwen38.mk` (una riga `include tools/qwen38_r1.mk`)
- Modify: `docs/devlog.md` (prima nota R0/R1)

**Interfaces:**
- Produces: `build/qwen38-capabilities` che stampa un oggetto JSON con chiavi
  `device`, `os`, `counter_sampling` (`stage`, `dispatch`, `draw`, `blit`, `tile`: bool),
  `counter_sets` (lista di nomi), `timestamp_counter` (bool),
  `command_buffer_gpu_times` (bool), `metal4_family` (bool),
  `mtl4_command_queue` (bool), `residency_set` (bool),
  `recommended_max_working_set_bytes`, `max_buffer_length`, `unified_memory`,
  `thermal_state` (stringa). Ogni API nuova si verifica con
  `respondsToSelector:`/`@available` prima di chiamarla: nessuna assunzione.
  Target `make build/qwen38-capabilities` e `make test-qwen38-r0-tools`.

- [ ] **Step 1: test che fallisce** — `tests/test_qwen38_capabilities.py`:
  `test_probe_emits_required_keys` esegue `build/qwen38-capabilities` sotto il lock e
  verifica presenza e tipo di tutte le chiavi sopra, `recommended_max_working_set_bytes > 0`.
- [ ] **Step 2:** `make test-qwen38-r0-tools PYTHON=../picoforge/tools/venv/bin/python` → FAIL (binario assente).
- [ ] **Step 3:** scrivere il probe (Objective-C, `-fobjc-arc`, framework Metal/Foundation).
- [ ] **Step 4:** stesso comando → PASS. Salvare l'output in
  `bench/qwen38_r0_<data>/capabilities.json` (cartella nuova).
- [ ] **Step 5:** nota devlog + commit `Add a Metal capability probe for R0`.

### Task 2: Modalità profilo nel grafo (R0.2)

**Files:**
- Create: `src/qwen38_profile.h`, `src/qwen38_profile.m`
- Modify: `src/qwen38_g_internal.h`, `src/qwen38_forward_gpu.m`, `src/qwen38_cli.m`,
  `tools/qwen38_g2.mk` (sorgenti nei target che compilano il grafo)
- Test: `tests/test_qwen38_profile.py`

**Interfaces:**
- Consumes: il probe del Task 1 decide il meccanismo. Se `counter_sampling.stage`
  è vero: in modalità profilo un compute encoder per gruppo, con timestamp di
  inizio e fine (sample buffer). Altrimenti: fallback dichiarato (un command
  buffer per gruppo, `GPUStartTime/GPUEndTime`), con overhead dichiarato.
- Produces:
  ```c
  /* qwen38_g_internal.h */
  typedef struct Q38Profile Q38Profile;   /* opaco, in qwen38_profile.h */
  Qwen38Status q38g_forward_set_profile(Qwen38GpuForward *, Q38Profile *, Qwen38Error *);
  /* chiusura del gruppo corrente e apertura del successivo: no-op senza profilo */
  void q38g_profile_mark(Qwen38GpuForward *, Qwen38GpuCommand *, uint32_t layer,
                         const char *op, uint32_t gguf_type, uint64_t weight_bytes);
  ```
  CLI `--profile FILE.csv` (rifiuta un file esistente). CSV, una riga per
  gruppo e per append: `phase,token,layer,op,format,weight_bytes,dispatches,barriers,gpu_us,gbps`.
  Gruppi minimi: `embed`, `ple`, `gdn` (blocco C intero), ciascuna `linear()` del
  grafo con il proprio nome (`hc_attn.down`, `qsa.qg`, `qsa.k`, … `lm_head`),
  `qsa.core`, gli op C dell'HC, `moe` (intero), `hc_add`, `final`.
  `layer` = `global` per embed/final/lm_head.
- Vincolo: senza `--profile` il percorso di codice e i logits non cambiano.

- [ ] **Step 1: test che falliscono** (`tests/test_qwen38_profile.py`, sul grafo
  sintetico `build/qwen38-g2/cli.gguf`): `test_profile_does_not_change_logits`
  (logits `--logits` con e senza `--profile` uguali bit per bit);
  `test_profile_csv_covers_every_layer` (ogni layer del grafo sintetico ha ≥ 1
  riga, `gpu_us > 0`, colonne esatte); `test_profile_rejects_existing_file`.
- [ ] **Step 2:** eseguirli sotto lock → FAIL.
- [ ] **Step 3:** implementare registratore, marcatori nel grafo e flag CLI.
- [ ] **Step 4:** test → PASS; poi `make test-qwen38-g2 test-qwen38-g2-state test-qwen38-cli` → PASS.
- [ ] **Step 5:** nota devlog (overhead misurato sul grafo sintetico) + commit
  `Add an opt-in per-op GPU profile to the Qwen3.8 graph`.

### Task 3: Byte, dispatch e barriere per op (R0.2–R0.3)

**Files:**
- Modify: `src/qwen38_profile.m`, `src/qwen38_forward_gpu.m`
- Create: `tools/eval/qwen38_profile_summary.py`, `tests/test_qwen38_profile_summary.py`

**Interfaces:**
- Consumes: Task 2.
- Produces: `weight_bytes` calcolati dalle binding reali per **ogni** gruppo,
  compresi i gruppi composti: `gdn` = somma dei `Q38CW_*` letti; `moe` = router +
  `top_k` × `expert_bytes` per gate/up/down per token + condiviso; `ple` = key/value
  + righe n-gram lette. `dispatches` e `barriers` contati senza toccare i
  componenti (per esempio un proxy `NSProxy` dell'encoder che conta
  `dispatchThreads`/`dispatchThreadgroups`/`memoryBarrier*` e inoltra tutto).
  `qwen38_profile_summary.py CSV... --out DIR` scrive `summary.csv`
  (`op,format,calls_per_token,us_per_token,share,weight_bytes_per_token,gbps`),
  `totals.json` (µs per token, byte per token, dispatch e barriere per token,
  pavimento `dispatch × 1,4 µs`, rapporto somma-gruppi / `gpu_s` normale) e una
  tabella Markdown.

- [ ] **Step 1: test che falliscono:** `test_summary_aggregates_tokens` e
  `test_summary_reports_dispatch_floor` su un CSV fixture scritto nel test;
  `test_profile_counts_dispatches` sul grafo sintetico (dispatch > 0, barriere > 0,
  stessi conteggi in due token di decode consecutivi).
- [ ] **Step 2:** FAIL. **Step 3:** implementare. **Step 4:** PASS + suite G2.
- [ ] **Step 5:** nota devlog + commit `Count bytes, dispatches and barriers per profiled op`.

### Task 4: Modo teacher-forced e harness contro DwarfStar (R0.4)

Diviso in 4a (CLI) e 4b (harness), commit separati.

**Files:**
- Modify: `src/qwen38_cli.m`, `tests/test_qwen38_cli.py`
- Create: `tools/eval/qwen38_compare.py`, `tests/test_qwen38_compare.py`

**Interfaces:**
- Produces (4a): `--teacher-forced N` con `--tokens`: i primi `count−N` ID sono
  il prompt (prefill a chunk), gli ultimi N sono appesi **uno alla volta** come
  in `ds4_session_eval`, ciascuno con la sua riga di logits e argmax. Nel JSONL
  di `--runs-jsonl`: `"mode":"teacher-forced"`, `decode_tokens = N`, `decode_s` =
  somma degli N append (argmax incluso). Velocità confrontabile con
  `gen_tps` di `ds4-bench` = `gen_tokens / gen_sec`.
- Produces (4b): `tools/eval/qwen38_compare.py`
  - `prepare OUT`: tokenizza con picoforge (`--complete-file --encode-only`) e con
    `ds4 --dump-tokens` lo stesso testo e richiede ID identici. Frontiera 8K:
    `../ds4-qwen38-baseline/speed-bench/promessi_sposi.txt`, primi 8063 + 128 ID.
    Frontiera breve: prompt coding della suite congelata
    (`bench/qwen38_mtp_20261013/paired/coding`) seguito dalla sua continuazione
    greedy registrata, 128 ID forzati. Deviazione dalla spec (DwarfStar
    `ds4-bench` teacher-forced anche alla frontiera breve, invece della CLI:
    la CLI si ferma a EOS e non garantisce 128 token) annotata in `R0-R1-decode.md`.
  - `run SUITE OUT --frontier {short,8k} --warmup W --measure M --block B`:
    blocchi alternati picoforge/DwarfStar, ogni processo sotto il lock, attesa
    di rilascio (pressione normale, wired ≤ 1/4 RAM, come
    `qwen38_native_bounded.py`) prima di ogni caricamento, `process_memory`
    attorno a ogni processo, `pmset -g batt`/`pmset -g therm` prima e dopo,
    verifica che `../ds4-qwen38-baseline` sia a `fc80bd6` e pulito. Rifiuta
    un `OUT` esistente. Scrive JSONL/CSV grezzi, comandi, hash dei binari.
  - `summarize OUT`: mediana, p10/p90 (campione più vicino, come native-next),
    rapporto picoforge/DwarfStar.
  - `ssd-check OUT`: un solo tentativo `ds4 --ssd-streaming` sul file Qwen3.8,
    log ed exit status conservati.

- [ ] **Step 1 (4a): test che fallisce** in `tests/test_qwen38_cli.py`:
  `test_teacher_forced_matches_plain_appends` — sul `cli.gguf` sintetico, i logits
  della modalità teacher-forced coincidono bit per bit con prefill + append
  ordinari degli stessi ID, e `decode_tokens == N`.
- [ ] **Step 2–4 (4a):** FAIL → implementare → PASS + `make test-qwen38-cli`.
  Commit `Add a teacher-forced decode mode to the Qwen3.8 CLI`.
- [ ] **Step 5 (4b): test che falliscono** in `tests/test_qwen38_compare.py` (nessun
  GPU): `test_schedule_alternates_blocks`, `test_stats_nearest_sample_percentiles`,
  `test_prepare_rejects_token_mismatch` (tokenizzatori finti),
  `test_run_rejects_existing_output`.
- [ ] **Step 6–8 (4b):** FAIL → implementare → PASS; prova breve reale
  `run … --frontier short --warmup 1 --measure 2 --block 1` in `build/` (non in bench).
  Commit `Add the paired picoforge/DwarfStar decode harness`. Nota devlog.

### Task 5: Baseline registrata (R0.5)

**Files:** Create `bench/qwen38_r0_<data>/` (README, CSV, JSONL, comandi, hash). Modify `docs/devlog.md`.

- [ ] **Step 1:** verificare AC, GPU libera, swap; `caffeinate -i` per tutto.
- [ ] **Step 2:** profilo baseline: frontiera breve (8 token di decode
  teacher-forced) e frontiera 8K (1 run, 4 token), `--profile`; summary.
- [ ] **Step 3:** benchmark frontiera breve: picoforge 3 warm-up + 20 misure,
  DwarfStar 3 + 20, blocchi alternati.
- [ ] **Step 4:** frontiera 8K: DwarfStar 3 + 20; picoforge 2 misure senza
  warm-up separato (≈ 35 min ciascuna, dichiarato).
- [ ] **Step 5:** `ssd-check`. Logits di riferimento per il controllo oracolo:
  `--logits` sul prompt breve (8 righe) salvati come `oracle-logits.f32` con SHA.
- [ ] **Step 6:** README: tabella del profilo (op → µs, GB/s, quota), byte e
  dispatch per token reali contro la stima 6,1 GB, baseline contro DwarfStar
  con mediana e p10/p90, limiti. Commit `Record the R0 decode baseline`.
- [ ] **Step 7 (orchestratore):** confronto profilo/ordine di R1; se diverso,
  aggiornare `R0-R1-decode.md` con la motivazione e riordinare i task qui sotto.

---

## R1 — Decode (ordine provvisorio: lo decide il profilo di R0)

Ogni task R1 che cambia il percorso predefinito chiude con: suite completa
(`make test-qwen38`, `make test-qwen38-mtp test-qwen38-speculative`,
`make test-qwen38-fast-kernels`), `make qwen38-g3 G3_OUTPUT=build/<nuova>`,
`Q38_REF_KERNELS=1` bit-identico a `oracle-logits.f32`, e una misura della
scala: frontiera breve, picoforge solo, 3 warm-up + 20 misure, più il profilo.
Il numero e il commit vanno in `bench/qwen38_r1_<data>/ladder.csv`.

### Task 6: Selettore oracolo e harness kernel-contro-oracolo (R1.1)

**Files:**
- Create: `src/qwen38_ref.h`, `tests/test_qwen38_fast_kernels.m`,
  `tests/test_qwen38_fast_kernels.py`, `tests/qwen38_kernel_tolerances.json`
- Modify: `src/qwen38_packed_gpu.m`, `src/qwen38_b_internal.h`, `tools/qwen38_r1.mk`

**Interfaces:**
- Produces: `q38_ref_kernels()`, `q38_packed_gpu_set_reference()`,
  `q38_packed_encode_fast()` (per ora sempre `Q38_UNSUPPORTED`). Target
  `test-qwen38-fast-kernels` (lock, `MTL_DEBUG_LAYER=1`, ASan/UBSan sul harness).
  Il harness apre il checkpoint reale con lo store esistente (solo i tensori
  scelti vengono toccati), sceglie ≥ 3 tensori per formato da layer diversi,
  genera x con seme fisso, calcola oracolo GPU, veloce GPU e riferimento FP64
  CPU (`q38_packed_decode_row`), più `Σ|w·x|` per riga; scrive i binari in una
  cartella nuova. Il Python calcola max abs, max rel, p50/p99 e le condizioni
  (a)/(b), e scrive `report.json`.
- [ ] **Step 1:** test `test_reference_against_itself_is_exact` (oracolo contro
  oracolo: errore 0) e `test_selector_parsing` (in C, valori `""`, `0`, `1`,
  `all`, `linear,moe`). FAIL → implementare → PASS.
- [ ] **Step 2:** commit `Add the reference-kernel selector and the fast-kernel harness`. Nota devlog.

### Task 7: Matvec Q8_0 cooperativo, solo kernel e test (R1.2)

**Files:** Create `src/qwen38_matvec.metal`. Modify `src/qwen38_packed_gpu.m`,
`tools/qwen38_r1.mk`, `Makefile` (AIR nella metallib),
`tests/test_qwen38_fast_kernels.{m,py}`, `tests/qwen38_kernel_tolerances.json`.

**Interfaces:**
- Produces: kernel `q38_matvec_q8_0` (stesso `PackedArgs` dell'oracolo);
  `q38_packed_encode_fast` lo usa per `gguf_type == 8`. Schema: un simdgroup su
  una o più righe; ogni lane legge blocchi interi da 34 byte in modo coalescente
  (scala fp16 + 32 int8), accumula in FP32, `simd_sum` finale; x letto una volta
  per threadgroup; con righe > 1 i pesi decodificati si riusano su più righe di x.
  Tiling e dimensioni scelti misurando (annotare le alternative provate).
- [ ] **Step 1: test che falliscono:** i cinque test della Review Focus (1–5) per
  Q8_0, più `test_q8_0_real_tensors_within_tolerance`.
- [ ] **Step 2:** FAIL. **Step 3:** kernel. **Step 4:** PASS; fissare `tol["q8_0"]`
  dalla distribuzione misurata. Riportare anche GB/s del kernel isolato su un
  tensore reale grande (LM head) rispetto a 307 GB/s.
- [ ] **Step 5:** commit `Add a cooperative Q8_0 matvec kernel with oracle tests`. Nota devlog.

### Task 8: Q8_0 nel grafo, gate end-to-end, misura (R1.2)

**Files:** Modify `src/qwen38_packed_gpu.m` (scelta in `q38_packed_encode`),
i target Make dei test che **certificano l'oracolo B** contro la CPU bit per bit
(aggiungere `Q38_REF_KERNELS=linear,indexed` solo a quelli, elencandoli nel commit).
- [ ] **Step 1:** test `test_default_selects_fast_q8_0` / `test_reference_env_selects_oracle`.
- [ ] **Step 2:** scelta veloce per `gguf_type == 8`, qualsiasi numero di righe.
- [ ] **Step 3:** chiusura R1 standard (suite, g3, controllo oracolo, scala).
- [ ] **Step 4:** commit `Use the cooperative Q8_0 matvec by default`. Nota devlog.

### Task 9: Matvec F16 e F32 (R1.3)

Come Task 7 + 8 per `gguf_type` 1 (HC) e 0 (router), stessi test e chiusura.
Due commit: kernel+test, poi selezione nel grafo.

### Task 10: Multi-GEMV (R1.4)

**Interfaces:**
```c
/* qwen38_b_internal.h: n proiezioni dello stesso x in un dispatch. Ogni y_i è
 * bit-identica a q38_packed_encode_fast(binding_i, x, y_i). */
Qwen38Status q38_packed_encode_multi(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *const *bindings,
    const Q38PackedLayout *layouts, size_t count, Qwen38GpuF32 x,
    const Qwen38GpuF32 *y, Qwen38Error *);
```
Usi: 5 proiezioni QSA su `mixed`; `QKV/Z/AP/BP` nel blocco GDN
(`q38c_block_encode`); gate+up dell'esperto condiviso. Test
`test_multi_bit_identical_to_single`. Chiusura R1 standard.

### Task 11: Esperti indicizzati veloci (R1.5)

11a: IQ2_XXS (`16`) e Q2_K (`10`) in `q38_packed_encode_indexed`; 11b: Q4_K
(`12`) e MXFP4 (`39`) per il layer misto e l'MTP. Oracolo
`q38_packed_linear_indexed`. Test della Review Focus + `test_expert_id_bounds`.
Route identiche salvo near-tie documentati come in `G2.md`. Chiusura R1 standard.

### Task 12: Gate+up+SwiGLU fuso (R1.5)

`src/qwen38_moe_fused.metal`: legge x una volta, per ogni esperto selezionato
calcola le righe gate e up e scrive `h = SiLU(g)·u`. Oracolo: sequenza attuale
(`IDX gate`, `IDX up`, `ACT SILU`, `MUL`). SiLU con la stessa formula FP32 di C.
Famiglia `moe` del selettore. Chiusura R1 standard.

### Task 13: Down e riduzione fusi (R1.5)

Down degli esperti, somma pesata in ordine di slot 0..K−1 e poi condiviso
(stesso ordine di `q38f_reduce`). Oracolo: `IDX down` + `q38f_reduce`.
Se il profilo mostra `q38f_route` > 2% del token: top-k parallelo con stessa
regola dei tie (ID minore a parità esatta), task separato 13b.

### Task 14: Barriere e fusioni elementwise (R1.6)

14a: encoder `MTLDispatchTypeConcurrent` nella sessione (API privata) con
barriere solo dove un op legge ciò che un op precedente ha scritto; la famiglia
`barrier` del selettore ripristina encoder seriale + barriera dopo ogni op.
Test: logits bit-identici fra i due modi su 3 esecuzioni ripetute. 14b: fusioni
RMSNorm / mixer HC / residui dove il profilo le giustifica, famiglia `fuse`.

### Task 15: Claim una volta, residency set, argmax su GPU (R1.6)

15a: claim READ dei pesi registrati una volta per sessione (API additiva G0 con
test in `test-qwen38-g0`), famiglia `claim`. 15b: `MTLResidencySet` per i buffer
dei pesi, se il probe lo conferma. 15c: argmax su GPU (tie al primo indice,
non finiti → bit 1) con readback dei logits solo con `--logits`, MTP o gate;
famiglia `argmax`. Pre-encoding solo se l'encode misurato lo giustifica.

### Task 16: GDN, QSA, n-gram (R1.7)

Solo le voci sopra ~5% del tempo nel profilo dopo il Task 15.

### Task 17: Chiusura R1 (R1.8)

Benchmark con protocollo completo contro DwarfStar a entrambe le frontiere
(3 + 20, blocchi alternati), memoria (picco Metal ≤ 48–50 GiB, nessuna crescita
swap), prefill non peggiorato, tutti i gate, `bench/qwen38_r1_<data>/README.md`
con la scala completa, README del progetto e devlog.
