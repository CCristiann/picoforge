# R0–R1 — Misura e decode veloce di Flash-Next Q2

Data: 2026-10-16. Stato: **bozza, in attesa di approvazione**.
Parte di [roadmap.md](roadmap.md), di cui valgono tutte le regole comuni:
correttezza, precisione, protocollo e modo di lavoro.

## Obiettivo

Portare il decode ordinario di Qwen3.8-Flash-Next Q2 da ~1,07 tok/s alla
parità con DwarfStar sullo stesso Mac e sullo stesso file, senza toccare il
gate di correttezza. Prima si misura (R0), poi si ottimizza nell'ordine
indicato dalla misura (R1).

## Criteri di uscita

1. **Velocità.** Decode ordinario, mediana ≥ 0,95 × la mediana di DwarfStar
   `fc80bd6`, a entrambe le frontiere del protocollo:
   - prompt breve, 128 token generati;
   - 8063 token di prompt + 128 token teacher-forced, come `ds4-bench`.
2. **Correttezza.** Tutti verdi:
   - `make qwen38-g3`: logits ≤ 1e-3 contro DwarfStar CPU FP32, argmax e
     margini;
   - `make test-qwen38`, `make test-qwen38-mtp test-qwen38-speculative`;
   - regressione legacy `make test-all`;
   - ogni kernel nuovo contro il suo oracolo entro la tolleranza documentata.
3. **Memoria.** Picco Metal ≤ 48–50 GiB, pressione normale, nessuna crescita
   swap sostenuta (`build/process_memory`).
4. **Nessun peggioramento.** Il prefill non rallenta: i nuovi kernel servono
   anche le righe > 1, oppure il prefill resta sul percorso attuale finché
   arriva R3.
5. **Evidenze.** Cartelle `bench/qwen38_r0_<data>/` e `bench/qwen38_r1_<data>/`
   con CSV, comandi, hash e README. Sezioni nuove in `docs/devlog.md`.

## Stato di partenza (fatti verificati)

- `run()` in `src/qwen38_forward_gpu.m` usa un command buffer per chiamata e
  un solo compute encoder, con una barriera `MTLBarrierScopeBuffers` dopo
  ogni op. In stima sono ~40 dispatch per layer, ~1.900 per token (da
  misurare in R0).
- Ogni proiezione passa da `linear()` → `q38_packed_encode` →
  `q38_packed_linear` (`src/qwen38_packed.metal:95`):
  - un thread per riga di output, con un ciclo seriale su K;
  - `packed_value()` decodifica un peso alla volta e rilegge l'header del blocco;
  - `isfinite` più un atomico per ogni elemento;
  - `#pragma METAL fp contract(off)`.
- Il MoE (`q38f_routed_encode`) fa il routing su GPU senza readback e usa
  `q38_packed_linear_indexed`.
- I pesi sono un mmap no-copy per tensore (`src/gpu_weights.m`), senza
  residency set e senza coda Metal 4.
- Ogni chiamata a `qwen38_gpu_forward()` fa il claim di tutti i 1.256 tensori
  di pesi, lato CPU.
- Per ogni token tutti i logits (248.320 float) tornano alla CPU.
- Decode misurato: `gpu_s` ≈ 0,93 s per token, ~6,6 GB/s effettivi. Lettura
  n-gram ed encoding stanno sotto lo 0,5%.
- Formati del decode, stimati per byte: Q8_0 ~3,0 GiB (GDN, QSA, LM head,
  esperto condiviso), F16 1,2 GiB (HC), esperti IQ2_XXS/Q2_K 0,67 GiB,
  F32 0,24 GiB (router).

## R0 — Misura

**R0.1 Probe delle capacità.** Un piccolo programma Objective-C stampa:
- `supportsCounterSampling:` al confine di stage e al confine di dispatch;
- la disponibilità dei timestamp GPU e della coda Metal 4;
- `recommendedMaxWorkingSetSize`.

Va salvato con le evidenze. Il punto è non presumere che il campionamento per
dispatch esista: su Apple GPU di solito c'è solo quello per stage.

**R0.2 Profilo per op.** Una modalità opt-in (flag CLI o variabile d'ambiente,
a scelta del builder) attribuisce il tempo GPU a ogni gruppo di op:
- se il probe lo consente, con un encoder per gruppo e timestamp di inizio e
  fine;
- altrimenti con un command buffer per gruppo e `GPUStartTime`/`GPUEndTime`.

Uscita: un CSV con token, layer, op, formato, byte di pesi letti, µs e GB/s.
La somma si confronta con `gpu_s` della modalità normale, e l'overhead della
profilazione va dichiarato. La modalità normale non cambia.

**R0.3 Byte e dispatch per token.**
- Byte letti per op, calcolati dalle binding reali: sostituiscono la stima
  della roadmap.
- Conteggio dei dispatch e delle barriere per token, confrontato con il
  pavimento di ~1,4 µs per dispatch.

**R0.4 Harness di confronto.** Uno script in `tools/eval/` esegue picoforge e
DwarfStar con il protocollo della roadmap, sugli stessi ID di token e in ordine
alternato.
- Ogni esecuzione passa da `tools/qwen38_gpu_lock.py`.
- DwarfStar è il checkout pinned in `../ds4-qwen38-baseline`, senza modifiche:
  `ds4-bench` per la frontiera 8K, CLI per il prompt breve.
- Il prompt breve è il caso coding della suite congelata in
  `bench/qwen38_mtp_20261013/`.
- Se a picoforge manca un modo teacher-forced, va aggiunto: serve a misurare
  lo stesso lavoro di `ds4-bench`.
- La baseline lenta (8K a ~34 minuti per run) si misura con le ripetizioni
  possibili, dichiarate.

Nello stesso passo si verifica se `ds4 --ssd-streaming` accetta il file Qwen3.8.
Serve alla roadmap e non costa nulla.

**R0.5 Baseline registrata.** Profilo, byte, dispatch e confronto, prima di
qualsiasi ottimizzazione. Questa tabella decide l'ordine di R1.

## R1 — Decode

L'ordine qui sotto è quello atteso dai byte. Se il profilo di R0 dice
altrimenti, vince il profilo.

**R1.1 Infrastruttura oracolo.**
- Un selettore globale del percorso di riferimento (per esempio
  `Q38_REF_KERNELS=1`) più eventuali selettori per kernel, come l'esistente
  `Q38_QSA_SCALAR`.
- Un harness che confronta kernel veloce e oracolo su slice reali del GGUF e
  su dati sintetici, riportando max abs, max rel e la distribuzione.
- Un file di tolleranze per kernel, con il valore fissato dalla misura e
  motivato.

Il grafo non cambia interfaccia: il percorso veloce si sceglie dentro
`q38_packed_encode` in base a formato e righe.

**R1.2 Matvec Q8_0 cooperativo**, il formato più pesante.
- Un simdgroup (32 lane) lavora su una o più righe.
- Ogni lane legge blocchi interi da 34 byte, cioè la scala fp16 più 32 int8,
  in modo coalescente, li decodifica e accumula in FP32.
- La somma finale si riduce con `simd_sum`.
- x viene letto una volta per threadgroup.
- Il controllo di finitezza passa dal singolo elemento all'output. Quello per
  elemento resta nell'oracolo.
- Con righe > 1 (MTP, chunk piccoli) i pesi decodificati si riusano su più
  righe di x.

Tiling, righe per simdgroup e dimensioni del threadgroup li sceglie il builder
misurando.

**R1.3 F16 e F32.** Matvec F16 per le hyper-connections e F32 per il router,
con lo stesso schema.

**R1.4 Multi-GEMV.** Un solo dispatch per più proiezioni che leggono lo stesso
x:
- le 5 proiezioni QSA su `mixed`;
- le proiezioni GDN che condividono l'input;
- gate+up dell'esperto condiviso.

**R1.5 Esperti e MoE fuso.**
- Matvec indicizzati IQ2_XXS e Q2_K, più Q4_K e MXFP4 per il layer misto e
  per l'MTP.
- Un kernel fonde gate, up e SwiGLU: legge x una volta e scrive h.
- Un secondo kernel fa down, somma pesata dei 10 esperti ed esperto condiviso.

Oracolo: il MoE device-routed attuale. Le route devono restare identiche
tranne nei near-tie documentati come in `G2.md`.

**R1.6 Grafo.**
- Barriere solo sulle dipendenze dati reali.
- Fusione di RMSNorm, mixer HC e residui dove il profilo lo giustifica.
- Claim e binding dei pesi calcolati una volta per sessione, non a ogni token.
- Residency set per i buffer dei pesi.
- Argmax su GPU, con readback dei logits solo se richiesto da `--logits`, dai
  gate o dal sampling.
- Pre-encoding (indirect command buffer o argument table Metal 4) solo se il
  costo di encoding misurato lo giustifica; per Metal 4 si verifica prima
  l'API con un probe.

**R1.7 GDN, QSA, n-gram in decode.** Si interviene solo sulle voci sopra
~5% del tempo nel profilo dopo R1.6.

**R1.8 Chiusura.** Benchmark con protocollo completo contro DwarfStar, gate
completi, aggiornamento del README, sezione del devlog. Una riga della scala
per ogni passo misurato.

## Passi previsti (anteprima del piano)

Ogni passo ≤ ~150 righe, con test e commit propri. Si lavora in un worktree su
`perf/qwen38-decode` da `main`, con `models/` raggiungibile per percorso
assoluto.

1. Probe delle capacità (R0.1)
2. Modalità profilo nel grafo (R0.2)
3. CSV, byte e dispatch per op (R0.2–R0.3)
4. Modo teacher-forced e harness contro DwarfStar (R0.4)
5. Run baseline e README delle evidenze (R0.5)
6. Selettore oracolo e harness kernel-contro-oracolo (R1.1)
7. Kernel matvec Q8_0 con test sulle slice reali (R1.2)
8. Q8_0 inserito nel grafo, gate end-to-end, misura (R1.2)
9. F16 e F32 (R1.3)
10. Multi-GEMV (R1.4)
11. Esperti IQ2_XXS e Q2_K indicizzati (R1.5)
12. Gate+up+SwiGLU fuso (R1.5)
13. Down e riduzione fusi (R1.5)
14. Barriere e fusioni elementwise (R1.6)
15. Claim una volta per sessione, residency set, argmax su GPU (R1.6)
16. GDN/QSA se necessario (R1.7)
17. Benchmark finale, documentazione e scala (R1.8)

## Lasciato al builder

Tiling e occupazione dei kernel; valori delle tolleranze, dalla misura;
nome e forma dei selettori; formato esatto dei CSV; meccanismo di profilazione
secondo il probe; se e come fare pre-encoding.

## Rischi

- **Il campionamento GPU non distingue le op.** Si passa al fallback con un
  command buffer per gruppo; l'overhead è dichiarato e serve solo a ordinare
  le priorità.
- **Pavimento dei dispatch.** Se dopo i kernel il tempo resta dominato dai
  dispatch, R1.6 diventa prioritario.
- **Near-tie del routing.** Un riordino delle somme FP32 può cambiare una
  route a pari merito. Si tratta come in `G2.md`: replay con gli stessi score,
  documentazione, nessuna tolleranza allargata.
- **Termica.** Benchmark lunghi a ordine alternato, `powermetrics` annotato.
