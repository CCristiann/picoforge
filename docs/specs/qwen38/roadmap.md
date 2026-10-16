# Roadmap — la famiglia Qwen3.8 su questo M5 Pro, dal terminale

Data: 2026-10-16. Stato: **bozza, in attesa di approvazione**.
Prima spec eseguibile: [R0–R1, misura e decode](R0-R1-decode.md).

## Obiettivo

Portare un modello open-weight di semi-frontiera a girare **bene** su questo
MacBook Pro (M5 Pro, 64 GiB, 307 GB/s), con tutte le ottimizzazioni che
questa macchina permette, usato **dal terminale come DwarfStar**. Il modo
resta quello di picoforge: codice leggibile, ogni tecnica validata contro un
oracolo e misurata da sola.

La famiglia scelta è **Qwen3.8**. Analisi del 16/10: è l'unica famiglia di
semi-frontiera che entra in 64 GiB. Le alternative valutate o non entrano
(DeepSeek V4 Flash, GLM-5.3-Flash, gpt-oss-120b), o non hanno ancora pesi
confermati (Nemotron 3.5 Bolt, Mistral Large 4), oppure sono di una
generazione precedente (Qwen3.6-35B-A3B).

Tre configurazioni, una alla volta, in quest'ordine (scelta mia, riordinabile):

| # | Configurazione | Perché | Riferimento |
|---|---|---|---|
| 1 | **Flash-Next Q2, residente** (41,7 GiB) | Il più veloce. Il grafo nativo è già corretto | DwarfStar sullo stesso Mac: 30,9 tok/s decode, 600,7 tok/s prefill a 8K |
| 2 | **Flash-Next Q4K, esperti in streaming da SSD** (69,7 GiB di pesi) | Qualità più alta sullo stesso modello | DwarfStar: «Q4 needs a larger machine» |
| 3 | **Qwen3.8-27B denso**, Q6_K/Q8_0 (Apache-2.0) | Qualità quasi originale, nessuna compressione aggressiva | Nessun runtime di riferimento in DwarfStar |

**Finito** significa: le tre configurazioni girano dal terminale, tutti i gate
sono verdi, i numeri sono misurati con il protocollo e il writeup è pubblicato.

## Cosa è originale, e cosa no

Verificato il 16/10 nel checkout pinned `fc80bd6`. DwarfStar usa già matvec
simdgroup, MoE fusi, TensorOps/Neural Accelerator nel prefill (kernel `*_nax_*`),
residency set, coda Metal 4 e MTP adattivo. **Le tecniche in sé non sono
nostre.** Sono originali di picoforge, da ricontrollare prima di qualsiasi
affermazione pubblica:

1. Streaming degli esperti da SSD per Qwen3.8, cioè Flash-Next Q4K su un Mac
   da 64 GiB. La documentazione di DwarfStar dà lo streaming su Metal solo
   per DeepSeek e GLM. Da confermare in R0 provando `ds4 --ssd-streaming`.
2. Il 27B denso nello stesso motore.
3. La **scala**: il guadagno misurato di ogni singola tecnica sullo stesso
   Mac, con l'oracolo che garantisce la correttezza di ogni gradino.
4. Il registro di correttezza: ogni kernel veloce con il suo oracolo e la sua
   tolleranza misurata.

## Il budget della macchina

Stime da confermare in R0. Valori calcolati da `inventory.json` e dalle model card.

| | Byte letti per token | Tetto decode (307 GB/s) | Note |
|---|---:|---:|---|
| Flash-Next Q2 | 6,1 GB | ~50 tok/s | DwarfStar ne usa il 62% |
| Flash-Next Q4K | ~7,7 GB di cui ~1,3 GB di esperti | dipende dagli hit in cache | Vedi sotto |
| 27B Q8_0 / Q6_K / Q4_K_M | ~29 / ~22 / ~16 GB | ~11 / ~14 / ~19 tok/s | Denso: tutti i pesi a ogni token |

- **Prefill**: ~2 FLOP per parametro attivo per token, cioè ~12 GFLOP per
  Flash-Next e ~54 GFLOP per il 27B. Il nostro miglior matmul misurato fa
  7,8 TFLOPS (TensorOps, `bench/matmul_m5pro.csv`).
- **Q4K in streaming**: con un budget di 48–50 GiB, tolti ~8 GiB di pesi non
  esperti e il runtime, restano ~36 GiB di cache. Ci stanno circa il 60% dei
  62 GiB di esperti. Con un tasso di miss m, ogni token legge da SSD
  m × 1,3 GB. La velocità dipende da m e dalla banda reale dell'SSD, che va
  misurata (R0/R5) prima di qualsiasi previsione.
- **Memoria**: 64 GiB fisici, working set Metal raccomandato 51,84 GiB, singolo
  buffer al massimo 38,88 GiB. Picco dell'applicazione 48–50 GiB, nessuna
  crescita swap sostenuta.
- **Disco**: 134 GiB liberi oggi. Il file Q4K pesa 165 GiB e **non ci sta**
  senza liberare spazio o senza un SSD esterno (più lento, da misurare).
  Decisione dell'utente all'inizio di R5.
- **Dispatch**: misurato ~1,4 µs minimo per dispatch dentro un command buffer
  (`bench/dispatch_m5pro.csv`). Con ~1.500–2.000 dispatch per token sono 2–3 ms
  dei ~32 ms disponibili a 31 tok/s. La fusione conta.

## Catalogo delle ottimizzazioni

Tutto ciò che conosciamo per questa macchina, e la tappa in cui entra.
"Già" significa che esiste ed è da mantenere.

| Tecnica | Perché qui | Tappa |
|---|---|---|
| Matvec quantizzati cooperativi: simdgroup, letture coalescenti, decodifica per blocco, `simd_sum` | Il decode è limitato dalla banda; oggi usiamo ~2% | R1 |
| Multi-GEMV: più proiezioni dello stesso x in un dispatch | Una sola lettura di x, meno dispatch | R1 |
| MoE fuso: gate+up+SwiGLU; down+somma pesata+esperto condiviso | Meno dispatch e meno traffico intermedio | R1 |
| Fusione elementwise (RMSNorm, mixer HC, residui) | ~40 dispatch per layer oggi | R1 |
| Barriere solo dove c'è una dipendenza dati | Oggi c'è una barriera dopo ogni op | R1 |
| Encoding CPU minimo: claim e binding precalcolati, eventuale pre-encoding | Oggi si fa il claim dei 1.256 tensori a ogni token | R1 |
| Residency set per i pesi mmappati | Pesi caldi, niente fault alla prima lettura | R1 |
| Argmax su GPU, readback dei logits solo se richiesto | 1 MB per token oggi | R1 |
| GEMM quantizzato `simdgroup_matrix`, poi TensorOps `matmul2d` sui Neural Accelerator | Il prefill è compute-bound | R3 |
| GEMM raggruppato per esperto | MoE in prefill | R3 |
| GDN a chunk (delta rule chunkwise) | Ricorrenza token-per-token in 36 layer | R3 |
| Attention di prefill a blocchi e QSA block-sparse | Contesto 8K | R3 |
| MTP adattivo 1–3, verifica multi-riga in un passaggio | Ammortizza la lettura dei pesi | R4 |
| Decisione MTP consapevole del costo (~1,4 GiB per bozza) | Una bozza vale ~¼ di token | R4 |
| Lettura n-gram sovrapposta al calcolo | Già bounded; da tenere fuori dal percorso critico | Già, poi R5 |
| Cache esperti con politica di sostituzione, `pread` parallele, `F_NOCACHE`, prefetch, I/O sovrapposto al calcolo | Q4K in 64 GiB | R5 |
| Thread di I/O con QoS adeguata | 18 core CPU liberi durante il decode | R5 |
| Limite wired `iogpu.wired_limit_mb` | Solo spiegato: decide e lancia l'utente | R5 |
| Attivazioni half / KV FP16 | Solo modalità opzionale con gate proprio | R3, R6 |
| Pesi mmap no-copy | Zero copie in memoria unificata | Già |
| Termica: `powermetrics`, ordine alternato, GPU mai ferma dentro un token | Misure oneste | Tutte |

## Tappe

| Tappa | Contenuto | Uscita |
|---|---|---|
| **R0** Misura | Probe delle capacità, profilo per op, byte per token reali, harness di confronto con DwarfStar, baseline | Tabella "op → µs, GB/s" e baseline in `bench/` |
| **R1** Decode Q2 | Matvec cooperativi per formato, MoE fuso, grafo snellito | Decode ≥ 0,95 × DwarfStar, gate verdi |
| **R2** Terminale | `./picoforge` interattivo come `./ds4`: chat multi-turno in streaming, `/help`, `/read FILE`, `/ctx N`, `/quit`, Ctrl+C, `-p`, `--mtp`, comando bench | Uso quotidiano dal terminale; UI web secondaria |
| **R3** Prefill | GEMM simdgroup poi TensorOps, esperti raggruppati, GDN a chunk, attention a blocchi | Prefill 8K ≥ 0,95 × DwarfStar |
| **R4** MTP | Profondità adattiva, verifica multi-riga | Confronto con `ds4 --mtp` sugli stessi prompt |
| **R5** Streaming SSD | Banda SSD misurata, cache esperti, prefetch, overlap; Flash-Next Q4K | Q4K gira su 64 GiB, numeri misurati, nessuno swap |
| **R6** 27B denso | Grafo 27B (riuso tokenizer, GDN, attention gated, MTP), Q6_K/Q8_0, prefill su NA | 27B dal terminale, gate contro un oracolo Transformers |
| **R7** Qualità | Concordanza top-1 e KL di Q2 contro Q4K e del 27B quantizzato contro Q8_0, suite di task congelata | Tabella qualità/velocità/memoria |
| **R8** Chiusura | Benchmark finali con il protocollo, scala delle tecniche, writeup, README | Progetto concluso |

Dipendenze:
- R2 parte dopo R1 (a 1 tok/s il terminale non è usabile).
- R5 riusa i kernel Q4_K/MXFP4 di R1 e il prefill di R3.
- R6 riusa i kernel di R1 e R3.
- R7 richiede R5 e R6.

## Non obiettivi

Agente di coding e server API (come `ds4-agent`/`ds4-server`): possibili dopo
R8, non necessari per "finito". Esclusi anche: visione, sessioni batched, più
Mac, CUDA. Il gate resta a contesto 8K; contesti più lunghi vengono misurati
in R8 se la memoria lo consente.

## Anti-obiettivi (falliamo anche se è veloce)

- Più veloce ma fuori dal gate, o con una deriva non spiegata.
- Kernel copiati da DwarfStar. È il riferimento delle tecniche e l'oracolo di
  velocità; i kernel sono nostri. Un'idea adattata si cita (MIT, commit pinned).
- Numeri senza protocollo, o un'esecuzione favorevole spacciata per risultato.
- Regressioni sui percorsi esistenti: Qwen3-0.6B, Qwen3-30B-A3B, test Qwen3.8.

## Regole comuni a tutte le tappe

**Correttezza.**
- I kernel oracolo attuali restano nel codice come percorso di riferimento
  selezionabile.
- Un kernel veloce cambia l'ordine delle somme, perché in FP32
  (a+b)+c ≠ a+(b+c). Per questo la bit-identità con la CPU cade, e lo
  dichiariamo come divergenza spiegata. Ogni kernel veloce si confronta con il
  suo oracolo con una tolleranza fissata **dalla distribuzione misurata**
  dell'errore e documentata.
- Il gate end-to-end **non si allarga**: logits ≤ 1e-3 contro il riferimento
  FP32 e greedy identico sulla suite congelata. I near-tie di routing si
  trattano come in `G2.md`.

**Precisione.** Di default attivazioni, accumulatori, KV e stato restano in
FP32: nel decode il costo è trascurabile. Half/FP16 entra solo come modalità
opzionale, con un nome esplicito e un gate proprio.

**Protocollo.**
- Stesso file, `--ctx 8192`, chunk 1024, greedy, GPU libera, alimentazione
  AC, `powermetrics` annotato.
- Almeno 3 warm-up e 20 misure, mediana e p10/p90, ordine alternato contro
  DwarfStar `fc80bd6`.
- Due frontiere: prompt breve, e 8063 + 128 token teacher-forced come
  `ds4-bench`.
- CSV grezzi e script dei grafici in `bench/`.
- Una baseline troppo lenta per 20 ripetizioni si misura con meno
  ripetizioni, e lo si dichiara.

**Modo di lavoro.**
- Prima di ogni componente: il concetto in 5–15 righe, poi il piano, poi il
  codice.
- Passi da ≤ ~150 righe, ognuno verificato e committato.
- Comando di verifica e domande di controllo nel riepilogo.
- Si lavora in un worktree su un branch da `main`, senza toccare
  `chore/repo-cleanup`, che ha modifiche non committate di un'altra sessione.

## Decisioni dell'utente lungo la strada

- Inizio R5: spazio su disco per il file Q4K da 165 GiB (liberare spazio, ad
  esempio i 86 GiB di Qwen3-30B-A3B, oppure SSD esterno) ed eventuale limite wired.
- Inizio R6: quale GGUF del 27B usare. Non c'è un repository ufficiale Qwen in
  GGUF: o un'altra conversione pubblicata, o la conversione nostra dai
  safetensors ufficiali.
- Ogni modalità a precisione ridotta prima di diventare disponibile.
