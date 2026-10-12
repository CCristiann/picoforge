# Qwen3.8 Q2 — base integrata A–F

2026-10-12, Mac M5 Pro 64 GiB. Branch `coord/qwen38-integrate`, da
BASE_CONCORDATA `c674b23bd1377b2ce0acef2d06fac8b160809446`.
Il commit che aggiunge questo report è BASE_INTEGRATA; lo SHA viene
restituito nella consegna, senza un hash autoreferenziale nel file.
Nessun merge/push su main. Direzione: Qwen3.8-Flash-Next Q2, C11/Metal.

## Risultato e perimetro

| Gate sulla base combinata | Esito / copertura |
|---|---|
| Build C11/ObjC + metallib legacy/B/C | PASS |
| Regressioni `make test-all` | PASS completo, zero skip |
| A | 20 test; config/catalogo, encode/decode/stream/chat; ASan/UBSan |
| B CPU | 25 test, 245 prodotti, sei formati, decode bit-esatto |
| C CPU | 35 test, 102 misure oracle/native; harness ASan/UBSan |
| D CPU | 40 test, core su input preparati e lifecycle; ASan/UBSan |
| E CPU | 15 test, hash/reader/gate; fault injection e ASan/UBSan |
| F CPU | 23 test, routing/grouping; ASan/UBSan |
| Contratti / runner | Header pubblici C11+ObjC, 4 test lock, strict runner PASS |
| Collegamenti pubblici | A catalogo→B proiezione→F route, C RMS→E gate, config A→D/E PASS |
| A storage Metal | Attach/byte-span/owner su piccolo GGUF sintetico PASS |
| B Metal | 25 test, 245 prodotti, metallib comune e Metal API validation PASS |
| C Metal | Primitive, stato, alias/errori, GDN T≤32768 piccolo e T=1/65 reale PASS |

**158 test Python CPU A–F, zero saltati.** A viene rieseguita anche nel target
GPU prima del piccolo harness storage: non è un ulteriore gate numerico GPU.
La suite CPU aggregata è stata rieseguita con successo per verificare i
target finali e i report senza sovrascrittura; preparazione asset ripetibile.
Regressioni Qwen3 e utility: `make test-all` concluso con exit 0 dopo la
correzione del percorso Python negli shard; rieseguita l’intera suite.
Include 55 run speculativi e routing fino a 4096 token. Dettagli in
`summary.json` e `regression.log`.

B: 137 slice reali, **400164 byte packed** totali, escluse tabella PLE e MTP.
Massima frazione del bound analitico B: **0.005362122088118356** per entrambi
i backend, senza allargare le soglie. La parità decode riguarda i bit FP32
nel dominio dichiarato; underflow FTZ Metal registrato a parte nel report.
Non si trasferisce questa evidenza al forward di un intero layer.

C: massimo errore scalato CPU nelle 102 misure **1.4290722042934374e-6**;
Metal **3.60608101e-5**, entro gate locali/lunghi congelati. Il picco
**87130112 byte** è `currentAllocatedSize` del harness Metal, non footprint,
RSS o residenza dei pesi. E: max |gate−ref| **2.68220901e-7**, |dot−ref|
**3.33786011e-6** nelle fixture previste, non PLE completo.

## Revisione e correzioni

- I sei branch discendono dalla base concordata; le consegne non modificano
  file in comune e rispettano l'ownership. Provenienza in `deliveries.json`.
- Header pubblici/ABI e soglie invariati. Config A, padding B, shape C,
  lifecycle D, reader E e routing F controllati sui confini condivisi.
- C GPU non riconosceva due oggetti Metal no-copy sulla stessa memoria
  Shared. Riprodotto con assertion fallita prima della correzione; ora
  controllo per indirizzo, incluso alias dell'error word, e test passato.
  Per l'in-place esatto si usa il medesimo oggetto buffer e offset.
- Il target contratti include solo header pubblici; quelli privati con
  funzioni static inline sono verificati nei rispettivi harness.
- A usa ora il runner strict e UBSan fatale. Un test unitario del runner
  verifica che PASS, errore numerico e skip abbiano exit status corretti.
- Il test shard Qwen3 usava un percorso venv fisso; corretto in
  `sys.executable`, mantenendo interprete e dipendenze scelti dal caller.
- Tutti i workload Metal usano lo stesso lock nel git-common-dir.
  Nessun kernel nuovo viene considerato verificato dal solo link riuscito.

## Riproduzione

Prerequisiti già presenti: Xcode/Metal 4, venv del progetto, checkout
DwarfStar fissato e GGUF verificato della fase 0. Nessuna nuova dipendenza.
Dal worktree integrato:

```sh
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
make qwen38-test-assets PYTHON="$Q38_PY" \
  Q38_REFERENCE_ROOT=/Users/cristiancirje/Desktop/Dev/picoforge \
  Q38_DWARFSTAR=/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline \
  Q38_SOURCE_DIR=/Users/cristiancirje/Desktop/Dev/picoforge-f-v1/build/qwen38-f/upstream
make -j4 picoforge PYTHON="$Q38_PY"
make test-qwen38-cpu PYTHON="$Q38_PY"
make test-qwen38-gpu PYTHON="$Q38_PY"
```

`Q38_CHECKPOINT` può indicare esplicitamente il GGUF. La preparazione copia
solo modeling/configuration/LICENSE upstream dopo hash-check; senza
`Q38_SOURCE_DIR` scarica esplicitamente i tre file fissati. Nessun download
nei test. B verifica identità del GGUF già autenticato, legge slice bounded
e costruisce nel build locale il decoder DwarfStar immutato con sanitizer.
Il replay verifica hash, decodifica e prodotti FP64 dei file già preparati;
non rifà un checksum completo di 137 GiB.

I target GPU acquisiscono il lock internamente: **non** avvolgerli in un
altro wrapper. Per la regressione legacy `test-all`, che non lo acquisisce,
si usa invece il wrapper esterno:

```sh
"$Q38_PY" -B tools/qwen38_gpu_lock.py --timeout 60 -- \
  env HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 UBSAN_OPTIONS=halt_on_error=1 \
  make test-all PYTHON="$Q38_PY" \
    -o models/Qwen3-0.6B-q8_row-embed \
    -o models/Qwen3-0.6B-q4_g32 \
    -o models/Qwen3-0.6B-q4_g32-embed
```

Per questa esecuzione soltanto, i quattro percorsi `models/Qwen3-0.6B` e
varianti quant sopra sono symlink ignorati ai corrispondenti asset già
presenti nel checkout principale, usati in lettura. I tre `-o` impediscono
rigenerazioni nei percorsi condivisi; non saltano alcun test. Le fixture
`build/tiny-qwen3-moe*` vengono invece generate in questo worktree. Nessun
symlink al venv richiesto. I workload legacy restano nei limiti dei loro
gate già esistenti; non cambiano la precisione scelta per Qwen3.8.

## Evidenza e limiti

`summary.json`: esiti, metriche, ambiente e hash degli artefatti.
`cpu.log`, `gpu.log`, `regression.log`: output completi delle esecuzioni
finali. `packed-cpu.json`, `packed-gpu.json`, `recurrent-cpu.json`: metriche
per caso e provenienza. `api.json`: **59 funzioni pubbliche implementate,
23 dichiarate soltanto**; `source-sha256.json` identifica i sorgenti verificati.
Non sono committati checkpoint, payload delle slice, build o sorgenti upstream.

D manca ancora `qwen38_attention_cpu`; E manca lifecycle pubblico e PLE;
F manca create/work/forward. C non ha ancora verificato HC/GDN composti con
proiezioni Q2 reali. Restano grafo 48 layer, lifecycle GPU pubblico composto,
commit globale/reset/replay, logits e generazione Qwen3.8 nativi, qualità Q2
contro BF16, memoria e prestazioni end-to-end. La baseline DwarfStar della
fase 0 rimane il riferimento; questo commit non sostituisce il suo runtime.

Ripartenza autorizzata C/D/E/F: [prompt specifici](../../docs/specs/qwen38/resume.md).
Contratti: [integration.md](../../docs/specs/qwen38/integration.md#11-base_integrata-v1--2026-10-12).
