# MTP nativo FP32 — prove e pilot appaiato

M5 Pro 64 GiB, 13 ottobre 2026. Incremento su `coord/qwen38-native-next`
dopo `888f47f`; ogni run identifica sorgenti, diff, binario e metallib effettivi.
Il checkpoint Q2 esistente non è stato scaricato, convertito o ricalcolato
integralmente. Il predictor usa i pesi originali del blocco 48 e cache FP32.
Percorso ordinario disponibile; MTP opt-in, una proposta greedy per ciclo.

## Risultati

| Caso | Prompt / risposta | Decode ordinario | Decode MTP | Speedup decode | Accettazione |
| --- | --- | --- | --- | --- | --- |
| Coding completion | 9 / 32 token | 1,068 token/s | 1,675 token/s | 1,568× | 45/48 (93,75%) |
| Italiano chat | 31 / 64 token | 1,065 token/s | 1,262 token/s | 1,185× | 78/108 (72,22%) |

Un warm-up escluso + tre ripetizioni residenti per modalità, ctx 8192, chunk
128. Ordine ordinario→MTP nel coding, MTP→ordinario nell'italiano. Tutte le
16 sequenze sono identiche, warm-up e replay inclusi. Risultati mediani, due
pilot brevi: nessuna generalizzazione del vantaggio a tutti i prompt.

Richiesta coding 30,939→20,473 s (1,511×); italiano 63,984→54,846 s (1,167×).
TTFT coding 1,917→1,968 s, italiano 4,818→4,930 s. Il load è una sola
osservazione per processo: 0,194/0,202 s e 0,187/0,195 s ordinario/MTP.
Non comprende la residenza di tutte le pagine; non equivale a un avvio cold.

Il decode parte dopo la prima scelta greedy e termina dopo l'ultima scelta
o stop; velocità = (token generati − 1) / decode_s, stop incluso se scelto.
Include proposta, verifica, snapshot e replay. Prefill comprende tutti i
token del prompt e tutte le coppie MTP; TTFT della richiesta comprende reset,
prefill e prima scelta. I/O/stdout e caricamento hanno confini separati:
[contratto temporale](../../docs/specs/qwen38/native-next.md).

Metal ordinario 42,387 GiB, MTP 42,454 GiB; snapshot host MTP 588.849.160 byte
(561,57 MiB) separato. Footprint massimo MTP 1,410 GiB, pressione sempre 1,
nessuna crescita dello swap osservata. Le variazioni finali swap per coding
ordinario/MTP sono −96/−672 MiB, italiano MTP/ordinario −8/−32 MiB.
Footprint, Metal e wired di sistema non si sommano. Cache OS/SSD non controllate.
N-gram disk-only con pread bounded: i byte logici non misurano I/O fisico SSD.

## Correttezza e prove conservate

- `gate/report.json`: DwarfStar CPU FP32 invariato, commit
  `fc80bd695da76ee14bcb86820bf900a5cdfda806`. Coding 14 righe trunk + 13 MTP,
  italiano 10 + 9, cinque passi di decode ciascuno. Tutti i 24/22 argmax e
  margini PASS; gate congelato 1e-3, massimo MTP per riga 1,43e-5. Logits
  ordinari chunk/decode bit-identici. Raw f32 restano in
  `build/qwen38-mtp/gate-real-01`, SHA e provenienza registrati nel report.
- `packed/`: manifest di 18 righe Q4_K (25.920 byte) e 9 MXFP4 (3.060 byte)
  reali, riferimento pinned e licenza. Dati bounded in `build/qwen38-mtp`.
- `paired/`: configurazioni, comandi, ID, output grezzi, osservazione memoria,
  hash e statistiche. `report.json` originale immutato; `summary.json`
  corregge esclusivamente `load_s.n` da 3 a 1, perché un solo processo ha
  caricato il modello. Contiene nota e SHA del report di origine.
- `browser/`: chat reale e report desktop 1280×850/mobile 390×844, evidenze
  e screenshot. Un worker, storia Ada preservata fra turni, streaming,
  stop e reset. Il report mostra prove registrate; la chat è il servizio locale.
- `tests/`: log completi, inclusi fallimenti intermedi e successive correzioni.

Il runner gate v1 sovrascriveva i campioni della prima attesa di rilascio Metal
con quelli della seconda prima dell'oracle CPU. Il sorgente esatto è conservato
in `gate/runner-source.py`; output dei processi e gate restano completi.
Il runner corrente separa le due attese. Il benchmark v1 è conservato in
`paired/runner-source.py`; la sola correzione dei metadati load non richiede
nuova inferenza. Nessuna evidenza preesistente è sovrascritta.

## Test eseguiti

- Packed CPU: 27 PASS; Metal finale: 27 PASS, nessuno skip, comprese slice
  reali Q4_K/MXFP4. Ricostruzione MXFP4 bit-esatta anche per subnormal/signed zero.
- G0, G2 (5), G2 state (1), CLI (4), session (2), server ordinary/MTP (7): PASS.
- MTP teacher-forced 9 e 33 token: chunk/reset/replay bit-identico, tutte le
  history e PLE dopo rollback, ticket invalidi/busy/stale, guasti e recupero: PASS.
- Speculativa (5): accettazione/rifiuto/falso stop, stop a ogni posizione,
  capacità 0–5, chunk 1, SIGUSR1 e reset: PASS. Fixture controllate via pesi,
  senza alterare il percorso runtime.
- A: config/binding/metadata e 20 test Python, 36 fixture chat HF con 364
  encode e 524 decode: PASS. Link e 9 controlli tooling nativo: PASS.
- Werror, ASan/UBSan e validazione Metal dei relativi harness; sintassi JS e
  Python verificata. `test-all` legacy Qwen3 non eseguito.

Il primo test packed CPU falliva perché l'harness non ammetteva i due nuovi
tipi; corretto e rieseguito. Il primo test speculativo ha trovato un buffer ID
QSA insufficiente ai contesti 1/2: correzione verificata, allocazione a 8192
invariata. Un primo giro Metal aveva skip sulle slice: la prova finale le
include. Il comando cumulativo delle regressioni terminava con un target
inesistente `test-qwen38-a-cpu`; i target precedenti passavano, poi il target
corretto `test-qwen38-a` è stato eseguito con successo nel log separato.

## Riproduzione e limiti

```sh
make picoforge picoforge.metallib build/process_memory build/qwen38-mtp/mtp_real
make test-qwen38-mtp test-qwen38-speculative
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_mtp_bench.py bench/MIO-BENCH-MTP
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_mtp_gate.py build/MIO-GATE-MTP
tools/venv/bin/python tools/ui/qwen38/server.py --mtp
```

Le cartelle di output devono essere nuove. Target Make GPU e servizio
acquisiscono il lock comune: non annidare wrapper, non avviare benchmark
finché la chat possiede la GPU. Nessuna dipendenza aggiunta o limite wired cambiato.

MTP reale a 8K ancora non verificato: il contesto è allocato, i prompt MTP
misurati sono brevi. Il percorso ordinario ha invece completato il prompt
8042 + 28 token, prefill 34,1 minuti. Nessun sampling o draft multiplo.
La qualità resta separata dalla correttezza: errore aritmetico 1002 vs 1633
nella suite ordinaria. Il gate storico DwarfStar Metal stock FP16 resta FAIL;
il PASS CPU FP32 è distinto e le tolleranze non sono state allargate.

[Report interattivo completo](../qwen38_native_20261013/report-mtp-final.html)
e [contratto MTP](../../docs/specs/qwen38/MTP.md).
