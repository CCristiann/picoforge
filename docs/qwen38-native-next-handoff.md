# Consegna nativa e MTP — 2026-10-13

I cinque incrementi sono implementati e verificati entro il perimetro descritto
qui. L’utente ha successivamente chiesto di portare i risultati su main,
superando il precedente vincolo di pubblicazione sul solo branch di lavoro.
Alla preparazione di questo commit, origin/main è `7c0b237`, antenato del
branch, senza divergenza o modifiche locali nel checkout principale.
L’esito dell’integrazione è verificabile nella cronologia Git/PR.

## Checkout e vincoli mantenuti

Worktree `/Users/cristiancirje/Desktop/Dev/picoforge-native-next`, branch
`coord/qwen38-native-next`, base `31d3460`. Incrementi precedenti pubblicati:
`d5b1732`, `44ec203`, `888f47f`; il commit che contiene questa consegna aggiunge
MTP, gate, benchmark appaiato e report finale. Nessun worktree altrui modificato.

`tools/venv` è un symlink locale non tracciato al venv del checkout principale;
non aggiungerlo. `models` riusa gli asset esistenti. Nessuna nuova dipendenza,
download, conversione completa, checksum integrale o modifica del limite wired.
DwarfStar locale `../ds4-qwen38-baseline` rimane al commit invariato
`fc80bd695da76ee14bcb86820bf900a5cdfda806`.

Ogni workload GPU/pesante usa il lock comune `.git/qwen38-gpu.lock`.
I target Make GPU e il servizio chat lo acquisiscono: non annidare wrapper.
Arrestare il servizio chat prima di avviare benchmark o test GPU.

## Risultati consegnati

1. Suite coding, ragionamento e italiano, risposte estese, contesti progressivi
   fino a 8042 token reali di prompt + 28 generati. Risposta JSON 8K corretta;
   prefill 2046,586 s (34,1 minuti), decode 25,296 s, Metal 42,387 GiB,
   footprint 0,837 GiB, pressione 1 e variazione swap zero. Chunk/reset/replay,
   stop e gate CPU bounded separati dalla valutazione di qualità.
2. Benchmark nativo con confini temporali, warm-up, ripetizioni, hash,
   configurazioni, stdout/stderr e osservazione memoria; evidenze append-only.
   Cache OS/SSD non controllate, nessuna dichiarazione cold. I/O n-gram logico
   e tempo di preparazione misurati; nessuna attribuzione a letture SSD fisiche.
3. Chat locale reale con worker unico residente, streaming, stop, reset,
   capacità/errori, storia renderizzata una volta da A e reset/replay per turno.
   Browser desktop/mobile verificato sul percorso ordinario e MTP.
4. QSA cooperativa FP32 bit-identica su sei forme, ATTEND 27,55–55,93×.
   Pilot nativo ordinario 1,058→1,116 token/s, richiesta 22,181→20,303 s.
5. MTP reale, decoder Q4_K/MXFP4, predictor con cache distinta, verifica
   greedy di una proposta, snapshot completo e rollback/replay. CPU indipendente:
   24 righe trunk e 22 MTP, argmax/margini PASS, massimo MTP per riga 1,43e-5
   contro gate 1e-3 invariato. Pilot appaiati: coding 1,068→1,675 token/s
   (1,568×, 93,75% accettazione), italiano 1,065→1,262 (1,185×, 72,22%).
   Tutte le 16 sequenze identiche. MTP Metal 42,454 GiB, snapshot host
   561,57 MiB separato; footprint massimo 1,410 GiB, pressione 1 e nessuna
   crescita swap osservata nei quattro processi.

## Evidenze e comandi

- `bench/qwen38_native_20261013/README.md`: suite estesa, 8K, QSA, benchmark,
  gate bounded precedenti, browser ordinario e tentativi incompleti.
- `bench/qwen38_mtp_20261013/README.md`: gate/pilot MTP, raw, provenienza,
  test e browser. Correzioni dei soli metadati descritte senza cancellare v1.
- `bench/qwen38_native_20261013/report-mtp-final.html`: report interattivo
  cumulativo; gli snapshot precedenti restano conservati.
- `docs/specs/qwen38/MTP.md`: contratto proposta/verifica/cache/rollback.

```sh
make picoforge picoforge.metallib build/process_memory build/qwen38-mtp/mtp_real
make ui-qwen38
# In alternativa, servizio MTP (stessa porta locale 8001):
tools/venv/bin/python tools/ui/qwen38/server.py --mtp
# Solo dopo aver arrestato la chat, output nuovo obbligatorio:
tools/venv/bin/python tools/qwen38_gpu_lock.py -- tools/venv/bin/python tools/eval/qwen38_mtp_bench.py bench/MIO-BENCH-MTP
```

## Verifiche finali e limiti

PASS: packed CPU/Metal 27 test ciascuno, slice reali incluse senza skip nel
run Metal finale; A, G0, G2, state, CLI, session, server ordinary/MTP; MTP
teacher-forced 9/33 token e rollback completo; cinque test speculativi con
accept/reject/falso stop, capacità 0–5, cancellazione e reset; link e tooling
nativo. Harness con Werror, ASan/UBSan e validazione Metal dove previsti.
Log completi, inclusi errori intermedi corretti, nella raccolta MTP.
`test-all` legacy Qwen3 non eseguito.

MTP non ancora verificato su prompt reale 8K: è un limite aperto, distinto
dalla prova ordinaria a 8K completata. Solo greedy, un draft; niente sampling
o draft multipli. I due pilot non garantiscono vantaggi su ogni input.
La qualità non è certificata: errore aritmetico 1002 anziché 1633 e violazioni
di formato coding restano registrati. Il FAIL DwarfStar Metal stock FP16
(max 0,0413082) resta distinto dal PASS CPU FP32; nessuna tolleranza allargata.
I precedenti tentativi bounded interrotti per memoria/riavvio restano incompleti.

Prossimo lavoro utile: gate/parità MTP con contesti progressivi reali,
campione prestazionale più ampio, attribuzione del divario GPU/wait nel
prefill lungo e riduzione del costo di rollback senza cambiare precisione.
