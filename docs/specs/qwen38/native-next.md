# Incrementi nativi dopo G2/G3

Base verificata `31d346056db6ba446fb8c10d76f0347dffaab8da`, nuovo worktree
`picoforge-native-next`, branch `coord/qwen38-native-next`. Gli altri worktree
non sono modificati. Venv e checkpoint sono quelli locali già presenti.

## Suite estesa

`tools/eval/qwen38_native_suite.py prepare` congela prompt, SHA256, token ID e
budget **prima** delle risposte. Il conteggio usa A tramite `--encode-only`,
senza aprire Metal. Coding Python/C e due file, ragionamento verificabile,
testo italiano; archivio sintetico con rettifica a contesti 1K/2K/4K/8K.
Il caso maggiore contiene 8042 token di prompt e riserva 128 nuovi token:
8170 <= 8192. Nessun troncamento silenzioso.

```sh
make picoforge build/process_memory
python3 tools/eval/qwen38_native_suite.py prepare bench/MIA-SUITE
# Esecuzione reale seriale, cartella di evidenze obbligatoriamente nuova.
python3 tools/qwen38_gpu_lock.py -- \
  python3 tools/eval/qwen38_native_suite.py run \
  bench/MIA-SUITE/suite.json bench/MIA-ESECUZIONE
# Reset nella stessa sessione e prefill con diverso chunk:
python3 tools/qwen38_gpu_lock.py -- \
  python3 tools/eval/qwen38_native_suite.py run \
  bench/MIA-SUITE/suite.json bench/MIO-REPLAY \
  --cases ragionamento --repeat 2 --replay-chunk 7
# Oracle indipendente CPU FP32, usa NumPy già presente nel venv:
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_native_bounded.py \
  bench/MIA-SUITE/suite.json build/MIO-GATE
```

Il gate bounded usa 24 token di ciascuno di tre prompt e fino a 4 nuove
scelte greedy, confrontate con DwarfStar CPU FP32 al commit congelato. Soglia
1e-3 invariata, controllo di argmax e margine. I controlli lunghi verificano
completamento finito, limiti, stop e memoria; non sono un oracle numerico
indipendente a 8K. I test sintetici G2 verificano tutte le history e guasti.
La qualità è separata: JSON verificabile; codice da revisionare prima dei test
eseguibili già disponibili; prosa italiana valutata manualmente.

`--chat-file`/`--complete-file` accettano testo fino a 4 MiB, senza NUL;
`--repeat N --warmup W` riutilizza pesi/grafo, resetta history a ogni passaggio.
`--replay-chunk` varia il chunk dopo il primo passaggio. `--stop-after N` e
SIGUSR1 interrompono fra completamenti GPU: nessuna cancellazione di un comando
in volo. `--runs-jsonl` rifiuta file esistenti e conserva ogni passaggio, inclusi
warmup, stop e token. `--token-ids`/`--logits` mantengono il comportamento CLI
precedente e, con ripetizioni, descrivono l'ultimo passaggio soltanto.

## Confini dei timer introdotti nella CLI

Tutti i timer usano `CLOCK_MONOTONIC`. `load_s`: ingresso nella CLI fino al
grafo/logits pronti; include metadata, tokenizer, mapping e allocazioni, esclude
prefill e il costo del lancio del processo. Non garantisce la residenza di ogni
pagina. `reset_s`: azzeramento degli stati per ripetizioni successive.
`prefill_s`: intorno a tutti gli append del prompt, incluso I/O n-gram,
encode/submit/wait e readback dell'ultima riga. Il primo token è scelto da
questa riga. `decode_s`: dopo scelta/emissione del primo token fino all'ultima
scelta/emissione o stop; numeratore `generated_tokens-1`, incluso lo stop.
`ttft_request_s`: inizio della richiesta (reset incluso) fino a prima scelta ed
emissione/flush; null se nessun token. Può trattarsi di uno stop senza testo.
`first_token_from_entry_s` include load, soltanto per la prima richiesta.
`request_s` comprende reset/prefill/generazione ma esclude serializzazione delle
statistiche e teardown. Il runner conserva anche tempo processo e primo byte
stdout osservato: non sono gli stessi confini.

Memoria: `currentAllocatedSize` Metal separato da buffer del grafo, footprint,
wired/pressione/swap di sistema. Il sampler esistente conserva i contatori a
0,5 s e arresta dopo tre campioni critici o crescita swap >2 GiB. I byte letti
dal processo non sono automaticamente letture fisiche del solo n-gram.
Cache OS e shader non controllate; nessuna dichiarazione di cache cold.

## Stato della consegna

La prima prova estesa ha completato C, ragionamento e italiano (373 token,
stop incluso). Il prefill scalar a 892 token è stato interrotto dopo 618,55 s:
`extended/manifest.json` rimane **incompleto**, con output e memoria conservati.
Le sole prove precedenti a questa interruzione sono valide. Il vecchio valore
`reset_replay_identical: true` nei casi a singola esecuzione era vacuo: il
summarizer lo corregge a `null`; il runner corrente richiede almeno due passaggi.

Per rendere praticabile la progressione è stato anticipato un intervento QSA
circoscritto, verificato e misurato. Il kernel scalar assegnava un'intera testa
e 256 accumulatori a un thread. Il kernel cooperativo assegna una testa a un
threadgroup: prodotti e 16 dot product paralleli, ma ogni somma, softmax e
accumulatore conserva l'ordine FP32 precedente. Nessuna riduzione tree, FMA o
precisione ridotta. `Q38_QSA_SCALAR=1` mantiene il riferimento originale.

`qsa-paired-v3/report.json`: sei forme con 64–8192 posizioni, fino a 2048 ID
selezionati, warm-up di almeno 0,5 s GPU e 25 misure; output **bit-identici**,
speedup del solo ATTEND 27,55–55,93×. `qsa-paired/` conserva la prima variante,
più lenta (6,2–6,45×), senza sostituire le evidenze.

`native-paired/`: confronto completo sul prompt di ragionamento, 61 token,
14 generati incluso stop, contesto 8192 e chunk 128; un warm-up e tre misure
residenti per variante. Mediane scalar → cooperativo: prefill 9,889 → 8,647 s;
decode 1,058 → 1,116 token/s; richiesta 22,181 → 20,303 s. Tutte le otto
sequenze coincidono. È un **pilot di un solo prompt**, con cache non controllate.
Il risultato aritmetico è errato in entrambe le varianti (1002 anziché 1633).

Metal 42,387 GiB, grafo 0,665 GiB, footprint lifetime 0,801 GiB nel processo
cooperativo; pressione normale e nessuna crescita swap. Footprint/memoria sono
osservati sull'intero processo, warm-up compreso. Nella mediana cooperativa,
prefill `prepare_s` 0,000607 s contro `gpu_s` 8,640 s: su questo caso caldo
l'I/O n-gram non è il collo di bottiglia. Nessuna conclusione su SSD cold.

La suite estesa cooperativa prosegue in `extended-cooperative/`, chunk 128.
Le prove reali bounded CPU FP32 aggiuntive e la progressione 8K non sono ancora
completate. Un primo tentativo bounded è rimasto incompleto durante un riavvio
del Mac; la causa del riavvio non è stata determinata. Non è un PASS.

Passano QSA Metal (13 test), grafo G2 (5), stato/fault (1), CLI (4), sessione
residente (2), HTTP con fixture reale (2), observer (6) e contratti (4).
Il test di template inizialmente fallito aveva omesso il blocco `<think>` vuoto
nelle risposte precedenti, conservato dal template ufficiale. La correzione
è nel test, non nel renderer A.

La chat è implementata e i test del servizio passano. Il browser ha verificato
caricamento/attesa del lock e layout desktop/mobile; lo scambio sul modello
reale resta da verificare dopo la suite seriale. **MTP non implementato.**
Il FAIL DwarfStar Metal FP16 rimane quello documentato in [G2.md](G2.md), distinto
dal PASS indipendente CPU FP32. Nessun allargamento dei gate.

## Benchmark e profilo

```sh
make picoforge build/process_memory build/qwen38-attention-bench
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_native_bench.py \
  bench/qwen38_native_20261013/suite/suite.json bench/MIO-BENCH \
  --repeat 3 --warmup 1 --chunk 128 --compare-scalar
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/bench_qwen38_attention.py bench/MIA-QSA
```

`prefill_profile`/`decode_profile`: `prepare_s` include pread bounded,
deduplicazione, decodifica BF16 e staging; `encode_s` include claim, encoding e
submit; `wait_s` attende il completamento; `gpu_s` usa i timestamp Metal e si
sovrappone a wait, quindi **non va sommato**. `ngram_calls`, `ngram_bytes` e
`ngram_rows` contano chiamate pread, byte logici letti e righe uniche; non sono
il traffico fisico SSD. Metadati, argv/env espliciti, SHA dei piccoli artefatti,
revisione/diff e campioni grezzi identificano ogni esecuzione. Il GGUF non viene
rihashato. Percentili p10/p90: campione ordinato più vicino, non intervalli di
confidenza. I warm-up sono conservati e esclusi dalle statistiche di velocità.

## Sessione locale

```sh
make ui-qwen38
# http://127.0.0.1:8001 — il worker acquisisce già il lock comune.
make test-qwen38-session test-qwen38-server
```

Il servizio Python standard library espone una sola conversazione locale,
condivisa dalle schede aperte. Un solo processo `--session` mantiene pesi e
grafo residenti e detiene il lock; gli accessi sono serializzati. Ogni richiesta
renderizza una volta l'intera storia tramite il template A, resetta gli stati
e ne esegue il replay. Non c'è riuso incrementale del prefisso. Il budget è
prompt renderizzato + massimo generabile: se supera il contesto la richiesta
fallisce senza troncamento e la storia precedente resta disponibile.

Stop tramite SIGUSR1, osservato tra append: il comando GPU in corso termina.
Il testo parziale effettivamente emesso rimane nella storia; stop prima del
primo testo non aggiunge un turno. Nuova conversazione azzera storia, decoder
e cache. Disconnessione HTTP richiede stop e drena il worker prima di rilasciare
l'accesso. Il servizio ascolta solo su loopback e verifica Host/Origin.
`ttft_request_s` del worker esclude parsing, template e tokenizzazione della
richiesta; non equivale alla latenza percepita dal click nel browser.
