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

La suite CLI estesa (4 test) e la validazione record passano. Le prove reali
sono in esecuzione sotto `bench/qwen38_native_20261013/extended`; il manifest
`complete` e i risultati registrati sono l'autorità, non questo testo.
Benchmark con ripetizioni, chat live, ottimizzazioni e MTP restano aperti.
Il FAIL DwarfStar Metal FP16 rimane quello documentato in [G2.md](G2.md), distinto
dal PASS indipendente CPU FP32. Nessun allargamento dei gate.
