# MTP FP32 — predictor e verifica greedy

Il contratto è stato fissato prima del codice. Il predictor ora esegue il
blocco reale del checkpoint, con cache propria e verifica greedy di una proposta.
`--mtp` è esplicito; il percorso ordinario resta disponibile e predefinito.
Gate indipendente, test di stato e confronto prestazionale completati.
Non sono implementati draft multipli, sampling o riuso incrementale fra turni.

## Riferimenti verificati nel checkout locale

DwarfStar invariato, commit `fc80bd695da76ee14bcb86820bf900a5cdfda806`,
`ds4.c`: `qwen4_ref_mtp`, `qwen4_graph_mtp_steps`, `qwen4_ref_row`.
Il sorgente è in `../ds4-qwen38-baseline/` rispetto al worktree. Il percorso
diagnostico `--cpu --first-token-test` con `DS4_QWEN4_MTP_OUT` salva i logits
teacher-forced `[n-1,vocab]` del predictor. Le attivazioni/cache sono FP32;
l'oracle CPU usa anche accumuli double nelle riduzioni, come il gate G3.
Non usare il backend Metal stock a cache FP16 come gate FP32.

L'inventario A descrive un blocco MTP QSA dopo i 48 blocchi ordinari. I ruoli
aggiuntivi sono `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` e il mixer
`nextn.hc_head`. Gate/up degli esperti sono Q4_K, down MXFP4; i decoder B
ora li eseguono con decoder dedicati CPU/Metal. Il predictor non ha PLE. I pesi MTP sono già
nelle viste storage del GGUF: non serve un secondo caricamento del modello.

Per il predictor all'indice i:

1. Prendere il residuo **prima del mixer finale** del trunk all'indice i e
   l'embedding del token confermato i+1.
2. RMS dell'embedding su E; RMS del residuo su **H×E**, non per ramo HC.
3. Proiettare `[e_norm, 0]` una volta e `[0, h_norm_s]` per ciascun ramo s
   con lo stesso `eh_proj`; sommare i due risultati per ramo.
4. Eseguire HC-attention, QSA all'indice i nella cache **propria del MTP**,
   HC-FFN e MoE, poi il mixer `nextn.hc_head` e la LM head condivisa.
5. Argmax deterministico sull'intero vocabolario, tie al token più piccolo.

## Una proposta per ciclo, verifica di due token

Il primo token a viene dall'ordinario e può essere emesso subito. MTP propone
b usando `(R_i,a)`. Il trunk verifica `[a,b]` in un solo append causale,
restituendo due righe di logits. L'argmax della prima deve essere uguale a b
per accettarlo; la seconda fornisce il prossimo token ordinario c.
Non usare soglie di confidenza per accettare un token diverso dall'argmax.

Prima della verifica va salvato lo stato del trunk: GDN, convolution history,
K/V, indexer/pooling QSA, PLE convolution e history/hash n-gram, posizione e
validità. Nessun peso o tabella SSD entra nello snapshot. Se b è rifiutato,
ripristinare il punto precedente alla verifica e ricalcolare solo a; il token
corretto viene dalla prima riga verificata. La cache non deve contenere b.
I ticket anteriori al ripristino diventano stale tramite cambio di epoch.
Il ripristino è ammesso solo a GPU idle, sulla stessa istanza, dopo append
riusciti e entro la finestra prevista; un errore GPU richiede reset completo.

Il predictor avanza solo su coppie confermate. Dopo l'accettazione di b,
elabora insieme `(R_a,b)` e `(R_b,c)`, leggendo soltanto la seconda riga
della LM head per la proposta successiva. Dopo un rifiuto non ha ancora elaborato b, quindi non deve
conservare alcuna sua history. Il prefill inizializza tutte le coppie del
prompt, anche al confine dei chunk. Nessun warm-start con cache MTP vuota a
una posizione arbitraria presentato come prefill completo.

## Stop e capacità

Tutti i limiti contano prompt + token emessi, stop incluso. Nessuna proposta
fuori dal contesto residuo. Con meno di due posti utili o limite di risposta
insufficiente, usare l'append ordinario. Stop proposto non è stop effettivo
finché la verifica non lo conferma. Se a è stop, terminare senza MTP.
Se b è accettato ed è stop, non emettere c né mantenere una conversazione
oltre lo stop. Cancellazione cooperativa tra append, mai pubblicazione di
proposte non verificate. Reset azzera entrambi i grafi e invalida snapshot.

## Contratto delle risorse e degli errori

`qwen38_gpu_mtp_enable` ammette esattamente un blocco MTP su grafo fresco.
`qwen38_gpu_mtp` legge solo un intervallo contiguo di residui dell'ultimo append
trunk completo; la posizione deve coincidere con quella del predictor. Gli
operatori e gli scratch sono condivisi, i ledger G0 e le cache QSA distinti.
Una sola chiamata sincrona alla volta; nessuna seconda mappatura dei pesi.

Lo snapshot host viene allocato una volta e riutilizzato. Copia le history
canoniche complete del trunk (anche le porzioni di cache inutilizzate), il
residuo dell'ultimo append e gli ultimi token usati dall'hash PLE. Esclude pesi,
workspace PLE e tabella n-gram. Posizione del trunk e contatori vengono
ripristinati; l'epoch avanza e gli stati E adottano la nuova epoch. La memoria
host è esposta separatamente da `qwen38_gpu_forward_bytes`/allocazioni Metal.

Il ticket dello snapshot contiene owner, epoch, seriale e posizioni: non può
essere modificato, riutilizzato o applicato dopo un altro save/reset/restore.
Restore rifiuta una finestra oltre due token o un predictor avanzato. Gli
errori GPU invalidano entrambi i grafi e non pubblicano logits; serve reset.
Un fallimento durante l'allocazione MTP richiede invece destroy.

La CLI verifica una proposta di stop facendo append soltanto del token a;
non inserisce lo stop nelle history. Il fallback terminale ordinario può
lasciare la cache MTP indietro: ogni richiesta successiva resetta e rigioca
entrambi i grafi. Durante cancellazione non vengono pubblicati draft non
verificati; se la coppia è già stata eseguita ma non emessa, il trunk torna
allo snapshot. I turni chat conservano il testo effettivamente emesso e
applicano una sola volta il template alla storia completa.

## Verifiche

- Decoder: Q4_K con scale/minimi a 6 bit e piani nibble, MXFP4 E8M0/E2M1,
  signed zero, subnormal e nonfinite; confronto bit per bit con DwarfStar.
  Metal richiede ricostruzione intera dei bit MXFP4 per evitare FTZ in decode.
  I test non allargano la tolleranza degli operatori FP32.
- Slice reali: 18 righe Q4_K (25.920 byte) e 9 MXFP4 (3.060 byte), tre esperti
  e tre righe per proiezione; decoder e prodotti CPU/Metal. Solo pread bounded,
  identità/header confrontati con la verifica esistente, nessun hash integrale.
- `test-qwen38-mtp`: teacher-forced chunk/reset/replay bit-identico, snapshot
  di tutte le history, PLE replay, ticket foreign/stale/alterati/busy, finestra
  superata, predictor avanzato e guasto tardivo MTP con output trattenuto.
- `test-qwen38-speculative`: fixture non degenere e transizioni controllate
  tramite i pesi reali del piccolo grafo, senza override nel codice runtime.
  Accettazione perfetta, rifiuto, falso stop, stop verificato, budget 0–5,
  chunk 1, cancellazione SIGUSR1 e reset della sessione residente.
- Gate reale CPU FP32: coding, 14 righe trunk + 13 MTP; italiano, 10 + 9.
  I prompt sono completamenti tokenizzati dal runtime con sei scelte greedy;
  l'oracle include cinque passi di decode. Tutti i 24/22 argmax e margini
  superano il gate 1e-3. Errori scalati trunk 2,40e-6 / 1,92e-6, MTP
  1,26e-5 / 9,88e-6. Massimo MTP per riga 1,43e-5. Logits ordinari
  bit-identici fra append di chunk e decode. CPU 83,55/58,83 s, pressione 1,
  variazioni swap -8 MiB / 0. Nessuna valutazione di qualità da questi casi.

## Pilot prestazionale misurato

M5 Pro 64 GiB, contesto allocato 8192, chunk 128. Un warm-up residente escluso
e tre ripetizioni misurate per modalità; ordine ordinario→MTP nel coding,
MTP→ordinario nell'italiano. Tutte le 16 sequenze (warm-up inclusi) sono identiche.

| Caso | Prompt / generati | Decode ordinario | Decode MTP | Rapporto | Accettazione |
| --- | --- | --- | --- | --- | --- |
| Coding | 9 / 32 | 1,068 token/s | 1,675 token/s | 1,568× | 45/48, 93,75% |
| Italiano chat | 31 / 64 | 1,065 token/s | 1,262 token/s | 1,185× | 78/108, 72,22% |

Mediane della richiesta: coding 30,939→20,473 s; italiano 63,984→54,846 s.
Il decode include proposta, verifica, snapshot e rollback/replay, senza
sottrarne i costi. Il prefill MTP è 1,963 / 4,925 s. Nel caso italiano il
replay mediano costa 9,46 s: il vantaggio dipende dall'accettazione.

Allocazioni Metal: ordinario 42,387 GiB, MTP 42,454 GiB. Snapshot host separato
561,57 MiB; footprint massimo MTP 1,410 GiB. Pressione sempre 1, nessuna
crescita dello swap in tutti e quattro i processi. Footprint, Metal e wired
non si sommano. Cache OS/SSD non controllate; nessuna misura dichiarata cold.
Il load è una sola osservazione per processo, non tre caricamenti indipendenti.

Raw, manifest, correzioni dei soli metadati e limiti:
[evidenze MTP](../../../bench/qwen38_mtp_20261013/README.md).
La chat reale MTP è verificata anche nel browser desktop/mobile; la continuità
deriva dalla storia renderizzata una volta e rigiocata, non da cache fra turni.

## Riproduzione

Usare il venv e gli asset già presenti. I target Make di test GPU acquisiscono
il lock; non aggiungere wrapper esterni a quei target.

```sh
make picoforge picoforge.metallib build/process_memory build/qwen38-mtp/mtp_real
make test-qwen38-mtp test-qwen38-speculative
# Cartelle di output nuove obbligatorie:
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_mtp_gate.py build/MIO-GATE-MTP
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  tools/venv/bin/python tools/eval/qwen38_mtp_bench.py bench/MIO-BENCH-MTP
# Generazione speculativa locale:
tools/venv/bin/python tools/qwen38_gpu_lock.py -- \
  ./picoforge --qwen38 models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --chat 'Scrivi cinque consigli per verificare un backup.' --max-new 128 \
  --context 8192 --chunk 128 --mtp --runs-jsonl build/MIE-MISURE.jsonl
# Il servizio acquisisce il lock da sé:
tools/venv/bin/python tools/ui/qwen38/server.py --mtp
```

`mtp.prefill_s` è dentro il prefill complessivo; `draft_s`, `verify_s`,
`snapshot_s`, `replay_s` sono dentro il decode. Replay comprende restore e
append corretto. `verified_tokens` conta anche proposte poi rifiutate;
`target_decode_appends` include replay e fallback. `accepted/proposed`
include gli stop proposti e confermati. `mtp_profile.gpu_s` comprende prefill
e draft e si sovrappone a wait, non va sommato a esso.

La prova ordinaria a 8K resta distinta: MTP non è ancora misurato con un
prompt reale a 8K. I casi bounded e i pilot non garantiscono parità o vantaggi
su ogni input possibile. Il FAIL storico DwarfStar Metal stock FP16 resta
visibile e non è stato usato come riferimento né corretto allargando i gate.
