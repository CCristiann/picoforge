# R0 — baseline registrata del decode Qwen3.8-Flash-Next Q2

M5 Pro 64 GiB, macOS 27.0.1 (build 26A434), 17 ottobre 2026 (tentativi 8K
fino al 18). Misure prima di qualsiasi ottimizzazione R1: picoforge `44d04bf`
(diff vuoto, `picoforge` sha256 `d10ae768…`, `picoforge.metallib` `dcbb957b…`),
DwarfStar `fc80bd6` pinned e non modificato (`ds4-bench` `27e2c366…`, `ds4`
`411ab7ff…`). Checkpoint `Qwen3.8-Flash-Next-Q2.gguf`, 147.207.127.040 byte,
SHA256 in `docs/weights.lock`, non ricalcolato. Spec: [R0-R1-decode](../../docs/specs/qwen38/R0-R1-decode.md).

**Esito in una riga:** a 9 token di prompt picoforge fa 1,064 tok/s contro i
33,075 di DwarfStar (0,0322×, DwarfStar 31,1× più veloce). Il decode legge
6,10 GB di pesi per token ma usa 6,5 GB/s, il 2,1% dei 307 GB/s. La baseline
appaiata a 8K **non c'è**: tre tentativi non validi, misura rinviata (sezione
dedicata).

## Risultati

| Frontiera | Motore | Mediana tok/s | p10 | p90 | n | Rapporto |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| breve (9 + 128 ID) | picoforge | 1,0641 | 1,0619 | 1,0663 | 20 | 0,0322× |
| breve (9 + 128 ID) | DwarfStar | 33,075 | 23,62 | 33,22 | 20 | 1× |
| 8K (8063 + 128 ID) | picoforge | profilo, non appaiato: ≈ 1,03 | — | — | 4 token | — |
| 8K (8063 + 128 ID) | DwarfStar | non misurato | — | — | — | — |

Rapporto = mediana picoforge / mediana DwarfStar (`summary.json`:
0,03217208…). In tempo: 939,8 ms contro 30,2 ms per token.

Protocollo (`manifest.json`): un processo per campione per entrambi i motori
(`ds4-bench` misura una frontiera per processo); prefill fino alla frontiera,
poi 128 ID forzati aggiunti uno alla volta (teacher-forced: stesso lavoro,
nessuna dipendenza da cosa sceglierebbe ciascun motore). tok/s = ID forzati /
secondi di decode (`decode_s` di picoforge, `gen_tps` di `ds4-bench`). Blocchi
alternati di 1 campione, 3 warm-up per motore (6 esclusi), 20 misure. Mediana e
percentili p10/p90 sul campione più vicino, mai interpolati. Contesto 8192 e
chunk 1024 per entrambi (a frontiera breve `ds4-bench` riceve `--ctx-alloc 8192`,
ruling R11: stessa allocazione di picoforge). Cache file dell'OS calda, non
svuotata: nessuna affermazione su avvii a freddo. Il prompt breve è il caso
coding congelato `bench/qwen38_mtp_20261013/paired/coding`; i 128 ID forzati
sono la continuazione greedy di picoforge e i due tokenizzatori li riproducono
identici (`suite/`, ruling R7).

Gli 8K della tabella sono il profilo di `profile-8k/` (picoforge da solo,
4 ID forzati dopo 8063 di prompt, `--profile`): 974,0 ms/token di tempo GPU
(somma dei gruppi; mediana 974,2), cioè ≈ 1,03 tok/s; a parete `decode_s`
3,950 s / 4 = 987,6 ms (1,01 tok/s). Il prefill di 8063 ID ha richiesto
1406,1 s (23,4 minuti, 5,73 ID/s). Non è il protocollo appaiato.

**DwarfStar, valori lenti.** Fra i 20 campioni misurati, tre consecutivi
(23,62; 22,19; 23,20 tok/s, campioni 7–9 in ordine di esecuzione) e uno a
27,57 (campione 4) sono molto sotto i circa 33. Pressione 1, swap 0 e nessun
avviso termico in tutti; i processi picoforge adiacenti (1,063–1,067 tok/s)
non mostrano cali. Sono transitori di cui non conosciamo la causa; non sono
stati scartati. La mediana li assorbe, il p10 = 23,62 li mostra. picoforge ha
una dispersione minima (1,0613–1,0666).

Memoria: picoforge alloca 43,11 GiB di Metal (buffer del grafo 1,39 GiB;
raccomandati 51,84 GiB), picco di footprint di processo 1,618 GB (10⁹ byte);
DwarfStar 1,312 GB. Un campione picoforge dura circa 123 s, uno DwarfStar
4,5–7,0 s.

## Dove va il tempo di un token (frontiera breve)

`profile-short/summary/summary.md`, output di
`tools/eval/qwen38_profile_summary.py` su 8 append di decode, confrontato con
la stessa esecuzione senza profilo (`oracle/runs.jsonl`). Metodo: un compute
encoder per gruppo nello stesso command buffer, timestamp ai bordi dell'encoder
(il probe `capabilities.json` dice che il campionamento esiste solo al bordo di
stage, non per dispatch). Somma dei gruppi 935.452,0 µs/token (mediana
935.623,2; p10 934.420,0; p90 936.474,6); `gpu_s` del decode normale
934.528,4 µs; rapporto 1,0010, cioè l'overhead del profilo è +0,10%.
Colonna 8K: stessa tabella da `profile-8k/summary` (4 append).

| Op | Chiamate/token | µs/token | Quota | MB pesi | GB/s | µs/token a 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdn` | 36 | 323.960,3 | 34,6% | 2.716,1 | 8,38 | 324.480,3 |
| `moe` | 48 | 262.743,3 | 28,1% | 1.218,0 | 4,64 | 262.975,3 |
| `hc_ffn.down` | 48 | 102.575,8 | 11,0% | 314,6 | 3,07 | 102.692,8 |
| `hc_ffn.inject` | 48 | 96.115,2 | 10,3% | 3,9 | 0,04 | 96.200,8 |
| `hc_attn.down` | 12 | 25.574,1 | 2,7% | 78,6 | 3,08 | 25.622,4 |
| `hc_attn.inject` | 12 | 23.996,2 | 2,6% | 1,0 | 0,04 | 24.032,2 |
| `qsa.out` | 12 | 16.871,5 | 1,8% | 200,5 | 11,89 | 16.890,3 |
| `hc_ffn.rms` | 48 | 13.954,4 | 1,5% | 0 | 0,00 | 13.961,6 |
| `lm_head` | 1 | 12.985,2 | 1,4% | 675,4 | 52,02 | 12.928,8 |
| `qsa.qg` | 12 | 9.328,5 | 1,0% | 401,1 | 43,00 | 9.378,9 |
| `qsa.k` | 12 | 7.212,4 | 0,8% | 16,7 | 2,32 | 7.202,1 |
| `qsa.v` | 12 | 7.176,4 | 0,8% | 16,7 | 2,33 | 7.175,0 |
| `qsa.ik` | 12 | 7.164,4 | 0,8% | 4,2 | 0,58 | 7.174,4 |
| `qsa.iq` | 12 | 7.138,0 | 0,8% | 16,7 | 2,34 | 7.158,1 |
| `hc_ffn.up` | 48 | 3.814,6 | 0,4% | 314,6 | 82,47 | 3.805,5 |
| `hc_attn.rms` | 12 | 3.478,2 | 0,4% | 0 | 0,00 | 3.480,4 |
| `qsa.core` | 12 | 3.102,7 | 0,3% | 0 | 0,00 | **40.531,8** |
| `ple` | 1 | 2.442,8 | 0,3% | 34,8 | 14,26 | 2.432,6 |

Le altre 14 voci (`final.*`, `embed`, `hc_*.hc_read/hc_low/hc_inject`,
`hc_add`, `hc_attn.up`) valgono ciascuna meno dello 0,3%: elenco completo nei
due `summary.md`. `MIXED` è il gruppo composito (più tensori); `gdn` sono le
proiezioni del layer GDN, Q8_0 secondo la spec. `qsa.core` legge le cache KV e
indice, non pesi: 0 byte. I byte di `moe` sono esatti per un token (router,
esperto condiviso, top_k esperti).

A 8K il token costa 38,5 ms in più (973.955,7 contro 935.452,0 µs) e quasi
tutto è `qsa.core`: +37,4 ms (3,1 → 40,5 ms, dallo 0,3% al 4,2%). Il resto
della tabella non cambia. A 8K la quota di `qsa.core` si avvicina alla soglia
del ~5% oltre la quale R1.7 interviene.

## Byte, dispatch e pavimenti

| | Riferimento precedente | Misurato in R0 |
| --- | --- | --- |
| Byte di pesi letti per token | 6,1 GB (roadmap) | 6.104,78 MB, dalle binding reali |
| Tetto a 307 GB/s | ~50 tok/s | pavimento 19.885,3 µs → 50,3 tok/s |
| Dispatch per token | ~1.900 (spec, "da misurare") | 2.986 (media di 8 token) |
| Barriere per token | una dopo ogni op | 3.033 (media) |
| Pavimento dei dispatch | 1,4 µs ciascuno | 2.986 × 1,4 = 4.180,4 µs |
| Tempo GPU per token | ~0,93 s (spec) | 935.452,0 µs: 47,0× il pavimento di banda, 223,8× quello dei dispatch |
| Banda effettiva picoforge | ~6,6 GB/s (spec) | 6,5 GB/s = 2,1% dei 307 GB/s |
| Banda DwarfStar (stima) | 62% (roadmap) | ≈ 201,9 GB/s = 65,8% |

I byte sono quelli della stima: la roadmap aveva ragione sui 6,1 GB (e quindi
sul tetto di ~50 tok/s), mentre i dispatch sono 1,57× la stima della spec. I due
pavimenti sono limiti inferiori indipendenti, non si sommano. Dispatch e
barriere sono una media: 2.977 / 3.024 nei token ordinari, 3.013 / 3.060 in
due token su otto (posizioni 11 e 15), dove `qsa.core` aggiunge 36 dispatch
(12 layer × 3); la causa non è stata indagata in R0.

La banda di DwarfStar è una **stima**, non una misura: byte per token di
picoforge × tok/s di DwarfStar = 6,10478 GB × 33,075 = 201,9 GB/s (144,2 GB/s
se si usa il p10, 23,62 tok/s). Presuppone che DwarfStar legga gli stessi byte; non è stato
misurato.

## Cosa decide l'ordine di R1

Il profilo conferma l'ordine della spec. `gdn` (34,6%, 8,4 GB/s) sono le
proiezioni Q8_0: il matvec cooperativo Q8_0 di R1.2 è la prima mossa, e porta
con sé `qsa.*`, `lm_head` e `qsa.out`, anch'essi Q8_0 (7,3% in tutto). Le
hyper-connections F16 (R1.3) valgono insieme 248,3 ms (26,5%, la somma delle
quote arrotondate dà 26,6%). Quasi la metà è `inject`, con n_out = 4 righe di
output a 0,04 GB/s: con un thread per riga (spec, `q38_packed_linear`) lavorano
4 thread in tutto, quindi il costo è di parallelismo, non di byte (4,9 MB per
token tra FFN e attenzione); anche `down` fa solo 3,1 GB/s. Seguono i MoE (R1.5, 28,1%, 4,6 GB/s).
Il pavimento dei dispatch pesa lo 0,45% del token, quindi R1.6 (barriere e
fusioni) non è ancora il collo di bottiglia; `qsa.core` va riguardato dopo
R1.6, perché a 8K cresce 13×.

## Condizioni e limiti

- **Alimentazione.** AC in tutti i 46 lanci di `paired-short` (prima e dopo).
  La batteria è scesa dal 37% al 28% pur con "AC Power" e "charging": il
  caricatore ha erogato meno del carico. Nello stage del profilo 46% → 37%.
- **Termica.** `pmset -g therm`: "No thermal warning level has been recorded",
  "No performance warning level has been recorded", "No CPU power status has
  been recorded" ovunque; `NSProcessInfo.thermalState` `nominal` prima e dopo
  `paired-short` e in `capabilities.json`. Il valore era `fair` all'inizio del
  primo e del terzo tentativo 8K (non validi). `powermetrics` richiede sudo e
  non è stato eseguito (ruling R6): la termica non ha misura di potenza.
- **Memoria.** In tutti i lanci validi pressione 1 e swap 0 byte a ogni
  campione ogni 0,25 s (`paired-short`: 9.615 campioni nei 46 lanci; suite,
  oracle, profili, ssd: tutti 1 e 0). Intervallo massimo fra campioni in
  `paired-short`: 0,349 s.
- **Guardia.** Ogni lancio che mappa il checkpoint passa dal `Launcher`:
  attesa del rilascio Metal prima e dopo (`release-before/`, `release-after/`),
  guardia stretta che ferma il figlio a pressione ≥ 2 o crescita swap
  > 256 MiB su due campioni consecutivi (ruling R9). Il motivo: avviare il
  processo successivo mentre i ~46 GiB wired del precedente erano ancora
  trattenuti ha bloccato il Mac due volte. Gli stage 1 e 2 non avevano
  ancora l'abort per salto temporale (`gap_s`, R0.5a) né il precheck di
  risveglio completo; i tentativi 2 e 3 sì.
- **Contesto breve.** `ds4-bench --ctx-alloc 8192` anche a 9 token (ruling R11).
- **Non coperto.** Nessuna misura a freddo; nessun confronto 8K appaiato;
  la banda DwarfStar è stimata; profilo e confronto sono a decode a un token,
  senza MTP né sampling; un solo Mac.
- **Rinviato.** DwarfStar 3 + 20 e picoforge R0 0 + 2 a 8K, quando la memoria
  lo permette e al più tardi nel Task 17. Il binario equivalente a R0 è
  salvato fuori dal repository (`build/r0-baseline/bin-302c490`, non
  versionato): `picoforge` sha256
  `0f19b10ae9f19838d1c457838569c9617bb35273e6f7f7eec14aab3b2cc72fb1`,
  `picoforge.metallib` `dcbb957b…` (lo stesso della baseline); il suo percorso
  di decode è risultato bit-identico a `oracle/oracle-logits.f32`.

Righe del controller (`build/r0-baseline/stage*.log`, non versionati):

| Stage | Batteria prima | Batteria dopo | Swap |
| --- | --- | --- | --- |
| 1 (suite, profili, oracle) | 46%; charging; 6:27 remaining | 37%; charging; (no estimate) | 0,00M → 0,00M |
| 2 (`paired-short`) | 37%; charging; (no estimate) | 28%; charging; 7:47 remaining | 0,00M → 0,00M |
| 3 (notte, non valido) | 38%; charging; (no estimate) | 83%; charging; (no estimate) | 0,00M → 0,00M |
| 3 r2 | 90%; AC attached; not charging | 83% (`conditions.json`) | 0 → 0,25 MiB |
| 3 r3 | 87%; charging; (no estimate) | 87% (`conditions.json`) | 0,25M (da prima) |

## Tentativi 8K non validi

Nessuno produce un risultato; ognuno ha `manifest.json` con `complete: false`
e `process.json` con `stop_reason`. Sono conservati, mai sovrascritti.

| Cartella | Lancio | Esito | Causa |
| --- | --- | --- | --- |
| `paired-8k/` | 18 ott 00:11 CEST (stage 3 avviato 23:57) | `timeout` a 3600,04 s, SIGTERM (−15) | Il Mac era andato in sleep alle 23:42:04 (`pmset sleep` = 1 min, `caffeinate` già terminato) e stage 3 è partito in DarkWake ("CPU Network", senza Graphics); `caffeinate -i` non tiene un DarkWake. picoforge è avanzato solo in finestre di manutenzione da ~20 s ogni ~15 min: 532 campioni di memoria con intervalli di 899,1, 898,7, 898,9 e 742,7 s. Coperchio chiuso alle 01:50. Batteria 38% → 83% (Mac quasi sempre in sleep). |
| `paired-8k-r2/` | 18 ott 10:52 CEST | guardia di memoria a 829,5 s | Pressione 2 su due campioni consecutivi (wired 48,4 GiB, compressore 1,3 → 7,5 GiB, swap +0,25 MiB, ben sotto i 256 MiB) mentre un worker compilava e testava → ruling R19: nessun worker durante i run reali. |
| `paired-8k-r3/` | 18 ott 17:49 CEST | guardia di memoria dopo 6,7 s (22 campioni) | Nessun worker attivo. Il compressore teneva già ~9,3 GiB di app dell'utente (`vm_stat` 610.680 pagine; in `process.json` 381.834 pagine prima del lancio, 684.803 allo stop; la notte prima 109.668), wired 43,9 GiB: non c'era margine per 8K. La guardia ha fatto il suo lavoro, nulla si è bloccato. |

Per il primo la correzione è il ruling R13 (commit `789ce5a`): abort se due
campioni di memoria distano > 5 s (`gap_s`, `stop_reason: "sleep gap"`) e
precheck di risveglio completo (Graphics presente, coperchio aperto o
clamshell accettato, AC). `ssd/` della stessa notte è valido.

`ssd/ssd-check.json`: `ds4 --ssd-streaming` su Qwen3.8 esce con codice 1 e
risponde che SSD streaming non è supportato per Qwen3.8 (insieme a tensor
parallelism, pipeline, DSpark, MTP esterno, power throttling): conferma
l'ipotesi della roadmap che lo streaming di DwarfStar non copra questo modello.

## Prove conservate

- `capabilities.json`: probe R0.1. Unico counter set `timestamp`, campionamento
  solo al bordo di stage (dispatch, draw, blit, tile: falso); famiglia
  Metal 4, coda `MTL4CommandQueue` e residency set disponibili; working set
  raccomandato 55.662.788.608 byte.
- `suite/`: `prepare` (testi, ID di stop, ID breve 9 + 128 e 8K 8063 + 128,
  accordo dei due tokenizzatori, lancio greedy). `suite.json` sha256
  `d0bcf8bf…`, registrato nei quattro `manifest.json`.
- `profile-short/`, `profile-8k/`: `profile.csv` grezzo, `summary/`
  (`summary.md`, `summary.csv`, `totals.json`), `runs.jsonl`, `ids.json`.
- `oracle/`: gli stessi 9 + 8 ID senza profilo, `oracle-logits.f32` = 9 righe ×
  248.320 FP32 (ultima riga del prefill + 8 del decode), 8.939.520 byte, sha256
  `36e67ce6d114413ce504b54798faaf9a956054f3528c3dc3ee2e92acdd7f07c9`. È il
  riferimento di R1: `Q38_REF_KERNELS=1` deve riprodurlo bit a bit (ruling R12).
- `paired-short/`: `samples.csv`, `samples.jsonl`, `summary.json`,
  `manifest.json` e una cartella per campione (`NNN-picoforge[-warmup]`,
  `NNN-dwarfstar[-warmup]`).
- Ogni lancio ha `process/{process.json,memory.jsonl,events.jsonl,stdout.log,stderr.log}`,
  `release-before/`, `release-after/`, `conditions.json`.
- Il prefisso assoluto della home del Mac di origine, fino alla cartella
  `Dev/`, è sostituito con `/path/to/` nei file di testo (62 file, 481 occorrenze), come
  in `d44c732`; nessun numero e nessun file binario (`oracle-logits.f32`)
  modificati; `suite.json` non conteneva percorsi, quindi il suo SHA256 nei
  manifest resta valido.

## Riproduzione

Dalla radice del worktree, con il checkpoint in `models/`. Le cartelle di
output devono essere nuove, ogni comando passa dal lock GPU comune e nessun
altro processo deve usare la GPU né ampia memoria (ruling R19).

```sh
PY=tools/venv/bin/python
B=bench/qwen38_r0_20261017
make picoforge picoforge.metallib build/process_memory build/qwen38-capabilities
$PY tools/qwen38_gpu_lock.py -- build/qwen38-capabilities > $B/capabilities.json
$PY tools/qwen38_gpu_lock.py -- $PY tools/eval/qwen38_compare.py prepare $B/suite
# profilo, oracle e profilo 8K: stesso comando picoforge registrato in
# profile-short/process/process.json, profile-8k/... e oracle/..., avviato dal
# Launcher di qwen38_compare.py (helper del controller, non versionato);
# --teacher-forced 8 (4 a 8K) con ID da suite.json, --profile profile.csv
# oppure --logits oracle-logits.f32, in lanci separati
$PY tools/eval/qwen38_profile_summary.py $B/profile-short/profile.csv \
  --out $B/profile-short/summary --runs-jsonl $B/oracle/runs.jsonl
$PY tools/eval/qwen38_profile_summary.py $B/profile-8k/profile.csv --out $B/profile-8k/summary
$PY tools/qwen38_gpu_lock.py -- $PY tools/eval/qwen38_compare.py run $B/suite $B/paired-short \
  --frontier short --warmup 3 --measure 20 --block 1
$PY tools/qwen38_gpu_lock.py -- $PY tools/eval/qwen38_compare.py ssd-check $B/ssd
# 8K appaiato (rinviato, tre tentativi non validi in paired-8k*):
$PY tools/qwen38_gpu_lock.py -- $PY tools/eval/qwen38_compare.py run $B/suite NUOVA-CARTELLA \
  --frontier 8k --warmup 3 --measure 20 --picoforge-warmup 0 --picoforge-measure 2 --block 1
```

Le statistiche di una cartella si ricalcolano con
`tools/eval/qwen38_compare.py summarize <cartella>`, che riscrive il solo
`summary.json`. Dopo `d44c732` i comandi registrati in `process.json` hanno
`/path/to/` al posto del prefisso del Mac di origine. L'8K appaiato richiede
circa 26 minuti per campione picoforge: usarlo solo quando il Mac è a
riposo, con alimentazione e display attivi.
