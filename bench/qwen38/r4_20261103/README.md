# R4 MTP — evidenze (2026-11-03)

M5 Pro 64 GiB, Qwen3.8-Flash-Next-Q2, contesto 8192, chat template senza thinking, greedy, 128
token nuovi. DwarfStar `fc80bd6` da `../ds4-qwen38-baseline` (albero pulito, verificato dal
manifest). Ogni processo passa da `tools/qwen38_gpu_lock.py` e dal Launcher (release wait,
watchdog). Spec: [R4-mtp.md](../../docs/specs/qwen38/R4-mtp.md); note e lezioni:
[devlog](../../docs/devlog.md), sezione "Roadmap stage R4".

## Condizioni

AC collegato per tutte le run qui sotto, batteria 97% → 81%, mai in scarica sotto carico (il
problema di R3 non si è ripresentato). `pmset -g therm`: nessun avviso. Swap invariato (3 MiB).
`powermetrics` richiede sudo: non usato (regola R6).
Le misure M1/M2 della spec sono state prese prima, a batteria 6–8%, e valgono solo come rapporti.

## Confronto appaiato con `ds4 --mtp` (`paired-mtp/`)

`tools/eval/qwen38_compare.py mtp OUT --warmup 1 --measure 8 --block 1`: un processo per
campione, motori alternati, picoforge `--mtp-draft auto` (il default di `--mtp`), DwarfStar
`--mtp --mtp-timing --temp 0 --nothink`. **Ripetizioni ridotte, dichiarate: 1+8 invece di
3+20** (vedi "Tentativo 3+20 interrotto"). Mediane e p10/p90 (campione reale più vicino).

| Prompt | picoforge tok/s | DwarfStar tok/s | Rapporto |
|---|---:|---:|---:|
| codice | 33,20 [33,16; 33,22] | 45,80 [45,74; 45,88] | 0,725 |
| chat | 25,64 [25,62; 25,65] | 39,48 [39,44; 39,52] | 0,649 |
| narrativa | 27,03 [26,96; 27,07] | 35,67 [35,45; 35,79] | 0,758 |
| conteggio | 40,99 [40,60; 41,03] | 53,08 [52,25; 53,58] | 0,772 |

Accettazione: picoforge bozze accettate/proposte 0,77 / 0,53 / 0,59 / 1,00; DwarfStar bozze
accettate per ciclo di verifica 0,92 / 0,67 / 0,51 / 1,00 (definizioni diverse: DwarfStar conta
per ciclo, con 1–2 bozze). La generazione è libera: i due motori non emettono per forza gli
stessi token.

## Senza MTP, stesse condizioni (`paired-plain/`, 1+3)

| Prompt | picoforge | DwarfStar | Rapporto | Guadagno MTP picoforge | Guadagno MTP DwarfStar |
|---|---:|---:|---:|---:|---:|
| codice | 24,14 | 32,81 | 0,736 | 1,375× | 1,396× |
| chat | 24,41 | 33,20 | 0,735 | 1,050× | 1,189× |
| narrativa | 24,41 | 33,12 | 0,737 | 1,107× | 1,077× |
| conteggio | 24,41 | 33,50 | 0,729 | 1,679× | 1,584× |

Il divario con DwarfStar è quello del decode di base (0,73×, come R2); il guadagno dell'MTP è
alla pari: migliore su narrativa e conteggio, peggiore sulla chat.

## Profondità per tipo di testo (`depth-m3/`, `depth-m3b/`)

Un processo per modo, ordine invertito a prompt alterni; `python3 depth_report.py depth-m3b`.
M3: plain e `--mtp-draft 0,1,2,3,auto` con la prima tabella di costi (proxy M2). M3b: plain, 1 e
`auto` dopo la ricalibrazione sui costi veri della verifica (1,27/1,56/1,91 per 2/3/4 righe,
bozza 0,095, allineamento del predictor 0,03). **36/36 run emettono esattamente i token del
greedy plain.** `keep` costa 0,02–0,22 ms per ciclo (prima: snapshot host 9,9 ms + replay 51 ms
per rifiuto).

| Prompt | k=1 | k=2 | k=3 | auto (M3b) |
|---|---:|---:|---:|---:|
| codice | 1,28× | 1,40× | 1,35× | 1,40× |
| chat | 1,08× | 1,12× | 0,92× | 1,09× |
| narrativa | 1,15× | 1,05× | 0,94× | 1,09× |
| conteggio | 1,42× | 1,66× | 1,77× | 1,65× |

## Tentativo 3+20 interrotto (`aborted-3x20/`)

Il primo confronto 3+20 è stato fermato dal watchdog al campione 33 (`033-code-dwarfstar`):
al caricamento di DwarfStar le pagine wired sono salite a ~51 GiB e la pressione di memoria è
arrivata a 2. I 32 campioni precedenti erano sani, nessuno swap. Chrome era aperto; non si è
insistito: l'utente lo ha chiuso e il confronto è ripartito da capo con 1+8. Il parziale (codice,
14/13 misure: 32,96 contro 45,66 tok/s, 0,722) concorda con il risultato 1+8.

## Gate alla chiusura (log in `build/r4-logs`, non versionati)

`make test-qwen38`, `test-qwen38-mtp` (verify/keep bit-identici per ogni coppia righe/keep, a
chunk 3 e 33), `test-qwen38-speculative` (profondità 1, 2, 3 e auto), CLI, compare: verdi.
fast-kernels 93/93 in default e con `Q38_REF_KERNELS=1`. G3 PASS, scaled 2,08e-7, argmax 12/12.
Oracolo con `Q38_REF_KERNELS=1` bit-identico a R0 (sha256 36e67ce6…); default: logits
bit-identici al build di R3 (785ee5c), cioè decode senza `--mtp` invariato al bit.
Regressioni `make test-all` (Qwen3-0.6B, MoE a forma Qwen3-30B-A3B) verdi a inizio e fine tappa.

## Limiti

- 1+8 e 1+3, non 3+20; prompt brevi (decode a contesto ~40–170): MTP non è misurato a 8K.
- Quattro prompt, uno per tipo: l'accettazione dipende dal testo e non generalizza.
- Il proxy M2 dei costi è sbagliato per la verifica (la sua lezione è nel devlog); la causa
  precisa della differenza non è stata isolata.
