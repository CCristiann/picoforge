# R4 — MTP adattivo e consapevole del costo

Tappa R4 della [roadmap](roadmap.md). Branch `perf/qwen38-mtp-adaptive` da `main` (66a6c2d).
Contratto di partenza: [MTP.md](MTP.md) (predictor `nextn.*`, verifica greedy esatta, una bozza).

## Punto di partenza (misurato, non stimato)

Misure del 2026-11-03 su M5 Pro, AC ma batteria al 6–8% "in carica": la macchina rallenta
durante le serie, quindi i tempi assoluti **non** sono da protocollo. Contano i rapporti
presi nello stesso processo o in ordine alternato. Evidenze in `build/r4-checks/` (non
versionate), riassunte nel README di chiusura.

**M1, accettazione della bozza a profondità 1** (chat template, 128 token, greedy; stesse
sequenze di token con e senza `--mtp` in 4/4):

| Testo | Accettate/proposte | Tasso | tok/s plain | tok/s `--mtp` |
|---|---:|---:|---:|---:|
| Codice (Python CSV) | 57/70 | 0,81 | 22,96 | 23,22 |
| Chat (consigli backup) | 46/80 | 0,57 | 20,20 | 14,78 |
| Narrativa (faro) | 49/77 | 0,64 | 19,82 | 10,55 |
| Conteggio 1–80 | 63/63 | 1,00 | 17,28 | 16,25 |

Costi per ciclo nel percorso di oggi (codice, il run meno rallentato): bozza 4,5 ms,
snapshot host 9,9 ms (copia di 561 MiB, quasi tutta KV cache che l'append non tocca),
verifica a 2 righe 52,8 ms, replay dopo un rifiuto 51,0 ms (restore + un append a 1 riga).
Anche al 100% di accettazione (conteggio) `--mtp` è più lento: 92,6 ms di GPU per l'append a
2 righe contro 52,5 ms per quello a 1 riga dello stesso testo, cioè 1,76×.

**M2, costo GPU di un append a n righe** (prefill di 132 ID con `--chunk n`, ordine
1,2,3,4,4,3,2,1, 2+5 ripetizioni per processo; mediane):

| Righe | GPU ms/append | Rapporto con 1 riga | ms per riga |
|---:|---:|---:|---:|
| 1 | 46,5 | 1,00 | 46,5 |
| 2 | 75,7 | 1,63 | 37,9 |
| 3 | 103,1 | 2,22 | 34,4 |
| 4 | 124,8 | 2,68 | 31,2 |

Profilo per op a 1 e 2 righe: lineari densi ~1,4× (il matvec decodifica ogni blocco una
volta per le righe del tile, ma moltiplicazioni e somme FP32 senza FMA sono per riga: il
kernel è limitato dal calcolo, R1.9b), MoE instradata ~2,1× (kernel per coppia
token-esperto, nessuna deduplica), `gdn.scan` 1,85×, `qsa.core` 1,9×.

**Esperti in più.** Con `Q38_ROUTE_LOG` (diagnostico nuovo, spento di default) gli esperti
distinti in n token consecutivi di decode sono 1,53–1,79× (n=2), 1,98–2,43× (n=3),
2,38–2,76× (n=4) rispetto a una riga (top-10 di 512, 48 layer). I kernel per coppia ne
leggono comunque n×: deduplicarli risparmierebbe al massimo il 20–37% della quota MoE.

## Modello di costo

Unità: il tempo di un append a 1 riga. Con k bozze per ciclo:

    costo(k) = C(1+k) + k·δ + μ        token attesi(k) = 1 + Σ_{j=1..k} Π_{i≤j} α_i

- C(n): tabella M2 (1; 1,63; 2,22; 2,68), sovrascrivibile con `PICOFORGE_MTP_COST`.
- δ: una bozza (blocco MTP a 1 riga + LM head su tutto il vocabolario), ~0,10.
- μ: tenere il predictor allineato a ogni token anche con k=0 (blocco MTP senza LM head).
- α_j: accettazione della j-esima bozza data la precedente, stimata a runtime.

Si sceglie il k ∈ {0..3} che massimizza token attesi / costo. Con le C misurate, k=1 conviene
solo per α₁ > ~0,75 e k=3 solo vicino a α ≈ 0,9. Il tetto teorico ad accettazione perfetta
è ~1,14× con k=1 e ~1,32× con k=3. La vittoria realistica di R4 è quindi: **non perdere**
su testo libero (k=0) e guadagnare un po' sul testo deterministico.

## Passi

1. **Stati per riga, niente snapshot host né replay.** La verifica di 1+k righe salva, per
   le righe 0..k-1, lo stato ricorrente GDN (scan spezzata per riga e copia nello slot),
   le history delle convoluzioni GDN e PLE (history vecchia + ingressi della verifica) e la
   history n-gram. `keep(m)` tiene le prime m righe: scambia il buffer di stato GDN con lo
   slot (zero copie), riscrive le history piccole, riporta posizione e ledger. Le cache
   QSA sono indicizzate per posizione: il blocco compresso si ricalcola quando si completa
   di nuovo. Finestra di rewind del ledger da 2 a 4 token.
2. **Bozze a catena.** Bozza j ≥ 2: il blocco MTP sulla propria uscita residua e sulla bozza
   precedente, alla posizione successiva della propria cache (speculativa, poi riscritta
   dal recupero con i residui veri del trunk). Idea: DwarfStar fc80bd6,
   `qwen4_graph_mtp_chain_step` (MIT).
3. **Verifica multi-riga in un append**, k fino a 3, con i matvec sotto le 32 righe.
4. **Politica.** `--mtp-draft N|auto` e `PICOFORGE_MTP_DRAFT` (N = 0..3 fisso, `auto` =
   modello di costo). Stime α con media mobile; con k=0 una bozza di sonda ogni 8 token,
   confrontata gratis con il token vero.
5. **Chiusura**: run appaiate con `ds4 --mtp` fc80bd6, gate completi, README in
   `bench/qwen38_r4_20261103/`.

## Correttezza

- Greedy con MTP = greedy senza MTP, token per token (le bozze sono solo proposte).
- `keep(m)` dopo una verifica di T righe deve lasciare history e stato **bit-identici** a m
  append ordinari degli stessi token: test dedicato in `test-qwen38-mtp`.
- La scan GDN spezzata per riga fa le stesse operazioni FP32 della scan su T righe.
- Decode senza `--mtp` invariato al bit; G3 e la bit-identità `Q38_REF_KERNELS=1` non si
  allargano.

## Fuori da R4 (backlog)

Deduplica degli esperti fra le righe della verifica, testa ridotta per la bozza (DwarfStar
`DS4_QWEN4_MTP_DRAFT_VOCAB`), argmax della bozza su GPU.

## Stato

In corso.
