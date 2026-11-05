# R5 — Flash-Next Q4K con gli esperti in streaming da SSD

Data: 2026-11-05. Stato: **approvata dall'utente il 2026-11-05. R5.0 chiusa (gate superato, vedi devlog); R5.1 è il passo successivo**.
Contesto: [roadmap](roadmap.md), tappa R5. Si appoggia su Decode II (comandi spezzati in più
command buffer) e sul registro delle tracce di routing (`Q38_ROUTE_LOG`).

## Intento

Far girare Flash-Next **Q4K** su questo Mac da 64 GiB: i pesi non-esperti (~8 GiB) e una cache
degli esperti più usati restano residenti, gli altri esperti si leggono dal GGUF su SSD quando
il router li sceglie. Meccanismo riusabile per ogni MoE (in particolare per il fratello di
Flash-Next di QN4, quando uscirà), non una soluzione cucita sul Q4K.

Perché: qualità più alta dello stesso modello (Q4K contro Q2), e una tecnica che DwarfStar non ha
per Qwen3.8 (`ds4 --ssd-streaming` esce con codice 1 e dice che non è supportato, prova di R0).

## Numeri di partenza (misurati o derivati dall'inventario del Q2)

- Q2: esperti instradati 35,4 GiB (49 layer × 512 esperti, 1,42 MB l'uno; gate 412 KB, up 412 KB,
  down 630 KB); tabella n-gram 95,4 GiB su disco; il resto 6,3 GiB. File 137,1 GiB.
- Q4K (roadmap, da verificare sul file): ~62 GiB di esperti, ~2,6 MB l'uno; ~8 GiB non-esperti;
  file 165 GiB; ~1,3 GB di esperti letti per token (48 layer × 10 esperti).
- Costo per token = `m × 1,3 GB`, con m il tasso di miss della cache; il tempo dipende dalla banda
  dell'SSD e dalla parte dello stallo che non si riesce a nascondere.

## Obiettivi, non-obiettivi, anti-obiettivi

Obiettivi: Q4K che gira su 64 GiB senza swap; logits **identici bit per bit** tra esecuzione
residente e in streaming a ogni dimensione di cache; numeri col protocollo di `docs/BENCHMARKS.md`.

Non-obiettivi: velocità pari al residente Q2; contesti oltre 8K; più sessioni; ottimizzare il
prefill in streaming oltre quello che serve a usare il modello.

Anti-obiettivi: un risultato più veloce ma con deriva numerica non spiegata; kernel copiati da
DwarfStar; numeri senza protocollo; leggere i pesi con `mmap` del file intero (l'RSS e le pagine
wired devono restare entro il budget).

## Fasi e gate

**R5.0 — Misure, senza scaricare nulla.** (a) Banda e latenza dell'SSD con letture casuali da
256 KB / 1,4 MB / 2,6 MB, 1-16 thread, sulla regione n-gram del Q2 (fredda), con `F_NOCACHE`;
lettura sequenziale per il picco. (b) Tracce di routing reali (`Q38_ROUTE_LOG`) su testi di più
tipi, e un simulatore di politiche di cache (LRU globale, LRU per layer, statica per frequenza,
LFU) che dà il tasso di miss in funzione della cache. (c) tok/s previsto = f(miss, banda).
**Gate:** se la previsione migliore con la cache da ~36 GiB sta sotto i 3 tok/s, si ferma e si
riporta, prima di scrivere codice di motore.

**R5.1 — Streaming sul Q2 emulato.** Pool di slot per gli esperti, tabella esperto→slot per layer,
thread di I/O con `pread`, kernel MoE che leggono tramite indirezione. Il Q2 ha 35,4 GiB di
esperti: si prova con la cache al 10%, 50% e 100%. **Gate:** logits e id bit-identici al
residente a ogni dimensione, nessuno swap, memoria entro il budget.

**R5.2 — Prefetch e sovrapposizione.** Sostituzione con la politica migliore di R5.0, prefetch
dagli esperti del token precedente, I/O sovrapposto al calcolo (attenzione, esperto condiviso)
sfruttando i command buffer spezzati. **Gate:** tempo di stallo per miss misurato con e senza.

**R5.3 — Q4K.** Verifica fonte e hash del file (chiedere prima di scaricare 165 GiB), kernel Q4_K e
MXFP4 degli esperti su fette reali, esecuzione, misure col protocollo, KL e top-1 contro il Q2
(alimenta R7). **Gate:** gira su 64 GiB, nessuno swap, correttezza su fette e su logits.

**R5.4 — MTP consapevole del costo di streaming.** Il costo di verifica diventa il numero di
esperti mancanti da leggere; si ricalibra la tabella dei costi di R4 sul costo misurato (non su un
proxy). Direzione originale scelta all'inizio del progetto.

## Vincoli

Un solo processo col checkpoint alla volta; ogni processo reale passa dal lancio protetto
(attesa dopo l'uscita, guardiano di memoria, `caffeinate`); lock GPU. `iogpu.wired_limit_mb` lo
decide e lo lancia l'utente. Niente download senza chiedere. FP32 di default e gate end-to-end non
allargato. Ogni fase ≤ ~150 righe nuove per passo, test verdi prima di ogni commit.

## Decisioni dell'utente lungo la strada

- Prima di R5.3: come far posto ai 165 GiB (liberare circa 146 GiB sul disco interno, SSD esterno più
  lento da misurare, o sostituire temporaneamente il Q2). Oggi sono liberi ~19 GiB.
- Prima di R5.3: eventuale limite wired.
- Conferma del download del Q4K e della sua fonte.
