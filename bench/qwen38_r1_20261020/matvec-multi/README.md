# R1.4 — multi-GEMV, kernel isolati

M5 Pro 64 GiB, macOS 27.0.1 (build 26A434), 20 ottobre 2026, 09:51:41–09:51:43
UTC; alimentazione AC (82%, in carica), `thermal_state` nominal prima e dopo,
swap 0. Kernel `q38_matvec_multi_*` di `src/qwen38_matvec.metal` e
`q38_packed_encode_multi` nell'albero non ancora committato su `5cdcc2e`
elencato in `manifest.txt` (il commit che aggiunge i kernel contiene lo stesso
codice: `picoforge.metallib` sha256 `b60e5ffd…` ricostruito identico).
Comando: `make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261020/matvec-multi`.
Spiegazione: [devlog, R1.4](../../../docs/devlog.md).

**Esito in una riga:** un dispatch per gruppo invece di uno per proiezione fa
risparmiare il 5–6% sul gruppo QSA (5 × Q8_0) e il 17–18% sul gruppo
dell'esperto condiviso (tensori piccoli); sul gruppo GDN (2 Q8_0 grandi + 2 F32
piccoli, quindi 2 dispatch contro 4) è un pareggio entro p10/p90.

## Risultati

Dati sintetici, mai il checkpoint. Mediana di 20 misure dopo 3 warm-up, fra
parentesi p10/p90. Ogni misura è un command buffer con 20 chiamate di fila,
ciascuna seguita da una barriera, come nel grafo; tempo per chiamata.
"singole" = le `q38_packed_encode_fast` di ogni proiezione, una barriera dopo
ciascuna (il grafo prima di R1.4); "multi" = una `q38_packed_encode_multi` e una
barriera. K = 2560 ovunque.

| gruppo (n_out) | righe | multi (µs) | singole (µs) | multi / singole |
| --- | ---: | ---: | ---: | ---: |
| QSA: qg 12288, k 512, v 512, iq 512, ik 128 (Q8_0) | 1 | 139,1 (135,4/172,5) | 147,7 (145,4/155,9) | 0,94 |
| QSA | 4 | 220,2 (214,8/234,1) | 231,9 (228,4/237,0) | 0,95 |
| GDN: QKV 10240, Z 6144 (Q8_0) + alpha 48, beta 48 (F32) | 1 | 182,1 (175,0/191,3) | 180,4 (171,4/192,5) | 1,01 |
| GDN | 4 | 266,7 (260,1/279,1) | 271,2 (265,3/275,6) | 0,98 |
| condiviso: router 512, slogit 1 (F32) + gate 640, up 640 (Q8_0) | 1 | 26,1 (26,0/27,3) | 31,5 (31,3/31,7) | 0,83 |
| condiviso | 4 | 42,1 (41,8/48,8) | 51,6 (50,2/57,8) | 0,82 |

Note.

- Il gruppo dell'esperto condiviso pesa 8,4 MiB e viene riletto 20 volte di
  fila: in parte resta in cache, per questo supera i 307 GB/s (334,8). Nel
  grafo ogni strato ha pesi diversi, quindi il valore assoluto non si
  trasferisce; il risparmio di ~5 µs a chiamata (dispatch e barriere in meno)
  sì.
- Il kernel multi costa per byte un po' più del singolo: con una sola
  proiezione da 16384 righe, prove esplorative non committate, 159 contro
  155 µs. La prima versione (indice dinamico su tutti i campi di `p[j]` e
  ricerca della proiezione con un ciclo) era a 163–165 µs e rendeva il gruppo
  GDN più lento del 2–4% (172–175 contro 168–171 µs); da qui la scelta di
  leggere il lato x da `p[0]` e di calcolare la proiezione senza ciclo.
- L'ordine multi/singole si alterna fra 1 e 4 righe.
