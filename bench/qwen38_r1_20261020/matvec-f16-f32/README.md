# R1.3 — matvec F16 e F32 cooperativi, kernel isolati

M5 Pro 64 GiB, macOS 27.0.1 (build 26A434), 20 ottobre 2026 ora locale
(19 ottobre 22:02:02–22:02:59 UTC). Kernel `q38_matvec_f16` e `q38_matvec_f32`
di `src/qwen38_matvec.metal` nell'albero non ancora committato su `8449a40`
elencato in `manifest.txt` (il commit che aggiunge questa cartella contiene lo
stesso kernel). `picoforge.metallib` sha256 `c136ae9c…`, harness
`build/qwen38-r1/test_fast_kernels` sha256 `2aedcae4…`. Spiegazione e
alternative provate: [devlog, R1.3](../../../docs/devlog.md).

**Esito in una riga:** su tensori da 500 MiB il kernel veloce legge i pesi a
286,6 GB/s in F16 e 287,7 GB/s in F32, il 93–94% dei 307 GB/s. Sulle forme HC
reali è 88× (down) e 385× (inject) più rapido dell'oracolo, ed è più veloce
dell'oracolo a ogni numero di righe misurato, da 1 a 1024, su tutte e quattro
le forme della tabella delle righe.

## Risultati

Dati sintetici, mai il checkpoint. Mediana di 20 misure dopo 3 warm-up, fra
parentesi p10/p90. GB/s = (byte del tensore + x + y) / mediana.

Banda, 1 riga di x (µs):

| Tensore (n_out × K) | veloce, 1 per buffer | veloce, 20 per buffer | veloce, 200 per buffer | oracolo |
| --- | ---: | ---: | ---: | ---: |
| F16 25600 × 10240, 500 MiB | 1855,7 (1807,7/1910,8), 282,6 GB/s | **1829,9 (1806,7/1838,6), 286,6 GB/s** | — | 5903,4 (1 per buffer), 88,8 GB/s |
| F32 51200 × 2560, 500 MiB | 1835,2 (1800,3/1847,9), 285,8 GB/s | **1823,2 (1814,8/1853,8), 287,7 GB/s** | — | 2649,1 (1 per buffer), 198,0 GB/s |
| HC down F16 320 × 10240, 6,25 MiB | 26,6 (26,3/92,6) | 29,4 (24,6/49,7) | **24,5 (24,5/24,6), 268,9 GB/s** | 2154,5 (200 per buffer), 3,1 GB/s |
| HC inject F16 4 × 10240, 80 KiB | 35,3 (28,7/35,8) | 5,3 (5,3/5,3) | **5,2 (5,1/7,8), 23,8 GB/s** | 2002,3, 0,1 GB/s |
| HC up F16 10240 × 320, 6,25 MiB | 206,1 (203,9/206,8) | 42,3 (42,2/42,3) | **42,1 (42,1/42,1), 156,6 GB/s** | 72,9, 90,5 GB/s |
| router F32 512 × 2560, 5 MiB | 54,9 (54,3/55,7) | 19,0 (17,3/33,6) | **12,6 (12,6/12,6), 417,9 GB/s¹** | 439,6, 12,0 GB/s |
| ssm_alpha F32 48 × 2560, 480 KiB | 25,1 (24,8/25,3) | 18,0 (17,6/18,4) | **3,6 (3,6/3,6), 138,8 GB/s** | 399,4, 1,3 GB/s |

¹ Sopra il tetto: 200 esecuzioni di fila dello stesso tensore da 5 MiB lo
tengono nella cache di sistema. I numeri onesti di banda sono quelli dei
tensori da 500 MiB. L'oracolo delle forme reali è a 200 per buffer. A 20 per
buffer è uguale entro l'1% (`runs.jsonl`); a 1 per buffer i dispatch brevi
sono più lenti per lo stesso effetto di clock (HC up 351,1 µs, router 872,0).

Veloce contro oracolo per numero di righe di x (µs). Fino a 32 righe 20
encode per buffer, a 128 e 1024 righe 1 per buffer; la riga 1 è dalla colonna
a 20 per buffer qui sopra:

| Forma | 1 | 2 | 4 | 8 | 32 | 128 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HC down veloce | 29,4 | 29,5 | 37,0 | 78,2 | 348,3 | 1576,7 | 42984,5 |
| HC down oracolo | 2152,9 | 2148,4 | 2146,3 | 2146,0 | 2203,0 | 7901,3 | 58208,5 |
| rapporto | 73× | 73× | 58× | 27× | 6,3× | 5,0× | 1,35× |
| router veloce | 19,0 | 21,6 | 22,4 | 22,1 | 94,2 | 500,3 | 4941,1 |
| router oracolo | 439,8 | 439,1 | 439,3 | 443,3 | 693,8 | 2320,8 | 17296,5 |
| rapporto | 23× | 20× | 20× | 20× | 7,4× | 4,6× | 3,5× |
| HC up veloce | 42,3 | 49,9 | 65,4 | 127,6 | 498,0 | 1981,4 | 15911,4 |
| HC up oracolo | 72,9 | 141,0 | 249,6 | 477,8 | 1834,2 | 7216,2 | 57462,4 |
| rapporto | 1,7× | 2,8× | 3,8× | 3,7× | 3,7× | 3,6× | 3,6× |
| HC inject veloce | 5,3 | 12,3 | 15,9 | 8,1 | 9,2 | 138,4 | 587,9 |
| HC inject oracolo | 2015,9 | 1967,0 | 1917,9 | 1907,8 | 1960,2 | 2030,8 | 6559,4 |
| rapporto | 380× | 160× | 121× | 236× | 213× | 15× | 11× |

Sotto i ~30 µs i tempi a 20 per buffer sono rumorosi (inject: 2 righe più
lente di 8). Il confronto con l'oracolo non cambia. Il caso più stretto è HC
down a 1024 righe (1,35×): lì il kernel è limitato dai load di x, non dai
pesi.

## Protocollo

`make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261020/matvec-f16-f32`,
un solo GPU lock per tutta la corsa, `caffeinate -i`, senza validation layer.
La modalità `bench` dell'harness è quella di R1.2a con un tipo in più:
- genera il tensore e x nel processo, senza stdin: xorshift64 con seed 3813;
  F16 con esponenti 2^-14..2^-1, F32 con esponenti 2^-20..2^-1, segno e
  frazione casuali; x uniforme in [-1, 1);
- fa 3 warm-up e 20 misure, ognuna un command buffer con `reps` encode di
  fila; il tempo GPU (`GPUStartTime`/`GPUEndTime`) è diviso per `reps`;
- calcola mediana e p10/p90 con la regola lineare di numpy.

Rispetto a R1.2a c'è la colonna a 200 per buffer: i dispatch delle forme HC e
router durano 4–40 µs, e a 20 per buffer la GPU non sempre alza il clock (HC
down 29,4 µs con p90 49,7; a 200 per buffer 24,5 con p90 24,6). Il grafo di
decode mette circa 3000 dispatch per command buffer.

L'harness è compilato con ASan/UBSan, che toccano solo la CPU. La correttezza
degli stessi kernel (oracolo, FP64, Review Focus 1–5, decodifica dei 65536
pattern fp16) è in `make test-qwen38-fast-kernels`, non qui.

## Condizioni

`before.txt` e `after.txt` contengono, prima e dopo la corsa:
- `pmset -g therm`: nessun avviso termico né di prestazioni;
- `pmset -g batt`: alimentazione AC, batteria 90% prima e dopo;
- `sysctl vm.swapusage`: swap 0 prima e dopo;
- l'output del probe `build/qwen38-capabilities`: `thermal_state` `nominal`
  prima e dopo. `powermetrics` richiede sudo e non è stato usato.

## File

- `runs.jsonl`: una riga JSON per misura (modo, tipo, n, k, m, reps, byte,
  mediana, p10, p90, GB/s), nell'ordine di esecuzione.
- `before.txt`, `after.txt`: le condizioni.
- `manifest.txt`: HEAD, `git status --porcelain` al momento della corsa e
  sha256 di harness e metallib.
