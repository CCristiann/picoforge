# R1.2a — matvec Q8_0 cooperativo, kernel isolato

M5 Pro 64 GiB, macOS 27.0.1 (build 26A434), 19 ottobre 2026, 06:47:57–06:48:56
UTC. Kernel `q38_matvec_q8_0` di `ab4b0a8` (`src/qwen38_matvec.metal` identico)
con le correzioni di review non ancora committate elencate in `manifest.txt`.
Nessuna tocca il kernel. `picoforge.metallib` sha256 `374689ce…`, harness
`build/qwen38-r1/test_fast_kernels` sha256 `90409451…`. Spiegazione e
alternative di tiling provate: [devlog, R1.2a](../../../docs/devlog.md).

**Esito in una riga:** sulla LM head (644 MiB) il kernel veloce legge i pesi a
280,2 GB/s, il 91% dei 307 GB/s. L'oracolo arriva a 52,3 GB/s, quindi il
veloce è 5,4× più rapido. È più veloce dell'oracolo a ogni numero di righe
misurato, da 1 a 1024.

## Risultati

Dati sintetici Q8_0, mai il checkpoint. Mediana di 20 misure dopo 3 warm-up,
fra parentesi p10/p90. GB/s = (byte del tensore + x + y) / mediana.

| Tensore (n_out × K) | veloce, 1 encode per command buffer | veloce, 20 per buffer | oracolo, 1 per buffer |
| --- | ---: | ---: | ---: |
| LM head 248320 × 2560, 644,1 MiB | 2518,4 µs (2413,2/2542,6), 268,6 GB/s | **2413,8 µs (2390,4/2427,9), 280,2 GB/s** | 12933,4 µs (12929,1/12944,0), 52,3 GB/s |
| attn_qkv 10240 × 2560, 26,6 MiB | 320,2 µs (314,8/329,5), 87,2 GB/s | 80,2 µs (79,5/81,0), 347,7 GB/s¹ | 625,2 µs (619,6/1115,6), 44,6 GB/s |

¹ Sopra il tetto. In 20 esecuzioni di fila parte dei 26,6 MiB resta nella
cache di sistema. Nel decode ogni tensore si legge una volta per token, fra
circa 6 GB di pesi, quindi il numero onesto di banda è quello della LM head,
24 volte più grande.

Veloce contro oracolo per numero di righe di x, su attn_qkv (µs). Fino a 32
righe la misura è a 20 encode per buffer per entrambi i kernel; a 128 e 1024
righe è a 1 per buffer (lavoro lungo, clock stabile):

| Righe di x | 1 | 2 | 4 | 8 | 32 | 128 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| veloce | 80,2 | 107,1 | 151,4 | 298,7 | 2127,9 | 10010,5 | 87942,4 |
| oracolo | 633,5 | 1272,7 | 2310,9 | 4462,4 | 17019,3 | 66884,9 | 538160,6 |
| rapporto | 7,9× | 11,9× | 15,3× | 14,9× | 8,0× | 6,7× | 6,1× |

Con 1 encode per buffer il veloce resta davanti a ogni conteggio:
- 1 riga: 320,2 contro 625,2 µs (2,0×)
- 2 righe: 117,3 contro 1261,8 µs (10,8×)
- 4 righe: 161,5 contro 2288,9 µs (14,2×)
- 8 righe: 409,1 contro 4378,4 µs (10,7×)
- 32 righe: 2284,4 contro 17105,1 µs (7,5×)

Le altre forme Q8_0 reali, veloce contro oracolo (µs); 1, 4 e 32 righe a 20
per buffer, 1024 righe a 1 per buffer:

| Forma (n_out × K) | 1 riga | 4 righe | 32 righe | 1024 righe |
| --- | ---: | ---: | ---: | ---: |
| shexp down 2560 × 640, 1,66 MiB | 8,5 / 146,0 (17,2×) | 15,4 / 155,9 (10,1×) | 109,4 / 1120,8 (10,2×) | 4944,3 / 33925,5 (6,9×) |
| ssm_out 2560 × 6144, 15,9 MiB | 43,3 / 1446,7 (33,4×) | 84,1 / 1483,5 (17,6×) | 1358,3 / 10708,8 (7,9×) | 55573,9 / 321596,4 (5,8×) |
| shexp gate/up 640 × 2560, 1,66 MiB | 29,4 / 555,7 (18,9×) | 12,4 / 559,7 (45,1×) | 99,1 / 1253,2 (12,6×) | 5453,1 / 33594,2 (6,2×) |

Sotto i ~50 µs i tempi sono rumorosi: su 640 × 2560, 4 righe costano meno di 1.
Il dispatch è troppo corto per dire di più. Il confronto con l'oracolo non
cambia.

## Protocollo

`make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261019/matvec-q8_0`, un
solo GPU lock per tutta la corsa, `caffeinate -i`, senza validation layer.

La modalità `bench` dell'harness:
- genera il tensore Q8_0 e x nel processo, senza stdin: xorshift64 con seed
  3813, scale fp16 finite in [2^-7, 2^-6), x uniforme in [-1, 1);
- fa 3 warm-up e 20 misure, ognuna un command buffer;
- prende il tempo GPU da `GPUStartTime`/`GPUEndTime`;
- calcola mediana e p10/p90 con la regola lineare di numpy.

Con reps = 20, nello stesso command buffer ci sono 20 encode di fila e il
tempo è diviso per 20. Serve perché un dispatch corto e isolato gira con la GPU
a clock basso: su attn_qkv lo stesso binario ha dato da 142 a 325 µs in corse
diverse. Il grafo di decode mette circa 3000 dispatch per command buffer,
quindi la colonna a 20 è quella rappresentativa. Quella a 1 è il protocollo
richiesto e resta riportata.

L'harness è compilato con ASan/UBSan, che toccano solo la CPU: i tempi GPU non
cambiano. La correttezza dello stesso kernel (oracolo, FP64, Review Focus 1–5)
è in `make test-qwen38-fast-kernels`, non qui.

## Condizioni

`before.txt` e `after.txt` contengono, prima e dopo la corsa:
- `pmset -g therm`: nessun avviso termico né di prestazioni;
- `pmset -g batt`: alimentazione AC, batteria 90% e poi 89% (in carica);
- `sysctl vm.swapusage`: swap 0;
- l'output del probe `build/qwen38-capabilities`: `thermal_state` `nominal`
  prima e dopo. `powermetrics` richiede sudo e non è stato usato.

## File

- `runs.jsonl`: una riga JSON per misura (modo, n, k, m, reps, byte, mediana,
  p10, p90, GB/s), nell'ordine di esecuzione.
- `before.txt`, `after.txt`: le condizioni.
- `manifest.txt`: HEAD, `git status --porcelain` al momento della corsa e
  sha256 di harness e metallib.
