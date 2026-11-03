# R3 — chiusura (Prefill)

M5 Pro 64 GiB, 2–3 novembre 2026. Codice: `785ee5c` (branch `perf/qwen38-prefill-nax`, da `main`
1700582 con R3.1–R3.3). Suite `bench/qwen38_r0_20261017/suite`, harness
`tools/eval/qwen38_compare.py` (alternato, `--block 1`), DwarfStar `fc80bd6`.

**Esito in una riga:** prefill 8K da ~215 s (R2) a **61,0 s in FP32 (132 tok/s)** e
**37,6 s con `--prefill-precision half` (214 tok/s)**. L'obiettivo di roadmap (≥ 0,95× DwarfStar)
**non è raggiunto**: 0,36× in FP32 e 0,59× in half contro la DwarfStar della stessa sessione
(365,9 tok/s, degradata), 0,23× e 0,37× contro i ~580 tok/s misurati in R2/R3.

| Misura (8063 ID di prompt + 128 forzati) | Motore | Campioni | Prefill tok/s mediana | p10 | p90 | Decode tok/s |
|---|---|---|---:|---:|---:|---:|
| `paired-8k-r3/` | picoforge FP32 | 1+5 | 132,2 | 107,7 | 132,5 | 23,10 |
| `paired-8k-r3/` | DwarfStar | 3+6 (fermata dal guard al 015) | 365,9 | 320,5 | 552,8 | 17,53 |
| `alternating-8k/` coppia 1 | picoforge FP32 / half | 1 / 1 | 132,5 / 214,4 | — | — | — |
| `half-gate/` greedy e teacher-forced | picoforge FP32 / half | 1 / 1 | 123,7 / 213,6 e 124,5 / 195,3 | — | — | — |

Condizioni, da leggere prima dei numeri:

- **Alimentazione degradata.** Il caricatore non reggeva il carico: batteria dal 12% al 6% "charging"
  durante le run, un distacco dell'AC in mezzo (`paired-8k-r3-aborted/`: campione 004 fermato dal
  guard, poi Mac a batteria). Verso il 6% la macchina rallenta: nelle coppie alternate
  (`alternating-8k/`) FP32 passa da 60,8 a 106,8 e 120,6 s e half da 37,6 a 75,2 e 88,0 s (half
  resta 1,37–1,62× più veloce in ogni coppia). Per questo `paired-8k-r3-half/` (half 1+5, prefill
  57,4 s, decode 15,0 tok/s contro 23,1 della run FP32 sullo stesso binario) **non è confrontabile**
  ed è conservata solo come evidenza; il numero half della riga 1 è la coppia sana.
- **DwarfStar degradata.** Decode 17,5 tok/s contro 30,1 in R2, prefill instabile (p10–p90
  320–553); il memory guard l'ha fermata al campione 015 (pressione), quindi 6 misure invece di 20.
  Il rapporto usa comunque la DwarfStar della stessa sessione, come in R1/R2; il confronto con i
  ~580 tok/s storici è dato a fianco.
- Nessun avviso termico (`pmset -g therm`), swap delta 0 sui campioni misurati picoforge.

Altro in questa cartella:

- `half-gate/logits.csv` (da `tools/eval/qwen38_half_gate.py`): half contro FP32 su N = 64…4000 ID
  del prompt 8K + 16 forzati e sull'8K + 128: scaled 0,06–0,16, KL max 0,04–0,37, argmax cambiati
  solo con margine FP32 top-1/top-2 ≤ 0,27, NLL media +0,0011 nat su 264 posizioni (+0,045 sulla
  finestra 8K). `greedy-*`: short identico (64/64), 8K identico per 11 token su 64.
  `fast-kernels-half-report.json`: gate sintetico (max (b) 4,1e-4, tol 2^-11).
- `profile-1000/`: profilo per op a N 1000, FP32 contro half (somma gruppi 6,66 → 3,74 s).
- Correttezza alla chiusura (log in `build/r35/logs`, non versionati): `make test-qwen38`,
  fast-kernels in default e con `Q38_REF_KERNELS=1` (93 test ciascuno), MTP, speculative, compare,
  CLI: tutti verdi. G3 PASS, scaled 2,08e-7, argmax 12/12 (come R2). Logit oracolo con
  `Q38_REF_KERNELS=1` bit-identici a 360acf0 (sha256 36e67ce6…), anche con
  `PICOFORGE_PREFILL_PRECISION=half`; default: differenza massima 4,95e-5 (scalata 1,96e-6), argmax 9/9.
