# R2 — chiusura

M5 Pro 64 GiB, 30 ottobre 2026. Codice: `9ae6e02`. Suite `bench/qwen38_r0_20261017/suite`,
harness `tools/eval/qwen38_compare.py` (alternato, `--block 1`). Le run appaiate sono su rete
(`on battery False` nei summary), swap residuo 0,25 MB o 0 (ruling R45), nessun avviso termico.

**Esito in una riga:** R2 porta il decode short da 20,7 a 24,2 tok/s e quello a 8K da 11,7 a
21,7 tok/s: **0,747×** DwarfStar sul prompt corto (obiettivo 0,80×: non raggiunto) e
**0,721×** a 8K (obiettivo 0,60×: raggiunto).

| Frontiera | Motore | Warmup+misure | tok/s mediana | p10 | p90 | vs DwarfStar |
|---|---|---|---:|---:|---:|---:|
| short | picoforge | 3+20 | 24,151 | 23,077 | 24,233 | 0,747× |
| short | DwarfStar | 3+20 | 32,330 | 31,600 | 32,740 | — |
| 8K (8063 + 128) | picoforge | 1+5 | 21,732 | 21,491 | 22,102 | 0,721× |
| 8K (8063 + 128) | DwarfStar | 3+20 | 30,140 | 29,400 | 31,580 | — |
| short, R1 | picoforge | 3+20 (28 ottobre) | 20,716 | 20,692 | 20,741 | 0,623× |
| 8K, R1 | picoforge | 1+5 (28 ottobre) | 11,715 | 11,133 | 11,745 | 0,372× |

DwarfStar oggi è più lento che in R1 (32,33 contro 33,23 short, 30,14 contro 31,48 a 8K): il
rapporto usa la DwarfStar della stessa sessione, come in R1.

- `paired-short-r2/`, `paired-8k-r2/`: campioni grezzi, manifest e processi.
- `profile-folded/` (short, batteria): 38,63 ms/token di GPU (somma dei gruppi), 158 GB/s,
  1356 dispatch e 1271 barriere per token (R1: 44,99 ms, 2272 e 1789).
- `profile-8k-r29/` (8K, batteria): 40,55 ms/token di gruppi, modalità normale 49,1 ms;
  qsa.core 2,41 ms (R1: 39,6 ms in `../qwen38_r2_20261028/profile-8k-r1`).
- `profile-qsa-small/`, `speed-*`: passi intermedi R2.8 e R2.9 (`speed-*`: picoforge da solo,
  1+3, per dimensionare il divario; non appaiati).
- Correttezza alla chiusura (log in `build/r2-logs`, non versionati): `make test-qwen38`,
  MTP, speculative, fast-kernels in default e con `Q38_REF_KERNELS=1` (80 test), compare: tutti
  verdi. G3 PASS, scaled 2,08e-7, argmax 12/12. Logit oracolo con `Q38_REF_KERNELS=1`
  bit-identici a 360acf0 (sha256 36e67ce6…); default: differenza massima 4,95e-5
  (scalata 1,96e-6), argmax 9/9.
- Quello che resta sullo short (dal profilo): le MoE instradate (gate/up 3,3 ms a 123 GB/s,
  down 3,2 ms a 97 GB/s) sono limitate dall'ALU, circa 7 operazioni per peso decodificato;
  poi gdn.hc_pre e le matvec HC F16 (~6,9 ms), gdn.scan 1,85 ms, moe.route 1,3 ms.
