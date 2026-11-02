# R3 — Prefill

Tappa R3 della [roadmap](roadmap.md). Branch `perf/qwen38-prefill` da `main` (d47d47b).

## Punto di partenza (misurato, non stimato)

Profilo 8K di R2 (`bench/qwen38_r2_20261030/profile-8k-r29/profile.csv`, righe `prefill`,
8 chunk da 1024): 199,8 s di GPU per 8063 token, cioè ~40 tok/s. Le run appaiate a 8K
danno prefill 210–253 s per picoforge e 560–605 tok/s per DwarfStar: un divario di ~15×.

| Gruppo | Quota del prefill | Ops |
|---|---:|---|
| Lineari densi (Q8_0, F16, F32) | ~55% | gdn.proj, gdn.out, gdn.hc_pre, hc_*.down/up, qsa.proj, qsa.out |
| MoE instradate | ~23% | moe.gate_up, moe.down (microbatch da 32 token) |
| Attention QSA | ~7% | qsa.core |
| Ricorrenza GDN | ~4% | gdn.scan |

Causa principale: i kernel lineari sono matvec che leggono 4 righe di x per threadgroup
(`Q8_T`, `DENSE_T`). Con 1024 righe la stessa matrice viene riletta ~256 volte: il prefill
è trattato come 256 decode, non come una GEMM.

## Passi

- **R3.1 GEMM densa simdgroup.** `simdgroup_float8x8`, tile 32 righe × 64 uscite × 32 di K,
  pesi decodificati in FP32 in memoria threadgroup. Usata da `encode_linear` e dalla multi
  quando le righe sono ≥ 32; sotto (decode, verifica MTP) restano i matvec di R1/R2, bit per
  bit. `Q38_REF_KERNELS=gemm` torna ai matvec.
- **R3.2 MoE raggruppata.** Token ordinati per esperto, una GEMM per esperto attivo sul
  chunk invece di microbatch da 32 che rileggono gli stessi esperti.
- **R3.3 GDN a chunk** (delta rule chunkwise) se il profilo lo giustifica.
- **R3.4 Attention di prefill a blocchi** per qsa.core.
- **R3.5 TensorOps sui Neural Accelerator** (`matmul2d`): input half/bfloat, quindi è una
  modalità a precisione ridotta. Si fa solo come modalità opzionale con gate proprio e con
  il tuo ok (regola della roadmap).

## Correttezza

- La GEMM cambia l'ordine delle somme rispetto al matvec: niente bit-identità con il
  decode, divergenza spiegata. Ogni GEMM si confronta con l'oracolo con la tolleranza della
  sua famiglia in `tests/qwen38_kernel_tolerances.json` (condizioni (a) e (b)).
- Dentro la GEMM i bit di una riga non dipendono da M né dalla posizione della riga.
- Gate end-to-end invariato: G3, logits ≤ 1e-3 contro il riferimento FP32, greedy
  identico; `Q38_REF_KERNELS=1` resta bit-identico all'oracolo R0.

## Obiettivo e misura

Obiettivo di roadmap: prefill 8K ≥ 0,95× DwarfStar. Misure rapide per passo con
`picoforge bench` su un prompt lungo; il protocollo appaiato completo solo a chiusura.
