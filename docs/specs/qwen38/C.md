# C — Ricorrenza e operatori di base Qwen3.8

Modalità **SPEC**, 2026-10-12. Stato: proposta per il consolidamento, nessun
operatore implementato o test numerico eseguito da questa sessione.
`BASE_CONCORDATA`: **da determinare dal coordinatore**.
Base ispezionata: `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
Branch: `spec/qwen38-c`; worktree esterno:
`/Users/cristiancirje/Desktop/Dev/picoforge-spec-c`.

Le etichette **V** (verificato su documenti/codice), **P** (contratto proposto)
e **A** (decisione/evidenza aperta) distinguono lo stato delle affermazioni.
Un controllo del sorgente non è una verifica numerica dell'implementazione.

## 1. Obiettivo e confini

**P.** Realizzare, dopo il consolidamento, le primitive C11 e Metal per:

- convoluzione causale depthwise comune a GDN ed E, anche dilatata;
- preparazione, aggiornamento di stato e uscita del Gated DeltaNet (GDN);
- L2Norm, RMSNorm a gruppi, sigmoid/SiLU/softplus stabili;
- lettura e iniezione del residuo a più rami (Gated Residual, nomi GGUF `hc_*`).

Il percorso iniziale usa una sequenza senza padding, accumuli e stato FP32,
prefill limitato a chunk di 1.024 token e decode a un token. Le dimensioni
arrivano da A; i numeri del checkpoint qui sotto sono verifiche del contratto,
non costanti da incorporare nell'engine. Nessun limite a 8K deriva dalla
ricorrenza: 8K è il primo obiettivo del grafo completo.

C possiede la matematica e lo stato della convoluzione. E possiede hashing,
lookup, gating PLE e la composizione del suo risultato col residuo; alloca una
propria istanza dello stato conv e usa la primitiva C. D conserva QSA/indexer,
RoPE, causalità e cache attention. F conserva routing e composizione MoE.
Le proiezioni compresse sono di B; C non aggiunge decoder Q8_0, Q2_K o IQ2_XXS.
Il coordinatore possiede il grafo e la transazione di stato tra componenti.
MTP, rewind speculativo, batching di sessioni, riduzione di precisione e
benchmark sono incrementi successivi. Il percorso Qwen3 esistente va preservato.

### Revisioni e fonti effettivamente consultate

**V.** HEAD, staging e working tree principali erano puliti all'avvio; nessun
reset. È stato letto anche il `CLAUDE.md` ignorato nella repository principale
(SHA-256 `3e49a726cf993e65af1a55c2d0079f7375ba2f5065d12b848a1634e6970aa844`).
Le vecchie roadmap Qwen3 non prevalgono sull'incarico Qwen3.8.

| Rif. | Fonte fissata e punti pertinenti |
|---|---|
| P0 | `README.md`, `Makefile`, `docs/DESIGN.md`, `docs/BENCHMARKS.md` alla base sopra |
| P1 | `docs/qwen38-q2-session-prompt.md`, `docs/qwen38-q2-baseline.md`, `docs/qwen38-storage.md`, letti integralmente |
| P2 | `docs/large-model-research-2026-10-11.md`, sezioni architettura, oracle, memoria e ordine; contesto in `project-review-2026-10-11.md`, `phase4-expert-experiment.md`, `80b-100b-model-comparison-2026-10-11.md` |
| I | `bench/qwen38_q2_20261011_phase0/inventory.json`, SHA-256 `605d00ad17ae2fa66b10cf64d0dd0c3fac16122bb526dee10db064a0b8a67093`; inventario registrato, payload non verificato da questa sessione |
| Q | [Configurazione originale](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json), revisione `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| H | [Transformers `modeling_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py), revisione `812a950d7037ba93bb624b154030e9068afdce4d`, SHA-256 del raw letto `f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa`; RMSNorm/Gated 143–187, conv/L2 203–250, delta chunk/recurrent 254–450, GDN 456–616, GatedResidual 994–1022, PLE 1174–1246 |
| D0 | Checkout locale `/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline`, pulito a `fc80bd695da76ee14bcb86820bf900a5cdfda806`; confronto GitHub con `main` il 2026-10-12: `identical`, nessun commit successivo da esaminare |
| D1 | [DwarfStar `ds4.c`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c): `qwen4_ref_rms/grouped_rms/l2norm` 68008–68026, `qwen4_ref_hc_mix/combine` 68085–68120, `qwen4_ref_ple` 68146–68202, `qwen4_ref_linear` 68204–68268, layer/output 68439–68483 |
| D2 | [DwarfStar `metal/qwen4.metal`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/metal/qwen4.metal): HC 28–458, conv 460–646, prep/scan 648–763, gated norm 935–959, PLE 968–1100 |
| D3 | [Conversione pack/GGUF](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/qwen4_pack_to_qwen4exp.py): `tiled_v`, `prod_ssm_a`, `prod_gdn_conv1d_f32`, `prod_qkv_vperm`; `qwen4_exp_convert.py` per precisioni e trasformazioni delegate al convertitore |
| R | [Report Qwen3.8 v1](https://arxiv.org/html/2608.30320v1), §§2.1.1 e 2.2, eq. 4–11 e 30–34; corroborazione matematica, non specifica binaria del GGUF |

L'artifact resta `antirez/qwen3.8-flash-next-gguf` revisione
`d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`, file
`Qwen3.8-Flash-Next-Q2.gguf`, SHA-256 registrato
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
`checkpoint.verification.json` registra `matched=true`; la lettura dei soli
attributi del file in questa sessione conferma 147.207.127.040 byte e lo stesso
mtime `1791785892435960350`. Non è stato ricalcolato l'hash né letto il payload.

## 2. Stato corrente e codice riutilizzabile

**V.** Lo storage nativo esiste; la generazione Qwen3.8 nativa no [P1].

| Codice attuale | Riutilizzo / limite |
|---|---|
| `src/gguf.h`, `gguf.c`, `gguf_reader.c`, `gguf_plan.c` | Descrittori, forma fisica, formato, offset e ownership; nessuna semantica GDN implicita |
| `src/gpu_weights.m` | Viste no-copy multiple; ogni binding appartiene a un plan; nessun mapping Metal della tabella n-gram |
| `src/ops.c:rmsnorm`, `silu`, `matmul` | Convenzioni e regressioni Qwen3; RMS con gain BF16 non accetta direttamente i gain F32 del GGUF; la `silu` attuale non è la primitiva stabile proposta |
| `src/kernels.metal:rmsnorm_rows`, `swiglu` | Tecniche di riduzione; non forniscono grouped RMS F32, GDN o gated residual Qwen3.8 |
| `src/metal.m`, `src/picoforge.h` | Device, queue, library, pipeline e buffer; modifiche solo del coordinatore. Cache pipeline attuale di 32 nomi: contare i nuovi kernel prima dell'integrazione |
| `tools/oracle/qwen3_forward.py`, `tests/test_ops.{c,py}` | Struttura della catena di oracle e harness. Le tolleranze Qwen3 non valgono automaticamente per GDN |
| `src/config.c`, `model.c`, `gpu_forward.m` | Percorso Qwen3, da preservare; non estendere il grafo semplicemente aggiungendo un nome modello |

**V.** Il venv principale contiene NumPy 2.5.1, torch 2.13.0 e Transformers
5.13.0 (metadata letti, nessun aggiornamento). Il file Qwen4Exp di H **non è
installato**. Esiste `transformers/models/qwen3_next/modeling_qwen3_next.py`
(SHA-256 `af00199af4ed6c2a632b5ef42a60c424968a566946a683e43127c3f883ccc95c`),
con delta rule chunk/recurrent utile come controllo supplementare, ma non come
modello sostitutivo: output gate e proiezioni sono specifici di Qwen3.8.

## 3. Semantica matematica e layout

### 3.1 Dimensioni e binding

Notazione: `T` token, `E` hidden size, `H` rami residui, `Rk` rango HC,
`Hk/Hv` teste key/value GDN, `Dk/Dv` dimensioni, `K` tap,
`C = 2 Hk Dk + Hv Dv`. Le matrici logiche sono `[out,in]`;
GGUF elenca prima la dimensione contigua, quindi `[in,out]`.

**V.** Per l'artifact [I,Q,D1]:

| Parametro | Metadato / derivazione | Valore osservato |
|---|---|---:|
| E, H, Rk | `qwen4exp.embedding_length`, `hyper_connection.count`, `hyper_connection.low_rank` | 2560, 4, 320 |
| Hk, Hv | `ssm.group_count`, `ssm.time_step_rank` | 16, 48 |
| Dk, Dv | `ssm.state_size`, `ssm.inner_size / Hv` | 128, 128 |
| K, C | `ssm.conv_kernel`, formula sopra | 4, 10240 |
| epsilon RMS | `attention.layer_norm_rms_epsilon` | FP32 circa 1e-6 |
| epsilon L2 | Semantica H/D1/D2, non chiave GGUF rilevata | 1e-6 |
| backbone / MTP | `block_count - nextn_predict_layers` / `nextn_predict_layers` | 48 / 1 |

Ci sono 36 tensori `ssm_a` (layer zero-based 0,1,2,4,5,6,…,44,45,46),
12 QSA nel backbone e un blocco MTP ulteriore. A deve produrre il tipo di
ogni layer dai metadati e validarlo contro i tensori; C non inferisce il tipo
dal solo modulo quattro. `ssm.time_step_rank` qui conta value head, non `Rk`.

Formati verificati uniformi nei 36 GDN; i suffissi sotto sono relativi a `blk.L.`:

| Tensore | Forma fisica GGUF | Forma logica / uso | Formato |
|---|---|---|---|
| `attn_qkv.weight` | `[E,C]` | `[C,E]`, Q poi K poi V | Q8_0 |
| `attn_gate.weight` | `[E,Hv Dv]` | `[Hv Dv,E]`, z senza conv | Q8_0 |
| `ssm_alpha.weight`, `ssm_beta.weight` | `[E,Hv]` | `[Hv,E]`, a e b | F32 |
| `ssm_a`, `ssm_dt.bias` | `[Hv]` | A effettiva e bias dt | F32 |
| `ssm_conv1d.weight` | `[K,C]` | tap `[C,K]` | F32 |
| `ssm_norm.weight` | `[Dv]` | gain diretto, condiviso fra value head | F32 |
| `ssm_out.weight` | `[Hv Dv,E]` | `[E,Hv Dv]` | Q8_0 |
| `hc_{attn,ffn}_norm.weight` | `[H E]` | gain per ramo e canale | F32 |
| `hc_{attn,ffn}_down.weight` | `[H E,Rk]` | `[Rk,H E]` | F16 |
| `hc_{attn,ffn}_up.weight` | `[Rk,H E]` | `[H E,Rk]` | F16 |
| `hc_{attn,ffn}_inject.weight` | `[H E,H]` | `[H,H E]` | F16 |

`output_hc_{norm,down,up}.weight` ha le stesse forme del mixer, senza inject.
`blk.1.ple_conv1d.weight` è F32 `[4,10240]`; E usa `K=4`, dilatazione
`ngram_size=3`, `C=H E=10240`. A/E devono riconciliare `ple.layers=[1]`
zero-based con `ple_layer_ids=[2]` nella configurazione originale.
Il padding esperti 640/768 appartiene a B/F, non allo stato GDN di C.

### 3.2 Ordine delle teste e trasformazioni già nel file

**V.** In H le value head sono raggruppate: la head `j` usa
`floor(j / (Hv/Hk))`. Nel GGUF [D3] la head fisica `h` contiene la head HF

`p(h) = (h % Hk) * (Hv/Hk) + floor(h/Hk)`.

Quindi C usa `key_head(h) = h % Hk`. La permutazione coinvolge V nella
proiezione QKV e nella conv, z, a, b, `ssm_a`, dt e l'asse di ingresso
di `ssm_out`. Non ripermutare soltanto V, né ripetere contiguamente Q/K
sui tensori già in ordine GGUF. Validare `Hv % Hk == 0`. A certifica l'ordine
del binding; il confronto con H applica la permutazione inversa, con fixture
che distinguano ogni testa anche quando `Hk != Hv`.

**V.** `ssm_a` contiene già `-exp(A_log)`; usarlo come tale. I gain GGUF
normativi sono già effettivi: nessun `+1` nel kernel C. H usa `1+w` per
`Qwen4ExpTextRMSNorm` a gruppi e gain diretto per `RMSNormGated` del GDN;
l'adattatore dell'oracle deve rispettare questa distinzione. Non applicare
al GGUF attivo le trasformazioni del vecchio pack DS4, inclusi divisori HC.
In particolare il docstring di `prod_hc_mix_f32` e la sua moltiplicazione
per quattro non bastano a definire una nuova convenzione: fanno fede il
grafo GGUF D1/D2, corroborato dalle formule H; A verifica il binding.

### 3.3 Convoluzione causale comune

**V → P.** Conv1d qui è correlazione depthwise, senza bias. Per canale c:

`u[t,c] = sum(j=0..K-1) W[c,j] * x[t-(K-1-j)*d,c]`.

Il tap `K-1` moltiplica il token corrente; `d` è la dilatazione. L'uscita
è `u` oppure `SiLU(u)`, scelta esplicita dell'operatore. GDN: `d=1` e SiLU.
PLE: `d=3` osservato, SiLU, ingresso già normalizzato fornito da E.

Stato persistente FP32 `[Lhist,C]`, `Lhist=(K-1)*d`, **oldest first**.
Prima di un chunk a posizione p contiene `x[p-Lhist:p]`, con zeri per
posizioni anteriori all'inizio sequenza. Dopo T token contiene le ultime
Lhist righe della concatenazione `(stato_in, x[0:T])`, anche se `T<Lhist`.
Si conserva l'ingresso della conv, mai il suo output attivato. Per PLE ciò
significa `gated_value_normed`, non gli ID, gli embedding o il residuo.

**P.** Supportare `K>=1`, `d>=1` entro limiti/memoria validati; `K=1` non
alloca storia. Baseline separa input e output. Il kernel ottimizzato può
supportare alias esatto solo con contratto esplicito e test del mantenimento
dell'input grezzo. Niente letture future: conv parallelizzata per token legge
un halo immutabile e l'input grezzo; commit della storia dopo tutti i lettori.
Un confine di chunk non introduce padding a sinistra nuovo.

### 3.4 Preparazione GDN, ricorrenza e uscita

**V.** Da input `x[t,E]` del mixer HC, proiezioni senza bias:

`raw_qkv = Wqkv x`, `z = Wz x`, `a = Wa x`, `b = Wb x`.

Conv+SiLU di `raw_qkv`, poi split Q `[T,Hk,Dk]`, K identica, V `[T,Hv,Dv]`.
Per ciascuna key head:

`q = q_raw / sqrt(sum(q_raw^2) + eps_l2) / sqrt(Dk)`

`k = k_raw / sqrt(sum(k_raw^2) + eps_l2)`.

È una **somma**, non una media, e epsilon è dentro la radice (non
`max(norm,eps)`). Normalizzare una volta, applicare `1/sqrt(Dk)` una volta.
Nessuna RoPE nel GDN. Per value head h:

`log_decay[t,h] = ssm_a[h] * softplus(a[t,h] + dt[h])`

`decay[t,h] = exp(log_decay[t,h])`, `beta[t,h] = sigmoid(b[t,h])`.

Il nome `g` in H indica **log-decay**; in D2 `ga` contiene già **decay**.
L'API proposta usa il nome `log_decay` e fa l'exp nello scan, senza ambiguità.

**P.** Stato canonico in C e Metal `S[Hv,Dv,Dk]`, asse dk contiguo; ogni
test esporta questo ordine nel manifest. H/D1 usano `[Hv,Dk,Dv]`:
la trasposizione è obbligatoria nel confronto, non un errore numerico.
Per token, con q/k della key head associata, e vettori colonna:

1. `S_decay = decay * S_previous`.
2. `prediction = S_decay k` (`Dv` valori).
3. `delta = beta * (v - prediction)`.
4. `S_new = S_decay + delta k^T`.
5. `o = S_new q` (`Dv` valori; legge lo stato **dopo** l'update).

Equivalentemente `S_new = decay S_previous (I-beta k k^T) + beta v k^T`.
Decadere dopo il delta, leggere `S_previous` al punto 5 o omettere la
predizione sottratta cambia l'algoritmo.

Uscita per testa:

`y[t,h,v] = o[t,h,v] / sqrt(mean_v(o[t,h,v]^2)+eps_rms)`

`y[t,h,v] *= ssm_norm[v] * sigmoid(z[t,h,v])`.

Poi `block_out[t,E] = Wout flatten(y[t,Hv,Dv])`. Il gate z è sigmoid,
non SiLU; la norma precede il gate, la riduzione è per testa Dv, il gain
non è per head. **P.** Attivazioni, dot product, somme, stato, conv history,
log-decay, beta e output restano FP32; nessuna conversione F16/BF16 implicita
ai confini del chunk. FP64 appartiene all'oracle analitico, non al kernel.

Sigmoid usa la forma stabile a due rami; SiLU è `x*sigmoid(x)`;
softplus è `max(x,0)+log1p(exp(-abs(x)))`. L'approssimazione D2 a ±20 è
un riferimento da confrontare, non un vincolo bitwise. Accettare underflow
del decay a zero; rifiutare NaN/+Inf e domini invalidi. Le statistiche
numeriche devono rilevare overflow nelle riduzioni, non occultarlo.

### 3.5 Normalizzazioni e residui gated

**V.** Per `R[t,H,E]`, la grouped RMS è indipendente per ramo:

`N[t,s,e] = R[t,s,e] * gamma[s,e] / sqrt(mean_e(R[t,s,e]^2)+eps)`.

Le proiezioni vedono `flatten(N)` in ordine stream-major:

`lo = Wdown flatten(N)`; `a_lo = SiLU(lo/H)`;

`gate_logits = Wup a_lo`; `G = sigmoid(gate_logits)`;

`mixed[t,e] = sum_s(G[t,s,e] * N[t,s,e]) / H`;

`inj[t,s] = 2 * sigmoid((Winject flatten(N))[t,s] / H)`;

`R_after[t,s,e] = R_before[t,s,e] + inj[t,s] * block_out[t,e]`.

Due divisioni distinte per H nel mixer: prima di SiLU e nella media;
una divisione per H nell'iniezione. Non è un softmax fra rami e non vi è
una matrice H×H che rimescola il residuo. `inj` si calcola sullo stesso N
che ha prodotto `mixed` e resta valido fino al relativo combine.

Ordine del layer [D1,H]: eventuale PLE aggiunge il suo risultato a R;
HC-attn → GDN o QSA → combine; nuovo HC-ffn sul residuo aggiornato → MoE
→ combine. Inizio testo: embedding replicato sui H rami. Uscita backbone:
mixer `output_hc_*` senza inject, poi LM head; non aggiungere una norma
Qwen3 estranea a questo grafo. L'ordine del grafo è del coordinatore.

**P.** C espone grouped RMS riutilizzabile da E/D con gain F32 diretto;
D decide i suoi assi head e le eventuali fusioni RoPE. E resta responsabile
del gate PLE signed-sqrt e di `R += gated_value + SiLU(conv(normed))`.
Nessuna seconda implementazione della conv in E.

### 3.6 Stato, reset, append ed equivalenza a chunk

**P.** Ogni sequenza e layer GDN ha uno stato esclusivo composto da S,
history conv, `next_pos`, identità modello/configurazione e flag validità.
E ha uno stato conv separato. Il reset completo azzera S e history, riporta
`next_pos=0`, aggiorna la generazione dello stato; non azzera i pesi.
EOS non resetta automaticamente GDN/conv: il reset di una sessione è esplicito.
Le regole EOS per gli ID n-gram sono di E e non implicano reset GDN.

Prima di append richiedere `start_pos == next_pos`, shape/config identiche,
nessuna operazione in volo; successo avanza di T. `T=0` è no-op senza
letture dati né avanzamento, su descrittori e stato comunque validi.
Non basta cambiare `next_pos` per rewind: servono snapshot di tutto lo stato
oppure replay dal reset. Snapshot/rollback MTP non sono implementati qui.

Per due chunk X,Y, in aritmetica esatta:
`F(X||Y,S0).output = concat(F(X,S0).output, F(Y,F(X,S0).state).output)` e
`F(X||Y,S0).state = F(Y,F(X,S0).state).state`; vale anche per conv con history.
La baseline C/Metal scorre
token nello stesso ordine all'interno del chunk, parallelizzando teste e
righe dv, quindi non ha una diversa algebra per prefill/decode.

Un futuro scan parallelo può usare transizioni affini
`S' = S M + B`, `M=decay*(I-beta*k*k^T)`, `B=beta*v*k^T`.
La composizione ordinata è `(M1 M2, B1 M2+B2)`; ciò dimostra l'equivalenza
algebrica, **non** bitwise FP32. Non materializzare T matrici Dk×Dk nel
runtime. L'oracle chunk H usa invece un sistema triangolare: percorso
computazionale indipendente da confrontare già prima di ottimizzare C.

## 4. API proposte: dati, ownership, errori e durata

**P.** Nomi provvisori da congelare in `integration.md`. Header pubblici
materializzati dal coordinatore; nessun header scritto in modalità SPEC.
Le primitive operano su attivazioni/proiezioni già calcolate: ciò consente
verifiche reali senza simulare B o dichiarare il layer completo.

```c
typedef enum {
    Q38_OK, Q38_INVALID_ARGUMENT, Q38_BAD_SHAPE, Q38_BAD_STRIDE,
    Q38_OVERFLOW, Q38_CAPACITY, Q38_BAD_STATE, Q38_UNSUPPORTED,
    Q38_NONFINITE, Q38_GPU_ERROR
} Qwen38Status;

typedef struct {
    const float *data;
    size_t capacity_bytes, rank, dim[3], stride_bytes[3];
} Qwen38F32;
typedef struct {
    float *data;
    size_t capacity_bytes, rank, dim[3], stride_bytes[3];
} Qwen38F32Mut;

typedef struct {
    size_t channels, kernel, dilation;
    bool apply_silu;
} Qwen38ConvShape;
typedef struct {
    size_t key_heads, value_heads, key_dim, value_dim;
    float rms_eps, l2_eps;
} Qwen38GdnShape;
typedef struct {
    size_t streams, hidden_size, low_rank;
    float rms_eps;
} Qwen38HcShape;
typedef struct Qwen38ConvState Qwen38ConvState;
typedef struct Qwen38GdnState Qwen38GdnState;

Qwen38Status qwen38_conv_causal(const Qwen38ConvShape *shape,
    Qwen38F32 x, Qwen38F32 taps, Qwen38F32Mut out,
    Qwen38ConvState *state, uint64_t start_pos);
Qwen38Status qwen38_gdn_scan(const Qwen38GdnShape *shape,
    Qwen38F32 qkv_prepared, Qwen38F32 log_decay, Qwen38F32 beta,
    Qwen38F32Mut out, Qwen38GdnState *state, uint64_t start_pos);
Qwen38Status qwen38_gdn_prepare(const Qwen38GdnShape *shape,
    Qwen38F32 qkv, Qwen38F32 a, Qwen38F32 b,
    Qwen38F32 ssm_a, Qwen38F32 dt_bias,
    Qwen38F32Mut qkv_prepared, Qwen38F32Mut log_decay, Qwen38F32Mut beta);
Qwen38Status qwen38_gdn_norm_gate(const Qwen38GdnShape *shape,
    Qwen38F32 o, Qwen38F32 z, Qwen38F32 gain, Qwen38F32Mut out);
Qwen38Status qwen38_rmsnorm_groups(Qwen38F32 x, Qwen38F32 gain,
    Qwen38F32Mut out, float eps);
Qwen38Status qwen38_hc_lowrank_activate(const Qwen38HcShape *shape,
    Qwen38F32 lo, Qwen38F32Mut activated);
Qwen38Status qwen38_hc_read(const Qwen38HcShape *shape,
    Qwen38F32 normed, Qwen38F32 gate_logits, Qwen38F32Mut mixed);
Qwen38Status qwen38_hc_inject_weights(const Qwen38HcShape *shape,
    Qwen38F32 logits, Qwen38F32Mut inject);
Qwen38Status qwen38_hc_add(const Qwen38HcShape *shape,
    Qwen38F32Mut residual, Qwen38F32 block_out, Qwen38F32 inject);
```

L'header include `<stdbool.h>`, `<stddef.h>`, `<stdint.h>`. Per GDN il
pack QKV e le head sono sempre in ordine GGUF §3.2; altri ordini sono
rifiutati dal binding, non indovinati dal kernel. Lo stato GDN dello scan
contiene S; la history è nel separato `Qwen38ConvState`. Il coordinatore
li tratta come coppia inseparabile nel contesto di layer.

| API / operazione proposta | Forme e comportamento |
|---|---|
| `qwen38_conv_state_bytes`, `qwen38_gdn_state_bytes` | Calcolo checked di byte payload e allineamento; nessuna allocazione nascosta |
| `qwen38_conv_state_init`, `qwen38_gdn_state_init`, rispettivi `reset` | Costruiscono metadata e azzerano storage prestato; config immutabile copiata, stamp di identità e posizione |
| `qwen38_conv_causal` | x/out `[T,C]`, taps `[C,K]` F32, history canonica; forma minima sopra |
| `qwen38_gdn_prepare` | QKV dopo conv `[T,C]` → Q/K normalizzati, V invariato; a/b `[T,Hv]`, A/dt `[Hv]` → log-decay/beta `[T,Hv]`; in-place esatto consentito per QKV, a e b dopo validazione |
| `qwen38_gdn_scan` | qkv preparato `[T,C]`, log-decay/beta `[T,Hv]`, output `[T,Hv,Dv]`; stato `[Hv,Dv,Dk]`, distinto da input/output |
| `qwen38_gdn_norm_gate` | o/z `[T,Hv,Dv]`, gain `[Dv]` diretto, output stessa forma; ammesso output=o |
| `qwen38_rmsnorm_groups` | x/out `[T,G,N]`, gain `[G,N]` oppure `[N]` per gain condiviso; eps argomento; supporta alias esatto x/out |
| `qwen38_hc_lowrank_activate` | lo `[T,Rk]` → SiLU(lo/H), anche in-place |
| `qwen38_hc_read` | N/gate_logits `[T,H,E]` → mixed `[T,E]` |
| `qwen38_hc_inject_weights` | logits `[T,H]` → `2*sigmoid(logits/H)`, anche in-place |
| `qwen38_hc_add` | R `[T,H,E]` aggiornato in-place; out `[T,E]`, inj `[T,H]` disgiunti |

Le primitive numeriche ricevono shape/config, view di ingresso `Qwen38F32`
e view di uscita `Qwen38F32Mut`; restituiscono `Qwen38Status`. Grouped RMS
ricava G/N dalle ultime due dimensioni della view rank 3; gli altri operatori
validano view e shape insieme, compreso T uguale fra tutti gli argomenti.
Le query di dimensionamento restituiscono lo status e scrivono byte/allineamento
in parametri di uscita. Init riceve backing e capacità per payload e metadata
opachi, ottenuti dalle query; non alloca, restituisce un handle interno al
backing metadata. Reset prende l'handle ed è sincrono su CPU; per Metal è
un comando con la stessa disciplina di completamento degli altri operatori.
I prototypes completi saranno concordati prima del primo commit C11; semantica,
shape e alias non possono dipendere dall'implementazione scelta.

**Stride e capacità.** Byte, non elementi; ultimo asse contiguo (`4` byte),
assi esterni con stride positivo, multiplo di quattro e senza overlap.
È ammesso padding a fine riga/testa; niente stride negativo o broadcasting
implicito. Prima di ogni accesso calcolare checked
`4 + sum((dim[i]-1)*stride_bytes[i]) <= capacity_bytes` per view non vuote;
per vuote non sottrarre uno. Bounds, prodotti, `start_pos+T`, `Hv/Hk`,
`Lhist` e restringimenti ai campi kernel si validano senza overflow.
Le slice Q/K/V interne mantengono lo stride di riga C, non diventano
accidentalmente tensori contigui più piccoli.

**Ownership.** Il chiamante possiede pesi, backing degli stati e workspace;
gli operatori li prendono in prestito e non fanno I/O o allocazioni per token.
Lo stato lega il backing a un'identità modello/layer: non copiarlo con una
semplice assegnazione, non condividere buffer mutabili fra sessioni. I gain
F32 possono essere viste di pesi immutabili. Eventuali tap F16 di un artifact
futuro vengono convertiti una volta tramite la primitiva condivisa concordata
con B/coordinatore; non si estende silenziosamente il contratto F32.

**Metal.** Stesse shape/layout/precisione; un comando `encode_*` riceve un
encoder/command context C opaco del coordinatore e view GPU con risorsa,
offset byte, capacità e owner espliciti. Nessun Objective-C nella ricorrenza.
Il glue `.m` crea pipeline e lega risorse, senza ospitare la matematica CPU.
È riutilizzato `GpuWeightStore`: mai sommare offset a un unico buffer globale.
Tutte le risorse e gli stati restano vivi fino al completamento; niente reuse,
reset o liberazione durante operazioni in volo. In prima versione usare
command buffer con risorse trattenute; il successo dell'encoding non avanza
ancora `next_pos`. Il coordinatore committa le posizioni a completamento GPU.
I kernel possono limitare esplicitamente le forme supportate nella prima
versione (Dk=Dv=128, per esempio); errore `UNSUPPORTED`, mai scarto di code.

**Errori.** Validazione strutturale fallita lascia output/stato invariati.
Baseline CPU verifica domini finiti prima della mutazione; `log_decay<=0`,
`0<=beta<=1`, eps finiti positivi, A finita non positiva. Un errore aritmetico
o GPU dopo l'inizio invalida lo stato coinvolto e l'intera sessione del grafo;
output parziale non consumabile. Non promettere rollback atomico gratuito:
serve reset/replay. Prevedere flag diagnostico GPU per valori non finiti,
letto a fine comando senza sync per token; un errore di E invalida anche gli
stati C già avanzati. Il wrapper CLI può tradurre lo status in `die()`,
preservando il comportamento fail-fast del progetto.

## 5. Dipendenze e questioni aperte per il consolidamento

| Componente | Contratto necessario / offerto | Stato |
|---|---|---|
| A | Dimensioni logiche, layer GDN/QSA/MTP, binding e forme, gain già effettivi, ordine head GGUF, epsilon; Dk/Dv distinti nei tipi | A: congelare nomi/validazioni. Epsilon L2 e sigmoid sono semantica del modello se assenti dai metadati; A deve registrarli come tali |
| B | Q8_0 serve a tutte e tre le grandi proiezioni GDN; Q2_K/IQ2_XXS non bastano al layer. Proiezioni FP32 in/out, niente conversione attivazioni implicita | A: priorità Q8_0 e API lineare; non duplicare decoder |
| B/coordinatore | Proprietario delle GEMV/GEMM F16/F32 per HC e a/b, conversione dei piccoli gain/tap e API delle proiezioni | A: assegnazione esplicita; C può completare primitive post-proiezione indipendenti, non il layer completo senza queste operazioni |
| D | Consuma mixed `[T,E]`, restituisce block_out `[T,E]`; C fornisce grouped RMS e HC; D tiene posizioni/cache/indexer | A: stessi confini transazione e dtype FP32 |
| E | Conv comune d=ngram_size, history distinta; grouped RMS diretto. E conserva gate, lookup e `R += gated + conv_out` | A: layout `[T,H,E]` e reset congiunto; niente doppia SiLU; nessun reset implicito a EOS |
| F | Consuma mixed `[T,E]`, restituisce somma routed+shared `[T,E]` prima dell'iniezione HC | A: esattamente un combine; un'eventuale fusione richiede stesso contratto e test di composizione |
| Coordinatore | Header/types/status/view GPU, grafo, lifetime, build/PYTHON, invalidazione globale, registry pipeline e lock comune | A: indispensabile prima dei test GPU e dell'integrazione |

Non si sono lette o modificate spec in lavorazione di altre sessioni né
delegate attività. Queste dipendenze sono proposte al coordinatore, non
accordi già presi. Un binding A mancante o B assente non viene sostituito
da pesi fittizi per dichiarare il layer reale completato.

## 6. File posseduti e modifiche condivise richieste

**SPEC: unico file scritto e da committare `docs/specs/qwen38/C.md`.**

**P.** Assegnazione futura richiesta, da ratificare in `integration.md`:

| Proprietario C | Contenuto |
|---|---|
| `src/qwen38_conv.c` | Conv C11 e gestione storia comune C/E |
| `src/qwen38_gdn.c` | Prep, scan, gated norm e stato S |
| `src/qwen38_base.c` | Funzioni stabili, grouped RMS, operazioni HC |
| `src/qwen38_recurrent.metal` | Kernel dei tre ambiti, senza decoder/proiezioni di B |
| `src/qwen38_recurrent_gpu.m` | Solo glue Metal dei kernel C |
| `tools/oracle/qwen38_recurrent.py` | Adattatore di H, oracle analitico e manifest fixture; nessun runtime Python nell'engine |
| `tests/test_qwen38_recurrent.c` | Harness CPU e controlli stato/errori/guard |
| `tests/test_qwen38_recurrent.m` | Harness GPU, lettura output/stato e guards |
| `tests/test_qwen38_recurrent.py` | Confronti oracle/C/Metal e tolleranze |
| `tests/fixtures/qwen38/c/manifest.json` | Versione schema, seed, revisioni e hash delle fixture sintetiche |
| `tests/fixtures/qwen38/c/analytic.json` | Piccoli casi analitici espliciti; niente pesi del modello |

**P.** File condivisi da lasciare al coordinatore: `Makefile`,
`src/picoforge.h`, `src/gguf.h`, nuovo `src/qwen38_ops.h` e header config/view
concordati, `src/metal.m`, `src/kernels.metal`, `src/main.c`, grafo forward,
`README.md`, report aggregati e `integration.md`. Richieste concrete:

1. Materializzare tipi e API del §4 una volta sola; decidere API lineare con B.
2. Compilare il nuovo `.metal` in AIR e linkarlo alla stessa metallib;
   dependency rules per header nuovi e target CPU/GPU separati.
3. Rendere il percorso Python/fixture/modelli parametrizzabile, usando il
   venv e i modelli principali in sola lettura; nessun symlink/installazione
   obbligatoria nel worktree e nessuna conversione sotto `models/` principale.
4. Target `test-qwen38-recurrent-{cpu,gpu}` e collegamento a `test-all`;
   freeze numerico con fast-math disabilitato nel percorso di correttezza.
5. Collegare mixer/GDN/PLE e gestione errori di sessione solo quando le
   dipendenze reali passano; adattare cache pipeline se il conteggio lo richiede.

Le directory `build/qwen38-c/` con fixture generate, slice e output sono
locali al worktree di implementazione, ignorate e non committate. Nessuna
modifica a `ops.c`, alla relativa oracle Qwen3 o alle tolleranze esistenti.

## 7. Oracle indipendente, fixture e slice reali

### 7.1 Catena di riferimenti

**P.** La riscrittura NumPy dello stesso loop C da sola non è un oracle.
Usare tre evidenze complementari:

1. **H CPU PyTorch**, senza costruire il modello: funzioni chunk e recurrent,
   conv tramite `torch.nn.functional.conv1d`, RMS/GatedResidual selezionati
   dal sorgente upstream fissato. Poiché Qwen4Exp non è installato, preparare
   in ESEGUI un adattatore revisionabile che estrae via AST solo i nodi
   esplicitamente elencati, verifica SHA-256, elimina decoratori di dispatch
   opzionale e fornisce `torch`, `F`, `nn`, attivazioni e predicate necessari.
   I corpi matematici restano immutati; tracciare ogni modifica di plumbing,
   versione e licenza Apache-2.0. H forza internamente FP32: non chiamarlo
   con `.double()` e poi etichettarne il risultato come oracle FP64.
   Vietato `exec` indiscriminato dell'intero
   modulo remoto, download automatico di kernel o installazione di FLA/CUDA.
   L'estrazione non viene eseguita in SPEC. Se una funzione richiede API
   assenti nel torch locale, il test resta bloccato: niente upgrade implicito.
2. **Riferimenti algebrici FP64**, con `conv1d` o matrice Toeplitz per conv,
   e piccoli prodotti di transizioni affini per GDN; casi scalari risolti
   esattamente e proprietà di causalità/append. Il chunk H risolve un sistema
   triangolare anziché ripetere il loop scritto da C: confrontare entrambi
   contro i casi analitici. Un oracle che passa solo contro se stesso fallisce
   il criterio di indipendenza.
3. **C11 → Metal**, stessi input e stessi pesi; DwarfStar a revisione fissata
   come controllo esterno secondario quando si acquisiranno slice/tracce.
   I test DwarfStar `gdn_reference` e `kernel_qwen4_gdn_scan` condividono
   la formulazione e non costituiscono da soli due oracle indipendenti.

Per passare i tensori GGUF a H: inversione della permutazione delle head,
trasposizione dello stato, `log_decay` senza doppia exp, scala q una sola
volta, gain corretti. Il check dell'adattatore usa sentinel per testa/canale
e non dipende dal kernel C. Test H in FP32 sui valori **dequantizzati Q2**
è correttezza sul medesimo artifact; non misura la perdita rispetto a BF16.
Un futuro confronto BF16 deve usare i pesi originali, con revisione e slice
appaiate separate; la sola conversione Q2→FP32 non li ricostruisce.

### 7.2 Fixture sintetiche obbligatorie

- Conv: impulso e tap tutti distinti (rivela reversal), canali distinti,
  `K=1,2,4`, `d=1,3`, history non nulla quando presente,
  `T=0,1,max(0,Lhist-1),Lhist,Lhist+1`;
  partizioni e append, output con sentinelle e alias vietato, code C=1,31,33.
- GDN: `Hk=2,Hv=6,Dk=3,Dv=5` nell'oracle/C; dimensione reale 16/48/128/128
  in un test dedicato; `Hk=Hv` e `Hv/Hk=3`, stato iniziale non simmetrico,
  teste numerate per verificare mapping e trasposizione. Il kernel Metal
  deve dichiarare l'eventuale mancato supporto alle forme piccole.
- GDN analitico dello scan: Dk=Dv=1, `S0=2,k=1,v=4,q=0.5,decay=0.5,beta=0.25`
  dà `S1=1.75,o1=0.875`; usa ingressi già preparati. Aggiungere beta=0 e 1,
  decay=0 e 1 (log finito con exp underflow a zero, oppure log=0),
  k=0, q=0, v uguale alla predizione. Un log-decay -Inf è invece un caso
  negativo: il contratto pubblico accetta solo log-decay finito.
- Prep: Q/K zero e quasi zero, scala input da 1e-6 a 1e2; z/a/b in
  `[-100,-20,-1,0,1,20,100]`, dt nonzero, A negativa; beta vicino a 0/1;
  test che distinguano L2 da RMS e sigmoid da SiLU.
- HC: rami diversi, gamma nonunitario/negativo, rank non potenza di due;
  `down=up=inject=0` dà `mixed=mean(N)/2`, `inj=1` e `R'=R+out`;
  manca una divisione per H o il fattore 2 → fixture fallisce.
- Norm: grouped contro full-width, gain condiviso contro gain per gruppo,
  zero, piccole norme, epsilon variato e gain effettivo contro `1+w`.
- Sequenze: lunghezze 1,3,4,5,63,64,65,127,128,129,1023,1024,1025;
  stream 8192,16384,32768 a forme piccole, senza salvare ogni stato S.
  Forme reali 1024+1 e 8192 streaming sotto lock; controllare stati ai
  confini e output di **ogni** token, non solo ultimo/logits.
- Stato: reset dopo history nonzero, due sessioni alternate senza alias,
  split irregolari `[1,3,2,65,…]`, errore di posizione, shape diversa,
  overflow size/stride, buffer corto, input nonfinite e errore GPU simulato
  dal harness di gestione, senza simulare il risultato numerico del kernel.
- Causalità: cambiare un suffisso futuro lascia invariati tutti gli output
  precedenti. Poison NaN delle destinazioni e guards bitwise intorno ai
  buffer; test padding stride che non deve essere letto o scritto.

### 7.3 Slice del checkpoint e tracce di integrazione

**P, non acquisite.** Prima leggere solo piccoli controlli: layer GDN 0,
1 (con PLE) e 46, `ssm_a`, dt, norm, conv, alpha/beta, gain HC e
`blk.1.ple_conv1d.weight`. Controlli GDN completi per layer circa 1,095 MiB;
tre layer più conv/gain HC/PLE sotto 4 MiB di payload selezionato.

Poi, con B, una sola layer GDN di proiezioni Q8_0 (circa 58,4 MiB packed)
e un mixer HC F16 (circa 12,6 MiB), mediante `pread` limitati e manifest di
nomi, offset, shape, formato, revisione/hash artifact e SHA-256 delle slice.
Dequantizzare una proiezione alla volta, non l'intero layer né modello.
Finché mancano tali slice, distinguere “primitive con input sintetici” da
“layer Q2 verificato”.

In una fase successiva al lock comune e al grafo disponibile, acquisire
tracce di attivazioni reali: input HC, mixed, qkv grezzo e dopo conv, a/b/z,
log-decay/beta, S prima/dopo, output GDN, injection e residuo finale su un
breve prompt/continuazione deterministici. Il pilot esistente conserva
risposte e misure, **non** queste fixture interne. Estrarle sotto build
isolata di osservazione, senza modificare il checkout DwarfStar principale.
Il layer con input sintetico e pesi reali non sostituisce questa verifica.

## 8. Tolleranze numeriche e criteri di accettazione

**P.** Limiti iniziali da congelare al consolidamento; non misure già ottenute.
Il confronto usa `err_inf=max(abs(got-ref))` e, per ciascuna riga/head/stato
locale, `scale=max(1,max(abs(ref)))`. Accettazione `err_inf <= tol*scale`.
Riportare anche errore RMS normalizzato, peggior indice/token/head, errore
elementwise e max assoluto. Una scala globale dominata da una testa non può
nascondere errori delle altre. NaN/Inf inattesi falliscono sempre.

| Confronto su fixture finite bounded | Soglia proposta |
|---|---:|
| Indici, mapping teste, storia conv, reset, guards | Esatto bitwise (la storia è copia degli input della conv) |
| C FP32 e Metal vs oracle: conv breve, attivazioni, grouped RMS | `1e-5` |
| Mixer HC esclusi errori di proiezioni, inject/combine | `2e-5` |
| GDN prep/scan/output e stato finale, T≤1025 | `5e-5` |
| GDN 8K/16K/32K streaming, stato/output per finestra | `2e-4` |
| Baseline token/chunk stessa backend, riduzioni e dispatch invariati | Bitwise output/stato; altri kernel devono rispettare i limiti numerici sopra |
| Layer composto con proiezioni B, incluse F16/Q8_0 | Budget da concordare con B; gate separato, nessun “PASS layer” con solo test core |

Motivazione: epsilon macchina FP32 è circa 1,19e-7; riduzioni lunghe 128
e 2560, libm/Metal e FMA cambiano l'arrotondamento. DwarfStar testa core GDN
a `5e-5` e HC a `2e-5` [D0 `tests/test_qwen4_kernels.c:597–605,745–750`];
sono evidenza di plausibilità, non calibrazione del nostro codice. Per k
normalizzata e beta in [0,1], `I-beta*k*k^T` non espande la norma spettrale,
ma cancellazioni e arrotondamenti rendono comunque necessari test lunghi,
specialmente decay≈1. La soglia lunga non cresce liberamente con T.

Conv history è esatta solo confrontando lo **stesso input** della primitiva;
per una pipeline intera gli errori nelle proiezioni/norme vengono misurati
prima della conv e propagati esplicitamente. FP32 oracle H e analitico FP64
possono divergere: registrare entrambe le distanze per distinguere un limite
del riferimento da un bug C. Le soglie qui sopra valgono per test core con
pesi/input identici, non per BF16 originale vs Q2.

Un errore sopra soglia richiede riproduzione minima, revisione layout/ordine
operazioni e confronto ad alta precisione; non si cambia la tolleranza per
farlo passare. Ottimizzazioni, fast-math o mixed precision richiedono una
nuova evidenza rispetto al percorso conservativo, incluse continuazioni.
Successo C richiede oracle indipendente, stato+output, append/reset/errori,
sanitizer CPU e Metal validation verdi. Non equivale a generazione completa
Qwen3.8 né promette identità greedy fra runtime differenti.

## 9. Implementazione per incrementi verificabili

**P.** In ESEGUI partire da `BASE_CONCORDATA`, worktree nuovo, file assegnati;
spiegazione del concetto prima di ogni incremento, modifiche piccole e test
pertinenti prima di continuare. Nessuna implementazione durante SPEC.

1. **C0 — contratto/oracle.** Congelare mapping teste, gamma e log-decay;
   preparare adattatore H e casi analitici CPU. Output: manifest con revisioni,
   hash e confronti dei due algoritmi upstream; nessun peso reale richiesto.
2. **C1 — conv C11.** Tap, storia e dilatazione prima di SiLU; poi SiLU e
   stride. Confronto `conv1d`/Toeplitz, append/reset/guards, ASan/UBSan.
   Consegnare subito il contratto/test utilizzabile da E.
3. **C2 — norme e HC C11.** Gain diretto, assi per gruppo, attivazioni stabili,
   divisioni HC e combine su proiezioni fornite dalla fixture. Test H e
   analitici; proiezioni non ancora coperte dichiarate esplicitamente.
4. **C3 — GDN C11.** Prep, poi scan, poi norm+sigmoid, ciascuno con confronto
   locale e stato esportabile. Token/chunk e sequenze lunghe a forme piccole.
5. **C4 — slice Q2/composizione CPU.** Dopo A e B/F16/F32: verificare un
   GDN/mixer con pesi reali e indipendente dequant B, senza modello completo.
6. **C5 — Metal conservativo.** Conv, norme/HC, prep, scan e gated norm,
   separatamente; stessa FP32 e layout C, token sequenziali nel dispatch.
   Lock unico, risorse trattenute e validazione API; confronto C+H.
7. **C6 — integrazione.** Coordinatore collega E→HC→C/D→HC→F, verifica
   transazione dello stato e trace reali; poi regressioni Qwen3. Un C5 verde
   non autorizza dichiarazioni sulla generazione completa.
8. **C7 — ottimizzazioni successive.** Halo conv/chunk scan, riduzioni o
   fusioni dopo misure esclusive e invarianti numerici; commit separati.

## 10. Verifiche, memoria e rischi concreti

### 10.1 Comandi e precondizioni

**V.** In questa sessione: letture Git/documenti/sorgenti, statistiche
metadata/file, consultazione fonti primarie e `metal -help`; nessun build,
inferenza, test GPU, hashing completo del checkpoint o benchmark.
Controlli documentali finali:

```sh
git status --short --branch
git diff --cached --name-only
git diff --check
git diff --cached --check
```

**P.** CLI e file qui sotto sono il contratto dei test da implementare,
**non comandi già disponibili a questa base**. Dal futuro worktree C:

```sh
PF_MAIN=/Users/cristiancirje/Desktop/Dev/picoforge
PF_PY="$PF_MAIN/tools/venv/bin/python"
mkdir -p build/qwen38-c
"$PF_PY" -B tools/oracle/qwen38_recurrent.py \
  --synthetic --device cpu --out build/qwen38-c/fixtures
clang -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
  -ffp-contract=off -fsanitize=address,undefined -Isrc \
  tests/test_qwen38_recurrent.c src/qwen38_conv.c src/qwen38_gdn.c \
  src/qwen38_base.c -lm -o build/qwen38-c/test_cpu
"$PF_PY" -B tests/test_qwen38_recurrent.py --backend cpu \
  --driver build/qwen38-c/test_cpu --fixtures build/qwen38-c/fixtures
```

L'adattatore risolve H a revisione/hash espliciti in ESEGUI, conserva la
sorgente nel build locale e non la scarica da `main`. L'opzione CPU deve
impedire l'uso implicito di MPS. Il driver legge solo fixture dichiarate,
produce un manifest di risultati e non avvia il modello. Il test Metal
avrà gli stessi argomenti `--backend metal --driver ...`, con
`MTL_DEBUG_LAYER=1`, **solo attraverso il wrapper di lock del coordinatore**.

Dopo l'aggiornamento Makefile del coordinatore, sotto lo stesso lock per
GPU/carichi pesanti: `make`, target CPU/GPU C e `make test-all`, con Python
e asset principali esplicitamente parametrizzati. Il Makefile attuale usa
`./tools/venv/bin/python` e modelli relativi: passare `PYTHON=...` oggi non
li sostituisce. Non creare copie né convertire automaticamente modelli per
far girare le regressioni. Il coordinatore deve definire il comando reale
del wrapper e il percorso del lock **di sistema**, comune ai worktree;
questa sessione non inventa un lock privato né lo dichiara acquisito.

### 10.2 Budget calcolato, non misurato

Byte payload FP32, una sessione, 36 GDN, dimensioni dell'inventario:

| Voce | Formula | MiB |
|---|---|---:|
| S per layer GDN | `4 Hv Dv Dk` | 3,000 |
| Tutti gli S backbone | `36 * 4 Hv Dv Dk` | 108,000 |
| Tutte le history GDN | `36 * 4 (K-1) C` | 4,219 |
| History PLE, di E | `4 (Kple-1) d H E` | 0,352 |
| QKV chunk T=1024 | `4 T C` | 40,000 |
| z e output scan, ciascuno | `4 T Hv Dv` | 24,000 |
| a/log-decay e b/beta, insieme | `8 T Hv` | 0,375 |
| R, N, gate_logits, ciascuno | `4 T H E` | 40,000 |
| lo e inj | `4 T (Rk+H)` | 1,266 |
| mixed e block_out, ciascuno | `4 T E` | 10,000 |

Con tutti questi buffer simultaneamente vivi: circa **342,211 MiB** inclusi
stati e history PLE, esclusi pesi e altri componenti. Una conv fuori posto
richiede altri 40 MiB a T=1024; riserva C runtime proposta **≤512 MiB** per
stato+workspace, includendo halo/flag/allineamenti e senza copie per tutti i
layer. I buffer R/PLE condivisi vanno contati una sola volta nel budget globale.
Le 36 coppie stato/history sono persistenti; il workspace si riusa per layer.

Oracle/test: budget processo proposto ≤2 GiB, slice su disco ≤256 MiB per
invocazione; a forme reali elaborare T=64/128 per oracle chunk e streaming
lungo. Non conservare `[T,Hv,Dv,Dk]`: già a T=1024 sarebbe 3 GiB per layer.
Non usare i 36 layer contemporaneamente per un test di primitiva. La
dequantizzazione FP32 della sola QKV occupa 100 MiB, non 40 MiB (questi sono
le attivazioni); liberare la proiezione prima della successiva.

Il budget applicativo globale 48–50 GiB, raccomandazione Metal, singolo
buffer massimo e RAM fisica rimangono vincoli distinti [P1]. Il calcolo C
non dimostra il fit dell'engine completo; includere KV/indexer D, staging E,
scratch B/F, driver e pesi residenti. Non sommare ingenuamente RSS,
footprint e allocazioni Metal come se fossero categorie disgiunte.

### 10.3 Rischi da chiudere

1. **Layout plausibile ma sbagliato:** `h % Hk` contro repeat-interleave,
   trasposta S, QKV stride o tap invertiti producono numeri finiti errati;
   sentinel e stato non simmetrico sono obbligatori.
2. **Trasformazioni doppie:** `+1` su gain, `-exp` su ssm_a già trasformata,
   exp di decay anziché log-decay, scale q/HC duplicate; fixture dedicate.
3. **Stato stale:** reset parziale, storia attivata, append fuori posizione,
   PLE I/O fallito dopo avanzamento C, resource reuse prima di completion;
   flag validità di sessione e reset/replay, nessun riuso dei risultati parziali.
4. **Oracle non disponibile:** H non è nel venv; adattatore selettivo ancora
   da verificare. Qwen3-Next e DwarfStar non autorizzano ad ignorare questo gate.
5. **FP32 e parallelismo:** FMA, rsqrt e ordine di riduzione possono modificare
   continuazioni. Criteri di §8 e precisione conservativa precedono benchmark.
6. **Dipendenze di proiezione:** Q8_0 ordinario, GEMV F16/F32 e binding A
   bloccano il layer completo pur lasciando eseguibili gli incrementi core.
7. **Memoria:** workspace moltiplicato per layer, oracle che salva ogni S o
   dequant intero layer possono violare il budget pur con uno stato piccolo.
8. **Concorrenza tra sessioni:** il lock comune non è ancora concordato;
   fino ad allora niente test GPU o carichi pesanti sul checkpoint.

Consegna SPEC: questo documento soltanto; commit isolato senza push o merge.
La base comune, i prototypes definitivi, l'ownership delle proiezioni semplici,
il budget numerico del layer composto e il protocollo GPU restano decisioni
esplicite della fase CONSOLIDA.
