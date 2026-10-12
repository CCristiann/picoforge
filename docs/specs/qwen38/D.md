# D — Qwen3.8 QSA, indexer e attention tiled

Stato: **SPEC**, 2026-10-12. Nessun operatore implementato o test numerico
eseguito da questa sessione. `BASE_CONCORDATA`: da fissare in consolidamento.
Base ispezionata: `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
Branch: `spec/qwen38-d`; worktree esterno: `/Users/cristiancirje/Desktop/Dev/picoforge-spec-d`.
Le parole **verificato**, **proposta** e **aperto** distinguono evidenza,
contratto da consolidare e lavoro necessario. I numeri di memoria sono calcoli,
non misure. Le tolleranze sotto sono soglie iniziali proposte, non risultati.

## 1. Obiettivo, confini e fonti

Implementare l'attention del backbone Qwen3.8-Flash-Next Q2: preparazione
Q/K e gate, posizioni, indexer compresso, selezione causale, cache e GQA
con softmax online tiled. Una sessione testuale, append contiguo, inizialmente
8.192 token totali e chunk esterni fino a 1.024. Verificare anche 16K/32K
su operatori isolati prima di estendere il grafo completo.

D riceve le proiezioni lineari e restituisce le teste concatenate dopo il
gate; il grafo applica la proiezione di uscita. Non possiede tokenizer,
parsing/configurazione, dequantizzazione/GEMM, Gated DeltaNet, convoluzione,
hyper-connection, MoE o lookup n-gram. MTP, rollback speculativo, batching di
sessioni, padding/packing di sequenze, multimodalità, YaRN, KV quantizzata e
sliding window sono esclusi dalla prima versione: una richiesta incompatibile
deve fallire esplicitamente. Il supporto Qwen3 resta separato e invariato.

### Provenienza riproducibile

- **P**: picoforge alla base sopra. Letti integralmente
  `docs/qwen38-q2-session-prompt.md`, `docs/qwen38-q2-baseline.md`,
  `docs/qwen38-storage.md`, `README.md`, `Makefile`, `docs/DESIGN.md`,
  `docs/BENCHMARKS.md`; lette le sezioni pertinenti di
  `docs/large-model-research-2026-10-11.md` (112–174, 244–291),
  `docs/project-review-2026-10-11.md`, `docs/phase4-expert-experiment.md` e
  `docs/80b-100b-model-comparison-2026-10-11.md`.
- Letto anche `CLAUDE.md` ignorato nella repository principale; SHA-256
  `3e49a726cf993e65af1a55c2d0079f7375ba2f5065d12b848a1634e6970aa844`.
  La modalità SPEC e la direzione Qwen3.8 dell'incarico prevalgono sulle
  vecchie roadmap. Stato iniziale e staging di P puliti; nessun ripristino.
- **G**: inventario locale `bench/qwen38_q2_20261011_phase0/inventory.json`,
  SHA-256 `605d00ad17ae2fa66b10cf64d0dd0c3fac16122bb526dee10db064a0b8a67093`.
  Checkpoint `antirez/qwen3.8-flash-next-gguf` revisione
  `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`, file
  `Qwen3.8-Flash-Next-Q2.gguf`, 147.207.127.040 byte (stat locale confermato).
  SHA-256 registrato dalla verifica precedente:
  `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
  Non riletto/hashato il payload in SPEC. L'inventario da solo non lo autentica.
- **DS**: checkout pulito `/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline`,
  commit `fc80bd695da76ee14bcb86820bf900a5cdfda806`; il `main` remoto via API
  GitHub al controllo del 2026-10-12 coincide. Fonti:
  [kernel Qwen](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/metal/qwen4.metal)
  1244–1532 (prep/pooling), 1534–1710 (score), 1712–2088
  (select/expand), 2090–2460 (attention);
  [grafo](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c)
  `qwen4_graph_attention_core`, `qwen4_graph_attention_tail`,
  `qwen4_rope_configure`, allocazioni intorno a 58660;
  `ds4_metal.m:49128` e `tests/test_qwen4_kernels.c:773–985`.
  Consultati anche `docs/QWEN38_FLASH_NEXT.md`,
  `gguf-tools/qwen4_pack.py` e `gguf-tools/qwen4_pack_to_qwen4exp.py`.
- **HF**: configurazione originale fissata a
  [de4b8e4d43b917e7706784d8bb445c9af86a3540](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json).
  Codice ufficiale Transformers letto online, senza installarlo:
  [modeling_qwen4_exp.py](https://github.com/huggingface/transformers/blob/db2304e8694db3ecd6f39d353beba361688afe97/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
  commit `db2304e8694db3ecd6f39d353beba361688afe97`, SHA-256 del file
  `f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa`;
  righe 76–163 (RoPE/norm), 626–770 (indexer), 773–892 (GQA/gate).
- **Q**: [report architetturale v1, §2.1 QSA, eq. 12–19](https://arxiv.org/html/2608.30320v1).
  Conferma pooling prima di RoPE e selezione dei blocchi completi con tail.
  L'implementazione e i metadati fissati sopra risolvono i dettagli operativi.

La baseline DwarfStar documenta generazione funzionante, non parità degli
intermedi nativi. La revisione HF del codice è distinta da quella dei pesi;
non si presume che il pacchetto Transformers locale la contenga.

## 2. Stato corrente e codice riutilizzabile

**Verificato in P**:

| File/simbolo | Riutilizzo e limite |
|---|---|
| `src/gguf.h`, `gguf.c`, `gguf_plan.c`, `gpu_weights.m` | Descrittori fisici, formato distinto, binding con owner, viste multiple; nessuna semantica QSA implementata |
| `src/ops.c:rmsnorm`, `rope_apply` | Riferimenti C leggibili; RMSNorm attuale riceve pesi BF16, serve il contratto gamma F32 di C; `rope_apply` ruota l'intera testa passata, quindi non applicarlo a 256/128 quando R=64 |
| `src/kernels.metal:qk_norm_rope`, `attention` | GQA causale Qwen3, cache FP32; array score threadgroup proporzionale al contesto, non adatto come nuovo percorso 8K/32K |
| `src/gpu_forward.m` | Ordine dispatch e gestione Metal; grafo safetensors Qwen3, non riutilizzabile cambiando solo nomi o dimensioni |
| `src/metal.m` | Device/queue/library e cache di pipeline; integrazione nuova libreria/kernel da coordinare |
| `tests/test_attention.m` | Oracle scalare FP64, output inizialmente NaN, future righe contaminate; copre GQA Qwen3 fino a 4096, non QSA |
| `tools/oracle/qwen3_forward.py` | Convenzioni dei test, non oracle semantico Qwen3.8 |

G contiene 49 blocchi, di cui 48 backbone e 1 MTP. I 12 layer QSA ordinari
sono 3, 7, …, 47 (indici zero-based), da ricavare dalla configurazione di A.
Il blocco 48 non entra nel conteggio cache di questa spec.

## 3. Semantica matematica e layout

### 3.1 Parametri e binding

Nomi simbolici: E=hidden size, H=query heads, J=KV heads, D=head dimension,
Hi=indexer query heads, Di=indexer dimension, r=compression ratio,
K=token budget, R=rotary dimensions, L=capacità, T=righe nell'append.
Valori verificati in G/HF: E=2560, H=24, J=2, D=256, Hi=4, Di=128,
indexer KV heads=1, r=4, K=2048, R=64, theta=10^7, eps≈10^-6.
`qwen4exp.attention.indexer.top_k` è un budget di **token**, non 2048 blocchi.
Questi valori descrivono l'artifact, non costanti da mettere nei kernel.

Prefisso seguente `blk.<l>.`; tutte le forme fisiche sono GGUF, dimensione 0
contigua. Una matrice logica è W[out,in], usata come X W^T.

| Suffisso | Forma fisica verificata | Forma logica | Formato G |
|---|---|---|---|
| `attn_q.weight` | [2560,12288] | [2HD,E] | Q8_0 |
| `attn_k.weight`, `attn_v.weight` | [2560,512] | [JD,E] | Q8_0 |
| `attn_output.weight` | [6144,2560] | [E,HD] | Q8_0 |
| `indexer.q_proj.weight` | [2560,512] | [Hi Di,E] | Q8_0 |
| `indexer.k_proj.weight` | [2560,128] | [Di,E] | Q8_0 |
| `attn_q_norm.weight`, `attn_k_norm.weight` | [256] | [D] | F32 |
| `indexer.q_norm.weight`, `indexer.k_norm.weight` | [128] | [Di] | F32 |

Verificata la famiglia Q8_0 per tutte queste proiezioni nell'inventario, non
solo nel layer 3. Servono gli operatori Q8_0 di B anche se il file si chiama
Q2. Nessun padding 640/768 riguarda queste matrici: è il down degli esperti
di B/F. Non usare q8_row in sostituzione di Q8_0.

### 3.2 Proiezioni, norm, RoPE e gate

X[T,E] è l'ingresso del sottoblocco prodotto da C/grafo. Si proiettano
QG[T,H,2,D], Kraw[T,J,D], V[T,J,D], IQraw[T,Hi,Di], IKraw[T,Di].
La riga QG contiene **q_h seguito da gate_h per ogni testa**, non tutte le
query seguite da tutti i gate. HF `q_proj(...).view(...,H,2D).chunk(2,-1)`
e DS `qwen4_attn_prep_slot` concordano. Nessun bias e nessun dropout in inferenza.

Definire `N_gamma(x)=gamma*x/sqrt(mean(x*x)+eps)`, media sull'ultima dimensione.
Q=N_gammaq(Qraw), K=N_gammak(Kraw), IQ=N_gammaiq(IQraw), separatamente per testa.
Gamma è condiviso tra teste, non una norma sul vettore HD. V e gate non
subiscono norm o RoPE. Il peso HF è un offset: gamma=1+weight. Il runtime DS
usa direttamente gamma del GGUF. La vecchia conversione folda `+1` prima
del salvataggio (`qwen4_pack.py:1492`), poi converte BF16→F32 senza rifoldare
(`qwen4_pack_to_qwen4exp.py:165`). A deve consegnare gamma effettivo e indicare
la convenzione: **non aggiungere un altro 1 al peso GGUF**. Per un oracle HF
alimentato dal GGUF usare offset=gamma-1 e controllare il round-trip.

Per testo, posizione assoluta p=numero di token precedenti, dal primo token
del template, compresi i token speciali. Per 0≤i<R/2:

```
f_i = theta^(-2 i/R); a = p f_i
(x'_i, x'_(i+R/2)) = (x_i cos(a)-x_(i+R/2) sin(a),
                      x_(i+R/2) cos(a)+x_i sin(a))
x'_d = x_d   per d >= R
```

Si applica a Q, K, IQ e alle chiavi aggregate dell'indexer, con lo stesso
R=64 anche per Di=128. Non R=32 derivato da 0,25*Di. Il profilo testuale usa
le tre coordinate MRoPE uguali a p; le sezioni HF [11,11,10] e l'interleave
DS coincidono in questo caso. Proposta iniziale: solo RoPE default, fattore
di ampiezza 1; rifiutare configurazioni YaRN o coordinate multimodali invece
di ignorarle. Posizioni di cache e rotary non si azzerano al cambio chunk.

### 3.3 Indexer e insieme selezionato

Per ciascun layer, conservare IKraw non normalizzate/non ruotate. Il blocco
b contiene token br,…,br+r-1 ed è completo solo se br+r-1≤p.

```
u_b = (1/r) sum_(j=0..r-1) IKraw_(br+j)
c_b = RoPE(N_gammaik(u_b), br)
n_p = floor((p+1)/r)
s_pb = sum_(h=0..Hi-1) max(dot(IQ_ph, c_b), 0),  0 <= b < n_p
B = K/r                         (contratto iniziale: K % r == 0)
S_p = union_(b in top_B(s_p)) {br,...,br+r-1}
      union {r*n_p,...,p}
```

Se n_p≤B, selezionare tutti i blocchi visibili: S_p={0,…,p}. È un fast path
esattamente equivalente, non una sostituzione di QSA con attention densa.
Il primo token sparso è p=(B+1)r-1, qui **2051**, non 2048. Il tail ha
0,…,r-1 token; il massimo |S_p| è K+r-1=2051. Un blocco appena completato
partecipa al top-k: il token corrente non è garantito quando il tail è vuoto.
Nessun sink, finestra locale aggiuntiva, bonus di recenza o riuso dell'indexer
tra layer è presente nel contratto ordinario.

**Differenze verificate fra fonti, da non nascondere**:

- HF divide gli score per sqrt(Di), DS e Q eq.15 no. Il fattore positivo
  costante non cambia il ranking in aritmetica esatta; il confronto di score
  deve riportarli alla stessa scala e misurare separatamente arrotondamento
  e near-tie. D propone lo score non scalato di DS e una sola convenzione
  anche nel prefill. Non aggiungere pesi appresi per testa: non sono presenti.
- Q esprime ceil(K/r) con troncamento a K, HF/DS usano floor. Nel checkpoint
  K è divisibile per r. Per altri rapporti la prima versione risponde
  `UNSUPPORTED`; generalizzazione da decidere con A, non floor implicito.
- HF `torch.topk` non prescrive lo stesso tie-break di DS. DS sceglie gli
  indici di blocco minori tra score uguali. **Proposta**: ordine totale
  (score decrescente, indice crescente), confrontando esattamente gli FP32
  e trattando ±0 come pari. Dopo la scelta, serializzare blocchi in ordine
  temporale crescente; le riduzioni possono differire dal gather DS, che
  emette prima gli score sopra soglia e poi quelli uguali. La semantica è
  l'insieme. I pareggi al confine HF restano una differenza esplicita, §8.
- HF forma blocchi sulla lista dei token visibili della maschera. Blocchi
  assoluti br sono equivalenti solo per il prefisso testuale contiguo senza
  padding adottato qui. Maschere arbitrarie richiedono un altro contratto.

La selezione vede soltanto b<n_p, anche se il prefill ha già preparato
chiavi di blocchi successivi. Non trasformare l'invalidità in score zero e
affidarsi casualmente al tie-break. NaN/Inf visibili sono errori numerici;
slot futuri possono contenere poison e non vanno letti. Selezioni senza
duplicati, tail disgiunto dai blocchi, nessun token >p o fuori cache.

### 3.4 GQA e softmax online tiled

Per query h, testa KV j=floor(h/(H/J)); H deve essere divisibile per J.
Per u∈S_p: z_u=dot(Q_ph,K_uj)/sqrt(D). Attenzione e gate:

```
a_ph = sum_(u in S_p) exp(z_u-m) V_uj / sum_(u in S_p) exp(z_u-m)
y_phd = a_phd * sigmoid(gate_phd)
output_p = concat_h(y_ph) W_o^T       # eseguito dal grafo con B
```

La normalizzazione è unica su tutto S_p, non una media dei softmax per
blocco. Gli score dell'indexer non sono probabilità dell'attention.
Per un tile visibile U mantenere (m,l,a), inizialmente (-inf,0,0):

```
m' = max(m, max_(u in U) z_u)
l' = exp(m-m') l + sum_(u in U) exp(z_u-m')
a' = exp(m-m') a + sum_(u in U) exp(z_u-m') V_u
```

Saltare tile vuoti, senza calcolare -inf-(-inf). Per split multipli conservare
parziali (m_s,l_s,a_s), combinarli allo stesso modo con massimo comune e
solo alla fine dividere per l e applicare il gate una volta. Una query valida
ha insieme non vuoto; l=0 al termine è errore, non output silenziosamente zero.

**Proposta numerica iniziale**: FP32 per attivazioni, pooling, gamma,
norm, trigonometria, dot product, accumuli softmax e tutte le cache. Oracle
matematico FP64 distinto. FMA/ordine di riduzione dichiarati nei manifest.
Niente fast-math per stabilire la prima baseline. In particolare, niente
cast FP16 delle IQ nel solo prefill: DS `ds4_metal.m:49128` sceglie scorer
matrix per T>2 che arrotonda IQ a half (`qwen4.metal:1682` circa), mentre
decode mantiene IQ FP32. Il test upstream ammette errore 4e-3 sui relativi
score; questo non garantisce identità della selezione al confine top-k.

Profilo FP16 cache (K,V,c_b), con IKraw FP32, è un incremento successivo
esplicito per confronto DS e risparmio memoria, da accettare separatamente.
HF normalmente usa BF16: FP16 non equivale a BF16. Le conversioni di cache
non vanno mascherate aumentando la tolleranza dell'operatore FP32.

### 3.5 Cache, append, reset e chunk

Cache per layer/sessione: K[L,J,D] dopo norm/RoPE, V[L,J,D],
IKraw[L,Di], C[floor(L/r),Di] dopo pooling/norm/RoPE; FP32 iniziale.
Ordine token-major, dimensione finale contigua. Nessuna cache per query,
gate o selezione tra chiamate. Nessuna eviction: QSA può riselezionare un
token vecchio in un append futuro.

Stato valido: `length=P`, `complete_blocks=floor(P/r)`, generazione/epoch,
capacità immutabile, identità configurazione/layer/profilo numerico. Tutte
le righe <P sono valide. La coda IKraw del blocco incompleto sopravvive.
Append T prepara gli slot [P,P+T), poi crea esattamente i blocchi
`b in [floor(P/r), floor((P+T)/r))`. La chiave del blocco usa posizione br,
mai quella del suo ultimo token o dell'inizio chunk. Ogni query del chunk
usa n_p proprio. La cache può contenere il futuro; la visibilità non può.

Reset cambia epoch e length=complete_blocks=0, invalida selezioni/workspace
e impedisce letture stale; non occorre azzerare fisicamente tutti i byte.
Riutilizzo di prefisso ammesso solo mantenendo stato e identità invariati e
aggiungendo token: cambiare un token precedente richiede reset/replay.
Rewind, truncate e fork di stato non vengono esposti come se bastasse
accorciare K/V (C/E possiedono altro stato necessario).

## 4. API proposta: forme, stride, durata ed errori

Nomi e header da ratificare in `integration.md`; questo è un contratto, non
codice già disponibile. API C11, nessun oggetto Objective-C negli header.

```
Q38Status = OK | INVALID | RANGE | OVERFLOW | UNSUPPORTED |
            NOMEM | BUSY | STALE | NUMERIC | GPU_ERROR
Q38AttentionConfig:
    uint32_t heads, kv_heads, head_dim, index_heads, index_kv_heads,
             index_dim, ratio, token_budget, rotary_dim;
    float rms_eps, rope_theta;
    enum cache_dtype { F32, F16 };       // F16 rifiutato nel primo incremento
Q38F32View:
    float *data; size_t bytes;
    uint32_t rank, shape[4]; size_t stride_bytes[4];
Q38AttentionInput:
    const-view qg[T,H,2,D], k[T,J,D], v[T,J,D],
               iq[T,Hi,Di], ik[T,Di];
    const-view gamma_q[D], gamma_k[D], gamma_iq[Di], gamma_ik[Di];
    uint32_t pos0, tokens; uint64_t epoch;
Q38AttentionState, Q38AttentionWorkspace: opachi, non copiabili per valore
Q38AttentionTicket: opaco, lega stato/epoch/append in corso
```

`const-view` usa puntatore `const float *` nella firma concreta; nessun cast
che autorizzi scritture sugli ingressi. Tutti gli stride sono in **byte**,
anche sul lato Metal; gli indici sono uint32, prodotti/offset controllati in
size_t/uint64 prima di qualsiasi narrowing. Densità interna richiesta per
dimensioni testa/feature; ammesso pitch tra token maggiorato, multiplo di 4.
Niente stride negativo, zero broadcast, alias input-output o output-cache.
Controllare ultimo indirizzo raggiungibile contro bytes, non solo il prodotto
delle dimensioni. Le cache interne sono contigue; scratch plan allinea a 16 byte.
In C11 il futuro profilo F16 usa storage `uint16_t` IEEE binary16 e
conversioni RNE verificate, non presume disponibilità di un tipo C `_Float16`.

| Funzione proposta | Contratto |
|---|---|
| `q38_attention_plan(cfg,L,Tmax,&plan)` | Pura, senza allocazioni/pesi; byte cache/scratch, allineamenti, tile e limiti, overflow controllati |
| `q38_attention_create(cfg,plan,&state,&workspace)` | Copia config, possiede allocazioni proprie; fallimento libera parziali, output null |
| `q38_attention_begin(state,pos0,T,epoch,&ticket)` | T>0, pos0==length, T≤Tmax, P+T≤L, un append in flight; nessuna mutazione del prefisso pubblicato |
| `q38_attention_cpu(ticket,input,out[T,H,D],workspace)` | Prepara cache, blocchi e output nel profilo C; ritorna stato, non termina il processo |
| `q38_attention_encode(ticket,encoder,gpu_input,gpu_out,workspace)` | Glue Metal, stesso contratto; registra risorse trattenute e non pubblica length |
| `q38_attention_finish(ticket,success)` | Dopo lavoro CPU/completamento GPU e commit del grafo: pubblica length=P+T; su errore rende la sessione non riutilizzabile fino a reset |
| `q38_attention_reset(state)` / `destroy` | Solo senza operazioni in flight; reset nuovo epoch, destroy libera proprietà; ingresso BUSY non libera risorse ancora usate |

Il coordinatore può semplificare il ticket per un primo grafo sincrono, ma
deve preservare questi punti di commit. Controlli di argomenti falliti prima
del begin non invalidano il prefisso. Un errore dopo scritture non consente
di proseguire su uno stato parziale. Se un layer C/E/F fallisce, il grafo
invalida la sessione intera; D non può dichiararla recuperata da solo.

Versione GPU delle view: buffer handle opaco preso in prestito, offset,
bytes, shape/stride, dtype; no `float *` CPU interpretato come indirizzo GPU.
Il contratto comune del coordinatore deve esplicitare encoder, fence/completion
e status. Stato, scratch, input, output e gamma vivono fino al completamento;
nessuna sovrapposizione tra append della stessa sessione. D non assume che
il ritorno di encode equivalga a GPU completata. Usare risorse retained;
command buffer unretained non supportati inizialmente.

Binding di pesi: A/B/grafo conservano il `GpuWeightStore` e i
`GgufBinding` con owner; D non memorizza offset assoluti in un ipotetico
singolo buffer. Proiezioni eseguite da B, le piccole gamma F32 possono essere
passate come view di binding oppure copia limitata posseduta dal modello.
Il piano non viene copiato; lifetime del modello ≥ sessione ≥ dispatch.

Validazioni: dimensioni >0, H%J==0, index_kv_heads==1, R pari e R≤min(D,Di),
K≥r, K%r==0, eps/theta finite e positive, layer QSA e tutte le forme esatte,
dtype/profilo supportato, nessun bias ignorato, capacità entro limite della
configurazione. Cache dtype non modificabile a metà sessione. Errori includono
layer, operazione e dimensione/offset falliti, senza contenuti di prompt/pesi.

Per test, esporre attraverso il solo harness dump di IQ, IKraw, C, score,
blocchi/token selezionati, conteggi e cache valide. Non trasferire tutti gli
score GPU alla CPU nel percorso ordinario per fare il top-k.

## 5. Dipendenze e decisioni per il consolidamento

| Responsabile | Contratto richiesto da D | Avanzamento indipendente possibile |
|---|---|---|
| A | Config senza costanti DS, layer types/ratio, 48+1, gamma effettiva, nomi/forme/provenienza, RoPE testuale | Fixture già proiettate con config sintetica esplicita |
| B | Decode/matmul Q8_0 per sei proiezioni; F32 per gamma; row/stride e accumulo FP32 | Core attention/indexer senza pesi compressi |
| C | RMSNorm con gamma effettiva, eps e accumulo FP32; ingresso X[T,E], residuo esterno dopo W_o | Oracle indipendente e test di score/softmax già normalizzati |
| E | Nessuna lettura n-gram dentro D; propagazione errori/epoch coerente nel grafo | Intero core D isolato |
| F | Nessuna selezione esperti dentro D; restituzione X/attn output al grafo | Intero core D isolato |
| Coordinatore | Header/status/view comuni, owner file, lifetime GPU, commit sessione, build, lock di sistema, oracle HF riproducibile | Contratti/documento e test CPU sintetici futuri |

RMSNorm generica appartiene a C: D può integrare norm/RoPE in un kernel
fuso solo dopo averlo verificato contro la primitiva di C, senza creare un
secondo contratto di normalizzazione. Le proiezioni restano B; D specifica
solo split/reshape, gate e ordine operatori. Il gate sigmoid dell'attention
non è il residuo gated a quattro rami di C.

**Aperti da risolvere prima della relativa integrazione**:

1. Ratifica tie-break, score non scalato, FP32 cache iniziale e ordine
   deterministico delle selezioni. Ties ufficiali non autorizzano scelte
   dipendenti dalla dimensione del chunk.
2. API/ownership sopra e loro raccordo con errori attualmente fatal dei
   loader esistenti; nessuna modifica unilaterale a `picoforge.h`/`gguf.h`.
3. Modalità riproducibile di esecuzione del riferimento HF (§7): manca il
   modulo installato, non le primitive torch necessarie. Nessun upgrade
   implicito delle dipendenze.
4. Percorso e wrapper del lock comune: ancora non assegnati, niente lock
   privato per worktree. Ogni test GPU, anche sintetico, attende questo punto.
5. Gamma/fusione originale→GGUF validata da A su slice, e Q8_0 di B prima
   della parità delle proiezioni reali. Non blocca il core già proiettato.

## 6. File posseduti e modifiche condivise richieste

**In questa sessione unico file scritto/committato**:
`docs/specs/qwen38/D.md`.

Proposta di ownership esclusiva D per ESEGUI, subordinata a integration.md:

- `src/qwen38_attention.c`: prep C, cache, indexer/select, online attention.
- `src/qwen38_attention_gpu.m`: sole risorse/encode/completion Metal.
- `src/qwen38_attention.metal`: kernel prep, pooling, score/select, attention.
- `tools/oracle/qwen38_attention.py`: adapter delle fonti indipendenti,
  fixture, manifest e confronto; nessun modulo engine Python.
- `tests/test_qwen38_attention.c`, `tests/test_qwen38_attention.m`,
  `tests/test_qwen38_attention.py`: harness/test CPU, Metal e oracle.
- `tests/fixtures/qwen38/attention/manifest.json`: ricetta/hash dei piccoli
  casi deterministici; array generati in `build/qwen38-attention/` ignorato.

Il coordinatore possiede e materializza gli header condivisi (proposta
`src/qwen38_attention.h` e il comune header di config/view/status), le regole
`Makefile`, eventuale inclusione in `src/kernels.metal` oppure link della
nuova AIR, integrazione in `src/metal.m`, entry point/grafo, `README.md` e
report aggregati. D consegna proposte puntuali, non li modifica. Preferire
kernel nominati `q38_*` nello stesso metallib con build esplicita del
coordinatore; preservare kernel e test Qwen3. Tutti i file sopra sono nomi
proposti, non file già esistenti alla base.

## 7. Oracle indipendente e fixture

Tre riferimenti complementari; la traduzione C→Python o DS→C non basta.

1. **Semantica ufficiale**: eseguire i metodi HF fissati al commit sopra per
   norm, rotary, indexer e attenzione eager, in `eval`, con config minuscola
   valida e pesi sintetici. Confrontare prima un singolo forward senza cache
   con maschera causale esplicita; questo costruisce la selezione da tutti
   i token visibili e non riusa il nostro algoritmo incrementale. Per append
   ricalcolare il prefisso con l'oracle: evita di condividere gli stessi bug
   di cache. Nessuna istanziazione del modello completo.
2. **Algebra e primitive indipendenti**: NumPy FP64 per matrici di score
   esplicite, sort lessicografico e maschera densa ricavata dall'oracle HF;
   `torch.nn.functional.scaled_dot_product_attention` su CPU FP64 con GQA
   espanso solo nella fixture, maschera bool esplicita e `is_causal=False`,
   dropout=0, scala dichiarata. Confrontare con softmax denso, e usare casi
   analitici (score uguali→media di V, unica chiave→V) per controllare
   il riferimento stesso. Questo è indipendente dal tiled online di D.
3. **Baseline runtime**: kernel DS fissati, input identici già proiettati,
   dump intermedi e cache nel loro profilo F16; test differenziale ulteriore,
   non unico oracle. Confrontare insiemi e gate oltre all'output. Non importare
   automaticamente la tolleranza 4e-3 o la dipendenza di precisione da T.

Ambiente esistente verificato via metadati: Python di
`/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python`, NumPy 2.5.1,
torch 2.13.0, Transformers 5.13.0, **senza** `models/qwen4_exp`.
Proposta senza nuove dipendenze: adapter di test che carica solo definizioni
HF revisionate necessarie (AST allowlist/metodi, torch/nn/math e config
semplice), conserva hash/licenza e non modifica i corpi degli operatori.
Escludere decorator/framework non necessari solo con patch documentata;
verificare l'adapter contro import ufficiale quando disponibile. Niente
download/esecuzione automatica di codice remoto non fissato. Se non è
possibile eseguire fedelmente il modulo col torch presente, dichiarare
l'oracle ufficiale bloccato e chiedere al coordinatore la soluzione; non
etichettare come parità HF il solo NumPy scritto da D.

### Matrice minima delle fixture

- Forma minuscola H=4,J=2,D=8,Hi=2,Di=8,R=4,r=4,K=8,E=16;
  forma reale H=24,J=2,D=256,Hi=4,Di=128,R=64 senza modello residente.
- p=0,1,2,3,4,7; confini K-1,K,K+r-2,K+r-1; sul profilo reale
  p=2047,2048,2049,2050,2051,2052. Tail zero e massimo; blocco corrente
  deliberatamente escluso; cambio forte di ranking completando un blocco.
- Score tutti zero (ReLU), ties al k-esimo, near-ties, negativi prima di
  ReLU, grande dinamica, gamma non uniforme, gate 0 e ±80; sigmoid(0)=0,5.
- Test che separano pooling→norm→RoPE dall'ordine sbagliato: chiavi
  non collineari, posizioni br≠ultimo token, resto non rotatorio invariato.
- Prefill monolitico contro token-per-token e partizioni [1,3,2,5,…],
  chunk 1024 e non multipli di r (1023/1025 suddivisi rispettando Tmax).
  Append dopo prefissi non allineati, reset e due sessioni alternate.
- Token futuri/cache non valida riempiti con NaN o valori enormi: invarianti
  fino a p non cambiano. Poison dell'output per scoprire scritture mancanti;
  canary nei pitch e dopo ogni buffer. Risorsa/epoch straniero rifiutati.
- Split softmax 1/2/7 e tile piccoli dispari; lunghezze intorno ai tile,
  4095/4096/4097, 8191/8192, poi 16384/32768 in test operatori limitati.
  Non materializzare [T,H,L] per tutte queste lunghezze insieme.
- Forme/stride invalidi, overflow offset, cap superata, T=0, OOM simulata,
  errore command buffer e invalidazione; nessun riuso dopo errore/reset.

### Slice reali necessarie (ESEGUI, letture limitate)

Leggere con `pread` dal checkpoint esistente e read-only, senza copia GGUF:
4 norm del layer 3 (3.072 byte) e righe/blocchi Q8_0 di ciascuna proiezione
per test B→D. Per un attention block completo il totale dei sei tensori
Q8_0 è 54.661.120 byte più norm: cap di lettura **64 MiB**, dequantizzazione
per tensore/tile con cap temporaneo **256 MiB**, una sola layer alla volta.
Le righe q includono q/gate corrispondenti, non segmenti disaccoppiati.
Ripetere slice piccole ai layer 27 e 47 per binding/offset; niente MTP.

Fixture di integrazione reale richiesta al coordinatore, non disponibile
oggi: X in ingresso a un layer QSA, relativo prefisso di K/V e IKraw oppure
proiezioni per 2053 token, dump da esecuzione controllata del checkpoint Q2.
Cap fixture 256 MiB, più lunghezze corte e una continuazione a 8192; archivio
con layer, posizioni, dtype, revisioni, seed/token IDs, hash dei payload e
range file letti. Non rigenerare queste attivazioni caricando tutto il modello
in SPEC. Input sintetici su pesi reali verificano proiezioni/composizione,
ma non sono attivazioni reali di una richiesta.

## 8. Tolleranze e criteri di accettazione

Tutti i confronti numerici usano `abs(got-ref) <= atol + rtol*abs(ref)`,
controllo di finitezza e massimo errore (oltre a RMS/percentili diagnostici).
Un errore di selezione non viene diluito nella media degli output.

| Confronto | Soglia iniziale proposta | Motivazione/limite |
|---|---|---|
| Indici, insiemi, conteggi, posizioni, maschere, epoch | Esatti | Sono discreti; tolleranza zero |
| Norm/RoPE/pooling C FP32 vs oracle | atol=2e-5, rtol=2e-5 | Riduzioni corte, trig e rsqrt FP32, fixture di scala limitata |
| Score indexer non scalato FP32 vs FP64 | atol=2e-4, rtol=2e-5 | 4 dot di 128 termini; vicino a zero domina cancellazione |
| Attention/gate FP32 vs denso FP64 | atol=3e-5, rtol=2e-4 | Online/split cambiano riduzioni; V sintetico in [-1,1] |
| Metal vs C, stessi ingressi/cache FP32 | Stesse soglie per stadio | Nessuna concessione automatica a half o fast-math |
| Cache F16 futura vs oracle quantizzato agli stessi confini | Stesse soglie di calcolo; conversione F16 RNE esatta | L'errore della conversione si misura separatamente contro FP32 |

Le fixture di stress fuori scala richiedono bound analitico specificato
prima del test, non queste soglie applicate ciecamente. Per score vicino al
confine registrare gap `s_B-s_(B+1)` ed errore massimo E_s: gap>2E_s è
condizione sufficiente per stabilità del set rispetto all'oracle FP64.
Se gap≤2E_s, conservare il caso, confrontare la regola su **identici score
FP32** e quantificare divergenza; non dichiarare parità del set con FP64.
In tali casi non si accetta silenziosamente il componente completo: la
decisione numerica deve essere consolidata con una riproduzione minima.

Ties esatti hanno set deterministico nativo/DS; per HF senza tie-break
specificato richiedere che tutti gli elementi sopra soglia siano inclusi
e che gli scambi riguardino solo gli uguali al confine. Confrontare poi
attention sulla **stessa maschera** per isolare il calcolo; riportare il
confronto end-to-end HF ambiguo, non passato con tolleranza allargata.

Accettazione dell'incremento D:

1. Oracle ufficiale eseguito e distinto da C; test analitici indipendenti
   passati, versioni e adattamenti registrati.
2. C passa tutti i confini e ASan/UBSan, nessuna allocazione crescente per
   token dopo create. Nessun accesso futuro o stato stale.
3. Metal passa gli stessi casi e la validazione API; indici identici C/Metal
   su fixture con margine e sui tie costruiti. Su input già proiettati,
   partizioni diverse producono gli stessi indici/cache e output entro soglia.
4. Con B, differenze delle proiezioni sono misurate prima di score/selection;
   parità chunk del sottoblocco completo resta un test separato. Non promettere
   bit-identità di GEMM/GEMV senza prova.
5. Test Qwen3 invariati passati dal coordinatore. Grafo ordinario e logits
   confrontati in integrazione: superare D non dimostra generazione completa.

Soglie non ancora calibrate su questo componente: risultati effettivi
andranno allegati agli incrementi. Correttezza su Q2 dequantizzato non misura
perdita rispetto al BF16 originale. Né il profilo di cache HF BF16 né il
runtime DS FP16 costituiscono da soli una baseline di qualità dei pesi.

## 9. Implementazione per incrementi verificabili

1. Consolidare tipi/ownership/config e fonti. Costruire adapter oracle e
   casi analitici minuscoli, senza pesi/GPU; congelare manifest e tie policy.
2. C prep (split QG, norm tramite C, RoPE parziale), poi pooling dei blocchi
   e stato append/reset. Dump intermedi e sanitizers prima dell'attention.
3. C scorer FP32 e selettore deterministico semplice (sort totale o heap
   bounded); espansione tail. Test di maschera, ties e confine 2051.
4. C attention online con tile e merge, oracle SDPA/denso, gate e GQA.
   Verificare chunk, append e errori. Ancora niente pesi compressi simulati.
5. Metal prep/pooling, poi scorer/select, poi attention. Un kernel/stadio
   verificato alla volta contro C, lock comune per ogni test; prima decode,
   poi prefill con microbatch di query. Accumuli sempre FP32.
6. Collegare proiezioni Q8_0 di B e configurazione/binding A nel grafo del
   coordinatore; slice reali e dump di attivazioni. Validare commit sessione
   con C/E/F e propagazione errori. Nessun logit fittizio per dipendenze mancanti.
7. Solo dopo parità FP32: cache F16 esplicita, split paralleli e tile ottimizzati;
   TensorOps eventuale con nuova verifica della selezione. Benchmark separati,
   autorizzati nella fase opportuna, protocollo di `docs/BENCHMARKS.md`.

## 10. Comandi, memoria e rischi

### Verifica di questa SPEC (nessuna GPU)

```
git -C /Users/cristiancirje/Desktop/Dev/picoforge-spec-d status --short --branch
git -C /Users/cristiancirje/Desktop/Dev/picoforge-spec-d diff --check
git -C /Users/cristiancirje/Desktop/Dev/picoforge-spec-d diff --cached --name-only
```

Nessun `make`, `test-all`, benchmark o lettura payload lanciato: il file è
documentazione e diversi target esistenti attivano GPU/modelli. L'ispezione
ha riusato ambiente Python e inventari; consultate fonti pubbliche senza
installazioni, copie del checkpoint o modifiche ai checkout di riferimento.
Controlli documentali eseguiti: dieci sezioni presenti, fence Markdown
bilanciati, sei proiezioni layer 3 e somma byte riscontrate nell'inventario,
formule cache 8K/16K/32K ricalcolate con lo stesso venv. Il diff staged viene
controllato prima del commit, perché un file untracked non compare nel diff.

### Comandi futuri proposti, non ancora target esistenti

Da eseguire solo dopo che il coordinatore materializza CLI/target e lock.
`PYTHON` passa il venv condiviso read-only, output esclusivamente nel worktree;
non creare un nuovo venv e non scrivere fixture in `models/` principale.

```
make PYTHON=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python test-qwen38-attention-oracle
make PYTHON=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python test-qwen38-attention-cpu
make test-qwen38-attention-sanitize
```

Il target sanitize deve usare `-std=c11 -O1 -g -fsanitize=address,undefined`
e i warning del progetto. Dopo acquisizione del lock **comune definito in
integration.md**, eseguire `MTL_DEBUG_LAYER=1 make test-qwen38-attention-gpu`;
poi `make test-attention` e, in integrazione, `make`/`make test-all` con i
percorsi fixture/venv predisposti dal coordinatore. Le ultime sono verifiche
esistenti ma possono caricare modelli: non eseguirle in parallelo o in SPEC.
Nessun comando con un ipotetico lock locale è presentato come già sicuro.

### Budget calcolato

Per layer: cache FP32 = `4 L (2 J D + Di) + 4 floor(L/r) Di` byte.
Profilo F16 futuro = `4 L J D + 4 L Di + 2 floor(L/r) Di`.
Per 12 layer del backbone, senza MTP, con forme reali:

| Contesto | Cache FP32 iniziale | Cache F16 futura |
|---|---:|---:|
| 8192 | 444 MiB | 246 MiB |
| 16384 | 888 MiB | 492 MiB |
| 32768 | 1776 MiB | 984 MiB |

Proposta scratch D condiviso tra layer, **192 MiB massimo a T=1024, L=8192**,
comprensivo di ingressi X/output W_o nel conteggio conservativo: proiezioni
circa 54,5 MiB, Q/gate/output teste 72 MiB, IQ normalizzata 2 MiB, X/output
20 MiB, score 8 MiB, liste ~10,02 MiB e margine per tile/split/metadata.
Non moltiplicare scratch per 12 layer. Microbatch query per l'attention
`Tq≤32`, tile chiavi iniziale `Bk≤16` a D=256; K/V tile FP32 per una testa
KV occupano 32 KiB già a Bk=16, quindi **non** allocare entrambi interi in
threadgroup più scratch: iniziare Bk=8 (16 KiB K/V) o leggere V a stream.
Verificare static+dynamic threadgroup ≤ capacità device (32 KiB osservati)
e register pressure; forma del tile è scelta runtime verificata, non assunta.
Parziali split FP32 `[Tq,H,S,D+2]` in device scratch, S limitato dal piano
(es. 16, ~12,1 MiB a Tq=32), da computare nel margine prima di dispatch.

Obiettivo D 8K: cache+scratch ≤636 MiB (FP32), esclusi pesi ed eventuale
snapshot; il piano deve stampare byte effettivi e rifiutare budget superato.
16K/32K: mantenere bounded scratch abbassando Tq delle matrici di score,
senza allocare score densi [T,H,L]. Il test oracle FP64 materializza solo
piccoli casi o una query alla volta. Cap processo test CPU: 512 MiB per
fixture sintetica; slice reale con un layer può richiedere cap 1 GiB esplicito.

Viste pesi native già pianificate: 44.795.625.472 byte (~41,72 GiB), non
residenza misurata del nuovo grafo. Sommare nel piano runtime cache D,
stato C, scratch B/F, staging E e temporanei, senza sommare RSS a Metal come
se fossero insiemi disgiunti. Il limite per buffer e working set raccomandato
sono distinti. Nessuna tabella n-gram completa mappata da D.

### Rischi concreti e condizioni di arresto

- Doppio `+1` sulle norm, split globale q/gate, RoPE completa o R indexer
  dimezzata producono risultati plausibili ma sbagliati: fixture dedicate.
- Media di chiavi già normalizzate/ruotate e posizione ultimo token cambiano
  l'indexer; blocchi parziali non persistiti rompono append non allineati.
- Futuro incluso nel top-k, duplicati o tail omessa violano la semantica
  anche se la metrica dell'output medio è piccola: errore bloccante.
- FP16 solo nel prefill, sort non deterministico e FMA diverse possono
  cambiare i blocchi scelti. Preservare il caso e gli intermedi, non allargare
  tolleranze per ottenere un PASS.
- Softmax normalizzato per tile o gate applicato ai parziali altera l'output;
  test di merge, scale estrema e tile vuoti.
- Lifetime GPU, scratch riusato prima di completion e pubblicazione di
  length prima del successo rendono visibile stato incompleto: ticket/epoch.
- Modulo HF non installato e lock non concordato sono dipendenze reali;
  nessuna dichiarazione di parità HF o test GPU eseguito finché mancano.
- Cache 32K, dtype ridotti e prestazioni restano incrementi da verificare.
  La SPEC non conclude che un kernel funzionante basti alla generazione nativa.
