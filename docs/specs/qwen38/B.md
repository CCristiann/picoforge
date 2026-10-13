# B — Formati compressi Qwen3.8-Flash-Next Q2

Modalità **SPEC**, 2026-10-12. Base ispezionata:
`7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
`BASE_CONCORDATA`: **da determinare nel consolidamento**.
Branch di questa spec: `spec/qwen38-b`; worktree esterno:
`/Users/cristiancirje/Desktop/Dev/picoforge-spec-b`.

**V** indica fatti verificati su codice, metadati o evidenze registrate;
**P** indica contratti proposti, da consolidare; **A** una decisione aperta.
API, file futuri e comandi di implementazione qui descritti non esistono
ancora. Questa sessione modifica esclusivamente questo documento.

## 1. Obiettivo e confini

**P.** Rendere utilizzabili i byte del checkpoint scelto senza ricomprimerli:
prima oracle indipendente, poi decoder e prodotti matrice-vettore/matrice
C11, infine kernel Metal. Primo formato Q2_K; secondo IQ2_XXS. Completare poi
Q8_0 e gli adattatori F16/F32/BF16 necessari alla generazione ordinaria.
Non espandere un tensore di esperti completo, un layer o il modello in FP16/FP32.

B possiede interpretazione dei blocchi, dequantizzazione e prodotti lineari
con pesi nei formati supportati, inclusi i prodotti per esperto scelto.
Le attivazioni iniziali e gli output sono FP32. Il significato di un tensore,
il routing, le attivazioni MoE, il peso dei singoli contributi, l'ordine delle
riduzioni tra esperti e il grafo non appartengono a B.

Esclusi: tokenizer/configurazione A; ricorrenza e convoluzione C; attenzione
D; I/O n-gram E; routing/composizione MoE F; quantizzatori, calibrazione,
qualità rispetto a BF16, MTP e ottimizzazioni che restringano la precisione
delle attivazioni senza una decisione separata. Q4_K e MXFP4 restano
inventariabili ma non sono prerequisiti dell'esecuzione ordinaria.

Il percorso Qwen3 esistente resta disponibile. Nessun riuso di `QWeight`
ottenuto assegnando a IQ2_XXS un generico `bits=2`.

## 2. Fonti, stato corrente e codice riutilizzabile

### Revisioni e provenienza

**V.** Istruzioni lette nella repository principale: `CLAUDE.md`, ignorato
da Git; nessun `AGENTS.md` trovato nei percorsi applicabili presenti, oltre
alle istruzioni fornite nell'incarico. Letti integralmente il
[prompt](../../qwen38-q2-session-prompt.md), la
[baseline](../../qwen38-q2-baseline.md), lo
[storage](../../qwen38-storage.md), [README](../../../README.md),
[Makefile](../../../Makefile), [DESIGN](../../DESIGN.md) e
[BENCHMARKS](../../BENCHMARKS.md). Considerate le sezioni compressione,
Metal e validazione della [ricerca locale](../../large-model-research-2026-10-11.md),
la [revisione](../../project-review-2026-10-11.md) e il contesto degli
[esperimenti Qwen3](../../phase4-expert-experiment.md).

| Fonte | Revisione / riferimento verificato |
|---|---|
| picoforge | `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`; checkout principale e staging inizialmente puliti |
| DwarfStar locale | `/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline`, `fc80bd695da76ee14bcb86820bf900a5cdfda806`, sorgenti tracciati puliti |
| Checkpoint Q2 | `antirez/qwen3.8-flash-next-gguf`, revisione `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe` |
| Config/tokenizer originali | `Qwen/Qwen3.8-Flash-Next`, revisione `de4b8e4d43b917e7706784d8bb445c9af86a3540`, registrata dalla baseline |
| ggml/llama.cpp consultato | `8bb3cf0d216a5ae48360179f2931daf284be4d01`, sorgenti pubblici letti senza installazione |

L'API GitHub consultata in questa sessione restituiva ancora `fc80bd6` come
ultimo commit DwarfStar: non sono stati osservati fix successivi. Le fonti
sono fissate alle revisioni sopra; non dipendono da un futuro `main`.

Fonti primarie per la codifica:

- [DwarfStar, ricetta e padding](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/README.md),
  `qwen4_iq2.py:quantize/assemble`, `qwen4_pack.py:encode_weighted_experts`.
- [DwarfStar CPU](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c):
  `q2_k_value_f32` (riga 3991), `qwen4_ref_row` (67857),
  `qwen4_ref_matvec_rows`, `config_validate_qwen4_model`.
- [DwarfStar Metal](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/metal/qwen4.metal):
  rami Q2_K/IQ2_XXS del prodotto scalare (2644–2681) e decoder per tile;
  [test](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/tests/test_qwen4_kernels.c),
  `arena_q2_K`, `arena_iq2_xxs`, `test_moe_mm_tiles_iq2`.
- [ggml, strutture e tabelle](https://github.com/ggml-org/llama.cpp/blob/8bb3cf0d216a5ae48360179f2931daf284be4d01/ggml/src/ggml-common.h),
  [decoder canonici C](https://github.com/ggml-org/llama.cpp/blob/8bb3cf0d216a5ae48360179f2931daf284be4d01/ggml/src/ggml-quants.c)
  `dequantize_row_q2_K`, `dequantize_row_iq2_xxs`, e
  [decoder NumPy upstream](https://github.com/ggml-org/llama.cpp/blob/8bb3cf0d216a5ae48360179f2931daf284be4d01/gguf-py/gguf/quants.py)
  `Q2_K`, `IQ2_XXS`, `Q8_0`. Quest'ultimo è stato letto, non installato né eseguito.

SHA-256 dei sorgenti consultati:

| Sorgente | SHA-256 |
|---|---|
| DwarfStar `ds4.c` | `d137a9a5936d35152d1aa7abd9631462697abd69b429f52e51663292c1d063bd` |
| DwarfStar `metal/qwen4.metal` | `13224aeae23802defef678877146ae8047a45c2d1be824361efa2f4f5aa25b28` |
| DwarfStar `gguf-tools/qwen4_iq2.py` | `c590438d13f35ac884ba9f6e187d264bfa8b5195edf917b675ebfadd87fbe327` |
| ggml `ggml-common.h` | `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c` |
| ggml `ggml-quants.c` | `5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab` |
| ggml `gguf-py/gguf/quants.py` | `2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b` |

### Inventario verificato

**V.** Checkpoint riutilizzato in sola lettura:
`/Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf`.
Dimensione 147.207.127.040 byte. SHA-256 già verificato dalla
[baseline](../../../bench/qwen38_q2_20261011_phase0/checkpoint.verification.json):
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
In questa sessione `stat` coincide con inode e mtime registrati; nessun nuovo
hash dell'intero payload. Riletto soltanto l'header, 11.025.376 byte:
`ec931bcfabe3d85d69a0f7a07cd697b7205afd620b220d4b3099c7cced7a6e75`.
Metadati e tutti i 1.256 descrittori coincidono con
[inventory.json](../../../bench/qwen38_q2_20261011_phase0/inventory.json),
il cui SHA-256 è
`605d00ad17ae2fa66b10cf64d0dd0c3fac16122bb526dee10db064a0b8a67093`.

`block_count=49`, `nextn_predict_layers=1`: trunk `blk.0`…`blk.47`, MTP
`blk.48`. Questa separazione è confermata dal packer e dal runtime, non
dedotta dal solo formato. A deve derivarla dai metadati, non fissare 48.

| Formato GGUF (ID) | Valori/blocco | Byte/blocco | Tensori ordinari | Tensori solo MTP | Ruolo |
|---|---:|---:|---:|---:|---|
| Q2_K (10) | 256 | 84 | 48 | 0 | `ffn_down_exps` routed |
| IQ2_XXS (16) | 256 | 66 | 96 | 0 | `ffn_gate_exps`, `ffn_up_exps` routed |
| Q8_0 (8) | 32 | 34 | 327 | 10 | proiezioni dense/attention/SSM, esperti shared, indexer, PLE, `output.weight` |
| F16 (1) | 1 | 2 | 290 | 8 | proiezioni hyper-connection, incluse quelle di output |
| F32 (0) | 1 | 4 | 461 | 11 | router, norm, convoluzioni, controlli SSM |
| BF16 (30) | 1 | 2 | 2 | 0 | `token_embd.weight` e tabella n-gram disk-only |
| Q4_K (12) | 256 | 144 | 0 | 2 | gate/up routed di `blk.48` |
| MXFP4 (39) | 32 | 17 | 0 | 1 | down routed di `blk.48` |

1224 tensori ordinari, di cui 1223 residenti e uno n-gram; 32 MTP.
Q4_0, Q8_K, IQ2_XS e INT2 non compaiono nell'artifact. Il fatto che un
riferimento esterno sappia leggerli non li aggiunge al perimetro.

Forme fisiche GGUF e stride verificati:

| Tensore | Forma `[Kp,N,E]` (dimensioni mancanti = 1) | Byte/riga | Byte/esperto |
|---|---|---:|---:|
| trunk gate/up IQ2_XXS | `[2560,640,512]` | 660 | 422.400 |
| trunk down Q2_K | `[768,2560,512]` | 252 | 645.120 |
| shared gate/up Q8_0 | `[2560,640]` | 2.720 | 1.740.800 |
| shared down Q8_0 | `[640,2560]` | 680 | 1.740.800 |
| MTP gate/up Q4_K | `[2560,640,512]` | 1.440 | 921.600 |
| MTP down MXFP4 | `[640,2560,512]` | 340 | 870.400 |

`token_embd.weight` BF16 è `[2560,248320]`, `output.weight` Q8_0 è della
stessa forma: non assumere embedding e LM head legati. La tabella n-gram
è `[160,320001536]`, 320 byte/riga; E ne possiede l'accesso.

### Riutilizzo e limiti

- [gguf.h](../../../src/gguf.h), [gguf.c](../../../src/gguf.c): ID, forme
  fisiche, dimensioni dei blocchi, bounds dei descrittori. Sono contratti di
  storage, non decoder. `gguf_format` termina con `die` su ID sconosciuto.
- [gguf_plan.c](../../../src/gguf_plan.c),
  [gpu_weights.m](../../../src/gpu_weights.m): `GgufBinding`, owner, buffer ID,
  offset, viste no-copy multiple. Non convertire offset a 32 bit.
  `Gguf.header` non costituisce una vista generale del payload CPU.
- [ops.c](../../../src/ops.c), [picoforge.h](../../../src/picoforge.h):
  `bf16_to_f32` e convenzione `Y=X W^T` sono riutilizzabili. `matmul_q` e
  `dequant_row` interpretano `QWeight`, non i blocchi GGUF.
- [quant.c](../../../src/quant.c): utile il controllo dell'errore rispetto
  a `sum(abs(x*w))`; il suo budget empirico di 16 epsilon FP32 è misurato su
  Qwen3 e non costituisce automaticamente la tolleranza Q2.
- [test_qwen38_inventory.py](../../../tests/test_qwen38_inventory.py) e
  [test_gguf.py](../../../tests/test_gguf.py): layout/overflow/troncamenti,
  non correttezza numerica. La baseline Metal verifica byte e durata delle
  mappature, non prodotti scalari.

`q8_row/q4_row` hanno scale BF16 in un piano separato e codici signed; i
Q4 esistenti impacchettano nibble consecutivi in complemento a due. Anche
`q8_g32` non è byte-compatibile con Q8_0: la granularità 32 non basta, poiché
Q8_0 interlaccia scale **FP16** e codici. Il commento in
`tools/quant/formats.py` che richiama il layout GGUF non autorizza alias.

## 3. Semantica matematica e layout

### Convenzioni comuni

**P.** Byte little-endian, nessuna dipendenza dal padding di una struct C.
Caricare interi/FP16 tramite byte o `memcpy`, senza cast disallineati o
violazioni di aliasing. Tutta l'aritmetica per indirizzi è controllata prima
dell'accesso. La dimensione GGUF zero è contigua.

Per matrice di esperti `W[E,N,Kp]`, con input logico `K <= Kp`:

```text
row_bytes    = (Kp / values_per_block) * bytes_per_block
expert_bytes = N * row_bytes
address(e,n) = tensor_base + e*expert_bytes + n*row_bytes
Y[t,n]       = sum(k=0..K-1, X[t,k] * decode(W[e,n], k))
```

`Kp` deve essere multiplo del blocco; non arrotondare forme arbitrarie in
silenzio. `[Kp,N,E]` fisico equivale a `[E,N,Kp]` NumPy contiguo. Dimensioni
e numero di esperti arrivano da A; la spec riporta numeri del checkpoint
come evidenza, non costanti del decoder.

### Q2_K

**V.** Superblocco di 256 valori, 84 byte:

| Offset byte | Dimensione | Significato |
|---|---:|---|
| 0 | 16 | `scales[g]`: nibble basso scala `s_g`, alto minimo `m_g` |
| 16 | 64 | `qs`, quattro piani a 2 bit |
| 80 | 2 | scala globale `d`, FP16 little-endian |
| 82 | 2 | scala globale del minimo `dmin`, FP16 little-endian |

Per indice locale `i=16*g+l`, `0<=g<16`, `0<=l<16`:

```text
s_g   = scales[g] & 15
m_g   = scales[g] >> 4
qbyte = qs[32*(g//8) + 16*(g%2) + l]
q     = (qbyte >> (2*((g//2)%4))) & 3
w_i   = (float(d) * s_g) * q - float(dmin) * m_g
```

Il minimo si **sottrae**. `q` è unsigned, non signed INT2. Scale/minimi
cambiano ogni 16 valori; non leggere quattro codici consecutivi per byte
come se rappresentassero quattro colonne consecutive. La densità fisica
è 2,625 bit/valore, metadata inclusi.

**P.** La baseline di dequantizzazione usa i passi FP32 sopra, round-to-nearest
ties-to-even, senza fast-math/reassociation; i prodotti di FP16 per questi
interi sono esatti in FP32, la sottrazione può richiedere arrotondamento.

### IQ2_XXS

**V.** Blocco di 256 valori, 66 byte: `d` FP16 ai byte 0–1; otto gruppi
di 32 valori ai byte `2+8*h`, `0<=h<8`. Ogni gruppo contiene due parole
little-endian `G=read_u32(2+8*h)` e `S=read_u32(6+8*h)`.

Per `j=0..3`, `l=0..7`, indice locale `32*h+8*j+l`:

```text
grid_id = (G >> (8*j)) & 255
sign_id = (S >> (7*j)) & 127
a       = S >> 28
signs   = ksigns_iq2xs[sign_id]
v       = iq2xxs_grid[grid_id][l]       # byte l, dal meno significativo
d_h     = float(d) * (0.5 + float(a)) * 0.25
w       = d_h * float(v) * (-1 if ((signs >> l) & 1) else +1)
```

La griglia ha **256 vettori da 8 byte**, valori `{8,25,43}`; il fattore
`0.25` è parte della codifica. I quattro indici di segno occupano 28 bit;
gli ultimi quattro bit sono la scala locale, non segni aggiuntivi.
La tabella dei segni equivale a
`sign_id | ((popcount(sign_id) & 1) << 7)`: il bit 7 realizza parità pari.
Questa equivalenza è stata verificata per tutte le 128 entrate.

Tabelle DwarfStar e ggml confrontate integralmente, byte per byte:

| Sequenza canonica | Byte | SHA-256 |
|---|---:|---|
| `ksigns_iq2xs` | 128 | `a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31` |
| `iq2xxs_grid`, uint64 serializzati little-endian | 2048 | `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54` |

**P.** C e Metal possono condividere una sola definizione di queste costanti
di formato. L'oracle non deve importare la copia picoforge delle tabelle.
Non ricostruire una griglia "equivalente" ordinandone i vettori: l'indice
on-disk fa parte del formato. IQ2_XXS occupa 2,0625 bit/valore, ma non è
un tensore di interi signed a 2 bit e non si passa a TensorOps come INT2.

### Padding 640/768

**V.** Il packer azzera 128 colonne aggiuntive delle matrici BF16 **prima**
della quantizzazione e aggiunge zeri all'imatrix. Il contenitore conserva
tre blocchi Q2_K per riga. `expert_feed_forward_length=640` resta logico;
il runtime DwarfStar CPU fornisce padding zero alle attivazioni e Metal
usa lo stride fisico mascherando la coda.

**P.** A autorizza `K=640,Kp=768` solo sul binding down routed previsto;
B riceve entrambe le dimensioni e non le deduce dal nome del tensore.
L'API non richiede che l'input abbia spazio per 768 float. Per ogni load,
verificare `k<K` **prima** di leggere `X`; un tile può scrivere zero nel
suo scratch interno per `K<=k<Kp`. Nessun prodotto con valori della coda.

Non assumere `decode(W)[640:768] == 0`: il padding prima della quantizzazione
non è una garanzia sufficiente per omettere la maschera. In SPEC non è stato
letto/dequantizzato payload per misurare quei valori. Il contratto resta
corretto anche con codici finiti non nulli nella coda. Il terzo blocco ha
128 valori utili; la riga successiva inizia dopo **252**, non 210 byte.

Shared down Q8_0 e MTP down MXFP4 hanno `K=Kp=640`: non trasferire loro
automaticamente il padding dei routed Q2_K. L'effettivo costo Q2_K per
input logico di down è `252*8/640=3,15` bit, incluso il padding.

### Altri formati ordinari

**V/P.** Q8_0: 32 valori/34 byte, `d=FP16(bytes[0:2])`, 32 codici int8
signed, `w[i]=float(d)*int8(bytes[2+i])`. Accettare l'intero dominio signed
`[-128,127]` nel decoder, anche se un quantizzatore può emettere un subset.
F16: conversione IEEE binary16; BF16: bit in alto del binary32; F32:
binary32 little-endian. Nessuna reinterpretazione F16/BF16.
Non trasformare una proiezione F32 del router in BF16 per riusare un kernel.

La prima API restituisce errore di formato non implementato per il calcolo
Q4_K/MXFP4, mentre il loader continua a inventariarli. Per futura MTP,
Q4_K richiederà scale/minimi a 6 bit per gruppi da 32 e nibble a piani;
MXFP4 richiederà E8M0/E2M1 e ordine dei nibble GGUF. Sono incrementi
successivi, con oracle dedicati; nessuna falsa compatibilità con `q4_row`.

## 4. API proposte e durata dei dati

**P.** Prefisso `q38_packed_`, contratto C pubblico in
`src/qwen38_packed.h`, materializzato dal coordinatore. Nessun cambiamento
di `QWeight`/`DType` esistenti. Schema iniziale da fissare con A/F:

```c
typedef enum {
    Q38_PACKED_OK = 0,
    Q38_PACKED_EINVAL, Q38_PACKED_EFORMAT, Q38_PACKED_ELAYOUT,
    Q38_PACKED_EOVERFLOW, Q38_PACKED_EBOUNDS,
    Q38_PACKED_ENONFINITE, Q38_PACKED_EGPU
} Q38PackedStatus;

typedef struct {
    uint32_t gguf_type;
    size_t k_storage, k_logical, n_out, n_experts;
    size_t row_bytes, expert_bytes, tensor_bytes;
} Q38PackedLayout;

typedef struct {
    const unsigned char *data;
    size_t bytes;
    Q38PackedLayout layout;
} Q38PackedCpuView;

typedef struct {
    const float *data;
    size_t elements, rows, cols, row_stride;
} Q38F32Input;
typedef struct {
    float *data;
    size_t elements, rows, cols, row_stride;
} Q38F32Output;

Q38PackedStatus q38_packed_layout_init(
    const GgufTensor *tensor, size_t k_logical, Q38PackedLayout *out);
Q38PackedStatus q38_packed_decode_row(
    const Q38PackedCpuView *w, size_t expert, size_t row,
    float *out, size_t out_elements);
Q38PackedStatus q38_packed_matmul(
    const Q38PackedCpuView *w, size_t expert,
    Q38F32Input x, Q38F32Output y);
```

Semantica:

- `layout_init`: rank 1/2/3; rank 1 è una riga, rank 2 un esperto, rank 3
  `[Kp,N,E]`. Rank 4 non necessario e respinto dal calcolo. Confronto
  dell'estensione calcolata con `tensor.bytes`; prodotti/somme controllati.
  Il registro dimensioni dei formati resta quello di `gguf.c`. Serve una
  lookup non fatale condivisa, oppure filtro degli ID prima di `gguf_format`.
  Il formato registrato ma non implementato rimane distinguibile da layout
  corrotto. `0<K<=Kp`; A valida quali padding il modello ammette.
- `decode_row` scrive **Kp** float per ispezionare anche il padding;
  niente stride nell'output della singola riga. Verifica esperto/riga e
  capacità. Il chiamante seleziona i K valori logici per un embedding.
- `matmul`: `x=[M,K]`, `y=[M,N]`, stride in **elementi float**, colonne
  contigue; `row_stride>=cols`. `elements` comprende gli eventuali spazi
  tra righe e deve coprire `(rows-1)*row_stride+cols`, verificato senza
  overflow. `M=0` è no-op dopo validazione del layout. Per `M>0` puntatori
  validi; `K,N,E>0`; nessun broadcast implicito. Scrive solo le colonne
  logiche di y e non accumula nel contenuto precedente.
- Nessun alias tra input, pesi e output. Nessuna allocazione nascosta o
  cache globale mutabile; funzioni CPU sincrone e reentrant. Peso costante
  per tutta la chiamata, memoria di x/y posseduta dal chiamante.
- La vista CPU prende in prestito byte **già materializzati o mappati**
  dal chiamante. Non apre file, non mappa tutto il checkpoint, non assume
  che `Gguf.header` contenga i pesi. Le slice reali possono essere piccole
  matrici autonome; il manifest conserva gli indirizzi originali.
- Errori strutturali verificati prima di scrivere; valori/scale non finiti
  provocano `ENONFINITE`. Dopo errore numerico l'intero output è invalido
  e può essere parzialmente scritto: il consumer non lo può riutilizzare.
  Il decoder scalare F16/BF16, se esposto separatamente in futuro, può
  preservare le classi IEEE; il percorso lineare ordinario richiede finiti.

**P. Metal.** Interfaccia analoga, in Objective-C soltanto per preparazione
pipeline, binding ed encoding. Proposta di tipi aggiuntivi nel medesimo
contratto condiviso:

```c
typedef struct Q38PackedGpu Q38PackedGpu;
typedef struct {
    void *buffer;                 /* id<MTLBuffer> preso in prestito */
    size_t offset_bytes, bytes;
} Q38GpuSpan;
typedef struct {
    Q38GpuSpan span;
    size_t rows, cols, row_stride; /* stride in float, come CPU */
} Q38GpuF32Matrix;

Q38PackedStatus q38_packed_gpu_create(
    void *device, void *library, Q38PackedGpu **out);
void q38_packed_gpu_free(Q38PackedGpu *ctx);
Q38PackedStatus q38_packed_gpu_encode(
    Q38PackedGpu *ctx, void *compute_encoder,
    const GpuWeightStore *store, const GgufBinding *binding,
    const Q38PackedLayout *layout, size_t expert,
    Q38GpuF32Matrix x, Q38GpuF32Matrix y, Q38GpuSpan error_word);
```

`create` possiede pipeline e riferimenti Metal necessari; nessun peso o
scratch proporzionale al modello. Encoder/device/library hanno lo stesso
device; `free` va chiamato a lavoro concluso nella prima versione.
`encode` non crea/committa/attende un command buffer: il grafo può comporre
più operazioni e F può aggiungere il proprio gating. È il chiamante che
possiede command buffer, x/y e sincronizzazione.

Prima di encode validare identità di `binding.owner` rispetto a
`gpu_weights_plan(store)`, corrispondenza descriptor/layout, buffer e intero
intervallo; poi usare `gpu_weights_buffer`. Offset del binding più stride
esperto/riga in `size_t` host e `ulong`/64 bit MSL. Le dimensioni eventualmente
trasferite in `uint32_t` richiedono controllo prima del narrowing.
Nessun layout binario C/MSL implicito: struct dei parametri separata,
campi a larghezza fissa, offset e dimensione verificati nel test.

`error_word` è un uint32 allineato, >=4 byte, azzerato dal chiamante prima
del batch; i kernel OR-ano un flag in presenza di non finiti rilevati.
Gli errori di preflight ritornano immediatamente; il chiamante legge flag
e `MTLCommandBuffer.error` dopo completion. Un errore invalida gli output
del batch. Non dichiarare successo numerico al semplice ritorno di encode.
La prima versione tiene store/binding/ctx vivi sino alla completion;
anche x/y non sono riutilizzabili prima. Questo vincolo più semplice non
modifica le garanzie no-copy già verificate dello storage.

**P. Contratto minimo per F:** F fornisce ID esperto già validato e righe di
attivazioni contigue per un gruppo; B calcola una proiezione con quell'esperto.
F può iniziare con una chiamata per gruppo e orchestrarne gate/up/down.
Non sono necessari duplicati dei decoder per routing top-k, slot o shared.
Una successiva API indexed/grouped con gather/scatter GPU richiede accordo
B/F sui descrittori; non blocca la correttezza del prodotto minimo e non
introduce routing CPU in un grafo che lo ha già sulla GPU.

## 5. Dipendenze tra componenti

| Componente | Fornisce a B / riceve da B |
|---|---|
| A | Fornisce binding, `K` logico, significato degli assi, trunk/MTP e validazioni del modello; riceve schema dei formati, stride fisici e decoder |
| C | Usa proiezioni Q8_0/F32 e lettura dei parametri; possiede norm, convoluzione e ricorrenza; nessun vincolo di stato introdotto da B |
| D | Usa prodotti per proiezioni/indexer; possiede selezione, cache e attention; B non decide causalità o quantizzazione KV |
| E | Gestisce I/O e staging BF16; può usare il decoder per righe staged e proiezioni Q8_0; nessuna mappatura della tabella da parte di B |
| F | Fornisce gruppi di input/esperto; riceve output gate/up/down FP32; possiede attivazione, gating shared, pesi di routing e riduzione |
| Coordinatore/storage | Header/ABI, build Metal, gestione command buffer/errori, lock comune, integrazione test, politica di residenza |

La prima tranche oracle/C su righe sintetiche non dipende dal grafo A–F.
La validazione delle slice usa i descrittori reali esistenti; la generazione
non è dichiarabile finita senza A, C, D, E e F. Le tolleranze delle riduzioni
MoE e dell'intero grafo sono aggiuntive, non sostituite dai test di B.

## 6. File posseduti e modifiche condivise richieste

**P.** Assegnazione proposta da confermare in `integration.md`:

| Proprietario B | Scopo |
|---|---|
| `docs/specs/qwen38/B.md` | unica scrittura della modalità SPEC |
| `src/qwen38_packed.c` | validazione layout, decode e prodotti C11 |
| `src/qwen38_packed_gpu.m` | sola interfaccia Metal/encoding |
| `src/qwen38_packed.metal` | decoder e prodotti Metal |
| `src/qwen38_iq2_grid.inc` | costanti di formato, con provenienza/licenza |
| `tools/oracle/qwen38_packed.py` | manifest, fixture analitiche, confronto e prodotti FP64 |
| `tools/oracle/qwen38_packed_reference.c` | adapter CPU al riferimento DwarfStar immutato |
| `tests/test_qwen38_packed.c` | harness C, bounds ed errori |
| `tests/test_qwen38_packed.py` | differential test CPU, fixture e inventario di copertura |
| `tests/test_qwen38_packed_gpu.m` | harness Metal, span e lifetime |
| `tests/fixtures/qwen38_packed/manifest.json` | fixture sintetiche, seed, hash e revisioni; niente pesi |

Output derivati sotto `build/qwen38-packed/`, ignorati. Le slice reali non
si committano: si rigenerano dal checkpoint locale verificato. Nessun file
della baseline DwarfStar va modificato; oggetti dell'adapter nel worktree B.
Attribuzioni MIT dei riferimenti conservate nell'adapter e nelle costanti
derivate; la licenza dei pesi resta separata.

**Richieste concrete al coordinatore**, nessuna modifica di B a questi file:

1. `src/qwen38_packed.h`: fissare ABI, status, span, stride e lifetime
   con A/F; resta header condiviso a proprietario unico.
2. `src/gguf.h`, `src/gguf.c`: eventuale `gguf_format_try` non fatale
   delegata al registro esistente; nessuna duplicazione delle dimensioni.
3. `Makefile`: dipendenze dagli header nuovi, AIR/metallib del file Metal B,
   target test proposti sotto, variabili per Python/checkpoint/riferimento
   esterni. Il wildcard C/Objective-C attuale non compila automaticamente
   un secondo `.metal`: la regola corrente include solo `src/kernels.metal`.
4. Entry point, dispatcher GPU e test-all: collegamento a livello grafo e
   raccolta degli errori asincroni; nessun flag CLI aggiunto da B.
5. `integration.md`: protocollo lock e budget globale; valutare separatamente
   se escludere MTP dal piano residente. Il piano attuale lo include anche
   senza eseguirlo: non contabilizzare un risparmio non implementato.

`README.md`, `src/picoforge.h`, `src/metal.m`, `src/kernels.metal`,
`src/gpu_forward.m`, report aggregati e codice Qwen3 non sono posseduti da B.

## 7. Oracle indipendente e fixture

### Indipendenza

**P.** Tre riscontri complementari, da mantenere separati:

1. **Riferimento eseguibile esterno già disponibile:** adapter a
   `qwen4_ref_row` del DwarfStar fissato, compilando il sorgente originale
   con `DS4_NO_GPU`, senza fast-math e con contrazione disabilitata per
   i test di decode. L'inclusione di `ds4.c` segue il precedente
   `tests/test_qwen4_ngrams.c`; il wrapper costruisce piccoli
   `ds4_model`/`ds4_tensor` sopra byte in memoria e invoca la funzione,
   senza `model_open`, GPU o modello intero. Le dipendenze CPU del link
   sono quelle del target n-gram upstream; build isolata, non nel checkout
   della baseline. Non estrarre e ricopiare la funzione in Python.
2. **Expected analitici costruiti prima del decoder:** pattern con risultati
   calcolabili su carta, one-hot sulle attivazioni e somme esatte dyadiche.
   Il generatore non chiama il decoder C/MSL e non recupera gli expected
   decodificando i byte appena generati. Ciò rompe la dipendenza comune
   tra encoder e decoder che potrebbero condividere un errore d'indicizzazione.
3. **Conferma del formato upstream:** strutture/decoder ggml consultati e
   uguaglianza completa delle costanti. Eventuale esecuzione del pacchetto
   gguf-py sarebbe un'ulteriore conferma, non viene fatta passare per già
   disponibile: nell'ambiente esistente `gguf` non è installato. Non è
   necessario aggiungerlo per i punti 1–2; ogni nuova dipendenza richiede OK.

Python usa NumPy/Torch già presenti e prodotti FP64 sulle righe restituite
dal riferimento, poi confronta il C. B va scritto dal contratto binario;
se si importa direttamente il decoder DwarfStar nel codice di produzione,
quel medesimo decoder cessa di essere oracle indipendente e occorre un
altro riferimento eseguibile prima di superare il gate. C contro Metal
da solo non dimostra la correttezza del formato.

Il wrapper non è ancora compilato: la fattibilità dell'adapter è il primo
incremento e un fallimento non si risolve sostituendo l'oracle con B stesso.
Il modello originale Transformers/BF16 non decodifica il checkpoint Q2;
non serve caricarlo per questi test. Gli expected di Q2 non misurano danno
da compressione, NLL, logit drift BF16 o utilità di coding.

### Sintetiche obbligatorie

- Q2_K: ciascuno dei 256 indici, tutti i codici 0…3, nibble scala/minimo
  0…15, `d` diverso da `dmin`, zeri e FP16 subnormali/negativi/finiti grandi.
  Esempio leggibile: `d=1`, `dmin=0.5`, `s=3`, `m=2` produce
  `[-1,2,5,8]` per codici `[0,1,2,3]`. Scale/minimi distinti per gruppo
  e impulsi ai confini 15/16, 31/32, 127/128, 255/256.
- IQ2_XXS: copertura di tutti i 256 indici griglia, 128 indici segno,
  16 scale locali, quattro posizioni nel gruppo e otto gruppi; combinazioni
  esaustive griglia/segni/scala elaborate in streaming. `d=1`, griglia 0,
  scala locale 0, segno 0 dà otto valori 1; segno 1 nega le posizioni
  0 e 7. Griglia 1, stesso d/scala, dà primo valore `43/8` e altri 1:
  questo verifica ordine dei byte e fattore `1/4` senza usare il decoder.
- Q8_0: tutti i 256 codici, scale FP16 distinte dal medesimo bit pattern
  interpretato come BF16. F16/BF16: zero con segno, minimi subnormali,
  massimi finiti; non finiti producono l'errore numerico documentato.
- Più blocchi/righe/esperti con pattern diversi; base disallineata; buffer
  con offset non nullo, capacità esatta e un byte/elemento troppo corto;
  overflow dei prodotti, rank/format invalidi e ID esperto fuori range.
- Padding `K=640,Kp=768`: input di **esattamente 640 float**, canary/guard
  dopo il buffer, coda pesi deliberatamente non nulla, output invariato
  al mutare della sola coda. Verifica completa del terzo blocco e della
  prima riga dell'esperto successivo; nessuna lettura dopo x[639].
- Prodotti: `M=1,2,7,32,33,128`, `N=1,7,33` e finestre di forme reali;
  K a uno/più blocchi e 640/768; casi generici `K=1,255,257` con Kp
  allineato per verificare anche maschere parziali diverse dal checkpoint.
  Output riempito di NaN prima del dispatch,
  input one-hot, tutti zero, termini alternati con cancellazione e random
  a seed fissato. Stride con spazi, span vicini al limite del buffer,
  esperti 0/intermedio/ultimo; risultati per gruppo uguali a prodotti
  singoli, esclusa la sola differenza FP32 ammessa nell'ordine di riduzione.
- Metal: peso in almeno due viste, buffer_id diverso, binding di owner
  estraneo respinto; stessa matrice con diversi tile e confini M/N;
  lettura dei risultati soltanto dopo completion. Controlli asincroni
  dei non finiti e di mancata scrittura degli output.

### Slice reali necessarie, acquisizione futura

**P.** `pread` bounded di righe complete, mai mmap dell'intero artifact.
Per ogni trunk layer 0/23/47, proiezione gate/up/down, esperto 0/255/511
e riga 0/centrale/ultima: 81 righe, 42.444 byte compressi. Aggiungere
righe Q8_0 di output, shared down/gate/up, proiezioni attention/SSM/indexer;
righe F16 hyper-connection, F32 router/norm e BF16 token embedding.
E possiede le slice della grande tabella n-gram, non questo estrattore.

In un secondo livello, una sola matrice di un esperto alla volta per
gate/up/down, massimo 7,5 MiB per la versione FP32 fisica del down
(6,25 MiB per la sola parte logica), con prodotti
chunked e rilascio immediato. Non leggere 512 esperti per costruire una
fixture. I formati MTP possono essere acquisiti solo per futuri test dedicati.

Manifest: revisione e SHA globale già verificato, stat prima/dopo,
header hash, nome/formato/forma, indici, offset assoluto, byte letti e
SHA-256 di ogni slice, seed e dtype delle attivazioni, versione dell'oracle,
compiler/flag e SHA degli expected. Validare il range contro descrittore
e file; gestire short read/EINTR/EOF senza emettere fixture parziali valide.
L'estrattore offline non duplica lo scheduler I/O n-gram di E.

## 8. Tolleranze e criteri di accettazione

**P.** Criteri da congelare prima dell'implementazione; nessun risultato
numerico C/Metal di B è stato ancora misurato.

1. **Byte e layout:** uguaglianza esatta di blocchi, indici griglia/segni,
   stride e offset; tolleranza zero. Tutte le fixture analitiche e i test
   della coda devono passare. Il requisito non si sostituisce con un errore
   medio basso su matrici random.
2. **Decode:** riferimento e C FP32 con operazioni specificate, nessun
   restringimento intermedio: valori finiti identici bit per bit, ammettendo
   equivalenza di +0/-0 dopo aritmetica. Conversione BF16/F16 isolata:
   preservare anche il segno dello zero. IQ2_XXS e Q8_0 sono esatti in FP32
   per scale FP16 finite; Q2_K confronta il risultato FP32 arrotondato della
   sottrazione, non la formula in double non arrotondata. Metal deve
   riprodurre questi valori prima di accettare una fusione con il prodotto.
3. **Prodotti CPU/Metal:** baseline con input e pesi dequantizzati FP32,
   accumulo FP32, nessuna conversione implicita BF16/FP16. Oracle `r` in
   FP64 e `S=sum(abs(float64(x)*float64(w)))`. Per ogni elemento finito:

   ```text
   u = 2^-24
   gamma(n) = n*u / (1-n*u)
   abs(y-r) <= gamma(2*K+8)*S + 2*ulp32(float32(r))
   ```

   È un tetto analitico conservativo per prodotti/addizioni FP32, non un
   errore relativo al risultato cancellato. Vale per `n*u<1` e assenza di
   overflow/underflow dei prodotti; i casi estremi/subnormali hanno test
   separati con comportamento IEEE e FTZ esplicitato. Se `S=0`, il risultato
   deve essere zero; se `r=0` per cancellazione ma `S>0`, vale il limite
   sopra e non si pretende zero esatto. I test analitici dyadici piccoli che non
   richiedono arrotondamento esigono uguaglianza esatta anche nel prodotto.
4. Registrare inoltre massimo errore assoluto, ULP, `abs(y-r)/S`, percentile
   e peggior indice per forma/formato. Obiettivo iniziale diagnostico
   `<=1e-5*S` sulle fixture realistiche; non è un risultato misurato né
   sostituisce il tetto analitico. Prima delle ottimizzazioni fissare un
   limite empirico più stretto sul set congelato, motivato dai dati. Non
   alzare le soglie per far sparire una divergenza.
5. **Riduzione di precisione futura:** se TensorOps richiede tile half/BF16,
   registrare separatamente errore di conversione, accumulo e risultato
   rispetto al prodotto FP32. L'accordo tra due percorsi che restringono
   entrambi a half non prova parità col checkpoint. La scelta richiede
   consolidamento con F e controlli di composizione/logits.
6. **Robustezza:** ASan/UBSan senza errori CPU; Metal API validation; output
   e guard intatti dove non si deve scrivere; nessun errore mascherato;
   indipendenza dal contenuto oltre il limite logico delle attivazioni.

B è accettato per l'ordinario quando passano oracle→C→Metal su tutti i
sei formati ordinari, fixture e slice sopra, e F può usare lo stesso prodotto
per routed/shared senza duplicarne decode. Un decoder Q2_K corretto è un
incremento, non il completamento di B o della generazione Qwen3.8.

## 9. Incrementi di implementazione

**P.** Ogni incremento mantiene diff leggibile e verificabile; separare
ulteriormente se supera circa 150 righe di codice nuove. Nessuno viene
implementato in questa modalità SPEC.

1. Consolidare header minimo, owner, errori, build e lock. Preparare adapter
   dell'oracle indipendente; ottenere expected Q2_K su fixture analitiche.
   Gate: origine verificata e risultati analitici, nessun codice B chiamato.
2. Layout e decoder C Q2_K su blocchi/righe. Gate: bit equality, overflow,
   truncation, ASan/UBSan. Poi caso 640/768 e slice down reali limitate.
3. Prodotto C Q2_K con maschera K e stride. Gate: one-hot/cancellazioni,
   esperti/righe adiacenti, input della sola lunghezza logica.
4. Kernel Metal Q2_K diagnostico e GEMV FP32; poi batch semplice. Gate:
   confronto C/oracle, padding e viste multiple. Solo sotto lock comune.
5. IQ2_XXS: costanti con hash/attribuzione, expected indipendenti, C decode,
   C prodotto, Metal, in quest'ordine e commit separati. Gate: tutte le
   griglie/segni/scale, slice gate/up e confini dei gruppi.
6. Q8_0 e adattatori F16/F32/BF16 con identica catena. Gate: output head,
   proiezioni dense e shared senza ricodifica nei vecchi formati Qwen3.
7. Collegamento F e proiezioni degli altri componenti: prodotti senza
   routing/attivazioni incorporati; fixture di composizione del coordinatore.
   Gate: tutti i formati ordinari raggiunti, nessun dispatch su MTP.
8. Solo dopo parità: prefill tiled/grouped, riuso di tile e possibile
   gate/up congiunto mantenendo il decoder unico. Fusioni SiLU/pesi di
   routing sono accordi con F. Benchmark esclusivi e ablation separata
   per qualsiasi narrowing; Q4_K/MXFP4 aspettano il lavoro MTP.

## 10. Verifica, memoria, rischi e decisioni aperte

### Comandi

**Eseguiti in SPEC:** `git status`, HEAD e staging; lettura integrale dei
documenti richiesti; inventario header-only attraverso lo script esistente,
con confronto integrale in memoria all'inventario registrato; confronto
delle tabelle con ggml upstream; sei test CPU dell'inventario, tutti passati:

```sh
cd /Users/cristiancirje/Desktop/Dev/picoforge-spec-b
mkdir -p build/spec-b-tmp
TMPDIR="$PWD/build/spec-b-tmp" \
  /Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python -B \
  tests/test_qwen38_inventory.py
git diff --check
```

Nessun test GPU, benchmark, caricamento completo, copia del modello,
nuova dipendenza o modifica di SDK/memoria. `make test-all` della baseline
è evidenza precedente: non è stato rieseguito da questa spec.

**Comandi proposti per ESEGUI, non disponibili fino all'integrazione del
Makefile/header:** `PYTHON`, `Q38_CHECKPOINT`, `DS4_REFERENCE`, `BUILD_DIR`
sono variabili di build da introdurre dal coordinatore, non flag già esistenti.

```sh
make test-qwen38-packed-cpu \
  PYTHON=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python \
  DS4_REFERENCE=/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline \
  BUILD_DIR=build/qwen38-packed
make test-qwen38-packed-sanitize  # ASan/UBSan, no fast-math, CPU soltanto
make test-qwen38-packed-slices \
  Q38_CHECKPOINT=/Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
# Acquisire prima il lock comune definito in integration.md:
MTL_DEBUG_LAYER=1 make test-qwen38-packed-gpu
```

L'ultimo target esegue soltanto fixture bounded. Il target CPU costruisce
l'adapter originale DwarfStar in `BUILD_DIR` e l'harness C, produce manifest
ed expected, poi esegue i differential test. Se sorgenti/hash non coincidono
o manca l'oracle deve fallire, non saltare il confronto. I test delle slice
non lanciano il modello. Per la regressione finale il coordinatore esegue
`make`, `make test-all` e i test di composizione sulla BASE_CONCORDATA con
fixture Qwen3/Python riusate e lock; nessuna creazione automatica di nuove
copie di checkpoint da parte di B.

### Budget proposto, non misura prestazionale

- SPEC: nessun payload letto; header effettivo ~10,5 MiB, massimo del
  lettore esistente 128 MiB; fixture sintetiche minime.
- Oracle/test B: tetto 256 MiB di array/staging numerici e 256 MiB di
  artifact per suite; elaborazione streaming delle combinazioni IQ2,
  checkpoint letto per non più di 32 MiB per invocazione slice. Runtime
  Python, compilatore e driver si contabilizzano separatamente; non
  presentare questo tetto degli array come misura del footprint del processo.
- Decoder CPU: O(1) scratch (al massimo un blocco 256 FP32 = 1 KiB),
  esclusi output richiesti dal chiamante. Prodotto diretto senza matrice
  dequantizzata completa. Tabella IQ2: 2 KiB + eventuali 128 byte segni.
- Primo Metal: decode in registri, accumulo FP32; per prefill proporre
  tile K=256/N=16 di pesi FP32 = 16 KiB più X per M=8 = 8 KiB,
  massimo 24 KiB di threadgroup per singolo staging, sotto i 32 KiB
  registrati. Nessun double buffering implicito; limiti/device e uso
  effettivo della pipeline vanno verificati prima del dispatch. Lo schema
  tile è un candidato di correttezza, non una promessa di prestazioni.
- Nessuno scratch `[E,N,K]`. Con M=1024, N=2560 un output FP32 è 10 MiB;
  moltiplicarlo per i 10 slot routed richiede 100 MiB e appartiene al piano
  F/coordinatore. B non alloca né conserva questi dieci output da solo.

**V, calcolo sui descrittori:** payload residente ordinario 43.287.498.240
byte (40,314624 GiB), MTP 1.508.088.832 byte (1,404517 GiB), tabella disk-only
102.400.491.520 byte (95,367889 GiB). Il piano attuale include MTP e riporta
44.795.625.472 byte di span mappati, inclusi arrotondamenti alle pagine.
Questi non sono footprint fisico né totale runtime. Obiettivo globale
48–50 GiB e riserve per C/D/E/F restano del coordinatore. Non sommare RSS,
Metal allocated bytes e wired come se fossero categorie disgiunte.

### Rischi e questioni da chiudere in CONSOLIDA

| ID | Decisione / rischio concreto | Responsabile proposto |
|---|---|---|
| B-A1 | Fissare layout logico/fisico e whitelist del padding per binding; separazione trunk/MTP e embedding/head non tied | A + coordinatore |
| B-F1 | Accettare l'ABI minimo per esperto; successivo contratto grouped/indexed, ownership gather/scatter e riduzione FP32 | B + F |
| B-G1 | Fissare header, error-word batch e lifetime; chi raccoglie completion/errori GPU; approvare formato dei parametri C/MSL | coordinatore |
| B-O1 | Compilare l'adapter originale CPU come primo incremento; non è ancora un oracle eseguito. Se si riusa il suo decoder in produzione serve un altro oracle | B |
| B-L1 | Percorso assoluto e protocollo del **lock unico di sistema** ancora assenti nei documenti ispezionati. Nessun lock locale inventato; GPU e carichi pesanti aspettano `integration.md` | coordinatore |
| B-M1 | Decidere se lasciare MTP residente inizialmente o escluderlo nel planner; non confondere "non eseguito" con "non mappato" | coordinatore/storage |
| B-P1 | Eventuale half/BF16 nei tile TensorOps cambia i valori: test conversione/prodotto/composizione e autorizzazione del contratto prima di adottarlo | B + F + coordinatore |

Errori da prevenire con test dedicati: overflow degli offset oltre 4 GiB;
letture u32 disallineate nei blocchi da 66 byte; scambio FP16/BF16;
griglia IQ2 trasposta o segni trattati come indipendenti; dimenticare il
minimo Q2_K; stride down da 640 anziché 768; accessi alla coda x;
alias dei vecchi formati Qwen3; uso accidentale di Q4_K/MXFP4 del MTP nel
grafo ordinario; errori GPU scambiati per un output numerico valido;
espansione di tutti gli esperti per semplificare il prefill. Ogni errore
richiede una riproduzione minima, non un allargamento delle tolleranze.


## Incremento MTP — 2026-10-13

Il runtime successivo alla consegna B iniziale abilita anche Q4_K (12) e
MXFP4 (39), senza cambiare i formati ordinari né il contratto FP32. Entrambi
usano il decoder indipendente DwarfStar pinned, interpretazione analitica,
prodotti con stride/esperti/padding e slice bounded del checkpoint. Q4_K:
18 righe/25.920 byte; MXFP4: 9 righe/3.060 byte. I nibble MXFP4 sono low per
le prime 16 colonne e high per le successive; E8M0=0 usa 2^-127.
Metal ricostruisce i bit FP32 per preservare anche i subnormal e signed zero
nel decode, senza moltiplicazioni soggette a FTZ. Il gate nonfinite rimane.

L’ammissione dei due formati non li sostituisce ai pesi ordinari: A continua
ad esigere IQ2_XXS/Q2_K nei ruoli del trunk e Q4_K/MXFP4 soltanto negli esperti
MTP. La down MTP usa K=Kphysical=640. Implementazione e gate del predictor in
[MTP.md](MTP.md); nessun riuso del padding fisico 768 del trunk.
