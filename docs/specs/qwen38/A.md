# A — Configurazione, binding, tokenizer e template Qwen3.8

Stato corrente: **ESEGUI A completato nel perimetro §8 di integration**;
risultati, limiti e comandi al §11. Nessuna generazione Qwen3.8 completa.
`BASE_CONCORDATA`: `c674b23bd1377b2ce0acef2d06fac8b160809446`.
I §1–10 conservano la proposta SPEC del 2026-10-12: decisioni allora aperte
e comandi allora proposti non prevalgono su integration/header v1 e §11.
Base ispezionata: `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
Branch di questa spec: `spec/qwen38-a`; worktree esterno:
`/Users/cristiancirje/Desktop/Dev/picoforge-spec-a`.

Nel seguito **V** indica un fatto verificato su file/codice, **P** un contratto
proposto, **Aperto** una decisione o prova ancora necessaria. Le dimensioni
numeriche descrivono il checkpoint ispezionato: non sono costanti da inserire
nei kernel. Questa spec non certifica generazione, logits o perdita da Q2.

## 1. Obiettivo, confini e fonti

**P.** A trasforma metadati e descrittori GGUF in una configurazione validata,
un catalogo di tensori con semantica e layout espliciti e un tokenizer/template
C11 per il percorso testuale Qwen3.8. B–F consumano questo contratto senza
rideterminare dimensioni, nomi, padding o convenzioni degli ID. Si preservano
configurazione, tokenizer, chat e inferenza Qwen3 esistenti.

Sono esclusi: decoder numerici low-bit e kernel di B; operatori e stato di
C–F; orchestrazione del forward/generazione; implementazione MTP, immagini,
video, audio; conversione dei pesi; benchmark. Il template deve coprire chat
testuale multi-turno, thinking e tool call testuali. Supportare i marcatori
multimodali nel vocabolario non significa supportare i relativi modelli.

### Registro delle fonti

| Rif. | Fonte e revisione effettivamente consultata |
|---|---|
| R1 | Repository alla base sopra; [handoff](../../qwen38-q2-session-prompt.md), [baseline](../../qwen38-q2-baseline.md), [storage](../../qwen38-storage.md), [README](../../../README.md), [Makefile](../../../Makefile), [DESIGN](../../DESIGN.md), [BENCHMARKS](../../BENCHMARKS.md). Letti integralmente. |
| R2 | Istruzioni della sessione e `CLAUDE.md` ignorato della repository principale, letto direttamente; nessun `AGENTS.md` aggiuntivo trovato negli antenati o nel percorso della spec. La direzione Qwen3.8 prevale sulla roadmap Qwen3. |
| R3 | [Ricerca locale](../../large-model-research-2026-10-11.md), sezioni architettura, compressione, storage, validazione; [review](../../project-review-2026-10-11.md), [esperimento 30B](../../phase4-expert-experiment.md), [confronto 80B](../../80b-100b-model-comparison-2026-10-11.md), parti pertinenti. |
| R4 | [Config HF][hf-config] e [generation config][hf-generation], revisione `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Recuperati in memoria tramite HTTPS, senza pesi né installazioni. SHA-256: rispettivamente `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` e `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`. |
| R5 | [Tokenizer HF][hf-tokenizer] e [tokenizer config][hf-tokenizer-config], stessa revisione; copie preesistenti `build/qwen38-tokenizer.json` e `build/qwen38-tokenizer-config.json` nella repository principale. SHA-256: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` e `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`. |
| R6 | [Artifact Q2][q2], revisione `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`, SHA-256 registrato `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`. Inventari e verifica in `bench/qwen38_q2_20261011_phase0/`; inventario C in `bench/qwen38_q2_20261012_storage/native-inventory.json`. |
| R7 | Checkout locale `/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline`, pulito, commit `fc80bd695da76ee14bcb86820bf900a5cdfda806`; `git ls-remote` il 12 ottobre conferma lo stesso HEAD upstream. Consultati [ds4.c][ds4-code], `config_validate_qwen4_model`, `weights_bind_qwen4_layer`, `qwen4_ref_*`, `bpe_tokenize_text_qwen35`; [convertitore][ds4-converter] e [guida Qwen][ds4-guide]. |

**V.** Il file locale Q2 ha ancora 147.207.127.040 byte, inode 30631842 e
mtime_ns 1791785892435960350, uguali alla verifica registrata. Non è stato
ricalcolato l'hash dell'intero file: richiederebbe un carico fuori da SPEC.
Sono stati letti soltanto 10.945.596 byte di metadati per il confronto del
tokenizer; nessun payload. L'inventario registra `data_offset=11025376` e
header SHA-256 `ec931bcfabe3d85d69a0f7a07cd697b7205afd620b220d4b3099c7cced7a6e75`.

## 2. Stato corrente e codice riutilizzabile

| Codice alla base R1 | Riutilizzo e limite verificato |
|---|---|
| `src/config.c:config_load`, `src/picoforge.h:Qwen3Config` | Solo `qwen3`/`qwen3_moe`; scanner di scalari con `strstr`, non parser del `text_config` annidato. Non aggiungere semplicemente `qwen4_exp` alla whitelist. |
| `src/gguf.c`, `src/gguf_reader.c`, `src/gguf.h` | GGUF v3 bounded; formati, overflow, rank, sovrapposizioni già verificati. `GgufMetadata` conserva intervalli borrowed dell'header; mancano accessor semantici e schema del modello. |
| `src/gguf_plan.c`, `src/gpu_weights.m` | `GgufBinding` conserva forma fisica, formato, buffer, offset e owner. Il piano separa la tabella PLE; non valida ancora il grafo né la forma logica. |
| `src/model.c`, `src/gpu_forward.m` | Binding/forward Qwen3; `Tensor`/`Linear` e `DType` non rappresentano i layout GGUF. Nessun cast tra enum dei due contenitori. |
| `src/tokenizer.c` | Byte alphabet, hash del vocabolario, merge ranks, BPE e ricerca added tokens riutilizzabili. Carica `vocab.json`/`merges.txt` e config Qwen3; splitter privo di `\p{M}`; flag `special` non conservato separatamente dal riconoscimento added. |
| `src/normalize.c`, `src/unicode_tables.c`, `tools/gen_unicode_tables.py` | NFC e tabelle Unicode 16.0.0; da verificare contro il backend R5, non aggiornare globalmente il comportamento Qwen3. |
| `src/generate.c:chat_format` | Template Qwen3 single-turno. Qwen3.8 thinking aggiunge istruzioni system e prefisso `<think>\n`: non è intercambiabile. |
| `tests/test_tokenizer.py`, `tests/test_encode.py`, `tests/test_nfc.py` | Confronti completi di tabelle, encode HF e NFC Qwen3; mantenere le aspettative esistenti. |
| `tools/eval/check_qwen38_tokens.py` | Oracle indipendente HF del template/tokenizer, già applicato ai cinque prompt della baseline DwarfStar. Non misura il tokenizer C Qwen3.8, che manca. |

**V.** Il Makefile usa `src/*.c`, header condiviso `picoforge.h`, dipendenze
GGUF esplicite e Python hardcoded `./tools/venv/bin/python`. `make test-all`
include GPU e forward: non è un controllo da eseguire in SPEC. Python 3.14.2,
Transformers 5.13.0, tokenizers 0.22.2, Jinja2 3.1.6 e regex 2026.6.28 sono già
disponibili nel venv principale. La directory Transformers `models/qwen4_exp`
**non è presente**: disponibilità del tokenizer non implica un oracle del
forward ufficiale. Non installare dipendenze per risolvere questo punto.

## 3. Semantica, configurazione e layout

### 3.1 Autorità delle informazioni e schema

**P.** Per il runtime Q2 il GGUF è l'autorità di dimensioni, forme, dtype e
costanti PLE. Config HF e codice originale sono riferimenti semantici e oracle
offline; non serve scaricare un JSON a ogni avvio. Metadati presenti e
contraddittori sono un errore, non un invito a provare un default.

Un profilo esplicito `qwen4exp`/Qwen3.8 identifica le convenzioni non codificate
nel GGUF: gate sigmoid, attivazione SiLU, ordine delle teste, interpretazione
di `ssm_a`, normalizzazioni, tokenizer NFC e template riconosciuto. Queste sono
definizioni dell'architettura, non dimensioni implicite. Il profilo documenta
la fonte di ogni campo derivato; una variante semantica sconosciuta viene
rifiutata. B–F non scelgono fallback locali.

Nel confronto HF usare **solo** `text_config`; `vision_config.hidden_size=1152`
non deve contaminare `hidden_size=2560`. Il nome HF esterno è `qwen4_exp`, quello
testuale `qwen4_exp_text`, quello GGUF `qwen4exp`.

Nella tabella il prefisso GGUF è `qwen4exp.`. Tutti gli scalari interi elencati
sono UINT32, epsilon/base RoPE sono FLOAT32; gli array architetturali osservati
sono ARRAY di UINT64. Interi architetturali positivi salvo le eccezioni indicate.

| Campo canonico | Metadato GGUF | Confronto HF / valore osservato |
|---|---|---|
| `n_blocks`, `n_mtp`, `n_layers` | `block_count`, `nextn_predict_layers` | 49, 1, **48 = 49−1**; HF `num_hidden_layers=48`, `mtp_num_hidden_layers=1` |
| `hidden`, `vocab`, `max_positions` | `embedding_length`, `vocab_size`, `context_length` | 2560, 248320, 262144 |
| `q_heads`, `kv_heads` | `attention.head_count`, `attention.head_count_kv` | 24, 2; non usare `hidden/q_heads` per `head_dim` |
| `key_dim`, `value_dim` | `attention.key_length`, `attention.value_length` | 256, 256 |
| `norm_eps` | `attention.layer_norm_rms_epsilon` | `float32(1e-6)` |
| `rope_dim`, `rope_theta` | `rope.dimension_count`, `rope.freq_base` | 64, 10000000; HF partial_rotary_factor=0,25 |
| `full_interval`, `compress_ratio[l]` | `full_attention_interval`, `attention.compress_ratios` | 4; 49 voci: 0 GDN, 4 QSA, incluso blocco 48 MTP |
| `index_heads`, `index_dim`, `index_budget` | `attention.indexer.head_count`, `.key_length`, `.top_k` | 4, 128, 2048 **token**, non 2048 blocchi |
| `index_kv_heads` | derivato da righe `indexer.k_proj` / `index_dim` | 1; HF `indexer_kv_heads=1` |
| `experts`, `top_k`, `expert_hidden`, `shared_hidden` | `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length` | 512, 10, 640, 640; i due hidden restano distinti |
| `conv_kernel`, `gdn_key_dim`, `gdn_k_heads`, `gdn_v_heads`, `gdn_inner` | `ssm.conv_kernel`, `.state_size`, `.group_count`, `.time_step_rank`, `.inner_size` | 4, 128, 16, 48, 6144 |
| `gdn_value_dim` | `ssm.inner_size / ssm.time_step_rank` | 128; divisione esatta e confronto HF `linear_value_head_dim` |
| `hc_count`, `hc_rank` | `hyper_connection.count`, `.low_rank` | 4, 320 |
| `ple_layers[]` | `ple.layers` | `[1]`, zero-based; HF `ple_layer_ids=[2]`, conversione 1-based→0-based confermata da `layers.1.ple.*` |
| `ngram_size`, `heads_per_ngram`, `ple_kernel` | `ple.ngram_size`, `.heads_per_ngram`, `.conv_kernel` | 3, 8, 4 |
| `ple_history_eos` | `ple.eos_token_id` | 248044; **diverso** da chat EOS 248046 |
| `ple_head_dim`, `ple_rows` | `embedding_length_per_layer_input`, `ple.row_dimension`, `ple.row_count` | 160=160, 320001536 |
| provenance PLE | `ple.seed`, `.vocab_base`, `.vocab_divisor` | 1234, 20000000, 128; non rigenerare gli hash al posto degli array presenti |
| `ple_multipliers[]`, `ple_offsets[]`, `ple_vocab[]` | `ple.layer_multipliers`, `.head_offsets`, `.head_vocab_sizes` | lunghezze 3, 16, 16; preservare i 64 bit |
| `down_logical`, `down_physical` | `ds4.qwen4.down.logical_input`, `.physical_input` (senza prefisso qwen4exp) | 640, 768; applicabili ai routed down Q2_K del trunk, **non** a shared/MTP |

Parametri semantici presenti in R4 ma non esportati come chiavi autonome nel
GGUF scelto: `hidden_act=silu`, `output_gate_type=sigmoid`,
`mamba_ssm_dtype=float32`, `attention_bias=false`, `tie_word_embeddings=false`,
RoPE `rope_type=default`, `mrope_interleaved=true`, `mrope_section=[11,11,10]`.
Il profilo li espone con provenienza R4; nessun default Qwen3. Per testo le tre
coordinate RoPE coincidono con la posizione token, ma la trasformazione rimane
di D. Lo stato ricorrente FP32 è il punto di partenza di C, non una scelta
dedotta dal dtype Q2 dei pesi. Training dropout/aux-loss non sono operatori
di inferenza. Qualsiasi variante di questi parametri richiede validazione
esplicita del profilo prima dell'esecuzione.

Gli array PLE verificati, da includere nella fixture del config (non nel codice):

```text
multipliers = [23703573157769, 20109073645365, 8052911324071]
offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165,
           120000228, 140000297, 160000374, 180000455, 200000548,
           220000655, 240000802, 260000955, 280001114, 300001275]
vocab = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
         20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
         20000153, 20000159, 20000161, 20000171]
```

**P. Validazioni prima di allocare stato o aprire Metal:**

- `0 <= n_mtp < n_blocks`, `n_layers=n_blocks−n_mtp`; richiedere la chiave
  nextn nel profilo scelto. Supportare metadati sintetici con `n_mtp=0`, senza
  supporre un ultimo layer ordinario di tipo QSA.
- Array compressioni lungo esattamente `n_blocks`; i primi `n_layers`
  concordano con l'intervallo e con i tensori GDN/QSA presenti; MTP classificato
  separatamente. In HF `layer_types` ha 48 voci e MTP ha il proprio array.
  Non iterare 49 layer per la generazione ordinaria.
- `q_heads % kv_heads == 0`, `gdn_v_heads % gdn_k_heads == 0`, divisori non
  nulli; `0 < top_k <= experts`; `rope_dim` pari, entro key/index dim;
  epsilon/theta finiti e positivi; ratio QSA positivo, budget divisibile per
  ratio per il profilo iniziale. D stabilisce il contratto per altri rapporti.
  Il profilo iniziale richiede key_dim=value_dim e gdn_key_dim=gdn_value_dim,
  come R4; eventuali dimensioni asimmetriche sono UNSUPPORTED fino a un
  contratto concordato con C/D, non un reshape automatico.
- Prodotti e arrotondamenti controllati **prima** di conversioni a `size_t`,
  `int`, parametri Metal o allocazioni; limiti del contenitore non equivalgono
  alla fattibilità del runtime. Contesto richiesto e chunk sono opzioni runtime
  separate da `max_positions`; 8192/1024 sono target iniziali, non il config.
- `ple_layers` univoci e nel trunk; un solo layer PLE supportato inizialmente,
  altri rifiutati esplicitamente. `N=(ngram_size−1)*heads_per_ngram=16`;
  `N*ple_head_dim=2560`. Le tre lunghezze degli array devono corrispondere.
  Offset contigui, vocab > 0, somme checked. La somma finale dei vocab è
  **320001446**, arrotondata al multiplo di `vocab_divisor` successivo:
  `ple_rows=ceil(320001446/128)*128=320001536`. Le ultime **90 righe** sono
  padding fisico della tabella, fuori dai range hash delle teste; non imporre
  somma finale uguale alle righe fisiche. Nessuna conversione dei multiplier
  in FP32.
- `token_embd` e `output` distinti nel profilo (`tie_word_embeddings=false`);
  niente alias implicito se manca `output`. Ogni nome necessario compare una
  volta sola, forma e formato devono corrispondere al ruolo. Bias inattesi,
  layer mancanti, famiglia di tensori del layer sbagliato: errore.
- Metadati descrittivi non computazionali possono essere ignorati con report;
  chiavi computazionali nuove e tensori sconosciuti non vengono usati per
  dedurre un'architettura alternativa. Il profilo deve classificarli o rifiutarli.

### 3.2 Catalogo dei tensori

**V/P.** Di seguito sono verificati nomi, forme e dtype di R6; le espressioni
simboliche sono la regola proposta di validazione. `E=hidden`, `V=vocab`,
`H=hc_count`, `R=hc_rank`, `X=H*E`, `F=expert_hidden`, `Fs=shared_hidden`,
`Ne=experts`, `Q=q_heads*key_dim`, `KV=kv_heads*key_dim`,
`Gk=gdn_k_heads*gdn_key_dim`, `Gv=gdn_inner`, `Gc=2*Gk+Gv`.
Le forme sono sempre in **ordine GGUF, dimensione contigua per prima**.

| Nome (prefisso `blk.l.` dove indicato) | Forma fisica | Dtype Q2 osservato |
|---|---|---|
| `token_embd.weight`, `output.weight` | `[E,V]` = `[2560,248320]` | BF16, Q8_0 rispettivamente |
| `output_hc_norm.weight` | `[X]` = `[10240]` | F32 |
| `output_hc_down.weight`, `output_hc_up.weight` | `[X,R]`, `[R,X]` | F16 |
| `hc_attn_norm.weight`, `hc_ffn_norm.weight` | `[X]` | F32 |
| `hc_attn_down.weight`, `hc_ffn_down.weight` | `[X,R]` | F16 |
| `hc_attn_up.weight`, `hc_ffn_up.weight` | `[R,X]` | F16 |
| `hc_attn_inject.weight`, `hc_ffn_inject.weight` | `[X,H]` | F16 |
| GDN `attn_qkv.weight` | `[E,Gc]` = `[2560,10240]` | Q8_0 |
| GDN `attn_gate.weight`, `ssm_out.weight` | `[E,Gv]`, `[Gv,E]` | Q8_0 |
| GDN `ssm_alpha.weight`, `ssm_beta.weight` | `[E,gdn_v_heads]` | F32 |
| GDN `ssm_a`, `ssm_dt.bias` | `[gdn_v_heads]` | F32 |
| GDN `ssm_norm.weight` | `[gdn_value_dim]` | F32 |
| GDN `ssm_conv1d.weight` | `[conv_kernel,Gc]` = `[4,10240]` | F32 |
| QSA `attn_q.weight` | `[E,2*Q]` = `[2560,12288]` | Q8_0 |
| QSA `attn_k.weight`, `attn_v.weight` | `[E,kv_heads*key_dim]`, `[E,kv_heads*value_dim]` | Q8_0 |
| QSA `attn_output.weight` | `[q_heads*value_dim,E]` = `[6144,2560]` | Q8_0 |
| QSA `attn_q_norm.weight`, `attn_k_norm.weight` | `[key_dim]` | F32 |
| QSA `indexer.q_proj.weight`, `.k_proj.weight` | `[E,index_heads*index_dim]`, `[E,index_kv_heads*index_dim]` | Q8_0 |
| QSA `indexer.q_norm.weight`, `.k_norm.weight` | `[index_dim]` | F32 |
| MoE `ffn_gate_inp.weight` | `[E,Ne]` | F32 |
| MoE `ffn_gate_exps.weight`, `ffn_up_exps.weight` | `[E,F,Ne]` | IQ2_XXS nel trunk |
| MoE `ffn_down_exps.weight` | `[Fp,E,Ne]`, `Fp=768`, logico `[F,E,Ne]` | Q2_K nel trunk |
| Shared `ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | `[E,Fs]` | Q8_0 |
| Shared `ffn_down_shexp.weight` | `[Fs,E]` = `[640,2560]` | Q8_0, senza padding Q2 |
| Shared `ffn_gate_inp_shexp.weight` | `[E]`, prodotto scalare | F32 |
| PLE `blk.1.ple_key.weight`, `.ple_value.weight` | `[N*ple_head_dim,X]`, `[N*ple_head_dim,E]` | Q8_0 |
| PLE `blk.1.ple_norm_key.weight`, `.ple_norm_query.weight`, `.ple_norm_conv.weight` | `[X]` | F32 |
| PLE `blk.1.ple_conv1d.weight` | `[ple_kernel,X]` | F32 |
| `per_layer_token_embd.weight` | `[ple_head_dim,ple_rows]` = `[160,320001536]` | BF16, **solo disco** |

Ci sono 36 layer GDN e 12 QSA nel trunk (QSA a indici 3,7,…,47). `blk.48`
è QSA/MoE MTP con gli stessi gruppi HC/shared/indexer, gate/up routed **Q4_K**
`[E,F,Ne]`, down routed **MXFP4** `[F,E,Ne]`. A ne inventaria e valida i
descrittori, senza renderlo parte del percorso ordinario. I sei tensori extra:

| Nome in `blk.48.` | Forma | Dtype |
|---|---|---|
| `nextn.eh_proj.weight` | `[2*E,E]` | Q8_0 |
| `nextn.enorm.weight` | `[E]` | F32 |
| `nextn.hnorm.weight`, `nextn.hc_head_norm.weight` | `[X]` | F32 |
| `nextn.hc_head_down.weight`, `nextn.hc_head_up.weight` | `[X,R]`, `[R,X]` | F16 |

Conteggio verificato, utile per l'accettazione del **checkpoint reale**, non
per limitare le fixture sintetiche:

| Dominio | BF16 | F16 | F32 | Q8_0 | IQ2_XXS | Q2_K | Q4_K | MXFP4 | Totale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Generazione ordinaria residente | 1 | 290 | 461 | 327 | 96 | 48 | 0 | 0 | 1223 |
| MTP | 0 | 8 | 11 | 10 | 0 | 0 | 2 | 1 | 32 |
| PLE disco | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

Il planner attuale rende residenti anche i 32 tensori MTP: A non promette un
risparmio di memoria solo perché non li esegue. Un'eventuale esclusione dal
planner è modifica del coordinatore/storage, con nuovi test dei segmenti.

### 3.3 Indirizzi, stride e padding

**P.** Per una matrice GGUF `[Kp,N]`, la vista logica C è `W[N,K]`; per
`[Kp,N,Ne]` è `W[Ne,N,K]`. La relazione di una proiezione è:

```text
Y[t,n] = sum(k=0..K−1) X[t,k] * decode(W[n,k])
row_bytes = (Kp / block_elems) * block_bytes
expert_bytes = N * row_bytes
block_offset(e,n,k) = tensor.file_offset + e*expert_bytes
                    + n*row_bytes + floor(k/block_elems)*block_bytes
```

`k % block_elems` va al decoder B. Per i formati scalari `block_elems=1`.
Non esiste uno stride scalare intero in byte per IQ2_XXS o Q2_K: niente
`data[k*sizeof(float)]`, niente `nelem*2/8`. `row_bytes` e `expert_bytes`
sono sempre fisici; le dimensioni del calcolo restano logiche. Con più
dimensioni, gli stride esterni si ottengono moltiplicando le forme fisiche
precedenti con overflow checked. L'indirizzo Metal sostituisce `file_offset`
con `GgufBinding.offset` nel buffer del relativo owner.

Esempio vincolante Q2_K down: `K=640`, `Kp=768`, blocchi da 256 elementi/84
byte; `row_bytes=252`, `expert_bytes=645120`, totale 330301440 byte per layer.
Il terzo blocco contiene 128 ingressi logici e 128 posizioni padded: **non si
può troncare il blocco**. Il decoder B legge il blocco intero; la riduzione
usa solo 640 ingressi, oppure F/B azzerano esplicitamente `X[640:768]` in
uno scratch. Non assumere che i pesi quantizzati di padding valgano zero.
Gate/up IQ2_XXS: `row_bytes=10*66=660`, `expert_bytes=422400`.

Per la tabella PLE l'indirizzo è `file_offset + row_id*320`, con
`file_offset=44806635520`; dimensione 102400491520 byte. Il limite fisico è
320001536 righe, ma un ID prodotto dall'hash deve appartenere al range della
sua testa, nell'unione `[0,320001446)`: il padding non è un lookup valido.
A fornisce solo la descrizione; E possiede letture, staging e invalidazione.

### 3.4 Convenzioni semantiche già applicate ai pesi

**V.** R7 usa le seguenti convenzioni; sono parte del contratto GGUF, non
trasformazioni da riapplicare al caricamento:

- GDN: `qkv=[q(Gk),k(Gk),v(Gv)]`. Le teste value GGUF sono tiled: la testa
  `j` usa la key `j % gdn_k_heads`. Rispetto all'ordine HF indicato dal
  convertitore, `p(j)=(j % gdn_k_heads)*(gdn_v_heads/gdn_k_heads)
  + floor(j/gdn_k_heads)`. La stessa permutazione interessa righe di v/gate,
  alpha/beta, `ssm_a`, dt, canali v della convoluzione e colonne di `ssm_out`.
  A conserva l'ordine già scritto; C lo consuma. Una sola permutazione in
  meno o in più non viene scoperta dal controllo delle forme.
- `ssm_a` è già `−exp(A_log)` in F32 secondo `prod_ssm_a`; non è `A_log`.
  I coefficienti norm sono usati direttamente da `qwen4_ref_rms`: il binding
  non aggiunge `1` ai pesi. C deve confermare le trasformazioni contro la sua
  fonte indipendente prima dei test su pesi reali.
- QSA `attn_q` produce per ogni testa **query poi gate**, layout
  `[T,q_heads,2,key_dim]`, non `[T,2,q_heads,key_dim]`;
  `qwen4_ref_attention` legge `h*2*key_dim`. D possiede split e sigmoid.
- L'indexer GGUF contiene due proiezioni già separate; il convertitore divide
  `index_qk_proj` HF in 512 righe query e 128 key. Non ricomporle per errore.
- Conv GGUF `[kernel,channels]` significa tap contigui per canale:
  `w[channel*kernel+tap]`. Non trasporre in `[tap,channel]` durante il binding.
- HC: `qwen4_ref_hc_mix` divide l'uscita della down per H prima della SiLU,
  fa la media su H stream e divide l'inject per H prima di `2*sigmoid`.
  **Aperto con C:** il docstring `prod_hc_mix_f32` parla di `pack/4`, mentre
  il codice e `N_HC_DIVISOR` moltiplicano il pack per 4. Il runtime è chiaro;
  la relazione con il BF16 originale richiede prova indipendente. Vietato
  scegliere una scala dal commento o applicare un ulteriore fold in A.

La spec C/D/E/F deve fissare le equazioni operative e i layout di attivazioni
e stato. Queste convenzioni non autorizzano A a implementarne una seconda copia.

### 3.5 Tokenizer, ID e template

**V.** R5 è BPE senza dropout/UNK/prefix/suffix, `byte_fallback=false`,
`ignore_merges=false`, normalizzazione NFC, ByteLevel senza spazio iniziale.
Il pre-tokenizer applica in ordine questa alternanza, prima del ByteLevel:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

**P.** Identificare il profilo con `tokenizer.ggml.model=gpt2` e
`tokenizer.ggml.pre=qwen35`; stringhe diverse richiedono un profilo verificato.
Il merge BPE seleziona la coppia adiacente col rank minimo, tie a sinistra;
nessun merge attraversa pre-token o added token. Riutilizzare questa primitiva
dal tokenizer Qwen3, non clonarla in `qwen38_tokenizer.c`.

La differenza `\p{M}` include Mn/Mc/Me, non solo combining class nonzero.
Una sequenza che comincia con un singolo Mark può richiedere backtracking
dell'optional prefix: la trascrizione meccanica dello splitter Qwen3 non basta.
Mantenere i singoli `\p{N}`, gli spazi finali e CR/LF del pattern esatto.
Provare anche case folding delle contrazioni contro il backend, senza assumere
che ogni `(?i:...)` equivalga a `tolower` ASCII.

**V.** Confronto integrale del metadata reale con R5:

- 248044 token BPE e 247587 merge identici per ID/rank/spelling;
- 33 added token, ID 248044…248076, spelling identico;
- GGUF ha 248320 voci: aggiunge 243 `[PAD248077]`…`[PAD248319]` di tipo UNUSED;
- tipi GGUF: NORMAL 248044, CONTROL 27, USER_DEFINED 6, UNUSED 243;
- `tokenizer.chat_template` identico byte per byte a R5. SHA-256 del testo:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

**P.** Separare `model_vocab_size=248320`, vocabolario BPE, added tokens e
bitmap di ID effettivamente decodificabili. Gli UNUSED non entrano nel BPE né
nel matching added; la stringa letterale `[PAD248077]` resta testo ordinario.
Non mascherare i logits di tali ID in A: sarebbe una modifica della generazione.

Gli added token hanno tutti `normalized=false`, `single_word=false`,
`lstrip=false`, `rstrip=false`. Estrarli sul testo originale, poi NFC sui
segmenti ordinari: evitare composizioni attraverso il loro confine. Per
parità encode vanno riconosciuti sia CONTROL sia USER_DEFINED, compresi
`<think>`, `</think>` e tag dei tool. `skip_special_tokens` è una decisione
di decode distinta: HF marca **non special** anche i sei token FIM/repository
248060…248065, mentre GGUF li marca CONTROL. Conservare la maschera HF nel
profilo verificato o ricavarla dal tokenizer config validato; non usare
semplicemente `token_type==CONTROL`. Tutte le 33 proprietà vanno in fixture.

Input pubblico UTF-8 con lunghezza esplicita, anche NUL interno; UTF-8 malformato
è errore documentato (HF accetta stringhe Unicode, non byte arbitrari). Decode
grezzo concatena i byte dei token; decode UTF-8 applica sostituzione U+FFFD
come ByteLevel **dopo** la concatenazione. In streaming mantenere fino a tre
byte UTF-8 incompleti e sostituire solo quando invalido/alla fine; non decodificare
ogni token isolatamente. `decode_utf8` canonico omette gli UNUSED, come il
backend HF osservato per 248077/248319; ID negativi o >= model vocab sono
errore, differenza intenzionale rispetto a HF che ignora anche ID arbitrari.
L'API raw può esporre la classificazione UNUSED per la diagnostica, senza
stampare il placeholder come testo generato ufficiale.

ID distinti e provenienza:

| Ruolo | Valore osservato e regola |
|---|---|
| `endoftext`, padding, seed della storia PLE | 248044 |
| `im_start`, `im_end` | 248045, 248046 |
| BOS metadata | 248044 ma `add_bos_token=false`; HF tokenizer `bos_token=null` |
| Arresto della generazione testuale | `[248046,248044]` da R4 generation config; non fermarsi su `</think>` |
| Parametri di sampling dichiarati | temperature=1, top_k=20, top_p=0,95, do_sample=true; profilo da R4, non confondere con greedy della baseline |

Il GGUF non porta tutta la generation config: registrare la provenienza dei
default del profilo; un'eventuale generation config esplicita deve essere
validata e un override utente deve apparire nel manifest. A consegna gli ID di
stop, senza implementare sampling o cambiare la policy del coordinatore.

**P. Template nativo specializzato**, non interprete Jinja generale. Accettare
solo il template riconosciuto per uguaglianza dei byte/hash (o una revisione
aggiunta con fixture); su template diverso errore esplicito. Le costanti di
testo, comprese istruzioni reasoning/tools, derivano da R5 e sono versionate.

Il contratto delle opzioni distingue `enable_thinking` omesso/true/false,
effort omesso/xhigh/medium/low, `preserve_thinking` omesso/true/false e
`add_generation_prompt`. Semantica richiesta dal template letto:

1. Nessun messaggio: errore. System può essere solo iniziale. Deve esistere
   una query user il cui contenuto trimmed non sia interamente racchiuso in
   `<tool_response>…</tool_response>`; ricercarla dalla fine.
2. Thinking omesso/true: effort default xhigh; istruzioni system esatte per
   xhigh/low, nessuna istruzione aggiunta per medium. Effort non ammesso è
   errore quando thinking è attivo; con false il template non lo consulta.
3. Il system testuale e il contenuto dei messaggi vengono `trim` come Jinja/
   Python; non confondere questo insieme Unicode con `\p{White_Space}` del
   tokenizer. Con system vuoto e thinking=false il turno system è omesso.
4. User: `<|im_start|>user\n{content}<|im_end|>\n`. Generation prompt:
   `<|im_start|>assistant\n<think>\n` se thinking; con false finisce in
   `<think>\n\n</think>\n\n`. Con `add_generation_prompt=false` nessun suffisso.
5. Assistant storico: `reasoning_content` è un campo separato, trimmed; il
   blocco think è preservato per default e dopo l'ultima query reale anche con
   `preserve_thinking=false`; prima di quella query può essere rimosso. Non
   estrarre automaticamente reasoning da `content`: R5 non lo fa.
6. Tools: prefisso system con schema JSON e istruzioni esatte di R5; serializzare
   con la stessa semantica `tojson` di Transformers/Jinja fissati. Conservare
   ordine delle chiavi e parametri, escape, separatori e valori null/bool/numeri.
   `transformers/utils/chat_template_utils.py` nel venv sostituisce il filtro
   Jinja con `json.dumps(ensure_ascii=False, sort_keys=False)`, separatori
   predefiniti: niente escaping HTML aggiuntivo, virgola/spazio e due punti/
   spazio. Questa implementazione fissata è il riferimento, non il `tojson`
   Jinja generico. Numeri JSON finiti, preservando distinzione int/float;
   oggetti con chiavi duplicate e NaN/Inf sono rifiutati dall'API nativa.
   Tool call assistant come `<tool_call>\n<function=…>\n<parameter=…>…`, non
   JSON libero; primo separatore dipende dalla presenza di `content`. I tool
   response consecutivi sono raggruppati in un turno user. Parità degli errori
   sul sottoinsieme supportato, non reinterpretazione dei casi malformati.
7. Supportare contenuto stringa/null e liste di parti testuali; immagini,
   video, audio o tipi sconosciuti ricevono `UNSUPPORTED`, prima di mutare
   output o storia. Ruoli riconosciuti: system/user/assistant/tool.

Il rendering restituisce testo; l'encode avviene una volta sul prompt completo
senza BOS automatico. **Non** concatenare tokenizzazioni indipendenti dei
messaggi: normalizzazione e confini BPE non sono additivi. Nell'append il
coordinatore confronta gli ID del nuovo prompt renderizzato con quelli della
sessione e riusa solo il prefisso identico; eventuale replay/reset di C/D/E è
responsabilità del grafo. Un cambio thinking/template invalida la fingerprint.

## 4. API C11 proposte, ownership ed errori

Le dichiarazioni seguenti sono un contratto da materializzare dal coordinatore
in `src/qwen38.h`. Non sono header già esistenti. Niente tipo Objective-C o
oggetto Metal nel contratto condiviso.

```c
typedef enum {
    Q38_OK, Q38_MISSING, Q38_TYPE, Q38_RANGE, Q38_SHAPE,
    Q38_FORMAT, Q38_UNSUPPORTED, Q38_CAPACITY, Q38_OOM, Q38_OWNER
} Qwen38Status;
typedef struct {
    Qwen38Status code;
    char subject[128];             /* chiave o nome del tensore */
    char detail[256];              /* atteso/ottenuto; nessun payload */
} Qwen38Error;
typedef struct Qwen38Config Qwen38Config;
typedef struct Qwen38Weights Qwen38Weights;
typedef struct Qwen38Tokenizer Qwen38Tokenizer;

typedef enum { Q38_TRUNK, Q38_MTP, Q38_DISK_PLE } Qwen38Domain;
typedef enum {
    Q38_VECTOR, Q38_MATRIX, Q38_EXPERTS, Q38_CONV,
    Q38_GDN_TILED, Q38_Q_GATE_INTERLEAVED, Q38_DISK_ROWS
} Qwen38Layout;
typedef struct {
    GgufTensor storage;            /* copia, shape fisica invariata */
    uint64_t logical_shape[4];     /* stesso ordine GGUF; non byte */
    uint64_t block_elems, block_bytes;
    uint64_t row_bytes, plane_bytes; /* plane per expert nei rank 3 */
    Qwen38Domain domain;
    Qwen38Layout layout;
    const GgufBinding *gpu;        /* NULL finché non attached, sempre NULL PLE */
} Qwen38Tensor;

Qwen38Status qwen38_config_load(const Gguf *, Qwen38Config **, Qwen38Error *);
void qwen38_config_free(Qwen38Config *);
Qwen38Status qwen38_weights_describe(const Gguf *, const Qwen38Config *,
                                   Qwen38Weights **, Qwen38Error *);
Qwen38Status qwen38_weights_attach(Qwen38Weights *, const GgufPlan *,
                                 Qwen38Error *);
void qwen38_weights_free(Qwen38Weights *);
Qwen38Status qwen38_tokenizer_load(const Gguf *, const Qwen38Config *,
                                 Qwen38Tokenizer **, Qwen38Error *);
void qwen38_tokenizer_free(Qwen38Tokenizer *);
```

**P. Dati canonici:** `Qwen38Config` è immutabile dopo costruzione; contiene
tutti i campi di §3.1, array per layer `{kind, compress_ratio, has_ple, domain}`,
ID/token policy e identificatore del profilo semantico. Campi dimensionali
`uint32_t` dopo controlli; offset, conteggi byte e costanti hash `uint64_t`.
Epsilon/theta `float` con bit F32 del GGUF. Esporre accessor `const` per
dimensioni e layer, nessuna copia del config contenente puntatori.

`Qwen38Weights` possiede i record `Qwen38Tensor` e gruppi tipizzati globali,
HC, GDN/QSA, MoE, PLE per layer; i gruppi riferiscono indici al catalogo,
non puntatori a payload. Ruoli e combinazioni legali seguono integralmente
§3.2; il lookup nome→indice è solo setup. Non cercare nomi durante ogni token.
L'accessor per `(layer,role)` restituisce `const Qwen38Tensor *` borrowed
fino a `weights_free`; un ruolo assente restituisce MISSING, non un tensore zero.

Per rank 1, `row_bytes=storage.bytes`, `plane_bytes=storage.bytes`; per rank 2
`row_bytes` come §3.3, `plane_bytes=storage.bytes`; per rank 3 `plane_bytes`
è lo stride expert. Le sole dimensioni inutilizzate di `logical_shape` valgono
1; `storage` resta una copia invariata del descrittore. Viste non contigue aggiuntive
non sono supportate in questa prima API: slice per expert mantiene il
descrittore base e passa `expert_id`, con offset checked da B.

La proiezione vettoriale shared gate `[E]` è un caso dichiarato, non un reshape
arbitrario. Non esporre solo un generico `void *weights`: preservare tipo e
provenienza in ogni passaggio. `layout` qualifica la semantica; `storage.type`
resta il GGUF type ID originale, distinto da `DType` e dai formati row.

**Durata:** config, tabelle tokenizer e descrittori copiano ciò che serve;
nessun puntatore nell'header sopravvive per errore a `gguf_close`. L'header
borrowed resta confinato alla costruzione. `weights_describe` non apre Metal,
non legge payload, non dipende dalla presenza dei decoder B.

`weights_attach` valida una corrispondenza completa con il piano: identità del
checkpoint, nome, rank/forma/type, range file e offset nel buffer; tutti i
record residenti devono riferire lo stesso `GgufPlan` atteso. Il piano ha
indirizzo stabile e non viene copiato. Attach è atomico: su errore nessun
binding parziale pubblicato. PLE resta disk-only e coincide con `plan.disk_table`.
Due file con stessi descrittori non sono identità sufficiente: il coordinatore
deve mantenere l'apertura dal medesimo `Gguf` oppure aggiungere un'identità
immutabile a store/plan (device/inode/size/mtime, hash registrato nel manifest).

Ordine ordinario di teardown: fermare dispatch e completare GPU → liberare
viste A → chiudere store/plan → chiudere lettore, ove ancora aperto. I buffer
retained possono sopravvivere allo store come nei test esistenti, ma ciò non
mantiene vivi i **puntatori C** ai binding. Nessuna API A può usare il piano
dopo free. E duplica il descriptor o possiede una propria apertura read-only
con identità verificata prima che il `Gguf` venga chiuso. A non chiude FD di E.

**P. API testuale:** usare `Qwen38Bytes { const uint8_t *data; size_t len; }`,
ID `int32_t`, array e lunghezze esplicite. `qwen38_encode`,
`qwen38_decode_utf8`, `qwen38_chat_render`
ricevono buffer del chiamante, capacità e `size_t *needed`; una chiamata con
output NULL/cap 0 misura la capacità necessaria. Su CAPACITY nessun output
parziale è utilizzabile. `needed` esclude NUL finale, richiesto solo dal
convenience wrapper; mai usare `strlen` sul contenuto utente. Stesso contratto
per `qwen38_decode_bytes`, con opzione skip-special applicata tramite maschera
R5. Le capacità encode sono in **token**, quelle decode/render in **byte**.
Le chiamate di misura hanno successo con `Q38_OK` e non mutano stato. Una
chiamata con buffer insufficiente restituisce CAPACITY e `needed` completo.

Il template riceve array borrowed di messaggi tipizzati (ruolo, parti testuali,
reasoning, tool calls e parametri ordinati) e opzioni; gli oggetti JSON per
tools/argomenti conservano tipo, ordine e valore. Il parser dell'entry point
è del coordinatore: A possiede il renderer e la serializzazione del sottoinsieme
JSON, senza nuova dipendenza. Le opzioni sono enum, non bool con default
ambiguo. Nessun contenuto viene trattenuto dopo la chiamata.

Tokenizer/config immutabili possono essere condivisi; scratch e decoder
UTF-8 streaming sono per chiamata/sessione. `decoder_reset` azzera i byte
pendenti; `decoder_finish` applica la sostituzione finale e chiude la sequenza.
Template e tokenizer non possiedono cache KV né storia PLE.

**Errori:** le API A restituiscono status e diagnostica deterministica;
`*out=NULL` su fallimento, risorse parziali liberate. Gli errori di input non
producono fallback Qwen3. L'esistente `gguf_open`/planner e vari helper chiamano
`die`: il coordinatore può conservare la terminazione del processo CLI e i
test negativi via subprocess. A non promette di recuperare un errore del
parser GGUF esistente, né modifica in questa fase tutto il modello di errori.

## 5. Dipendenze dagli altri componenti

| Componente | Contratto consegnato da A / decisione da ricevere |
|---|---|
| B | Dtype GGUF, K logico/K fisico, stride di riga/expert, dominio. B possiede conversione/dequant e matmul. Confermare API che gestisce il terzo blocco Q2_K senza leggere attivazioni fuori limite. Q8_0/F16/F32/BF16 necessari anche fuori dal MoE; Q4_K/MXFP4 solo MTP nel checkpoint scelto. |
| C | Config GDN/HC, conv kernel, teste tiled e `ssm_a` già trasformato. C possiede conv comune e layout dello stato. Confermare scala HC e coefficienti norm contro oracle indipendente (§3.4). |
| D | Layer QSA, Q/gate interleaved, dimensioni indexer, budget in token, ratio per layer, RoPE parziale. D conferma split, causalità e cache; solo posizioni testuali, nessun YaRN/mRoPE multimodale implicito. |
| E | Array PLE uint64 verbatim, indice layer zero-based, tabella BF16 disk-only, EOS della storia separato dagli stop. E conferma regola hash/reset, range dei prodotti e layout del vettore concatenato; usa conv di C. |
| F | Ne/top-k/F/Fs e tensori routed/shared; A non assume softmax Qwen3. F fissa routing/pesi e consuma decoder B. Formati supportati determinati dal catalogo, non dal nome commerciale Q2. |
| Coordinatore | Header, opzioni CLI, dispatch per famiglia, budget runtime, lock GPU, gestione vita store/sessioni e integrazione Makefile. Deve materializzare `BASE_CONCORDATA` e i contratti prima di ESEGUI. |

Nessun test A richiede operatori B–F finti. Config, binding descrittivo e
tokenizer/template possono essere verificati in CPU prima dei kernel. La
composizione con B–F e la generazione sono gate separati del coordinatore.

## 6. File posseduti e modifiche condivise richieste

**In questa sessione l'unico file scritto/committato è questo documento.**
Ownership seguente proposta per CONSOLIDA; non è autorizzazione ad anticipare
l'implementazione prima della base comune.

| Proprietario proposto | File esatti | Intervento |
|---|---|---|
| A | `src/qwen38_config.c` | Accessor metadata interni, configurazione e validazioni semantiche |
| A | `src/qwen38_weights.c` | Catalogo tipizzato, forme logiche, attach al planner; nessun decoder |
| A | `src/qwen38_tokenizer.c` | Import GGUF, profilo qwen35, classificazione ID e adattatore al BPE comune |
| A | `src/qwen38_chat.c` | Renderer testuale Qwen3.8 e serializzazione JSON necessaria ai tool |
| A | `src/qwen38_unicode.c`, `tools/gen_qwen38_unicode.py` | Eventuali tabelle specifiche qwen35 generate e versionate; non duplicare NFC/BPE |
| A, previa assegnazione unica | `src/tokenizer.c` | Estrarre accesso al core comune, selezione profilo esplicita, import e decode necessari; default Qwen3 invariato |
| A | `tests/test_qwen38_config.c`, `tests/test_qwen38_bindings.c`, `tests/test_qwen38_text.c`, `tests/test_qwen38_a.py` | Harness CPU e driver oracle/negative tests |
| A | `tools/oracle/qwen38_config.py`, `tools/oracle/qwen38_text.py`, `tools/synth/make_qwen38_a.py` | Oracle config HF, uso diretto del tokenizer/template ufficiali, fixture piccole |
| A | `tests/fixtures/qwen38/a/manifest.json`, `tests/fixtures/qwen38/a/config.json`, `tests/fixtures/qwen38/a/text.json` | Metadati/attesi piccoli, seed e hash; nessun peso completo né copia integrale del tokenizer |
| Coordinatore | `src/qwen38.h` (nuovo), `src/picoforge.h` | Materializzare contratto A/B–F; dichiarazioni minime per BPE/profili senza sovrascrivere Qwen3Config |
| Coordinatore | `src/gguf.h`, `src/gguf_plan.c`, `src/gpu_weights.m` | Solo se necessaria identità del piano/filtraggio MTP; conservare ownership esistente |
| Coordinatore | `Makefile`, `src/main.c`, `src/generate.c` | Build/test, Python riusabile via variabile, selezione nuova famiglia, wiring chat/stop; `chat_format` Qwen3 rimane disponibile |
| Coordinatore | `README.md`, `docs/DESIGN.md`, `docs/BENCHMARKS.md`, report aggregati, `docs/specs/qwen38/integration.md` | Stato integrato, risultati complessivi, protocollo lock |

Non assegnati ad A: `src/config.c`, `src/model.c`, `src/gpu_forward.m`,
`src/kernels.metal`, `src/normalize.c`, `src/unicode_tables.c` e test Qwen3.
Se emergono correzioni indispensabili in questi file, proporre diff concreto
al proprietario; non applicarle per comodità. Se il coordinatore sceglie un
header privato comune al tokenizer, deve fissarne nome e owner in integration,
evitando dipendenze circolari con `qwen38.h`.

## 7. Oracle indipendenti, fixture e slice reali

**Config:** R4 originale è il riferimento indipendente per significato e
dimensioni; il parser Python standard `json` deve selezionare esplicitamente
`text_config`. Confrontare la configurazione C normalizzata con quella HF,
documentando campi solo-GGUF, trasformazioni PLE/MTP, tipo F32 e provenienza
dei default. L'inventario Python preesistente è indipendente dal lettore C
per il contenitore, ma una seconda traduzione della tabella A non prova da
sola la semantica del modello.

**Binding:** generare tensori con valori-indice in assi diversi e confrontare
le righe/slice restituite con array NumPy indicizzati come `[Ne,N,Kp]`, più
matrici identità/basi con risultato determinato analiticamente. Includere
permute GDN simboliche (bijettività e associazione key/value), query/gate
asimmetrici e tap conv tutti differenti. Per l'offset low-bit bastano byte
sentinella per blocco e riga; la decodifica reale è test B, non un decoder
Python copiato in A. Il planner C già testato non è da solo l'oracle semantico.

**Tokenizer/template:** usare direttamente `tokenizers.Tokenizer.from_file`
e `PreTrainedTokenizerFast.apply_chat_template` con R5. Nessuna riscrittura
Python del BPE/splitter o del template nativo da confrontare con se stessa.
Salvare testo UTF-8 completo, ID, decode completo e opzioni, oltre agli hash.
Il confronto tabelle deve coprire ogni ID e merge; i cinque prompt esistenti
sono regressione utile ma insufficienti per casi limite, tools e multi-turno.

Fixture sintetiche necessarie:

1. Config tiny con E/F/Fs diversi dai reali, forme compatibili con blocchi
   quantizzati, almeno un GDN e un QSA; variante senza MTP e con PLE a indice
   diverso da 1. Nessun kernel può dipendere dai numeri reali. Formati e policy
   unsupported devono fallire esplicitamente, non accettare una semantica diversa.
2. Missing key, tipo stringa al posto di numero, intero signed negativo,
   overflow in prodotti/arrotondamento, zero divisore, NaN/Inf, array corto/lungo,
   PLE duplicato/fuori range, zero vocab, offset non contigui, mismatch HF/GGUF,
   incoerenza n_layers/n_mtp e QSA con tensori GDN. Includere tabella PLE
   arrotondata correttamente e una che espone il padding come vocab di testa.
3. Nomi mancanti/duplicati/inattesi, rank/shape/type errati, padding attribuito
   a shared/MTP, piano di altro owner, offset/range alterati; PLE tra due viste
   residenti. Ereditare le fixture del planner senza modificarne i test.
4. Caso routed down `K=640,Kp=768`, guard page/canary dopo le 640 attivazioni,
   padding avvelenato e ultima riga/ultimo expert. A verifica gli indirizzi;
   B/F verificano in seguito che padding non contribuisca al prodotto.
5. Testi italiano/inglese/codice, digit singoli, hash nelle merge, CR/LF,
   whitespace Unicode/finale, emoji/ZWJ, NFC/NFD/Hangul, categorie Mn/Mc/Me,
   Mark iniziale/isolato, contrazioni case-insensitive, NUL e input vuoto;
   token added adiacenti a spazi/Mark, tutti i 33 added, letterali UNUSED.
6. Decode byte incompleti, tutti i tag FIM/tools/thinking, skip true/false,
   streaming diviso a ogni confine byte di sequenze UTF-8, reset/finish;
   encode concatenato confrontato con prompt intero, non assunto incrementale.
7. Template senza system, system vuoto/nonvuoto, trim ASCII/Unicode, tutti gli
   effort e flag omissione/true/false, più turni e reasoning preservato/rimosso,
   tools JSON annidati e ordinati, più call e response consecutive, null/testo
   multipart, input invalidi e contenuti multimodali rifiutati. Testare errori
   del riferimento come errori attesi, non output vuoto valido.

**Slice reali necessarie:** header/inventario completo e tokenizer R5 bastano
per l'accettazione autonoma A; nessun layer intero e nessun caricamento BF16.
Per integrazione con B/C/D/E/F servono slice minime, versionate per offset,
lunghezza e hash: righe/expert 0 e ultimo di routed down/gate; righe Q/gate
di due teste; vettori `ssm_a`/dt/alpha e canali conv; righe HC e indexer;
prime/ultime righe PLE per testa. Budget proposto <=16 MiB di payload totale,
non automaticamente autorizzato durante SPEC; letture future sotto protocollo
condiviso quando richiesto. Gli expected numerici devono provenire dagli
oracle B–F e da fonti indipendenti delle trasformazioni, non soltanto R7.

La corrispondenza a un oracle che usa **questo Q2** misura l'implementazione.
La perdita rispetto a BF16 richiede pesi originali comparabili ed esperimento
separato: nessuna parità tokenizer, inventario o dequant Q2 la quantifica.

## 8. Tolleranze e criteri di accettazione

| Oggetto | Criterio proposto e motivazione |
|---|---|
| Metadati interi, ID, rank, offset, stride, domini, ordine layer | Uguaglianza esatta; qualsiasi differenza cambia indirizzi o algoritmo |
| Float del config | Bit F32 uguali ai byte GGUF; per HF confrontare dopo conversione round-to-nearest a F32. `1e-6` decimale e 9.999999974752427e-7 non sono una divergenza |
| Vocabolario e merge | Zero mismatch su tutte le voci/rank del riferimento, con padding e maschera special verificati separatamente |
| Encode/template/decode UTF-8 | Zero mismatch su ID, lunghezze e byte, inclusi newline/trim; nessuna tolleranza “testo equivalente” |
| Binding/sentinel/copy di scalari | Bit/byte esatti; il binding non fa arithmetic numerica né conversione lossy |
| Slice numeriche attraverso B–F | Tolleranze dei proprietari concordate in integration, fissate prima dei test; A non ne introduce di alternative |
| Regressione Qwen3 | Aspettative esistenti invariate e test pertinenti verdi; non aggiornare goldens per assorbire una regressione |

A è accettabile quando il report canonico del checkpoint classifica tutti i
1256 tensori (1223 ordinari residenti, 32 MTP, una tabella disco), ogni binding
necessario è validato prima dell'uso, tutti i negativi falliscono chiaramente,
il tokenizer/template passa l'oracle indipendente sullo scope testuale completo
e i controlli CPU sotto ASan/UBSan non riportano errori. Header/tokenizer
possono essere chiusi/liberati nell'ordine dichiarato senza dangling borrow.

L'eventuale gate Metal dell'attach è un test storage sotto lock, con buffer
piccoli e MTL validation; non prova il forward. Nessuna metrica di performance,
qualità di coding o greedy end-to-end fa parte dell'accettazione A isolata.

## 9. Incrementi di implementazione verificabili

1. **Consolidare e congelare il contratto.** Il coordinatore risolve §10,
   assegna file/header/lock e comunica la base comune. Nuovo worktree ESEGUI
   da quella base; questa spec non sostituisce integration.
2. **Oracle config/testi prima del C.** Fixture da R4/R5 con hash, casi negativi
   e corpus testuale. Verifica che il runner non carichi `AutoModel`, pesi o GPU.
3. **Accessor e config C.** Tipi stretti, dimensioni/array derivati, riepilogo
   canonico; confronto HF e ASan/UBSan. Nessun binding GPU ancora.
4. **Catalogo descrittivo.** Un gruppo alla volta: globali/HC, GDN, QSA,
   routed/shared, PLE, MTP solo inventory. Sintetici prima, header reale poi;
   forme logiche e stride testati senza decoder. Attach separato al planner.
5. **Core tokenizer condiviso e import.** Preservare Qwen3 ad ogni estrazione,
   aggiungere importer GGUF e profilo qwen35, controllare tutte le tabelle,
   poi corpus NFC/Mark/special e decode. Nessuna nuova libreria C.
6. **Template.** Prima single-turno/thinking, poi cronologia/preserve_thinking,
   infine tools/JSON e casi di errore; ogni incremento confronta byte e ID con
   Jinja originale. Non dichiarare parità completa prima dell'ultimo gruppo.
7. **Integrazione del coordinatore.** Wiring CLI/API senza cambiare Qwen3;
   controlli storage/Metal sotto lock; regressioni generali e composizione
   con B–F quando disponibili. Il completamento di A non abilita da solo
   la generazione Qwen3.8.

Ogni incremento deve avere commit piccolo con concetto, comando e risultato
effettivo. L'ordine non richiede né wrapper DwarfStar né stub di operatori.

## 10. Verifiche, budget, rischi e decisioni aperte

### Verifiche effettive di questa sessione SPEC

- Stato principale iniziale pulito, staging vuoto, HEAD uguale alla base
  indicata; branch/worktree dedicato creato senza reset, stash o modifiche altrui.
- R1/R2 letti; ricerca pertinente e codice/config di R4–R7 ispezionati.
- Identità file confrontata con verifica archiviata; metadati token letti in
  sola lettura, confronto integrale con R5 con esito positivo (§3.5).
- Ricalcolati gli hash dei due file tokenizer locali e del template.
- Ripetuto il controllo HF CPU dei cinque prompt rispetto ai token DwarfStar
  archiviati: **5/5 identici**, lunghezze 96, 126, 173, 142, 7614. Sono prompt
  token checks, non una nuova esecuzione del modello.
- Probe CPU del riferimento: FIM/think restano con skip-special=true mentre
  endoftext sparisce; UNUSED decodificano in vuoto; `A\x00B` produce
  `[32,188,33]`; thinking default aggiunge system xhigh, medium no.
- Nessun benchmark, test GPU, modello completo, payload o nuova dipendenza.
  I test del futuro componente C non sono eseguibili: il componente non esiste.
- Documento controllato con parser Markdown: dieci sezioni richieste,
  blocchi di codice bilanciati, collegamenti locali risolti e whitespace
  verificato. Verificata anche la differenza tra righe PLE hash e padding.

Comando **esistente**, eseguito con output solo su stdout nella repository
principale, senza crearvi file:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 tools/venv/bin/python -B \
  tools/eval/check_qwen38_tokens.py \
  bench/qwen38_q2_20261011_phase0/suite \
  build/qwen38-tokenizer.json build/qwen38-tokenizer-config.json \
  --pilot bench/qwen38_q2_20261011_phase0/pilot
```

Comandi **proposti per ESEGUI**, da aggiungere al Makefile dal coordinatore;
questi target/variabili non esistono alla base R1:

```sh
PF_MAIN=/Users/cristiancirje/Desktop/Dev/picoforge
PF_PY="$PF_MAIN/tools/venv/bin/python"
make test-qwen38-a-cpu PYTHON="$PF_PY" QWEN38_REFERENCE_ROOT="$PF_MAIN" \
  SANITIZERS=address,undefined
make test-qwen38-a-header PYTHON="$PF_PY" \
  QWEN38_GGUF="$PF_MAIN/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf"
make test-nfc test-tokenizer test-encode PYTHON="$PF_PY" \
  QWEN3_REFERENCE_ROOT="$PF_MAIN"
git diff --check
```

`test-qwen38-a-cpu` deve compilare solo i sorgenti C necessari, senza prerequisito
`picoforge`/Metal, eseguire sintetici/negativi e oracle text, con output sotto
`build/qwen38/a/` del worktree. `test-qwen38-a-header` accede al solo header del
checkpoint read-only. Adeguare i runner Qwen3 per separare riferimenti read-only
e output locali è compito del coordinatore se necessario; i comandi sopra non
presumono che le variabili siano già supportate. Non copiare pesi o venv.

Il coordinatore eseguirà `make`, `make test-all`, test attach/storage e
composizione quando appropriati. **Lock GPU comune ancora non definito:**
nessuna sessione deve inventare un lock nel proprio worktree. Il protocollo
in integration deve specificare pathname assoluto condiviso, primitive di
acquisizione/rilascio, timeout e comportamento su crash. Qualunque test GPU
e carico pesante sul checkpoint lo acquisisce; benchmark con GPU esclusiva.
Finché manca, A può completare soltanto prove CPU piccole/header.

### Budget proposto (limiti di progetto, non picchi misurati)

| Operazione | Budget e vincoli |
|---|---|
| Costruzione config/catalogo | Header mapping CPU <=128 MiB come loader attuale; config/catalogo e indici aggiuntivi <=4 MiB per questo modello; nessun mapping payload |
| Tokenizer nativo | Tabelle/token/rank/added <=64 MiB; input/rendering <=16 MiB per chiamata, scratch <=256 MiB; cap espliciti e CAPACITY prima di superamenti |
| Test A CPU | Picco processo target <=512 MiB C; oracle Python/Transformers <=1 GiB, da misurare in ESEGUI; una fixture alla volta |
| Fixture/slice | Goldens sintetici piccoli e versionati, intermedi <=64 MiB; slice reali integrazione <=16 MiB totali |
| GPU A | Zero durante SPEC; successivo test attach solo fixture piccola sotto lock, senza resident set completo |
| Runtime globale | Target 48–50 GiB, 8K/chunk 1024 del progetto; A non riserva questo budget e non ignora costo MTP, staging o tokenizer |

Le copie del tokenizer sono MB, non copie del checkpoint. Nel runtime evitare
due vocabolari completi permanenti; stream degli array metadata nell'importer,
poi header liberabile. Lunghe sequenze senza confini possono rendere quadratico
il BPE esistente: imporre budget CPU/timeout nel test e misurare prima di
ottimizzarlo; nessuna troncatura silenziosa. Allocazioni Metal e footprint non
vanno sommate meccanicamente come se fossero categorie disgiunte.

### Decisioni aperte da portare in CONSOLIDA

| ID | Responsabile | Decisione/prova necessaria e impatto |
|---|---|---|
| A-01 | Coordinatore + A | Approvare header/tipi, ownership di `src/tokenizer.c`, nomi target e accessori del core comune; fissare `BASE_CONCORDATA`. Blocca il codice condiviso, non l'oracle indipendente. |
| A-02 | Coordinatore + storage | Come verificare identità GGUF→plan e durata dei binding; mantenere MTP residente inizialmente oppure filtrarlo con test nuovi. Nessuna promessa di risparmio prima della decisione. |
| A-03 | C + A | Risolvere prova indipendente della scala HC, norm e trasformazioni GDN, inclusa contraddizione del docstring R7. Necessaria prima di accettare binding semantico + operatori su slice reali. |
| A-04 | D + A | Confermare Q/gate interleaved, indexer budget in token e profilo RoPE testuale; vietare assunzioni di GQA/posizioni provenienti da Qwen3. |
| A-05 | E + A | Confermare one-based HF→zero-based GGUF, shape/vettore PLE, hash uint64 e EOS storia distinto dallo stop; gli array letti restano immutati. |
| A-06 | A + coordinatore | Congelare maschera special HF, policy UNUSED/decode, fingerprint template e serializzazione `tojson`; scope testuale tools incluso, multimodale rifiutato. |
| A-07 | A | Audit versione categorie Unicode/NFC del backend tokenizers rispetto alle tabelle locali; aggiungere fixture sui confini di versione. Non aggiornare Unicode Qwen3 senza prova di regressione. |
| A-08 | Coordinatore | Pubblicare protocollo unico del lock prima di qualsiasi GPU/heavy I/O. Nessun lock è stato acquisito perché non sono stati fatti tali carichi. |
| A-09 | Coordinatore + B–F | Oracle ufficiale del forward non disponibile nel venv corrente; ottenere fonti/fixture indipendenti senza installazioni implicite. A può verificare il testo con gli strumenti già presenti, ma R7 da solo non certifica la semantica numerica originale. |

Rischi prioritari: usare 49 layer come trunk; confondere PLE EOS con chat stop;
applicare padding a tutti i down; trattare tutti i CONTROL come special HF;
duplicare permutazioni/scaling già applicati; chiudere il piano prima dei
borrow C; derivare shape dal nome del modello; accettare un template diverso
con lo stesso renderer. I test indicati mirano a questi errori prima del
collegamento al grafo completo.

[hf-config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
[hf-generation]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json
[hf-tokenizer]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json
[hf-tokenizer-config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json
[q2]: https://huggingface.co/antirez/qwen3.8-flash-next-gguf/tree/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe
[ds4-code]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c
[ds4-converter]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/qwen4_pack_to_qwen4exp.py
[ds4-guide]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/QWEN38_FLASH_NEXT.md

## 11. Consegna ESEGUI sulla base consolidata

**Verificato il 2026-10-12.** Branch `impl/qwen38-a-v1`, worktree
`/Users/cristiancirje/Desktop/Dev/picoforge-a-v1`, creato dalla base sopra.
Preservati main, `picoforge-spec-a`, `picoforge-a-oracle` e gli altri worktree.
Recuperati soltanto i due commit oracle propri; nessuna nuova acquisizione
della spec o di implementazioni B–F. Nessun merge/push.

### Risultato e contratti effettivi

- `qwen38_config.c`: accessor v1, metadati a tipi stretti, dimensioni e array
  copiati, overflow/shape/profilo/errori verificati. Il profilo esplicito
  registra i parametri assenti dal GGUF; nessun default Qwen3. Riconosciuti
  E=2560, V=248320, 48 trunk (36 GDN/12 QSA), un MTP QSA, PLE al layer 1.
- `qwen38_weights.c`: tutti i 1256 descrittori classificati: 1223 ordinari,
  32 MTP, una tabella PLE disco. Sei ruoli extra MTP restano interni alla v1.
  `storage.shape[0]=768` e `logical_shape[0]=640` per routed down ordinario;
  shared down resta 640. Gain/scaling/permutazioni dei pesi non modificati.
  Lookup per ruolo senza ricerca del nome nel percorso token. Attach atomico
  sul medesimo `Gguf` vivo, `GgufPlan.source` e owner verificati; una seconda
  attach restituisce BUSY. PLE esclusa da tutte le viste Metal.
- `tokenizer.c` mantiene un unico algoritmo BPE e il profilo Qwen3 originale.
  `qwen38_tokenizer.c` implementa import GGUF, qwen35/NFC, NUL, aggiunti prima
  della normalizzazione, skip-special HF, UNUSED vuoti, stop 248044/248046,
  nessun BOS implicito. Decode lossy UTF-8 dopo concatenazione; streaming
  con carry <=3 byte, finish/reset e retry dopo CAPACITY senza mutazione.
- `qwen38_chat.c`: renderer del solo template fissato, confronto letterale
  completo al load, albero JSON ordinato v1, thinking/preserve/effort, tools,
  cronologia e grouping delle risposte. Float finiti serializzati con locale C
  privato e forma shortest-roundtrip; escape e ordine controllati contro HF.
  Schema sconosciuto, duplicati, cicli, nonfinite e multimodale rifiutati.

Config e tokenizer possiedono le copie necessarie: il reader può essere chiuso
dopo la costruzione. I puntatori degli accessor durano fino al free del relativo
oggetto. Il catalogo copia i descrittori, ma richiede il medesimo reader vivo
fino all'attach; successivamente prende in prestito il piano/binding. Piano e
store devono sopravvivere a tutti i consumatori dei binding. Il decoder prende
in prestito il tokenizer. Alberi JSON/input/output sono del chiamante, validi
per la chiamata. Misura con `out=NULL, capacity=0`; nessuna terminazione NUL
implicita. CAPACITY indica output insufficiente; RANGE indica limite di
ammissione. Errori e misura non pubblicano output parziali né avanzano decoder.

**Correzione Unicode verificata:** riutilizzare i dati NFC Unicode 16 di
Python/Qwen3 produceva differenze dal backend `tokenizers==0.22.2` (es. U+0897).
Le tabelle A ora provengono da NFD/NFC e Regex del backend effettivo, con
generator pin e fixture sui confini di versione. Non si è cambiata la tolleranza
né la tabella Qwen3. L'algoritmo nativo deriva da `normalize.c`; l'oracle è
il backend HF originale, non una seconda copia dell'algoritmo C.

### Verifiche passate e riproduzione

Il [manifest](../../../tests/fixtures/qwen38/a/manifest.json) conserva revisioni,
hash dei sorgenti/fixture, versioni, prove pre-base e risultati nativi distinti.
Tutti i confronti testuali sono esatti; float del config confrontati come bit
FP32 dopo conversione del config HF. Nessuna tolleranza numerica B–F modificata.

| Controllo eseguito | Risultato |
|---|---|
| Tre GGUF sintetici (`small`, `no-mtp`, `last-gdn`) e header reale | Config/catalogo/attach CPU, negativi e lifetime passati sotto ASan/UBSan |
| Config HF e inventario indipendente | 37 metadati, tutti gli accessor esportati e 1256 descrittori esatti |
| Testo nativo contro HF congelato | 364 encode, 524 decode (keep/skip e streaming), 36 chat di cui 8 errori attesi; byte e ID esatti |
| NFC esteso | Stream dei 1.112.064 scalar Unicode in cinque chunk identico al backend HF; non prova tutte le sequenze combinatorie possibili |
| Suite Python con tutti gli asset dichiarati | 20 test, zero skip; riproduzione delle fixture con HF originale |
| Storage Metal sintetico sotto lock, API validation | Attach/span CPU/owner estraneo/PLE esclusa passati, fixture <16 MiB |
| Regressione Qwen3 sotto lock comune | 151643 token e 151387 merge identici, encode 30/30, NFC 4017/4017 |
| Build e gate esistenti | `make -j4`; `test-qwen38-contracts` (4 lock test); `test-gguf` (7 test), passati |

Comando autonomo realmente eseguito, dal worktree A (nessun target Makefile
nuovo necessario):

```sh
PF_MAIN=/Users/cristiancirje/Desktop/Dev/picoforge
PF_PY="$PF_MAIN/tools/venv/bin/python"
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 "$PF_PY" -B \
  tools/synth/make_qwen38_a.py --verify \
  --checkpoint "$PF_MAIN/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf" \
  --reference-root "$PF_MAIN"
make -j4
make PYTHON="$PF_PY" test-qwen38-contracts test-gguf
```

Il runner compila i tre harness C11 con i warning del progetto come errori,
`-ffp-contract=off -fsanitize=address,undefined`, legge soltanto header/catalogo
reali, genera il dump config e inventario e confronta i riferimenti. Output
isolati in `build/qwen38-a/`. Per ripetere anche la prova storage già passata,
aggiungere `--gpu` dopo `make`: il runner acquisisce automaticamente il lock
canonico e abilita `MTL_DEBUG_LAYER=1` per la sola fixture piccola. Non avvolgere
questa opzione in un secondo wrapper. Nessun accesso al payload Qwen3.8:
il loader esistente mappa un prefisso CPU di 128 MiB, che comprende anche
byte successivi all'header, ma A non li legge né li rende residenti in Metal.

Regressioni Qwen3 eseguite senza copiare asset o venv, dopo la build locale:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TMPDIR="$PWD/build/qwen38-a" \
  "$PF_PY" -B tools/qwen38_gpu_lock.py --timeout 60 -- "$PF_PY" -B -c \
  'import subprocess,sys; [subprocess.run([sys.executable,"-B",t,"/Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3-0.6B"],check=True) for t in ("tests/test_tokenizer.py","tests/test_encode.py","tests/test_nfc.py")]'
```

### Memoria e limiti misurati

Ammissione runtime: config/catalogo/indici <=4 MiB; import tokenizer <=1 milione
di voci per tabella e blob <=16 MiB (tabelle permanenti complessive <64 MiB).
Encode: input <=4 MiB, segmento NFC <=8 MiB; scratch totale <256 MiB. Il renderer
ammette <=4 MiB di stringhe JSON, 100000 nodi, profondità <64 e output <=16 MiB.
Decode UTF-8: scratch raw <=64 MiB per chiamata; l'output appartiene al chiamante.
Gli estremi di ammissione sono limiti calcolati, non carichi misurati. BPE e
riordino di combining mark possono avere costo quadratico; nessuna promessa
di latenza per input avversari. Profilo iniziale: un sito PLE, GDN Dk=Dv,
ricetta Q2 fissata; varianti non rappresentate sono rifiutate esplicitamente.

Diagnostica singola `/usr/bin/time -l`, senza benchmark prestazionale: harness
testo con ASan/UBSan, RSS 62.930.944 byte, footprint 64.099.920 byte. Suite Python:
inizialmente RSS 1.176.829.952 e footprint 1.081.984.584 byte, oltre il target
1 GiB; riutilizzando le due istanze HF validate, RSS 585.760.768 e footprint
481.166.440 byte. Stessi 20 test verdi, hash ricontrollati a ogni accesso,
contatore swap dei processi zero. Non sono misure della pressione di sistema,
del runtime completo o del fit 48–50 GiB. Nessun modello BF16 caricato o
checksum del file da 137 GiB ripetuto; metadati tokenizer letti: 10.945.596 byte.

### Integrazione e dipendenze residue

Modificati esclusivamente i file A di integration §7: cinque sorgenti
`qwen38_{config,weights,tokenizer,chat,unicode}.c`, `tokenizer.c`, header **privato**
`qwen38_a_internal.h`, generatore Unicode, due oracle, runner sintetico, tre
harness C, test Python, fixture/manifest A e questo documento. Header pubblici,
Makefile, entrypoint, fonti B–F e report aggregati invariati.

Le decisioni A-01/02/04/05/06/08 sono chiuse dal contratto v1; A-07 è verificata
con backend e regressioni qui riportati. Le formule di A-03 sono congelate da
integration; la loro prova numerica con operatori e slice reali, insieme ad
A-09, resta a C–F e al coordinatore. Nessuna API alternativa o stub introdotto.

Al coordinatore restano: target `test-qwen38-a-cpu`/`test-qwen38-a-header`
che invochino il runner con venv/asset esistenti (e un target storage separato
che usi `--gpu`); wiring del grafo/CLI; composizione A+B–F sulle slice e sullo
stato; `make test-all`, logits/generazione, misure globali e qualità. La wildcard
C esistente include già i sorgenti A nella build. Il runner elenca esattamente
i sorgenti per ciascun harness; non è necessario ricopiare un BPE o decoder.
Il lock comune effettivamente usato è
`/Users/cristiancirje/Desktop/Dev/picoforge/.git/qwen38-gpu.lock`.
I controlli A passati non certificano operatori numerici, MTP o perdita Q2/BF16.

Commit di codice da integrare in ordine (lo SHA della consegna documentale è
comunicato separatamente): `5d918bb`, `cb48f84`, `4df72eb`, `c445a5b`, `e54c798`,
`7316db4`, `298fec6`, `71f1368`, `343d8b4`, `16f5add`, `bc1aca6`.
