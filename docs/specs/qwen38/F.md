# F — MoE Qwen3.8-Flash-Next Q2

Stato: SPEC, 2026-10-12. Nessun componente implementato o test numerico
eseguito in questa sessione. Base ispezionata:
`7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`; `BASE_CONCORDATA` ancora da
fissare nel consolidamento. Branch `spec/qwen38-f`, worktree esterno
`/path/to/picoforge-spec-f`.

Le affermazioni marcate **verificato** derivano da letture di codice,
configurazione e inventario; non sono misure di correttezza numerica.
**Proposta** indica contratti da consolidare. **Aperto** indica una decisione
o verifica ancora necessaria. La nuova direzione prevale sulla roadmap Qwen3
del `CLAUDE.md` ignorato, letto nella repository principale.

## 1. Obiettivo e confini

Implementare successivamente il blocco MoE ordinario: proiezione router,
softmax, selezione top-k, pesi, esperti routed, shared expert e loro somma.
Engine C11, Metal per il calcolo GPU, Objective-C esclusivamente per il binding
Metal. Tutte le dimensioni sono ricevute dalla configurazione validata da A.

Il confine è `X[T,H] -> Y[T,H]`: X è già il risultato del mixer/normalizzatore
FFN di C; Y è il contributo FFN, prima della reiniezione nel residuo gated.
F non aggiunge X a Y, non normalizza nuovamente X e non modifica i quattro
rami del residuo. C possiede mixer, normalizzazioni e reiniezione.

Esclusi: dequantizzazione e matmul packed (B), tokenizer (A), attention (D),
n-gram (E), MTP, pruning, modifica del top-k, expert streaming, scheduling
speculativo e benchmark in SPEC. Nessuna copia completa dei pesi.
Qwen3 dense/MoE e l'esperimento `auto-experts` restano invariati.

## 2. Fonti, stato corrente e codice riutilizzabile

### Provenienza verificata

| Fonte | Revisione / evidenza |
|---|---|
| Picoforge | HEAD sopra; repository principale e staging puliti all'avvio |
| DwarfStar locale | `../ds4-qwen38-baseline`, HEAD `fc80bd695da76ee14bcb86820bf900a5cdfda806`, stato pulito |
| Config originale | [config.json][config], revisione HF `de4b8e4d43b917e7706784d8bb445c9af86a3540`, sezione `text_config` |
| Artifact Q2 | `antirez/qwen3.8-flash-next-gguf`, rev. `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe` |
| Checksum completo | `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`, verifica precedente registrata nella baseline; non ricalcolato qui |
| Inventario Python | `bench/qwen38_q2_20261011_phase0/inventory.json`, SHA256 `605d00ad17ae2fa66b10cf64d0dd0c3fac16122bb526dee10db064a0b8a67093` |
| Inventario nativo | `bench/qwen38_q2_20261012_storage/native-inventory.json`, SHA256 `ba4276f7fe1cc5919188ed49025aefe680e92f94512bf7cc9b4787fd36bfb11d` |
| Transformers upstream | [modeling][hf-model] e [configuration][hf-config], commit `812a950d7037ba93bb624b154030e9068afdce4d`, sorgenti letti il 2026-10-12 |

SHA256 dei due file upstream letti: modeling
`f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa`;
configuration `ac0950ab34516d26fe4f6b28cffe0649d763449effe44e42b5913463806f7e0b`.
Questa revisione software non è la revisione dei pesi: va conservata nel
manifest dell'oracle. Non è stato eseguito codice remoto in SPEC.

Riferimenti puntuali DwarfStar alla revisione fissata:

- [ds4.c][ds4-c]: `qwen4_ref_moe` (68391), `qwen4_ref_layer` (68442),
  `qwen4_graph_moe` (59305), `qwen4_graph_forward_tokens`.
- [metal/qwen4.metal][ds4-metal]: `kernel_qwen4_router_topk` (1122),
  `kernel_qwen4_moe_mid` (2752), `kernel_qwen4_moe_down` (3033),
  `kernel_qwen4_moe_reduce` (3319).
- `gguf-tools/qwen4_iq2.py`: packing gate/up, padding down e metadati
  `ds4.qwen4.down.logical_input` / `physical_input` (124–125).

L'upstream HF conferma softmax FP32, rinormalizzazione condizionale, SwiGLU
senza bias e shared expert con gate sigmoid separato. La configurazione
software imposta `norm_topk_prob=True` quando assente; il JSON del checkpoint
non contiene questa chiave. `Qwen4ExpTextDecoderLayer` colloca il MoE tra
lettura e reiniezione del residuo gated. Il report [architetturale][paper]
fornisce contesto; codice e artifact fissati definiscono i dettagli operativi.

### Riutilizzo locale

| Codice attuale | Riutilizzo e limite |
|---|---|
| `src/model.c:230`, `moe_block` | Struttura CPU router/esperti; incorpora il residuo Qwen3, quindi non chiamabile direttamente per F |
| `src/ops.c`, `silu`, `softmax` | Primitive leggibili; verificare stabilità/non-finiti e contratto C prima del riuso |
| `src/kernels.metal:660–717` | Routing deterministico e grouping stabile; non copiare i matmul q8_row |
| `src/gpu_forward.m:407`, `encode_moe` | Esempio di gather/group/scatter; assume un solo buffer, esperti BF16 o q8_row, microbatch massimo 32 |
| `src/gguf.h`, `gguf_plan.c`, `gpu_weights.m` | Binding con owner, formato, offset e viste multiple; non forniscono operatori numerici |
| `tools/oracle/qwen3_forward.py:298`, `tests/test_routing_trace.py` | Convenzioni delle tracce e regressioni; semantica Qwen3 non è oracle Qwen3.8 |

Verificato: `config.c` accetta solo `qwen3`/`qwen3_moe`. I kernel attuali
limitano gli esperti a 512; `PF_MAX_TOPK=64`. Sono limiti del percorso vecchio,
non dimensioni da incorporare nel nuovo modello. La generazione Qwen3.8 manca.

Letti inoltre README, Makefile, DESIGN, BENCHMARKS, i tre documenti di handoff,
ricerca large-model, review, confronto 80–100B e parti sull'esperimento esperti.
La baseline DwarfStar e la validazione storage non provano correttezza di F.

## 3. Semantica matematica e layout

### Parametri e binding verificati

| Simbolo | Config HF / metadata GGUF `qwen4exp.*` | Valore artifact |
|---|---|---:|
| H | `hidden_size` / `embedding_length` | 2560 |
| E | `num_experts` / `expert_count` | 512 |
| K | `num_experts_per_tok` / `expert_used_count` | 10 |
| I | `moe_intermediate_size` / `expert_feed_forward_length` | 640 |
| Is | `shared_expert_intermediate_size` / `expert_shared_feed_forward_length` | 640 |
| Ip | forma fisica down; `ds4.qwen4.down.physical_input` | 768 |
| L | `num_hidden_layers`; GGUF `block_count-nextn_predict_layers` | 48 = 49−1 |

`hidden_act=silu`; rinormalizzazione top-k vera tramite default architetturale
fissato sopra. A deve materializzare il valore e la sua provenienza nella
configurazione, senza cercare una chiave GGUF inesistente o usare il default
Qwen3 falso. `output_gate_type` descrive un'altra parte del modello: non
determina il gate dello shared expert. I e Is sono parametri distinti.

Per ogni `blk.l`, `0 <= l < L`, le forme fisiche GGUF sono:

| Suffisso del tensore | Forma GGUF, asse 0 contiguo | Formato | Vista matematica |
|---|---|---|---|
| `ffn_gate_inp.weight` | `[H,E]` | F32 | router `[E,H]` |
| `ffn_gate_exps.weight` | `[H,I,E]` | IQ2_XXS | gate `[E,I,H]` |
| `ffn_up_exps.weight` | `[H,I,E]` | IQ2_XXS | up `[E,I,H]` |
| `ffn_down_exps.weight` | `[Ip,H,E]` | Q2_K | down `[E,H,Ip]` |
| `ffn_gate_shexp.weight` | `[H,Is]` | Q8_0 | gate shared `[Is,H]` |
| `ffn_up_shexp.weight` | `[H,Is]` | Q8_0 | up shared `[Is,H]` |
| `ffn_down_shexp.weight` | `[Is,H]` | Q8_0 | down shared `[H,Is]` |
| `ffn_gate_inp_shexp.weight` | `[H]` | F32 | gate scalare `[H]` |

Questo schema è stato controllato sull'inventario per tutti i 48 blocchi.
Q4_K compare solo nei due gate/up di `blk.48`; MXFP4 solo nel suo down:
sono MTP nell'artifact selezionato, non dipendenze del MoE ordinario.
Q8_0 è invece necessario subito. B possiede tutti questi formati, distinti
da q8_row/q4_row e dall'INT2 nativo.

Una riga IQ2_XXS gate/up occupa `(H/256)*66 = 660` byte; un esperto
`I*660 = 422400` byte. Down: `(Ip/256)*84 = 252` byte/riga,
`H*252 = 645120` byte/esperto. Queste sono verifiche del layout di B, non
decoder in F. Offset relativo dell'esperto e della riga:
`e*expert_stride_bytes + r*row_stride_bytes`, con overflow e limiti verificati.
Gate e up sono binding separati, non metà adiacenti presumibili nel GGUF;
HF invece li concatena lungo l'asse di output in `gate_up_proj`.

### Equazioni

Per ogni token t, tutti i prodotti seguenti usano X[t], non il residuo a 4H:

```text
z[e] = sum_h Wr[e,h] X[t,h]
p[e] = exp(z[e] - max(z)) / sum_j exp(z[j] - max(z))
s[0:K] = top_k(p), in ordine di probabilità decrescente
a[j] = p[s[j]] / sum_r p[s[r]]           (norm_topk_prob = true)

g_e = Wgate_e X[t];  u_e = Wup_e X[t]
m_e[i] = SiLU(g_e[i]) * u_e[i], 0 <= i < I
m_e[i] = +0.0, I <= i < Ip
v_e = Wdown_e m_e

gs = Wgate_s X[t];  us = Wup_s X[t]
vs = Wdown_s (SiLU(gs) * us)
q = sigmoid(sum_h Ws_gate[h] X[t,h])
Y[t] = sum_j a[j] v_s[j] + q * vs
SiLU(x) = x * sigmoid(x)
```

Se A ammette esplicitamente `norm_topk_prob=false`, `a[j]=p[s[j]]` senza
divisione: coprire il ramo sintetico, ma non usarlo per questo checkpoint.
Non introdurre sigmoid nel router routed, bias di bilanciamento, group top-k,
fattori di scala, clipping SwiGLU, normalizzazione con lo shared o token drop.
Lo shared è sempre eseguito e non compete nei K slot routed.

**Proposta deterministica:** confrontare le probabilità FP32; a parità esatta
vince l'ID minore, coerentemente con DwarfStar e Picoforge. HF usa `torch.topk`,
che non definisce questo stesso ordine per i pareggi. La parità HF sugli ID
richiede quindi fixture senza pareggi; i pareggi hanno un test contrattuale
separato. Anche logits diversi possono diventare probabilità FP32 uguali:
non sostituire top-k delle probabilità con top-k dei logits senza verifica.

Accumulatori e intermedi iniziali FP32, inclusi router, softmax, sigmoid,
SwiGLU, down e riduzione. Oracle ad alta precisione distinto in FP64.
Nessun cast implicito BF16 degli intermedi per usare TensorOps; eventuali
percorsi ridotti sono esperimenti successivi. Sommare routed in ordine slot
0..K−1, quindi shared. Una sigmoid stabile per argomenti negativi evita
overflow; gli input/intermedi non finiti producono errore numerico esplicito.

Padding: non leggere 768 valori da un buffer di 640. F prepara lo zero padding
dell'attivazione (anche dopo riuso dello scratch); B interpreta tutte le
colonne fisiche del tensore. Le colonne packed oltre I non devono essere
assunte esattamente zero dopo quantizzazione. Is non viene padded a Ip.

F non ha stato temporale: a X identico, decode, prefill e suddivisioni in
chunk producono le stesse route e Y entro tolleranza. Reset svuota validità
di scratch/tracce, non cache del modello. Append non riutilizza route vecchie.

## 4. API proposte

I nomi sono proposte, non header già disponibili. Il consolidamento decide
tipi condivisi una sola volta; F include A/B senza definirne copie.

```c
typedef struct Qwen38Moe Qwen38Moe;         /* config/binding immutabili */
typedef struct Qwen38MoeWork Qwen38MoeWork; /* scratch di una sola esecuzione */
typedef struct {
    float *data;
    size_t rows, cols, row_stride, capacity; /* stride/capacity in float */
} Qwen38MoeF32;
typedef struct {
    uint32_t *ids; float *weights;         /* [T,K] */
    size_t row_stride, capacity;           /* elementi per array */
} Qwen38MoeRoutes;
```

Operazioni C11 proposte, con status esplicito e diagnostica nel tipo errori
comune deciso dal coordinatore:

| Operazione | Contratto |
|---|---|
| `qwen38_moe_create(config, layer_bindings, linear_backend, out, err)` | Copia scalari config; prende in prestito binding A e backend B; valida tutte le forme/formati prima del calcolo |
| `qwen38_moe_work_size(moe, token_cap, bytes, err)` | Calcolo checked degli scratch, incluso padding e workspace richiesto da B |
| `qwen38_moe_work_create(moe, token_cap, byte_budget, out, err)` | Alloca scratch riusabile entro budget; nessuna allocazione dipendente dai pesi durante forward |
| `qwen38_moe_route(moe, logits, routes, work, err)` | Testabilità del solo router su logits FP32 `[T,E]`; no top-k hardcoded |
| `qwen38_moe_forward(moe, x, y, optional_routes, work, err)` | Prodotti tramite B; scrive Y senza residuo; T=0 è no-op senza accessi |
| `qwen38_moe_work_reset(work)` / `*_destroy` | Reset validità solo quando non in flight; distruzione dopo completamento GPU |

In ogni matrice `row_stride >= cols` e
`(rows-1)*row_stride + cols <= capacity` con moltiplicazioni checked.
Colonne contigue; stride negativi o trasposte implicite non supportati.
I puntatori input sono logicamente const. X, Y, route e scratch non si
sovrappongono. ID `uint32_t`, pesi FP32; capienza e prodotto T*K validati prima
del narrowing per Metal. Config valida `H,I,Is,E > 0`, `1 <= K <= E`,
`Ip >= I`, rank/bytes/format coerenti e layer ordinario. Limiti GPU espliciti
in admission, mai troncamento a costanti o route silenziosamente saltate.

**Contratto richiesto a B:** prodotto `[M,N] = [M,Kphysical] * W[N,Kphysical]^T`,
input/output FP32 con stride, matrice packed selezionabile tramite binding
e indice esperto, `Klogical` distinto da `Kphysical`, stride packed in byte.
F deve poter richiedere F32, Q8_0, IQ2_XXS, Q2_K e gruppi di righe senza
ottenere un'intera matrice dequantizzata. Decode può invocare K GEMV;
grouped GEMM è ottimizzazione successiva dello stesso contratto B.
Lo staging CPU dei byte packed e i decoder sono di B/storage, non di F.

Per Metal: `qwen38_moe_encode` riceve contesto/encoder e viste di attivazioni
del backend comune, restituisce stato di encoding e un completion status
leggibile dopo il command buffer. Non sincronizza o fa readback ad ogni
esperto. Un errore numerico GPU marca il risultato non valido; i consumatori
non devono applicarlo a stato persistente senza il protocollo d'errore comune.
Se il grafo ha già avanzato stato, il coordinatore invalida la richiesta
anziché continuare dal risultato parziale. Nessun fallback a route stale.

Ownership: create non copia pesi; il proprietario dei binding/plan/backend
vive fino a destroy e completamento dei comandi. F non chiama gguf_close o
gpu_weights_close. I descrittori `GgufBinding` identificano un plan stabile:
non ricostruire owner per valore. Buffer Metal, arena e trace devono restare
vivi finché in flight, anche se i command buffer trattengono i pesi.
Ogni esecuzione concorrente richiede un workspace separato. Tracce valide
fino al forward/reset successivo; copiarle se servono oltre tale durata.

Errori richiesti: argomento/shape/stride/capienza, overflow, config o formato
non supportato, owner errato, budget/OOM, errore del backend/I/O/Metal e
valori non finiti. Preflight fallito non scrive output; errore durante il
calcolo invalida l'intero output e tutte le route del passaggio.

## 5. Dipendenze e decisioni per il consolidamento

| Componente | Contratto richiesto |
|---|---|
| A | Config con H/E/K/I/Is, attivazione, norm_topk e provenienza del default; L ordinario distinto da MTP; binding degli otto tensori e Ip |
| B | Decoder/oracle indipendente, linear CPU/Metal F32/Q8_0/IQ2_XXS/Q2_K, padding 640/768, ownership e workspace; nessun kernel quantizzato duplicato in F |
| C | X `[T,H]` dopo mixer FFN, consumo Y prima della reiniezione; sigmoid/SiLU comuni se esposte, stesso contratto numerico |
| D/E | Nessuna dipendenza diretta; effetti sullo stesso X attraverso il grafo, verificati nell'integrazione |
| Coordinatore | Header/errori/viste GPU, file condivisi, lock globale, budget totale, oracle upstream, BASE_CONCORDATA |

Aperto: fissare firme reali A/B/C e ownership degli header. F può iniziare
oracle e routing sintetico senza decoder, ma non dichiarare MoE Q2 completo
finché B non fornisce i tre formati packed necessari. Il lock GPU comune non
è definito in questa base: non inventare un lock locale per worktree.

## 6. File posseduti e modifiche condivise richieste

In SPEC è scritto e committato **soltanto questo file**.
Assegnazioni proposte per ESEGUI, da ratificare in integration.md:

| Proprietario F | Contenuto |
|---|---|
| `src/qwen38_moe.c` | Validazione, routing, orchestration CPU, grouping |
| `src/qwen38_moe.m` | Solo binding e dispatch Metal |
| `src/qwen38_moe.metal` | Routing/group/gather/scatter e composizione, nessun decoder/matmul packed |
| `tools/oracle/qwen38_moe.py` | Driver oracle indipendente e manifest |
| `tests/test_qwen38_moe.c` | Harness C11, contratti e sanitizers |
| `tests/test_qwen38_moe.m` | Harness Metal e lifetime |
| `tests/test_qwen38_moe.py` | Differenziale, generazione sintetica e slice bounded |
| `docs/specs/qwen38/F.md` | Contratto e risultati del componente |

Header pubblico proposto `src/qwen38_moe.h`: proprietà del coordinatore come
gli altri header condivisi. Chiedere al coordinatore i target build/test e il
link della sorgente Metal separata (oggi Makefile compila solo kernels.metal).
Sono condivisi e non editati da F: Makefile, `picoforge.h`, `gguf.h`,
`kernels.metal`, `config.c`, `model.c`, `gpu_forward.m`, `metal.m`, entry point,
README, report aggregati e integration.md. Nessun refactor Qwen3 necessario.
Fixture/output generati restano nel build locale, hash e ricetta nei test.

## 7. Oracle indipendente, fixture e slice reali

### Catena di verifica

1. Eseguire le classi upstream fissate `Qwen4ExpTextTopKRouter`,
   `Qwen4ExpTextExperts`, `Qwen4ExpTextMLP`, `Qwen4ExpTextSparseMoeBlock`, su
   fixture piccole in CPU FP32. Usare implementazione eager degli esperti.
   Il driver NumPy di F è confrontato con questo codice mantenuto a monte,
   non con un secondo port della propria formula.
2. Per quantizzazione, i valori dequantizzati provengono dall'oracle
   indipendente di B (non dal suo decoder C sotto test), con hash e layout.
   Le classi upstream ricevono questi stessi valori FP32: ciò verifica F
   sul checkpoint Q2, non la perdita rispetto a BF16.
3. Confrontare intermedi C11 contro oracle; Metal contro C11 e fixture
   upstream congelate. DwarfStar CPU/GPU resta confronto aggiuntivo, non
   unica giustificazione di una traduzione del suo algoritmo.

**Disponibilità verificata:** `tools/venv` principale ha Transformers 5.13.0,
ma non il file `models/qwen4_exp/modeling_qwen4_exp.py`. Nessun pip install.
Proposta: il driver recupera la sorgente pinnata in ESEGUI, verifica gli hash
e usa una selezione AST delle quattro classi, senza modificare i corpi;
rimuove soltanto il decorator di dispatch opzionale dagli esperti e fornisce
`torch`, `nn`, `F`, `ACT2FN`/SiLU e config minimale. Registrare trasformazioni
e licenza Apache-2.0; niente import del modello multimodale completo.
Se questa esecuzione isolata non è compatibile con il torch esistente, è una
dipendenza aperta: mantenere test analitici, non spacciare il port NumPy per
oracle indipendente e non aggiornare dipendenze senza autorizzazione.

Fixture minime:

- Dense sintetica `H=8,E=4,K=2,I=5,Is=3`, seed fisso: forme asimmetriche
  rivelano trasposte e indebita uguaglianza I=Is. Attivazioni positive/negative,
  input zero, shared gate zero (fattore 1/2), saturazione positiva/negativa.
- Routing `E=512,K=10`, e varianti K=1/K=E; logits uniformi, pareggi alla
  soglia, vicino-pareggi in FP32, shift costante, dominanza e underflow.
  NaN/Inf e overflow della proiezione devono fallire esplicitamente.
- Esperti con uscite analiticamente distinguibili; scambio gate/up deve
  fallire. Token ripetuti e molti token sul medesimo esperto; esperti vuoti;
  nessuna capacità per esperto che scarti token. Grouping è una permutazione
  biunivoca dei P=T*K pair, scatter conserva slot/peso originali.
- Fixture packed B con H=256, I=640, Ip=768, Is=32, E=4/K=2: padding poison
  prima del forward, strided input/output, canary ai confini e peso padded
  nonzero. Calcolo ignorando le colonne oltre 640 deve coincidere col prodotto
  su 768 attivazioni zero-padded.
- T=0,1,2,31,32,33,64,65,1024 (1024 con pesi sintetici piccoli), chunk irregolari
  1/7/32 e append/reset; routes identiche e nessuna dipendenza dai buffer vecchi.
- Binding di owner diverso, offset oltre span, overflow T*K/stride, capienza
  insufficiente, formati misti per proiezione, Ip sbagliato, shared assente,
  budget insufficiente e fallimento backend a metà passaggio.

Slice reali future: leggere con pread bounded router F32, shared gate e tre
matrici shared, più gate/up/down degli esperti realmente selezionati da 1–4
X deterministici, per layer 0,3,47, uno alla volta. Includere esperto 0 e 511
come probe forzati separati dal routing naturale. Non ridurre E scegliendo
solo gli esperti letti: il router reale usa tutte le 512 righe. Hash di ogni
slice, offset assoluto, forma logica/fisica, nome e artifact hash nel manifest.
Evitare la materializzazione FP32 di tutti gli E esperti: eseguire le classi
upstream sui soli esperti selezionati rimappando gli ID dopo aver verificato
il router completo; marcare questa trasformazione. L'oracle del blocco
upstream completo resta disponibile sulla fixture sintetica piccola.

Slice di attivazioni da veri prompt richiedono più avanti un dump indipendente
limitato ai confini del blocco, con prompt/token/hash e revisione runtime.
Input sintetici su pesi reali non equivalgono a parità del grafo o dei logits.
Questi dump dipendono dall'integrazione e dal lock, non si producono in SPEC.

## 8. Tolleranze e criteri di accettazione

**Proposte da congelare prima dei test, non tolleranze calibrate qui.**
Confrontare anche tutti gli intermedi: logits, probabilità, ID/pesi, gate/up,
SwiGLU, down, contributo shared e Y. Non basta il testo finale.

| Quantità | Criterio iniziale |
|---|---|
| Shape, offset, ID, grouping, padding | Esatti; zero pair persi/duplicati, ID distinti per token |
| Routing su stessi logits FP32, margini non ambigui | Tutti gli ID uguali; pesi `atol=2e-6, rtol=2e-5`; somma 1 entro 2e-6 se normalizzata |
| Sigmoid/SiLU isolate, input finiti moderati | `atol=2e-6, rtol=2e-5`, test separati di saturazione |
| Proiezioni FP32 / prodotti packed esatti | Limite assoluto dipendente da condizionamento descritto sotto; decodifica deve già passare i test B |
| Y di fixture sintetiche scalate a RMS circa 1 | `atol=2e-5, rtol=2e-4`, NRMSE <= 1e-4 |
| Y su slice reali C/Metal vs FP64 | NRMSE <= 2e-4 e max errore <= `5e-4*max(1,RMS(Yref))`, con intermedi e route verificati |
| Chunk contro token singoli | ID uguali; medesimi limiti Y, zero carry-over dopo reset |

`NRMSE = RMS(Y-Yref)/max(RMS(Yref),1e-6)`; riportare max assoluto e relativo,
non solo media. Ogni gate è necessario, non alternative tra cui scegliere.

Motivazione: con unit roundoff FP32 `u=2^-24`, una dot di n termini ha limite
classico `gamma_n * sum(abs(w_i*x_i))`, `gamma_n=n*u/(1-n*u)`; per n=2560
gamma è circa 1.53e-4. Per il confronto iniziale usare al massimo
`2*gamma_(2n)*sum(abs(w_i*x_i)) + 2e-6` per coprire moltiplicazione/riduzione
di entrambi i percorsi; validare il bound rispetto all'ordine effettivo di B.
Il bound assoluto evita falsi relativi enormi vicino a cancellazioni. Le
soglie Y sono obiettivi più stringenti sul dato, non conseguenze del bound.

Route end-to-end: registrare margine K/K+1 e anche margini interni tra slot.
Se l'errore dei logits supera metà margine, analizzare con riferimento FP64.
Un disaccordo reale non è accettato semplicemente come “quasi pareggio”:
serve soluzione numerica o decisione esplicita del coordinatore. HF con
pareggi non ha una parità ID promessa; il comportamento deterministico
nativo va verificato indipendentemente. Differenze nell'ordine di riduzione
HF (per ID esperto) e nativo (per slot) ammettono errore FP32, non route diverse.

Accettazione finale F: oracle indipendente eseguibile, test C con sanitizers,
Metal API validation, slice reali, chunk/stride/lifetime/errori e regressioni
Qwen3 verdi. Nessun allargamento automatico delle soglie dopo un fallimento.
La correttezza su Q2 non misura dNLL/KL/perdita BF16 e non dichiara completa
la generazione Qwen3.8.

## 9. Incrementi implementativi verificabili

1. Consolidare config/binding e matrice delle dipendenze; costruire oracle
   upstream isolato e fixture analitiche, senza B né checkpoint completo.
2. Routing C11 da logits FP32, stride/errori e ID deterministici; poi routing
   Metal, soltanto col lock comune disponibile. Confrontare entrambi all'oracle.
3. Comporre un MoE sintetico F32 con B: routed + shared, nessun residuo;
   verificare intermedi e integrazione X/Y con C.
4. Collegare B Q2_K per down, poi IQ2_XXS gate/up e Q8_0 shared: padding,
   fixture packed indipendenti, slice bounded. Nessun decoder provvisorio in F.
5. Portare orchestration Metal riusando linear B e gli operatori C concordati;
   validare buffer/owner/errori e numerica su singolo token.
6. Group/gather/scatter e microbatch, prove delle permutazioni, chunk e
   append. Cambiare GEMV/GEMM solo a parità verificata; profiling successivo.
7. Consegnare al coordinatore test del confine C→F→C e comandi di regressione.
   Aggiornare questo documento con misure effettive, senza report aggregati.

Ogni incremento ha commit piccolo e verifica proporzionata. L'assenza di B
blocca gli incrementi packed, non viene nascosta con un mock “completo”.

## 10. Comandi, memoria, rischi e verifiche SPEC

### Comandi futuri da rendere disponibili in ESEGUI

I seguenti target/script/flag sono **proposti**, oggi non esistono. Il
coordinatore ne materializza build e header; non lanciarli per validare SPEC.
L'ambiente è quello esistente, l'output appartiene al nuovo worktree:

```sh
PF_MAIN=/path/to/picoforge
PF_PY="$PF_MAIN/tools/venv/bin/python"
"$PF_PY" tools/oracle/qwen38_moe.py --synthetic --out build/qwen38-f/oracle
make build/test_qwen38_moe
ASAN_OPTIONS=detect_leaks=1 ./build/test_qwen38_moe --synthetic
"$PF_PY" tests/test_qwen38_moe.py --cpu --fixtures build/qwen38-f/oracle
```

Il target CPU deve usare C11 e warning del Makefile, con
`-fsanitize=address,undefined -fno-omit-frame-pointer`. Dopo aver acquisito
il lock **di sistema fissato in integration.md**, in ESEGUI:

```sh
make build/test_qwen38_moe_metal
MTL_DEBUG_LAYER=1 ./build/test_qwen38_moe_metal --synthetic
"$PF_PY" tests/test_qwen38_moe.py --real-slices \
  "$PF_MAIN/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf" \
  --max-slice-bytes 268435456 --out build/qwen38-f/slices
```

Il lock copre tutti i test GPU e letture/calcoli pesanti su checkpoint,
anche cross-worktree; mancando il protocollo si eseguono solo test CPU piccoli.
Non lanciare benchmark contemporanei. Nessun benchmark è richiesto da SPEC.

Regressioni esistenti per il coordinatore: `make`, `make test-oracle-moe
test-forward-moe test-forward-moe-quant test-routing-trace test-spec-experts`,
poi `make test-all`. Questi target assumono `tools/venv` e fixture locali:
il coordinatore deve predisporre il riuso dell'ambiente e isolare le build,
senza conversioni/scritture nei modelli della directory principale. Anche
queste regressioni GPU richiedono il lock; non sono state eseguite in SPEC.

### Budget proposto, non misura

Tutti gli esperti compressi residenti appartengono allo storage. F non li
ricopia e non espande un layer intero. Usare inizialmente microbatch F di
R<=32 all'interno del chunk del grafo di 1024; nessuna dipendenza temporale
impedisce questa suddivisione. R è una capacità di workspace, non del modello.

Con P=R*K, una stima conservativa dello scratch FP32 è:

```text
4 * (R*E + 2*P*H + 2*P*I + P*Ip + 3*R*Is + 2*R*H + R)
+ 4 * (4*P + 3*E + 1) byte per route/group metadata
```

Comprende logits, input gathered, output per pair, gate/up/mid padded,
shared e due buffer risultato; metadata include ID, pesi e due permutazioni.
Per R=32: **10,153,092 byte, 9.68 MiB**, prima di allineamento, pipeline e
workspace B. Per 1024 sarebbero **309.67 MiB**: non allocarli implicitamente.
Proposta cap F+workspace B: 32 MiB iniziali, query/admission prima del forward;
se B non rientra, ridurre R o rifiutare, non sottrarre memoria alle cache.
L'ottimizzazione può eliminare gather/buffer intermedi, dopo i test.

Oracle/slice: massimo 256 MiB di payload residenti simultaneamente, un layer
e pochi esperti per volta; non include overhead dell'interprete/torch, da
misurare separatamente. Un routed expert occupa 1,489,920 byte packed; una
versione logica FP32 gate/up/down circa 18.75 MiB. Evitare duplicazioni FP64,
FP32 e torch contemporanee su decine di esperti: iterare in batch piccoli.
Il budget applicativo complessivo resta 48–50 GiB da misurare in integrazione,
non 48–50 GiB aggiuntivi per componente.

Rischi concreti: default norm_topk mancante nei metadati; pareggi e ordine
di riduzione; confusione gate routed/shared; Is indebitamente uguale a I;
padding non azzerato; stride esperto su asse GGUF errato; binding su viste
diverse; F32 router involontariamente quantizzato; cast BF16 dei matmul;
capacità per gruppo che scarta pair; command buffer/workspace riusato presto;
reiniezione doppia del residuo; oracle upstream non eseguibile nel venv.

### Evidenza di questa sessione

Controllati git status, staging, HEAD e worktree; lette istruzioni e fonti;
ricontati sull'inventario tutti i 48 schemi FFN e i tre tensori Q4_K/MXFP4
MTP; verificati hash dei piccoli inventari e sorgenti upstream. Nessun peso
letto, modello caricato, benchmark, test GPU, aggiornamento dipendenze o
modifica condivisa. Verifiche documentali finali: `git diff --check` e
controllo che il commit contenga esclusivamente `docs/specs/qwen38/F.md`.

## ESEGUI v1 — base consolidata e risultati

Base `c674b23bd1377b2ce0acef2d06fac8b160809446`, branch `impl/qwen38-f-v1`,
worktree `../picoforge-f-v1`; il precedente `../picoforge-f-oracle` è preservato.
L'[integrazione](integration.md) e gli header v1 prevalgono su tutte le proposte
precedenti: stride/capacità **in byte**, errori `Qwen38Status`, tie su probabilità
FP32 con ID minore, e down con **640 input logici letti da B**, senza scratch
di attivazioni padded a 768. Il bound dei prodotti è quello di B consolidato.

Recuperati soltanto tooling e test dai commit oracle F `d1a2788`, `9668aa9`,
`a779e7c`, rispettivamente in `a4ddfca`, `ad98fc4`, `222a91a`. Non riacquisite
spec, modifiche condivise o implementazioni altrui. I corpi delle quattro
classi upstream hash-pinnate e la licenza sono riusati nella cache locale di
build; il venv rimane quello principale (torch 2.13.0, NumPy 2.5.1).
I 16 test oracle sono ripassati su questa base.

Primo incremento C11: `qwen38_moe_route` implementa l'header pubblico immutato,
senza dipendere da A/B/C. Softmax FP32, selezione sulle probabilità (inclusi
pareggi dovuti all'arrotondamento), K distinti, rinormalizzazione configurata.
Nessuna allocazione. Scratch `[T,E]` del caller; logits, probabilità, ID e pesi
disgiunti. Preflight di config/shape/stride/capacità/overflow/allineamento e
alias prima delle scritture; scansione di finitezza di tutti i logits prima
del calcolo. Su questi errori output e guard rimangono invariati. T=0 valida
struttura/config senza dereferenziare dati. La differenza negativa overflow
tra logits finiti estremi può essere -Inf: `exp(-Inf)=0` è ammesso.

**20 test passati** con il harness C compilato ASan/UBSan e warning come errori.
Fixture E=1/4/17/512/513/1025, K=1/K=E/top-10, T fino a 1024; controllati
guard delle righe strided, output immutati su errori, NaN/Inf, ±3e38,
probabilità uniformi e quasi-pareggi, norm sì/no. Quattro seed di proiezioni
floating point upstream E=512 alimentano lo stesso routing C: tutti gli ID
coincidono e i pesi rispettano atol=2e-6/rtol=2e-5; somma entro2e-6.
Logits arbitrari sono confrontati anche con softmax torch e riferimento NumPy.
Chunk 1/7/32 e append conservano bitwise probabilità/ID/pesi a logits identici.

Comandi effettivi dal worktree, senza modificare Makefile:

```sh
mkdir -p build/qwen38-f
clang -std=c11 -O2 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
  -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
  tests/test_qwen38_moe.c src/qwen38_moe.c -lm -o build/qwen38-f/test_moe
./build/qwen38-f/test_moe
Q38_F_NATIVE="$PWD/build/qwen38-f/test_moe" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../picoforge/tools/venv/bin/python tests/test_qwen38_moe.py
```

Se manca la cache upstream, prepararla esplicitamente con
`../picoforge/tools/venv/bin/python tools/oracle/qwen38_moe.py --prepare-upstream`;
i test non fanno download. Senza `Q38_F_NATIVE` i test nativi sono marcati
SKIP, non PASS. Le fixture numeriche native usano un protocollo binario locale
little-endian nel harness (`--route`), solo per test, non un ABI dell'engine.

Dipendenze residue: create/work/forward attendono B scalar e attivazioni C;
il percorso packed attende IQ2_XXS/Q2_K/Q8_0 reali. Nessun fallback/stub o
decoder locale. API GPU pubbliche di composizione non ancora materializzate;
nessun test GPU, carico sul checkpoint o benchmark in questo incremento.
Piccoli test CPU bounded eseguiti senza lock, come consentito dal §9;
qualsiasi futuro GPU/workload pesante userà `tools/qwen38_gpu_lock.py`.

### Secondo incremento: grouping CPU delle coppie

`q38_moe_group_pairs` è un helper **privato** di F, dichiarato soltanto in
`src/qwen38_f_internal.h`. I contratti pubblici restano immutati. Data la
coppia `pair=t*K+slot`, ordina per esperto crescente e poi pair crescente:
`row_to_pair[row]=pair`, `pair_to_row[pair]=row`. Il gruppo e occupa
`[offsets[e],offsets[e+1])`. Counting/prefix/scatter, senza scartare token,
imporre capacità per esperto o cambiare pesi. Per la futura riduzione si
torna alle coppie originali, nell'ordine slot stabilito dall'integrazione.

ID strided `[T,K]` in sola lettura logica; quattro array caller-owned,
allineati a size_t e disgiunti: offsets `[E+1]`, cursors scratch `[E]`,
due permutazioni `[T*K]`. Tutte le capacità sono in byte e la durata è quella
della chiamata sincrona. Nessuna allocazione o stato persistente. Validazione
di shape, stride, capacità, alias, overflow, ID fuori range e duplicati nel
medesimo token prima delle scritture. T=0 permette puntatori NULL per ID e
permutazioni, ma richiede offsets/cursors e li azzera. I gruppi vuoti hanno
offset consecutivi identici. Non è un nuovo ABI di composizione MoE.

**23 test passati** con lo stesso comando ASan/UBSan riportato sopra, nessuno
saltato. Il grouping C è confrontato con `np.argsort(kind="stable")` e
`bincount`, indipendenti dall'implementazione counting/scatter nativa:
28 casi comprendono E=1/4/17/512/513, T=0/1/7/33/65/1024, K=1/4/5/10/11,
carico concentrato sugli stessi esperti, gruppi vuoti e K=E. Verificata
identità esatta delle permutazioni e degli inversi; guard di righe e array
intatti e output invariati sui nove errori del harness C.
Quattro fixture floating point del riferimento già confrontato con upstream
verificano route C → grouping C → ripristino bitwise degli output per coppia
→ somma pesata entro atol=2e-5/rtol=2e-4, anche a chunk 1/7/25.
Quest'ultimo test usa i prodotti dell'oracle: **non certifica un forward C**.

Budget runtime routing: logits+probabilità `8*T*E` byte, ID+pesi `8*T*K`;
grouping `sizeof(size_t)*(2*E+1+2*T*K)`. Con T=1024/E=512/K=10 su macchina
64-bit, totale 4.45 MB decimali (4.24 MiB), esclusi pesi e guard di test;
solo grouping 172040 byte. Il routing corrente è scalare e privilegia la
verifica: ricerca top-k O(T*E*K²); non è una misura prestazionale.
Il grouping impiega O(E+T*K²) tempo, inclusa la verifica dei duplicati.
Le API create/work/forward restano dichiarazioni senza stub: B scalar/F32
e C attivazioni non sono implementati nella base di questo worktree.
I successivi incrementi richiedono quei commit reali dal coordinatore,
poi IQ2_XXS/Q2_K/Q8_0 per pesi compressi e binding/slice da A.

[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
[hf-model]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L895-L991
[hf-config]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py#L162
[ds4-c]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L68391-L68451
[ds4-metal]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/metal/qwen4.metal#L1100-L1250
[paper]: https://arxiv.org/html/2608.30320v1

## ESEGUI v2 — composizione CPU sulla BASE_INTEGRATA

Base `6a6b886e985185d7c9104eda888e6a406553aedf`; nuovo worktree
`../picoforge-f-v2`, branch `impl/qwen38-f-v2`; nessun reimport delle consegne.
Create/work/forward implementati in `src/qwen38_moe.c`: config e descrittori
copiati, byte borrowed immutabili, validazione dei pesi tramite B con M=0.
I due router richiedono F32. La metadata del modello rimane viva finché esiste
un workspace; destroy invalida l'owner e work_reset lo rifiuta. Chiamate CPU
sincrone, un workspace per esecuzione, nessuna concorrenza sullo stesso handle.
Preflight strutturale preserva output; errore numerico durante forward rende
workspace/output invalidi fino a reset. Non promette rollback della sessione.

Scratch allocato solo da work_create, budget include metadata+arena; microbatch
min(token_cap,32), gather per coppia, scratch gate/up/SiLU condiviso tra gruppi.
B esegue tutte le proiezioni; C tutte le SiLU/sigmoid. Grouping stabile F,
riduzione slot crescente e shared aggiunto alla fine, senza residuo. Il privato
observer registra viste temporanee degli stadi per verificare gli intermedi;
non sostituisce operatori e non può mutare/ri-entrare/liberare il workspace.

Verificati 27 test (strict runner, zero skip), harness ASan/UBSan e warning
come errori. Floating T=0/1/2/31/32/33/65/1024, K=1/2/E, chunk 1/7/32,
rami azzerati separatamente, shared gate, stride, alias, budget, reset dopo
NaN nei pesi borrowed. Packed H256/I640/Ip768/Is32, IQ2_XXS gate/up,
Q2_K down e Q8_0 shared: T=1/2/33. Oracle decoder esterno di B, non F;
pad down nonzero e variante F32 con poison 1e30 danno lo stesso risultato.
Ogni prodotto confrontato al bound B su identico input; attivazioni ai gate F.

Slice A+B: layer 0/3/47, router completo 512 esperti e top-10 naturale,
un input sintetico RMS~0.1 per layer. 28,354,560 byte letti per layer, manifest
con offset/hash/config/provenienza; include 0/511 come **probe caricati**, non
come esperti aggiunti alla route. Non sono attivazioni estratte da prompt.
Il harness riserva indirizzi anonimi per il layout dei tre tensori routed,
abilitando solo le pagine degli esperti forniti: nessun mapping del checkpoint
intero, nessuna espansione FP32 del layer. Riserva virtuale e payload residente
sono distinti; il limite trasporto è 64 MiB, l'oracle elabora un esperto alla
volta. Il blocco upstream con E=1 è usato per verificare ciascun esperto
selezionato, dopo aver controllato il router originale completo; le slice
sono dequantizzate esclusivamente dall'oracle indipendente B.

| Layer | Y max abs vs FP64 | Y NRMSE | errore probabilità | margine minimo top-K/K+1 |
|---|---:|---:|---:|---:|
| 0 | 2.063e-9 | 1.209e-6 | 2.561e-9 | 1.308e-5 |
| 3 | 2.105e-9 | 1.285e-6 | 1.164e-9 | 5.753e-6 |
| 47 | 2.280e-9 | 1.379e-6 | 4.657e-10 | 7.309e-7 |

Tutti gli ID coincidono. Il massimo rapporto errore/bound B dei prodotti
routed è 0.004622. Controllati routed e shared, gate/up/down e attivazioni;
Y rispetta entrambe le soglie slice, senza allargarle. Workspace reale a T=1:
235592 byte. Le misure non stimano perdita rispetto al BF16 originale.

Comando CPU aggiornato (il target condiviso v1 non linka ancora B+C+A):

```sh
clang -std=c11 -O2 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
  -ffp-contract=off -fno-fast-math -fsanitize=address,undefined \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -Isrc \
  tests/test_qwen38_moe.c src/qwen38_moe.c src/qwen38_packed.c src/qwen38_base.c \
  src/qwen38_config.c src/qwen38_weights.c src/gguf.c src/gguf_reader.c src/util.c \
  -lm -o build/qwen38-f/test_moe
Q38_F_NATIVE="$PWD/build/qwen38-f/test_moe" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../picoforge/tools/venv/bin/python tools/qwen38_gpu_lock.py --timeout 60 -- \
  ../picoforge/tools/venv/bin/python -B tools/qwen38_test.py tests/test_qwen38_moe.py -v
```

Preparazione esplicita in Python: `qwen38_packed.build_reference` dal checkout
DwarfStar fissato verso `build/qwen38-f/packed-reference` con sanitize=True;
`qwen38_moe.prepare_real_slice(native,checkpoint,output,layer)` sotto lock per
0/3/47, output `build/qwen38-f/slice-<layer>`. Riusare i tre file upstream
hash-verificati nella cache F. Test offline; nessuno skip equivale a PASS.
Report numerico riprodotto in `build/qwen38-f/slice-results.json`.

### Metal privato post-routing e consegna v2

`src/qwen38_moe.m` compone **route CPU congelate → gather GPU → prodotti B →
SiLU/sigmoid C → prodotto gate/up F → riduzione routed/shared F**.
`src/qwen38_moe.metal` contiene solo gather/prodotto/riduzione e copia per
tracce di test, senza decoder, linear o attivazioni duplicati. Contesto e
scratch privati con cap 1..32, budget esplicito <=32 MiB; store/binding borrowed,
pipeline/buffer retained fino a destroy dopo completion. La funzione encode
non fa commit/wait, non azzera la parola errore e non alloca scratch per token.
Il caller serializza il riuso; nessun lifecycle GPU pubblico nuovo.

Limite esplicito: B encode riceve `expert` host-side; questo esperimento non
esegue routing/grouping dipendente da risultati GPU. Il harness esegue il
forward CPU per route/riferimento, poi un solo command buffer per microbatch
per tutti gli esperti selezionati e shared. Non è un engine ibrido pubblicato
né una misura di velocità. Proiezioni e attivazioni GPU sono catturate prima
del riuso scratch tramite copie nello stesso encoder, poi confrontate con
FP64 e upstream; le copie e le allocazioni diagnostiche appartengono al test.
Le route e le tracce router di questo percorso restano **CPU**.

Verifica finale GPU: **4 test, zero skip**, 42 casi differenziali registrati,
4280 stadi (include router CPU), ulteriori casi analitici di tie e fault
injection; 154 microbatch GPU completati con Metal API Validation e harness
ASan/UBSan. Floating T fino a1024, chunk 1/7/32, K=1/2/E; shared gate a
-100/0/+100 e norm_topk false sintetico. Packed IQ2_XXS/Q2_K/Q8_0, pad640/768
nonzero e variante poison. Preflight rifiuta alias X/Y/error word, view corta,
owner diverso, duplicati ID, pesi negativi e budget insufficiente; T=0 valida
senza dispatch. NaN introdotta **solo nel buffer GPU dopo il riferimento CPU**
produce `Q38_GPU_NONFINITE`, controllato dopo command completion.

Y sintetico GPU vs upstream: max abs **1.192093e-6**, max NRMSE
**4.836173e-7**, entro entrambi i gate. Snapshot di ogni prodotto B controllato
sullo stesso input al bound congelato; attivazioni ai gate F e prodotto
SiLU*up esatto rispetto agli snapshot. Allocazione Metal massima osservata
nel harness **4,866,048 byte**, trace massimo1,634,688 byte, scratch F massimo
445,312 byte sulle shape sintetiche provate; non è footprint del modello.
Report nel build: `gpu-final.json`, `gpu-final-allocations.jsonl`.
Slice reali verificate **CPU**, non Metal; il harness GPU rifiuta oltre64MiB
packed e non materializza lo storage dell'intero layer reale.

Comandi Metal effettivi, oltre alla preparazione CPU descritta sopra:

```sh
make build/qwen38-b/packed.air build/qwen38-c/recurrent.air
xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc \
  -c src/qwen38_moe.metal -o build/qwen38-f/moe.air
xcrun -sdk macosx metallib build/qwen38-b/packed.air \
  build/qwen38-c/recurrent.air build/qwen38-f/moe.air -o build/qwen38-f/moe.metallib
clang -std=c11 -O2 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
  -ffp-contract=off -fno-fast-math -fsanitize=address,undefined \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -fobjc-arc -Isrc \
  tests/test_qwen38_moe.m src/qwen38_moe.c src/qwen38_moe.m \
  src/qwen38_packed.c src/qwen38_packed_gpu.m src/qwen38_base.c src/qwen38_conv.c \
  src/qwen38_gdn.c src/qwen38_recurrent_gpu.m src/qwen38_config.c \
  src/qwen38_weights.c src/gpu_weights.m src/gguf_plan.c src/gguf.c \
  src/gguf_reader.c src/util.c -framework Foundation -framework Metal -lm \
  -o build/qwen38-f/test_moe_gpu
Q38_F_NATIVE="$PWD/build/qwen38-f/test_moe_gpu" \
  Q38_F_LIBRARY="$PWD/build/qwen38-f/moe.metallib" Q38_F_METAL=1 MTL_DEBUG_LAYER=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../picoforge/tools/venv/bin/python tools/qwen38_gpu_lock.py --timeout 60 -- \
  ../picoforge/tools/venv/bin/python tests/test_qwen38_moe.py \
  NativeForwardTests.test_float_composed_stages_chunks_and_shared \
  NativeForwardTests.test_float_nonfinite_and_zero_input \
  NativeForwardTests.test_packed_composition_and_640_768_padding \
  NativeForwardTests.test_native_tied_block_and_near_tie_margins
```

`Q38_F_REPORT` opzionale salva JSON in un percorso **nuovo**; il test rifiuta
sovrascrittura e restituisce nonzero per errori o skip. `Q38_F_ALLOCATIONS`
aggiunge record di allocazione nel build; scegliere un nome nuovo per ogni run.
CPU finale: suite estesa a28 test; `make test-qwen38-links
PYTHON=/path/to/picoforge/tools/venv/bin/python` passa
A→B→F, C→E e A→D/E. Nessuna esecuzione del modello completo o benchmark.

**Proposte concrete al coordinatore, file condivisi non modificati:**

1. Il target CPU F deve aggiungere i `.c` B/C/A/reader del comando sopra.
   Collegare AIR F alla libreria e aggiungere il target GPU privato sotto lock.
   Il Makefile corrente mappa **sia `qwen38_moe.c` sia `qwen38_moe.m` a
   `src/qwen38_moe.o`**: usare oggetti distinti (es. `.c.o` e `.m.o`) oppure
   assegnare a F il nome `src/qwen38_moe_gpu.m` e aggiornare ownership/build.
   `make`/`test-all` globali restano al coordinatore dopo questo wiring;
   i comandi isolati e test-links sopra sono verificati.
2. Per routing interamente GPU senza readback/commit per esperto, B deve
   possedere un encode indexed: stesso layout packed e input `[P,Klogical]`,
   output `[P,N]`, più `expert_ids[P]` uint32 contiguo in GPU span bounded.
   Ogni riga sceglie la propria matrice esperto, verifica ID<`n_experts` e
   conserva il bound numerico/logical padding B. F può così produrre route e
   pair sulla GPU e ridurre per slot, senza accedere al decoder B. Nome/API
   pubblica e lifecycle sono da fissare dal coordinatore, non introdotti qui.

Limiti residui: Metal completo con router GPU e slice reali, errori di
allocazione forzati, lifecycle globale/integrazione nel grafo, regressioni
Qwen3 complete e generazione Qwen3.8. Le prove C11 complete del blocco MoE e
Metal post-routing non certificano questi incrementi né la qualità Q2/BF16.
