# Contratto MTP — proposta di implementazione, non consegna

Stato: **non eseguibile**. Il percorso ordinario rimane il riferimento.
Questo documento fissa il contratto prima di aggiungere il grafo MTP.
L'implementazione e i gate elencati sotto sono ancora da eseguire.

## Riferimenti verificati nel checkout locale

DwarfStar invariato, commit `fc80bd695da76ee14bcb86820bf900a5cdfda806`,
`ds4.c`: `qwen4_ref_mtp`, `qwen4_graph_mtp_steps`, `qwen4_ref_row`.
Il sorgente è in `../ds4-qwen38-baseline/` rispetto al worktree. Il percorso
diagnostico `--cpu --first-token-test` con `DS4_QWEN4_MTP_OUT` salva i logits
teacher-forced `[n-1,vocab]` del predictor. Le attivazioni/cache sono FP32;
l'oracle CPU usa anche accumuli double nelle riduzioni, come il gate G3.
Non usare il backend Metal stock a cache FP16 come gate FP32.

L'inventario A descrive un blocco MTP QSA dopo i 48 blocchi ordinari. I ruoli
aggiuntivi sono `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` e il mixer
`nextn.hc_head`. Gate/up degli esperti sono Q4_K, down MXFP4; i decoder B
ordinari non li eseguono ancora. Il predictor non ha PLE. I pesi MTP sono già
nelle viste storage del GGUF: non serve un secondo caricamento del modello.

Per il predictor all'indice i:

1. Prendere il residuo **prima del mixer finale** del trunk all'indice i e
   l'embedding del token confermato i+1.
2. RMS dell'embedding su E; RMS del residuo su **H×E**, non per ramo HC.
3. Proiettare `[e_norm, 0]` una volta e `[0, h_norm_s]` per ciascun ramo s
   con lo stesso `eh_proj`; sommare i due risultati per ramo.
4. Eseguire HC-attention, QSA all'indice i nella cache **propria del MTP**,
   HC-FFN e MoE, poi il mixer `nextn.hc_head` e la LM head condivisa.
5. Argmax deterministico sull'intero vocabolario, tie al token più piccolo.

## Una proposta per ciclo, verifica di due token

Il primo token a viene dall'ordinario e può essere emesso subito. MTP propone
b usando `(R_i,a)`. Il trunk verifica `[a,b]` in un solo append causale,
restituendo due righe di logits. L'argmax della prima deve essere uguale a b
per accettarlo; la seconda fornisce il prossimo token ordinario c.
Non usare soglie di confidenza per accettare un token diverso dall'argmax.

Prima della verifica va salvato lo stato del trunk: GDN, convolution history,
K/V, indexer/pooling QSA, PLE convolution e history/hash n-gram, posizione e
validità. Nessun peso o tabella SSD entra nello snapshot. Se b è rifiutato,
ripristinare il punto precedente alla verifica e ricalcolare solo a; il token
corretto viene dalla prima riga verificata. La cache non deve contenere b.
I ticket anteriori al ripristino diventano stale tramite cambio di epoch.
Il ripristino è ammesso solo a GPU idle, sulla stessa istanza, dopo append
riusciti e entro la finestra prevista; un errore GPU richiede reset completo.

Il predictor avanza solo su coppie confermate. Dopo l'accettazione di b,
recupera `(R_a,b)` senza necessità di LM head; poi `(R_b,c)` propone il token
successivo. Dopo un rifiuto non ha ancora elaborato b, quindi non deve
conservare alcuna sua history. Il prefill inizializza tutte le coppie del
prompt, anche al confine dei chunk. Nessun warm-start con cache MTP vuota a
una posizione arbitraria presentato come prefill completo.

## Stop e capacità

Tutti i limiti contano prompt + token emessi, stop incluso. Nessuna proposta
fuori dal contesto residuo. Con meno di due posti utili o limite di risposta
insufficiente, usare l'append ordinario. Stop proposto non è stop effettivo
finché la verifica non lo conferma. Se a è stop, terminare senza MTP.
Se b è accettato ed è stop, non emettere c né mantenere una conversazione
oltre lo stop. Cancellazione cooperativa tra append, mai pubblicazione di
proposte non verificate. Reset azzera entrambi i grafi e invalida snapshot.

## Gate necessari prima di dichiarare la fase completata

- Decoder Q4_K/MXFP4 CPU e Metal: blocchi analitici, slice bounded del GGUF e
  confronto con il decoder indipendente; nonfinite, bounds e padding.
- Predictor: confronto numerico teacher-forced con CPU DwarfStar su casi
  congelati, gate 1e-3 invariato, controlli per token e margini.
- Snapshot: stato e logits identici dopo restore/replay, inclusa history
  n-gram e blocchi QSA parziali; rifiuto di snapshot stale/foreign/busy.
- Speculazione: sequenza identica al greedy ordinario, accettazione e rifiuto
  forzati nei test, stop in ogni posizione, cancellazione e contesto residuo.
- Misure: proposta, verifica, rollback, accettazioni/proposte, token per
  append, footprint/Metal/swap e tempi contro l'ordinario nelle stesse
  condizioni. Conservare una regressione prestazionale se osservata.

Parsing dei pesi, questa specifica e un loop senza predictor reale non sono
un'implementazione MTP completata.
