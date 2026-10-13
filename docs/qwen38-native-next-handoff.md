# Handoff operativo — MTP ancora da implementare

Obiettivo utente: completare cinque incrementi, non fermarsi al piano. Questa
è continuità di lavoro, **non una consegna che dichiara MTP completato**.

## Checkout e vincoli

Lavorare solo in `/Users/cristiancirje/Desktop/Dev/picoforge-native-next`, branch
`coord/qwen38-native-next`, da `31d3460` di g1-cpu. Commit già pubblicati:
`d5b1732` suite/sessione e `44ec203` benchmark/QSA/chat. Il commit che contiene
questo aggiornamento chiude le verifiche 8K/browser e il gate bounded aggiuntivo.
Nessun merge/push main, nessuna delega autorizzata, nessuna nuova dipendenza.
`tools/venv` è un symlink non tracciato: **non aggiungerlo**. Checkpoint e venv
sono quelli del repository principale tramite symlink; non ricalcolare il
checksum integrale, non convertire/scaricare pesi. Upstream DwarfStar locale:
`../ds4-qwen38-baseline`, fc80bd695da76ee14bcb86820bf900a5cdfda806 invariato.
Tutti i workload pesanti usano tools/qwen38_gpu_lock.py; Make GPU e server
acquisiscono già il lock, quindi non annidare wrapper.

## Processi e browser

Suite estesa, regressioni, oracle e replay sono terminati. Chat reale porta8001
fermata dopo le prove (server96299 / worker96310, entrambi usciti). Nessun modello
residente lasciato da questa sessione. Server statici ancora attivi:
porta8768 directorybench (sessione97615), porta8769 directorybuild (78042).
Browser T3 tab_1, viewport390×844, report-verified.html sulla8768. Usare
preview_status e preview_*; strumenti disponibili. I primi click fallivano
perché il target era fuori viewport: scrollIntoView con behavior:'instant',
poi click in una chiamata successiva funziona. Alcuni primi turni chat sono
stati attivati con eventi DOM; stop prefill/reset/recupero con tool click/type.
Nessun browser alternativo necessario. Skill frontend-design già letta/usata.

## Risultati e stato delle fasi 1–4

- Suite congelata9casi. Coding C/Python/duefile, italiano373token e contesti
  1K/2K/4K/8K reali completati nei gruppi extended + extended-cooperative.
- 8K:8042prompt +28generati (riservati128, budget8170≤8192), JSON IrisBARI29 /
  LotoVERDE82 corretto, stop. Prefill2046,586064s, decode25,296222s.
  Metal42,387GiB, grafo0,665GiB, footprintlifetime899174360byte (0,837GiB),
  wiredpeak49,262GiB, pressione1, swapdelta0;2761campioni. Wait2045,062s contro
  GPU1373,368s: differenza non attribuita senza altre misure.
- Scalar1K precedente interrotto dopo618,55s: evidenza incompleta conservata.
  Ragionamento completo errato1002 anziché1633, anche dopo ottimizzazione.
  C e merge-ranges passano oracle funzionali ma falliscono il formato (fence).
- Benchmark appaiato native-paired: warmup1 +3misure/modo, stesso prompt61token,
 14generati. Mediane scalar→cooperativo: prefill9,889→8,647s,
 decode1,058→1,116token/s, richiesta22,181→20,303s. Otto sequenze identiche.
 Cache OS non controllate, è un pilot. ATTEND solo:27,55–55,93× su6forme,
 output bit-identici; qsa-paired-v3. Nessuna modifica FP32/gate.
- chunk-replay: reset reale e chunk128/7, stessi14token. Non è oracle indipendente.
- bounded-lazy: tre prefissi24token contro CPU DwarfStar originale; tutti
 scelgono immediatamente stop. Una riga finale/caso, non continuazione multitoken.
 Errori2,0331e-6 /5,1447e-5 /1,9392e-6; argmax/margini PASS, gate1e-3 invariato.
 Circa130s/caso, pressione1, swap in diminuzione. Per rafforzare i controlli
 successivi usare prompt bounded chiusi con più token, non chiamare queste
 tre scelte stop una verifica indipendente di decode lungo.
- CPU oracle stock interrotto dal guard per pressione4/swapgrowth>2GiB.
 Primo wrapper lazy: primo caso PASS, secondo caricamento arrestato. Conservati
 in bounded-attempts/. Un precedente reboot aveva lasciato un tentativo vuoto,
 causa non determinata. Nessuno presentato come PASS completo.
- Nuovo tools/oracle/qwen38_dwarfstar_cpu.c include sorgente pinned senza modificarlo;
 model_open(private,lazy), bind/vocab + qwen4_first_token_test originali. Evita
 hint globale WILLNEED. Runner attende tre campioni normali/wired≤RAM/4 dopo
 l'uscita nativa, massimo60s, senza cambiare limiti/purgare cache. Questa versione
 ha completato i3casi. n-gram sempre disk-only. builder verifica dipendenze/hash.
- Chat reale: desktop1280×850 e mobile390×844, «Ada» ricordata al secondo turno,
 streaming, stop decode15token, stop prefill128/473senza testo, reset,
 contesto63000caratteri rifiutato con storia preservata, recupero «pronto».
 Unico worker96310. Prove/screenshot in browser/. Header sticky, prompt recuperato
 su errore/stop senza testo, stato e accessi condivisi tra schede. Reset/replay
 dell'intera storia con template A una sola volta; nessun riuso incrementale.
- report-verified.html / summary-verified.json: nuovi snapshot immutabili.
 Browser: selezione8K, rawlink200, nooverflow, gate bounded esplicito e FAIL
 stockFP16 visibile tramite click. Non sovrascrivere report-initial o storicoG3.

## Test

Ultime regressioni: G0 ledger/sessione11gruppi; G2 5; stato/fault1; CLI4;
worker2; HTTP3 +UTF8unit1. QSA13 già PASS sulla variante finale. B CPU25 PASS
anche dopo parametrizzazione del builder oracle. make test-qwen38-native-tools
PASS (record1, release-wait2, observer6). Syntax ObjC-Werror e JS node--check
PASS; git diff --check sui sorgenti/documenti (stdout grezzi preservati,
compresi spazi finali emessi dal modello). Log in bench/.../validation, fonti complete in build/.
Non eseguito test-all legacy Qwen3: non dichiararlo superato. Gate storico
DwarfStar Metal FP16 FAIL0,0413082 separato dal CPUFP32PASS2,404e-6.

## Prossima azione: MTP vero, non solo parsing

Leggere docs/specs/qwen38/MTP.md. Il contratto è già definito, codice MTP assente.
Riferimenti verificati: ds4.c qwen4_ref_mtp(68489), qwen4_graph_mtp_steps(59806),
qwen4_ref_row(67857). Il wrapper CPU appena introdotto può usare
DS4_QWEN4_MTP_OUT e DS4_QWEN4_FT_HIDDEN tramite ambiente esplicito del runner.
CPU statiFP32 con riduzioni double upstream; non usare oracle MetalFP16.

Ordine suggerito:
1. Ruoli A per6pesi nextn oggi inventariati Q38_ROLE_COUNT: EHproj, enorm,
 hnorm, HChead norm/down/up. Appendere enumprimaCOUNT preservando indici esistenti.
2. Decoder CPU/Metal Q4_K(type12,256/144) e MXFP4(type39,32/17) nel moduloB,
 con oracle indipendente adattatorepacked (oggi supporta0,1,8,10,16,30).
 Q4: d/dminF16,scales12bytes a4,qs a16; gruppog=lane/32. g<4 scale=sc[g]&63,
 min=sc[g+4]&63; g≥4 scale=(sc[g+4]&15)|((sc[g-4]&192)>>2),
 min=(sc[g+4]>>4)|((sc[g]&192)>>2). Nibble daqs[(g>>1)*32+lane%32].
 MXFP4:e8m0byte0,16packedbytesa1,lowlanes0–15/high16–31; valori
 [0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], scale bitsexp==0?
 0x00400000:exp<<23. Verificare subnormali/exp255/nonfinite senza scorciatoie.
 Test analitici + slice bounded checkpoint + gate CPU/Metal, padding/bounds.
3. MTP opzionale nel forwardGPU: layer48QSA, nessunPLE, pesi già residenti.
 f->r è residuo del trunk prima del mixerfinale. Normalizzare embedding suE,
 residuale suH×E (NON per ramo). EHproietta[e,0] e[0,h_s], somma perramo.
 PoiQSA/HCFFN/MoE/nextnHChead/LMhead condivisa. CacheQSA e ledgerMTP propri,
 scratch condivisibile solo sottof->busy. MoEdown MTPphysical640 anziché768.
 Due kernel wiring in forward.metal possono creareconcat e sommareproiezioni;
 RMS usaoperatoreC convista[T,H*E], proiezioniB giàesistenti.
4. PrefillMTP di tutte le coppie confermate (R_i,token_i+1), anche tra chunk:
 ultimo chunkT−1, altrimentiT (prossimo token prompt noto). PosMTP=P−1 altermine.
5. Snapshot/restore idle di tutte le historyGDN/conv/QSA/PLE e previoushashE,
 pos/epoch e residuo prehead. Non pesi/tabellaSSD. G0 private rewind validato
 stessaistanza/epoch/posizioneattesa, incrementaepoch, ticketstale. ErroriGPU
 richiedonoreset, mai rollback da statoinvalido. Snapshotconsumabile/stale/foreign
 testati. Per k1restoreammesso dopo≤2token. AllocazionehostsnapshotseparataMetal.
6. Loopgreedyk1: a ordinario emesso; MTP(Rprev,a)proponeb; snapshot; verifica[a,b]
 trunk2righe. Se argmaxriga0==b accettab e c=argmaxriga1. Emetteb,c entrobudget.
 Successiva proposta recupera(Ra,b),(Rb,c) inbatchMTP2 conheadsoloultima.
 Rifiuto: restore +append soloa, bcorretto=argmaxriga0; MTP(Ra,bcorretto) perprossima.
 MTPnonvede maiun tokenrifiutato. Se stopproposto, verificare soloa senzaappendere
 lostop; se residuocontesto/budgetinsufficiente usaregreedyordinario.
 Cancellazione: nonpubblicareproposte, cache/history coerenti oresetobbligatorio.
7. CLI--mtp opzionale; ordinarioimmutato. Bufferlogits2VsoloMTP. Metricheproposta,
 verifica,restore,accettazioni/proposte,tokenperappend,memoria. TinyfixtureMTP
 dev’esserenondegenere: oggi tuttiipayloadMTP sonozeri. Gatepredictorteacherforced
 CPU, sequenzegreedyidentiche, forcedaccept/reject/stop/capacity/fault e realistici.
 Conservarebenchprima/dopoequivalenti, inclusaeventualeregressione. NessunPASSMTP
 finché manca predictor reale e identitàverificata.
