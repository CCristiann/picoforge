# Oracle CPU dei formati Qwen3.8

Risultati del 2026-10-12, perimetro B indipendente dalla BASE_CONCORDATA.
Questo incremento verifica la decodifica del checkpoint Q2; non implementa
operatori del runtime, API condivise o Metal e non misura perdita rispetto a BF16.
La spec di partenza è `82f38ce6b8c8ab16e99b08dd86e2c96f729312b4`.

## Riferimenti e indipendenza

L'adapter C include il sorgente originale, immutato, di
[DwarfStar fc80bd6](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c)
ed esegue `qwen4_ref_row` su piccoli buffer. Non apre il modello e definisce
`DS4_NO_GPU`. La build verifica revisioni, contenuto Git e SHA-256 di tutte le
dipendenze locali; conserva compilatore, comando, checksum del binario e licenza.
Il linker elimina il runtime inutilizzato: non occorrono ulteriori oggetti CPU.

Gli expected analitici controllano indirizzamento, scale e segni senza usare
il decoder picoforge. La griglia IQ2 viene dal riferimento esterno ed è
verificata contro il checksum della tabella canonica
[ggml 8bb3cf0](https://github.com/ggml-org/llama.cpp/blob/8bb3cf0d216a5ae48360179f2931daf284be4d01/ggml/src/ggml-common.h),
confrontata integralmente durante SPEC B. Il package `gguf` non è installato
né eseguito. `mathematical_rows` interpreta separatamente i bit in NumPy/FP64:
questa seconda implementazione locale, da sola, **non** è un oracle indipendente.
Se in futuro il runtime copia il decoder DwarfStar, serve un altro riferimento
eseguibile prima di dichiarare verificato il percorso nativo.

## Risultati registrati

`manifest.json` conserva i due report effettivi della suite, la provenienza
delle slice e gli esiti del replay. `slices.jsonl` contiene una riga per slice
con tensore, offset, indici, forme e checksum dei cinque artefatti. I pesi e
gli expected binari rimangono esclusivamente nel `build/` ignorato da Git.

- **21 test passati** sia CPU ordinaria sia ASan/UBSan, zero skip; UBSan fatale.
- Q2_K: blocchi da 84 byte/256 valori, tutte le 1.024 combinazioni di scala,
  minimo e codice, 256 posizioni indipendenti, confini dei gruppi e delle righe.
- IQ2_XXS: blocchi da 66 byte/256 valori, ordine dei byte, 32 slot di griglia,
  256 griglie × 128 segni × 16 scale = 524.288 vettori, 4.194.304 valori.
  L'ottavo segno è ricavato dalla parità dei primi sette.
- Scale FP16 finite estreme/subnormali; Q8_0 con tutti i codici signed;
  tutti i pattern finiti F16 e BF16; classi nonfinite scalari trasportate
  come dati del riferimento, senza fissare una policy per il runtime.
- Errori di forma, input troncato/eccedente, budget, provenienza alterata,
  EINTR, short read, EOF, errore I/O e checksum corrotti vengono esercitati.
- Slice reali: **27 down Q2_K + 54 gate/up IQ2_XXS**, layer 0/23/47,
  esperti 0/255/511, prima/riga centrale/ultima. **158.976 valori FP32**
  uguali bit per bit; estrazione e replay con ASan/UBSan passati.
- Down usa `Klogical=640`, `Kphysical=768`, stride 252 byte. La coda è zero
  nelle 27 righe campionate: non è una garanzia per tutto il checkpoint.
  Fixture sintetiche con coda ±65504 verificano che il prodotto la ignori,
  usando input di esattamente 640 float. Gate/up usa K=2560, stride 660 byte.

Tolleranza della decodifica finita: zero, confronto dei bit FP32. Gli input dei
prodotti sono one-hot e numeri dyadici con seed 3811; output e somma dei termini
assoluti sono FP64. Il replay richiede uguaglianza esatta degli expected salvati.
Questi prodotti preparano i futuri test delle riduzioni; non validano già una
GEMV/GEMM nativa. Non viene fissata qui una tolleranza per le API da consolidare.

## Riproduzione

Dalla radice del worktree, con checkout DwarfStar e ambiente già presenti:

```sh
B_ORACLE_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
B_ORACLE_DS4=/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline
B_ORACLE_MODEL=/Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf

"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py build "$B_ORACLE_DS4" build/qwen38-packed/repro-cpu
"$B_ORACLE_PY" -B tests/test_qwen38_packed.py --reference build/qwen38-packed/repro-cpu/reference --report build/qwen38-packed/repro-cpu-tests.json -v
"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py build "$B_ORACLE_DS4" build/qwen38-packed/repro-sanitized --sanitize
"$B_ORACLE_PY" -B tests/test_qwen38_packed.py --reference build/qwen38-packed/repro-sanitized/reference --report build/qwen38-packed/repro-sanitized-tests.json -v

"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py slices build/qwen38-packed/repro-sanitized/reference "$B_ORACLE_MODEL" build/qwen38-packed/repro-q2 --kind 10
"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py replay build/qwen38-packed/repro-sanitized/reference build/qwen38-packed/repro-q2
"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py slices build/qwen38-packed/repro-sanitized/reference "$B_ORACLE_MODEL" build/qwen38-packed/repro-iq2 --kind 16
"$B_ORACLE_PY" -B tools/oracle/qwen38_packed.py replay build/qwen38-packed/repro-sanitized/reference build/qwen38-packed/repro-iq2
```

Attesi: `Ran 21 tests ... OK` per ciascuna suite; `PASS: 27 real rows, 6804
payload bytes, bit-exact decode`, poi 54 righe/35640 byte per IQ2; replay di
27 e 54 righe. Output e report devono essere nuovi: non vengono sovrascritti.
I checksum degli artefatti sono confrontabili con `slices.jsonl`; quelli dei
manifest/binari possono cambiare con percorsi e compilatore.

## Budget e limiti

L'adapter limita input compresso e output decodificato a 32 MiB ciascuno per
richiesta; Python usa temporanei aggiuntivi. È un limite di trasporto, non una
misura del picco RSS. Il caso esaustivo IQ2 procede in 16 batch da 1 MiB di
output ciascuno. Le selezioni reali leggono in totale **42.444 byte di payload**
per coppia di estrazioni, più 11.025.376 byte di header per estrazione. Nessun
mmap completo, carico GPU o benchmark viene eseguito.

Il SHA-256 dell'intero checkpoint proviene dalla verifica locale già registrata
in `bench/qwen38_q2_20261011_phase0/checkpoint.verification.json`: non viene
ricalcolato leggendo 137 GiB. L'estrazione controlla inode, dimensione, mtime,
hash dell'header e stabilità dell'identità del file aperto; una copia con inode
diverso viene rifiutata. I checksum delle slice attestano i byte letti qui.

Non vengono testati in questo incremento i formati MTP Q4_K/MXFP4, il grafo MoE,
la generazione, il driver Metal o la qualità rispetto ai pesi BF16. Q2_K e
IQ2_XXS rimangono distinti da `q4_row`/`q8_row` e dall'INT2 nativo.
