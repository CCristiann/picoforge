# Prove native estese e benchmark

M5 Pro 64 GiB, 13 ottobre 2026. Base `31d3460`, branch
`coord/qwen38-native-next`; sorgenti/binari effettivi identificati dagli hash
nei manifest, anche per le prove eseguite con modifiche non ancora committate.
Nessun download, nuovo checksum integrale o variazione del limite wired.

- `suite/`: prompt, token ID, SHA e budget congelati prima dell'esecuzione.
- `extended/`: scalar; C, aritmetica e italiano completati. Contesto 1K
  interrotto dopo 618,55 s. Manifest volutamente incompleto; 2K/4K/8K non avviati.
  Le risposte sono preservate. C passa oracle/sanitizer ma usa fence vietate;
  aritmetica errata; italiano 256 parole, valutazione manuale separata.
- `qsa-paired/`: prima variante cooperativa, output bit-identici, 6,2–6,45×.
- `qsa-paired-v3/`: variante adottata, 0,5 s warm-up GPU e 25 misure per cella;
  output bit-identici nelle sei forme, ATTEND 27,55–55,93×. Include forme reali
  H=24, J=2, D=256, N fino a 8192 e selezione limitata a 2048.
- `native-paired/`: un warm-up + tre misure residenti per variante, prompt
  aritmetico da 61 token e 14 generati. Prefill mediano 9,889 → 8,647 s,
  decode 1,058 → 1,116 token/s, richiesta 22,181 → 20,303 s. Otto sequenze
  identiche, reset a ogni richiesta. Un pilot, non un benchmark di qualità.
- `extended-cooperative/`: sei casi completati, inclusi 8042 token reali di prompt
  con 28 generati e JSON corretto. Prefill 8K 2046,586 s, decode 25,296 s;
  pressione 1, nessuna crescita swap. Oracle coding PASS; fence vietate
  nel caso merge-ranges, conformità di formato distinta.
- `summary-initial.json` / [report iniziale](report-initial.html): fotografia
  dei soli casi completati e delle misure sopra, con lo storico G2/G3 distinto.
- `bounded-lazy/`: tre prefissi da 24 token contro CPU indipendente originale,
  ultima scelta stop in tutti i casi. Errore massimo 5,14e-5 < 1e-3;
  non è un confronto di continuazioni multitoken. Dati f32 restano in
  `build/qwen38-native-bounded-settled`, identificati dagli SHA nel report.
- `bounded-attempts/`: tentativi stock/lazy arrestati dal guard memoria,
  incluso il primo caso lazy numericamente superato; suite incomplete.
- `chunk-replay/`: 14 token identici dopo reset, chunk 128 e 7.
- [Report verificato](report-verified.html) / `summary-verified.json`: nuovo
  snapshot dei risultati, con qualità, fallimenti, scope dei gate e link grezzi.
- `browser/`: conversazione reale desktop/mobile, streaming, stop, reset,
  errore di capacità e recupero. Un solo worker nativo residente.
- `validation/`: log dei test, inclusi tentativi falliti e correzioni.

Il riavvio intervenuto durante il primo oracle bounded CPU ha lasciato
`build/qwen38-native-bounded-20261013` incompleto. Causa non determinata;
nessun risultato numerico aggiuntivo è dichiarato superato per quel tentativo.

Cache file/shader OS non controllate. Il load della CLI non garantisce la
residenza delle pagine: il primo prefill può includerla. Il tempo GPU si
sovrappone a wait; footprint, allocazioni Metal e wired di sistema non si
sommano. Memoria osservata anche durante il warm-up. N-gram disk-only:
`ngram_bytes` misura byte logici pread, non letture fisiche dal dispositivo.

Comandi e confini in [native-next.md](../../docs/specs/qwen38/native-next.md).
Questa raccolta precede MTP: [incremento e misure](../qwen38_mtp_20261013/README.md),
[report completo successivo](report-mtp-final.html). Il vecchio gate DwarfStar Metal stock FP16 resta FAIL,
distinto dal PASS CPU FP32 indipendente già registrato in G2/G3.
