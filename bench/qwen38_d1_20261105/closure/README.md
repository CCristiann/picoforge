# Decode II closure — real-model logits, old binary against new

Question: do the command split (D1.1) and the fused shared-expert activation (D1.2) change a
single bit of the model's output? The old binary is the one in the main checkout, built at
15:07 on 2026-11-05 from the code of PR #7/#8 (before the tidy-up and before any Decode II
change); the new binary is this tree (main + D1.2). Same checkpoint, same 153 ids (the short
prompt plus 16 forced), `--teacher-forced 16 --logits`, each launched through the guarded
launcher (release wait before and after, strict memory guard), one at a time.

| Binary | logits.f32 (16 steps x 248320 floats) | sha256 (first 20 hex) | generated ids |
|---|---:|---|---|
| old | 16 885 760 bytes | b52666de47670b84e257 | 16 |
| new | 16 885 760 bytes | b52666de47670b84e257 | 16, identical |

`cmp` finds no differing byte. The decode steps of this run are exactly the appends that the
split and the fused activation touch.
