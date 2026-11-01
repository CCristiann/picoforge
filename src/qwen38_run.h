/* Host driver shared by the one-shot CLI and the REPL: one loaded model, one
 * graph, and one request = prefill of some tokens, then greedy decode with
 * optional MTP. Only orchestration lives here; the graph calls are the same. */
#ifndef PICOFORGE_QWEN38_RUN_H
#define PICOFORGE_QWEN38_RUN_H
#include "picoforge.h"
#include "qwen38_forward_gpu.h"
#include "qwen38_g_internal.h"
#include "qwen38_text.h"
#include <signal.h>
#include <stdio.h>

/* Cooperative stop, checked between appends (SIGUSR1, and SIGINT in the REPL). */
extern volatile sig_atomic_t q38_stopped;

typedef struct {
    Gguf g; Qwen38Config *config; Qwen38Weights *weights; Qwen38Tokenizer *tok;
    const Qwen38ConfigView *c; GpuWeightStore *store;
    void *device, *library;                 /* borrowed from the CLI's ARC scope */
    Qwen38GpuForward *graph; Qwen38Decoder *decoder; Q38Profile *profile;
    float *logits; Qwen38F32Mut y;          /* one row, two with MTP */
    uint32_t capacity, chunk; bool mtp;
    size_t last_count;                      /* tokens in the last trunk append */
} Q38Run;

typedef struct {
    uint32_t chunk, max_new, stop_after, forced;
    FILE *logfile;                          /* --logits rows; forces the logits path */
    bool session;                           /* JSON events instead of raw text */
    /* Optional text sink (REPL formatting); NULL writes the bytes to stdout. */
    void (*print)(void *, const uint8_t *, size_t); void *print_ctx;
} Q38TurnOptions;

typedef struct {
    Qwen38GpuForwardStats prefill, decode, mtp;
    size_t got, proposed, accepted, target_appends, rollback_count, verified_tokens;
    double prefill_start, prefill_end, first_token, decode_start, end;
    double mtp_prefill_s, draft_s, verify_s, snapshot_s, replay_s;
    const char *reason;                     /* "stop", "length" or "cancelled" */
} Q38Turn;

/* Append tokens[0..count) to the live graph, then decode up to max_new ids into
 * generated[]. The graph is not reset: the caller decides (one-shot: always;
 * REPL: only on rollback). With MTP the predictor must already be caught up. */
void q38_turn(Q38Run *, const int32_t *tokens, size_t count, Q38TurnOptions,
              int32_t *generated, Q38Turn *);
/* (Re)create graph, decoder and logits for a context size; the old ones go. */
void q38_run_graph(Q38Run *, uint32_t capacity);
void q38_check(Qwen38Status, const Qwen38Error *);
double q38_now(void);
/* Interactive chat on stdin (src/qwen38_repl.c). max_new 0 = until stop or full context. */
int q38_repl(Q38Run *, bool thinking, uint32_t max_new, const char *ids_path);
#endif
