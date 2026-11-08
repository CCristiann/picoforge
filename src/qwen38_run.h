/* Host driver shared by the one-shot CLI and the REPL: one loaded model, one
 * graph, and one request = prefill of some tokens, then greedy decode with
 * optional MTP. Only orchestration lives here; the graph calls are the same. */
#ifndef PICOFORGE_QWEN38_RUN_H
#define PICOFORGE_QWEN38_RUN_H
#include "common.h"
#include "qwen38_forward_gpu.h"
#include "qwen38_g_internal.h"
#include "qwen38_text.h"
#include <signal.h>
#include <stdio.h>

/* Cooperative stop, checked between appends (SIGUSR1, and SIGINT in the REPL). */
extern volatile sig_atomic_t q38_stopped;

/* R4 draft policy (docs/specs/qwen38/R4-mtp.md): k drafts per cycle, fixed or
 * chosen to maximize expected tokens / cost with measured n-row append costs. */
typedef struct {
    int fixed;                              /* 0..3, or -1 = auto (cost model) */
    double cost[QWEN38_VERIFY_ROWS+1];      /* cost[n]: n-row append in 1-row appends */
    double delta;                           /* one draft (predictor row + LM head) */
    double mu;                              /* keeping the predictor aligned, paid at every k */
    double alpha[QWEN38_VERIFY_ROWS-1];     /* P(draft j accepted | drafts < j accepted) */
    uint32_t idle;                          /* k=0 tokens since the last probe draft */
} Q38Policy;
/* value: "auto" or "0".."3" (NULL = auto); costs: NULL or "c2,c3,c4,delta[,mu]". Dies on bad input. */
void q38_policy_init(Q38Policy *, const char *value, const char *costs);
size_t q38_policy_choose(const Q38Policy *, size_t kmax);

typedef struct {
    Gguf g; Qwen38Config *config; Qwen38Weights *weights; Qwen38Tokenizer *tok;
    const Qwen38ConfigView *c; GpuWeightStore *store;
    void *device, *library;                 /* borrowed from the CLI's ARC scope */
    Qwen38GpuForward *graph; Qwen38Decoder *decoder; Q38Profile *profile;
    float *logits; Qwen38F32Mut y;          /* one row, two with MTP */
    uint32_t capacity, chunk; bool mtp;
    uint32_t verify_rows;                   /* rows a verify append may take (<= QWEN38_VERIFY_ROWS) */
    Q38Policy policy;
    size_t last_count;                      /* tokens in the last trunk append */
    struct Q38ExpertPool *pool; int pool_fd; /* R5.1b: PICOFORGE_STREAM_EXPERTS, else NULL */
} Q38Run;

typedef struct {
    uint32_t chunk, max_new, stop_after, forced;
    FILE *logfile;                          /* --logits rows; forces the logits path */
    bool session;                           /* JSON events instead of raw text */
    /* REPL only: before a plain decode append, pair the MTP predictor's pending
     * rows (no logits read back), so the next turn can reuse the cache. The
     * one-shot path leaves it off and stays call-for-call as before. */
    bool keep_predictor;
    /* Optional text sink (REPL formatting); NULL writes the bytes to stdout. */
    void (*print)(void *, const uint8_t *, size_t); void *print_ctx;
} Q38TurnOptions;

typedef struct {
    Qwen38GpuForwardStats prefill, decode, mtp;
    size_t got, proposed, accepted, target_appends, rollback_count, verified_tokens;
    double prefill_start, prefill_end, first_token, decode_start, end;
    double mtp_prefill_s, draft_s, verify_s, snapshot_s, replay_s;   /* R4: replay_s = keep (no replay) */
    size_t depth[QWEN38_VERIFY_ROWS];       /* cycles by drafts k (k=0: plain appends) */
    size_t proposed_at[QWEN38_VERIFY_ROWS-1], accepted_at[QWEN38_VERIFY_ROWS-1];
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
/* One measured repetition of `picoforge bench` (src/qwen38_bench.c). */
typedef struct { size_t prompt_tokens, decode_tokens; double prefill_s, decode_s, ttft_s; } Q38BenchRow;
/* Median and p10/p90 of prefill tok/s, decode tok/s and TTFT on stdout;
 * weight_mb > 0 (measured bytes/token, in MB) adds GB/s against 307. */
void q38_bench_report(const Q38BenchRow *, size_t, double weight_mb);
void q38_bench_csv(FILE *, const Q38BenchRow *, size_t);
#endif
