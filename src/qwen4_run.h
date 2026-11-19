/* Host driver shared by the one-shot CLI and the REPL: one loaded model, one
 * graph, and one request = prefill of some tokens, then greedy decode with
 * optional MTP. Only orchestration lives here; the graph calls are the same. */
#ifndef PICOFORGE_QWEN4_RUN_H
#define PICOFORGE_QWEN4_RUN_H
#include "qwen4_sample.h"
#include <stdlib.h>
#include "common.h"
#include "qwen4_forward_gpu.h"
#include "qwen4_graph_internal.h"
#include "qwen4_text.h"
#include <signal.h>
#include <stdio.h>

/* Cooperative stop, checked between appends (SIGUSR1, and SIGINT in the REPL). */
extern volatile sig_atomic_t qw4_stopped;

/* R4 draft policy (devlog, R4.3): k drafts per cycle, fixed or
 * chosen to maximize expected tokens / cost with measured n-row append costs. */
typedef struct {
    int fixed;                              /* 0..3, or -1 = auto (cost model) */
    double cost[QWEN4_VERIFY_ROWS+1];      /* cost[n]: n-row append in 1-row appends */
    double delta;                           /* one draft (predictor row + LM head) */
    double mu;                              /* keeping the predictor aligned, paid at every k */
    double alpha[QWEN4_VERIFY_ROWS-1];     /* P(draft j accepted | drafts < j accepted) */
    uint32_t idle;                          /* k=0 tokens since the last probe draft */
} QW4Policy;
/* value: "auto" or "0".."3" (NULL = auto); costs: NULL or "c2,c3,c4,delta[,mu]". Dies on bad input. */
void qw4_policy_init(QW4Policy *, const char *value, const char *costs);
size_t qw4_policy_choose(const QW4Policy *, size_t kmax);

/* The chat's system message, as ds4 sends it (ds4_cli.c: "You are a helpful
 * assistant"). Without one, Qwen3.8 in thinking mode invents its own settings
 * ("Desired oververbosity 9") and reasons until the context runs out.
 * PICOFORGE_SYSTEM overrides it; an empty value sends none (the tests' oracles). */
static inline __attribute__((unused)) const char *qw4_system_prompt(void) {
    const char *v=getenv("PICOFORGE_SYSTEM");
    return v ? v : "You are a helpful assistant";
}

typedef struct {
    Gguf g; Qwen4Config *config; Qwen4Weights *weights; Qwen4Tokenizer *tok;
    const Qwen4ConfigView *c; GpuWeightStore *store;
    void *device, *library;                 /* retained by qw4_run_gpu, released by qw4_run_close */
    Qwen4GpuForward *graph; Qwen4Decoder *decoder; QW4Profile *profile;
    float *logits; Qwen4F32Mut y;          /* one row, two with MTP */
    uint32_t capacity, chunk; bool mtp;
    uint32_t verify_rows;                   /* rows a verify append may take (<= QWEN4_VERIFY_ROWS) */
    QW4Policy policy;
    QW4Sampler sampler;                     /* temperature <= 0: greedy (bench, oracles) */
    size_t last_count;                      /* tokens in the last trunk append */
    struct QW4ExpertPool *pool; int pool_fd; /* R5.1b: PICOFORGE_STREAM_EXPERTS, else NULL */
    /* --power N / /power N: GPU duty cycle in percent (100: off). After each step (a prefill
     * chunk, a decode append or an MTP cycle) the host sleeps avg * (100 - N) / N, avg the
     * running mean of the step's wall time, so work / (work + sleep) = N%. It only spaces
     * the steps: the numbers do not change. Idea from DwarfStar fc80bd6 (MIT), ds4.c
     * graph_power_sleep. Measurements (bench, teacher forcing) refuse or skip it. */
    int power; double power_prefill_avg, power_decode_avg;
} QW4Run;
/* Sleep after a step of `work` seconds (avg: that kind of step's running mean). */
void qw4_power_pause(const QW4Run *, double *avg, double work);

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
    /* Optional, for a status line: after each prefill chunk (prefilled of
     * prompt tokens) and each emitted token (generated). Same ctx as print. */
    void (*progress)(void *, size_t prefilled, size_t prompt, size_t generated);
} QW4TurnOptions;

typedef struct {
    Qwen4GpuForwardStats prefill, decode, mtp;
    size_t got, proposed, accepted, target_appends, rollback_count, verified_tokens;
    double prefill_start, prefill_end, first_token, decode_start, end;
    double mtp_prefill_s, draft_s, verify_s, snapshot_s, replay_s;   /* R4: replay_s = keep (no replay) */
    size_t depth[QWEN4_VERIFY_ROWS];       /* cycles by drafts k (k=0: plain appends) */
    size_t proposed_at[QWEN4_VERIFY_ROWS-1], accepted_at[QWEN4_VERIFY_ROWS-1];
    const char *reason;                     /* "stop", "length" or "cancelled" */
} QW4Turn;

/* Append tokens[0..count) to the live graph, then decode up to max_new ids into
 * generated[]. The graph is not reset: the caller decides (one-shot: always;
 * REPL: only on rollback). With MTP the predictor must already be caught up. */
void qw4_turn(QW4Run *, const int32_t *tokens, size_t count, QW4TurnOptions,
              int32_t *generated, QW4Turn *);
/* Load a model (src/qwen4_cli.m): open = GGUF, config, weights, tokenizer;
 * gpu = Metal device, kernels, weights (SSD streaming when needed); close frees all. */
void qw4_run_open(QW4Run *, const char *path);
void qw4_run_gpu(QW4Run *, const char *path, const char *metallib, uint32_t capacity);   /* capacity: sizes the expert pool */
void qw4_run_close(QW4Run *);
/* (Re)create graph, decoder and logits for a context size; the old ones go.
 * Dies, before allocating, if the graph would not fit in the GPU budget. */
void qw4_run_graph(QW4Run *, uint32_t capacity);
void qw4_check(Qwen4Status, const Qwen4Error *);
double qw4_now(void);
/* Streamed reply text for a terminal (src/qwen4_repl.c): <think> tags dropped,
 * reasoning grey when color is on. Zero-initialize, set color, shade(in_think). */
typedef struct { bool color, in_think, newline; char pending[8]; size_t n; } QW4Printer;
void qw4_printer_shade(QW4Printer *, bool think);
void qw4_print_text(void *printer, const uint8_t *, size_t);
void qw4_print_end(QW4Printer *);
/* Bring the graph to ids[0..total) and decode up to max_new (0: the room left)
 * ids into generated[], streaming the text to print (e.g. qw4_print_text with a
 * QW4Printer). held: transcript length before this turn; a graph past it, or an
 * MTP predictor behind, resets and replays (*replayed). Returns the number of
 * ids prefilled. */
size_t qw4_chat_continue(QW4Run *, const int32_t *ids, size_t held, size_t total, uint32_t max_new,
                         void (*print)(void *, const uint8_t *, size_t), void *print_ctx,
                         void (*progress)(void *, size_t prefilled, size_t prompt, size_t generated),
                         int32_t *generated, QW4Turn *, bool *replayed);
/* Interactive chat on stdin (src/qwen4_repl.c). max_new 0 = until stop or full context. */
int qw4_repl(QW4Run *, bool thinking, uint32_t max_new, const char *ids_path);
/* One measured repetition of `pf-bench` (src/qwen4_bench.c). */
typedef struct { size_t prompt_tokens, decode_tokens; double prefill_s, decode_s, ttft_s; } QW4BenchRow;
/* Median and p10/p90 of prefill tok/s, decode tok/s and TTFT on stdout;
 * weight_mb > 0 (measured bytes/token, in MB) adds GB/s against 307. */
void qw4_bench_report(const QW4BenchRow *, size_t, double weight_mb);
void qw4_bench_csv(FILE *, const QW4BenchRow *, size_t);
#endif
