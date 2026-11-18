/* Host driver shared by the one-shot CLI and the REPL: one loaded model, one
 * graph, and one request = prefill of some tokens, then greedy decode with
 * optional MTP. Only orchestration lives here; the graph calls are the same. */
#ifndef PICOFORGE_QWEN38_RUN_H
#define PICOFORGE_QWEN38_RUN_H
#include "qwen38_sample.h"
#include <stdlib.h>
#include "common.h"
#include "qwen38_forward_gpu.h"
#include "qwen38_g_internal.h"
#include "qwen38_text.h"
#include <signal.h>
#include <stdio.h>

/* Cooperative stop, checked between appends (SIGUSR1, and SIGINT in the REPL). */
extern volatile sig_atomic_t q38_stopped;

/* R4 draft policy (devlog, R4.3): k drafts per cycle, fixed or
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

/* The chat's system message, as ds4 sends it (ds4_cli.c: "You are a helpful
 * assistant"). Without one, Qwen3.8 in thinking mode invents its own settings
 * ("Desired oververbosity 9") and reasons until the context runs out.
 * PICOFORGE_SYSTEM overrides it; an empty value sends none (the tests' oracles). */
static inline __attribute__((unused)) const char *q38_system_prompt(void) {
    const char *v=getenv("PICOFORGE_SYSTEM");
    return v ? v : "You are a helpful assistant";
}

typedef struct {
    Gguf g; Qwen38Config *config; Qwen38Weights *weights; Qwen38Tokenizer *tok;
    const Qwen38ConfigView *c; GpuWeightStore *store;
    void *device, *library;                 /* retained by q38_run_gpu, released by q38_run_close */
    Qwen38GpuForward *graph; Qwen38Decoder *decoder; Q38Profile *profile;
    float *logits; Qwen38F32Mut y;          /* one row, two with MTP */
    uint32_t capacity, chunk; bool mtp;
    uint32_t verify_rows;                   /* rows a verify append may take (<= QWEN38_VERIFY_ROWS) */
    Q38Policy policy;
    Q38Sampler sampler;                     /* temperature <= 0: greedy (bench, oracles) */
    size_t last_count;                      /* tokens in the last trunk append */
    struct Q38ExpertPool *pool; int pool_fd; /* R5.1b: PICOFORGE_STREAM_EXPERTS, else NULL */
    /* --power N / /power N: GPU duty cycle in percent (100: off). After each step (a prefill
     * chunk, a decode append or an MTP cycle) the host sleeps avg * (100 - N) / N, avg the
     * running mean of the step's wall time, so work / (work + sleep) = N%. It only spaces
     * the steps: the numbers do not change. Idea from DwarfStar fc80bd6 (MIT), ds4.c
     * graph_power_sleep. Measurements (bench, teacher forcing) refuse or skip it. */
    int power; double power_prefill_avg, power_decode_avg;
} Q38Run;
/* Sleep after a step of `work` seconds (avg: that kind of step's running mean). */
void q38_power_pause(const Q38Run *, double *avg, double work);

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
/* Load a model (src/qwen38_cli.m): open = GGUF, config, weights, tokenizer;
 * gpu = Metal device, kernels, weights (SSD streaming when needed); close frees all. */
void q38_run_open(Q38Run *, const char *path);
void q38_run_gpu(Q38Run *, const char *path, const char *metallib);
void q38_run_close(Q38Run *);
/* (Re)create graph, decoder and logits for a context size; the old ones go. */
void q38_run_graph(Q38Run *, uint32_t capacity);
void q38_check(Qwen38Status, const Qwen38Error *);
double q38_now(void);
/* Streamed reply text for a terminal (src/qwen38_repl.c): <think> tags dropped,
 * reasoning grey when color is on. Zero-initialize, set color, shade(in_think). */
typedef struct { bool color, in_think, newline; char pending[8]; size_t n; } Q38Printer;
void q38_printer_shade(Q38Printer *, bool think);
void q38_print_text(void *printer, const uint8_t *, size_t);
void q38_print_end(Q38Printer *);
/* Bring the graph to ids[0..total) and decode up to max_new (0: the room left)
 * ids into generated[], streaming the text to print (e.g. q38_print_text with a
 * Q38Printer). held: transcript length before this turn; a graph past it, or an
 * MTP predictor behind, resets and replays (*replayed). Returns the number of
 * ids prefilled. */
size_t q38_chat_continue(Q38Run *, const int32_t *ids, size_t held, size_t total, uint32_t max_new,
                         void (*print)(void *, const uint8_t *, size_t), void *print_ctx,
                         int32_t *generated, Q38Turn *, bool *replayed);
/* Interactive chat on stdin (src/qwen38_repl.c). max_new 0 = until stop or full context. */
int q38_repl(Q38Run *, bool thinking, uint32_t max_new, const char *ids_path);
/* One measured repetition of `picoforge bench` (src/qwen38_bench.c). */
typedef struct { size_t prompt_tokens, decode_tokens; double prefill_s, decode_s, ttft_s; } Q38BenchRow;
/* Median and p10/p90 of prefill tok/s, decode tok/s and TTFT on stdout;
 * weight_mb > 0 (measured bytes/token, in MB) adds GB/s against 307. */
void q38_bench_report(const Q38BenchRow *, size_t, double weight_mb);
void q38_bench_csv(FILE *, const Q38BenchRow *, size_t);
#endif
