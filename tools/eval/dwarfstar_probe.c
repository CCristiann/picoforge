/* A measurement client of pinned DwarfStar's public API, NOT a picoforge engine.
 * MODEL PROMPT_FILE OUTPUT_STEM MAX_NEW MODE REPEATS
 * MODE: nothink, think, or raw:N (first N tokens, throughput only).
 * Each repetition creates fresh state; only the model and GPU caches persist.
 * Build recipe and upstream provenance: docs/qwen38-q2-baseline.md.
 */
#include "ds4.h"
#include "ds4_gpu.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PICOFORGE_DWARFSTAR_OBSERVE
void picoforge_ngram_report(const char *phase);
#else
static void picoforge_ngram_report(const char *phase) { (void)phase; }
#endif

static double now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) exit(1);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int positive(const char *s) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || !*s || *end || n < 1 || n > INT_MAX) exit(2);
    return (int)n;
}

static char *read_prompt(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) exit(1);
    long size = ftell(file);
    if (size < 0 || size > 1024 * 1024 || fseek(file, 0, SEEK_SET)) exit(2);
    char *text = calloc((size_t)size + 1, 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) exit(1);
    if (memchr(text, 0, (size_t)size) || fclose(file)) exit(2);
    return text;
}

static FILE *output(const char *stem, int repeat, const char *suffix) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s.%d.%s", stem, repeat, suffix);
    if (n < 0 || (size_t)n >= sizeof(path)) exit(2);
    FILE *file = fopen(path, "wx");
    if (!file) { perror(path); exit(1); }
    return file;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL PROMPT STEM MAX_NEW nothink|think|raw:N REPEATS\n", argv[0]);
        return 2;
    }
    const int max_new = positive(argv[4]), repeats = positive(argv[6]);
    const bool raw = strncmp(argv[5], "raw:", 4) == 0;
    const ds4_think_mode mode = strcmp(argv[5], "think") == 0 ? DS4_THINK_HIGH : DS4_THINK_NONE;
    if (!raw && strcmp(argv[5], "think") && strcmp(argv[5], "nothink")) return 2;
    const int raw_count = raw ? positive(argv[5] + 4) : 0;
    if (max_new >= 8192 || repeats > 100 || raw_count >= 8192) return 2;
    char *text = read_prompt(argv[2]);
    const double start = now();
    const ds4_engine_options options = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 8192, .prefill_chunk = 1024, .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f, .power_percent = 100, .placement_session_count_hint = 1};
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options)) return 1;
    const double loaded = now();
    if (!ds4_engine_is_qwen4(engine)) return 2;
    ds4_tokens prompt = {0};
    if (raw) {
        ds4_tokenize_text(engine, text, &prompt);
        if (prompt.len < raw_count) return 2;
        prompt.len = raw_count;
    } else {
        ds4_encode_chat_prompt(engine, "", text, mode, &prompt);
    }
    free(text);
    if (prompt.len <= 0 || prompt.len + max_new >= 8192) return 2;
    FILE *input = output(argv[3], 0, "prompt.ids");
    for (int i = 0; i < prompt.len; i++) fprintf(input, "%d\n", prompt.v[i]);
    if (fclose(input)) return 1;
    fprintf(stderr, "probe: load_s=%.9f prompt_tokens=%d context=8192 chunk=1024 mtp=false\n",
            loaded - start, prompt.len);
    for (int repeat = 0; repeat < repeats; repeat++) {
        FILE *reply = output(argv[3], repeat, "txt"), *ids = output(argv[3], repeat, "ids");
        ds4_session *session = NULL;
        const double begin = now();
        if (ds4_session_create(&session, engine, 8192)) return 1;
        const double created = now();
        ds4_session_gpu_warmup(session);
        picoforge_ngram_report("before_prefill");
        const double prefill_start = now();
        char error[256] = {0};
        if (ds4_session_sync(session, &prompt, error, sizeof(error))) {
            fprintf(stderr, "probe: prefill failed: %s\n", error); return 1;
        }
        const double prefill_end = now();
        picoforge_ngram_report("after_prefill");
        const double decode_start = now();
        int generated = 0, evals = 0;
        double first = 0.0;
        bool stopped = false;
        while (generated < max_new) {
            int token = ds4_session_argmax(session);
            if (ds4_token_is_stop_for_think_mode(engine, token, mode)) { stopped = true; break; }
            size_t size = 0;
            char *piece = ds4_token_text(engine, token, &size);
            if (!piece || fwrite(piece, 1, size, reply) != size || fflush(reply)) return 1;
            free(piece);
            if (!generated) first = now();
            fprintf(ids, "%d\n", token);
            generated++;
            if (generated == max_new) break;
            if (ds4_session_eval(session, token, error, sizeof(error))) {
                fprintf(stderr, "probe: decode failed: %s\n", error); return 1;
            }
            evals++;
        }
        const double end = now();
        picoforge_ngram_report("after_decode");
        printf("{\"repeat\":%d,\"prompt_tokens\":%d,\"generated\":%d,\"decode_evals\":%d,"
               "\"engine_start_monotonic_s\":%.9f,\"prefill_start_monotonic_s\":%.9f,"
               "\"decode_start_monotonic_s\":%.9f,"
               "\"load_s\":%.9f,\"session_s\":%.9f,\"warmup_s\":%.9f,\"prefill_s\":%.9f,"
               "\"decode_s\":%.9f,\"request_s\":%.9f,\"first_token_from_start_s\":%.9f,"
               "\"ttft_session_s\":%.9f,\"stopped\":%s}\n", repeat, prompt.len, generated, evals,
               start, prefill_start, decode_start,
               loaded-start, created-begin, prefill_start-created, prefill_end-prefill_start,
               end-decode_start, end-begin, first > 0.0 ? first-start : -1.0,
               first > 0.0 ? first-begin : -1.0, stopped ? "true" : "false");
        fflush(stdout);
        ds4_gpu_print_memory_report("probe before session free");
        ds4_session_free(session);
        if (fclose(reply) || fclose(ids)) return 1;
    }
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
