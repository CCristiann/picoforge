/* config.c — read config.json without a JSON library.
 *
 * We are NOT writing a JSON parser. config.json is machine-generated, and
 * every value the forward pass needs is a top-level scalar, so a scanner
 * that locates "key" and reads the literal after the colon is enough:
 * ~60 lines instead of ~2000, and no dependency (CLAUDE.md principle #3).
 *
 * The price of that shortcut: strstr() would happily match a key nested
 * inside some other object. We pay it with strictness — every lookup that
 * fails aborts. An engine that silently starts on a default dimension is
 * worse than one that refuses to start at all.
 */
#include "picoforge.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("picoforge: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

char *slurp(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    if (fseek(f, 0, SEEK_END) != 0) die("cannot seek %s", path);
    long n = ftell(f);
    if (n < 0) die("cannot size %s", path);
    rewind(f);

    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory reading %s (%ld bytes)", path, n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("short read on %s", path);
    buf[n] = '\0';                     /* so every str* below is safe */
    fclose(f);

    if (len_out) *len_out = (size_t)n;
    return buf;
}

/* Position just past the colon that follows "key". Dies if absent. */
static const char *json_value(const char *json, const char *key) {
    char needle[64];
    snprintf(needle, sizeof needle, "\"%s\"", key);

    const char *p = strstr(json, needle);
    if (!p) die("config.json: key \"%s\" not found", key);
    p += strlen(needle);

    while (isspace((unsigned char)*p)) p++;
    if (*p != ':') die("config.json: no ':' after \"%s\"", key);
    p++;
    while (isspace((unsigned char)*p)) p++;
    return p;
}

static long json_int(const char *json, const char *key) {
    const char *p = json_value(json, key);
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) die("config.json: \"%s\" is not an integer", key);
    return v;
}

/* strtod, not strtol: rope_theta is written 1000000.0 and rms_norm_eps
 * is written 1e-06. Both are floats in JSON even when they look integral. */
static double json_num(const char *json, const char *key) {
    const char *p = json_value(json, key);
    char *end;
    double v = strtod(p, &end);
    if (end == p) die("config.json: \"%s\" is not a number", key);
    return v;
}

static bool json_bool(const char *json, const char *key) {
    const char *p = json_value(json, key);
    if (strncmp(p, "true", 4) == 0)  return true;
    if (strncmp(p, "false", 5) == 0) return false;
    die("config.json: \"%s\" is not a boolean", key);
    return false;                      /* unreachable; keeps the compiler calm */
}

static void json_str(const char *json, const char *key, char *out, size_t cap) {
    const char *p = json_value(json, key);
    if (*p != '"') die("config.json: \"%s\" is not a string", key);
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
    if (*p != '"') die("config.json: \"%s\" is unterminated or too long", key);
    out[i] = '\0';
}

void config_load(const char *model_dir, Qwen3Config *cfg) {
    char path[1024];
    snprintf(path, sizeof path, "%s/config.json", model_dir);
    char *json = slurp(path, NULL);

    /* Refuse anything that is not the architecture we implement. Qwen2 and
     * Qwen3 differ in ways (QK-norm, explicit head_dim) that would not
     * crash — they would quietly produce wrong numbers. */
    char model_type[32];
    json_str(json, "model_type", model_type, sizeof model_type);
    if (strcmp(model_type, "qwen3") != 0)
        die("not a Qwen3 config: model_type=\"%s\"", model_type);

    cfg->hidden_size             = (int)json_int(json, "hidden_size");
    cfg->num_hidden_layers       = (int)json_int(json, "num_hidden_layers");
    cfg->num_attention_heads     = (int)json_int(json, "num_attention_heads");
    cfg->num_key_value_heads     = (int)json_int(json, "num_key_value_heads");
    cfg->head_dim                = (int)json_int(json, "head_dim");
    cfg->intermediate_size       = (int)json_int(json, "intermediate_size");
    cfg->vocab_size              = (int)json_int(json, "vocab_size");
    cfg->max_position_embeddings = (int)json_int(json, "max_position_embeddings");
    cfg->bos_token_id            = (int)json_int(json, "bos_token_id");
    cfg->eos_token_id            = (int)json_int(json, "eos_token_id");
    cfg->rms_norm_eps            = (float)json_num(json, "rms_norm_eps");
    cfg->rope_theta              = (float)json_num(json, "rope_theta");
    cfg->tie_word_embeddings     = json_bool(json, "tie_word_embeddings");
    json_str(json, "torch_dtype", cfg->torch_dtype, sizeof cfg->torch_dtype);

    /* Invariants the rest of the engine will assume without re-checking. */
    if (cfg->num_attention_heads % cfg->num_key_value_heads != 0)
        die("GQA: %d Q heads is not a multiple of %d KV heads",
            cfg->num_attention_heads, cfg->num_key_value_heads);

    free(json);
}

void config_print(const Qwen3Config *cfg) {
    int q_dim  = cfg->num_attention_heads * cfg->head_dim;
    int kv_dim = cfg->num_key_value_heads * cfg->head_dim;

    printf("=== Qwen3 architecture summary ===\n");
    printf("layers                : %d\n", cfg->num_hidden_layers);
    printf("hidden_size           : %d\n", cfg->hidden_size);
    printf("vocab_size            : %d\n", cfg->vocab_size);
    printf("attention             : %d Q heads, %d KV heads (GQA: %d Q per KV head)\n",
           cfg->num_attention_heads, cfg->num_key_value_heads,
           cfg->num_attention_heads / cfg->num_key_value_heads);
    printf("head_dim              : %d (explicit; naive hidden/heads would give %d"
           " — the classic Qwen3 gotcha)\n",
           cfg->head_dim, cfg->hidden_size / cfg->num_attention_heads);
    printf("  Q proj              : %d -> %d\n", cfg->hidden_size, q_dim);
    printf("  K/V proj            : %d -> %d each\n", cfg->hidden_size, kv_dim);
    printf("  O proj              : %d -> %d\n", q_dim, cfg->hidden_size);
    printf("MLP (SwiGLU)          : %d -> %d -> %d\n",
           cfg->hidden_size, cfg->intermediate_size, cfg->hidden_size);
    printf("rope_theta            : %g\n", (double)cfg->rope_theta);
    printf("max positions         : %d\n", cfg->max_position_embeddings);
    printf("rms_norm_eps          : %g\n", (double)cfg->rms_norm_eps);
    /* Capitalised True/False, and the same parentheticals as the oracle,
     * so that `./picoforge DIR | diff - <(oracle DIR)` is EMPTY. The summary
     * is not decoration: it is the cheapest regression test in the project,
     * and it only works if both sides are byte-identical. */
    printf("tied embeddings       : %s (no separate lm_head tensor in the weights)\n",
           cfg->tie_word_embeddings ? "True" : "False");
    printf("weights dtype         : %s (oracle computes in fp32)\n", cfg->torch_dtype);
    printf("bos / eos             : %d / %d\n", cfg->bos_token_id, cfg->eos_token_id);
}
