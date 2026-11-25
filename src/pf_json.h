/* pf_json.h — JSON text <-> Qwen4Json trees, for pf-server.
 *
 * The chat template (qwen4_chat_render) takes a borrowed JSON tree; an HTTP
 * request arrives as JSON text. This is the parser in between, plus a writer
 * and a few builders for the normalized trees the server hands the template.
 * Every node lives in an arena: one request, one arena, one free. */
#ifndef PICOFORGE_PF_JSON_H
#define PICOFORGE_PF_JSON_H
#include "pf_util.h"
#include "qwen4_text.h"

typedef struct PfArenaBlock PfArenaBlock;
typedef struct { PfArenaBlock *head; } PfArena;
void *pf_alloc(PfArena *, size_t);          /* zeroed; dies when out of memory */
char *pf_strdup(PfArena *, const char *, size_t);
void pf_arena_free(PfArena *);

/* NULL on bad JSON, with a short reason in err. Strings come out as UTF-8,
 * escapes decoded; integers that fit int64 are QW4_JSON_INT. */
Qwen4Json *pf_json_parse(PfArena *, const char *text, size_t n, char *err, size_t errlen);
void pf_json_write(Buf *, const Qwen4Json *);   /* compact */

const Qwen4Json *pf_json_get(const Qwen4Json *object, const char *key);   /* NULL if absent or not an object */
bool pf_json_is(const Qwen4Json *, const char *string);                    /* a string equal to it */
/* A string member as a NUL-terminated copy (NULL if absent or not a string). */
const char *pf_json_str(PfArena *, const Qwen4Json *object, const char *key);
double pf_json_num(const Qwen4Json *object, const char *key, double fallback);

/* Builders: objects get a fixed capacity, members are appended in order. */
Qwen4Json pf_jstr(PfArena *, const char *s, size_t n);
Qwen4Json pf_jobject(PfArena *, size_t capacity);
Qwen4Json pf_jarray(PfArena *, size_t capacity);
void pf_jset(PfArena *, Qwen4Json *object, const char *key, Qwen4Json value);
void pf_jpush(Qwen4Json *array, Qwen4Json value);   /* within the capacity given */
#endif
