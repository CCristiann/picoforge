/* A: binary-safe tokenizer/template API. No implicit strlen or BOS. */
#ifndef PICOFORGE_QWEN38_TEXT_H
#define PICOFORGE_QWEN38_TEXT_H
#include "qwen38.h"
typedef struct Qwen38Tokenizer Qwen38Tokenizer;
typedef struct Qwen38Decoder Qwen38Decoder;
Qwen38Status qwen38_tokenizer_load(const Gguf *, const Qwen38Config *,
    Qwen38Tokenizer **, Qwen38Error *);
void qwen38_tokenizer_free(Qwen38Tokenizer *);
Qwen38Status qwen38_encode(const Qwen38Tokenizer *, Qwen38Bytes,
    int32_t *out, size_t token_capacity, size_t *needed, Qwen38Error *);
Qwen38Status qwen38_decode_bytes(const Qwen38Tokenizer *, const int32_t *,
    size_t count, bool skip_special, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen38Error *);
Qwen38Status qwen38_decode_utf8(const Qwen38Tokenizer *, const int32_t *,
    size_t count, bool skip_special, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen38Error *);
bool qwen38_token_is_stop(const Qwen38Tokenizer *, int32_t);
Qwen38Status qwen38_decoder_create(const Qwen38Tokenizer *, bool skip_special,
    Qwen38Decoder **, Qwen38Error *);
void qwen38_decoder_free(Qwen38Decoder *);
void qwen38_decoder_reset(Qwen38Decoder *);
Qwen38Status qwen38_decoder_append(Qwen38Decoder *, const int32_t *, size_t,
    uint8_t *, size_t byte_capacity, size_t *needed, Qwen38Error *);
Qwen38Status qwen38_decoder_finish(Qwen38Decoder *, uint8_t *, size_t byte_capacity,
    size_t *needed, Qwen38Error *);
/* Ordered, borrowed JSON tree. Finite numbers; no cycles, duplicate object
 * keys or unknown message schemas. NULL means absent, JSON_NULL explicit null. */
typedef struct Qwen38Json Qwen38Json;
typedef struct { Qwen38Bytes key; const Qwen38Json *value; } Qwen38JsonMember;
typedef enum {
    Q38_JSON_NULL, Q38_JSON_BOOL, Q38_JSON_INT, Q38_JSON_FLOAT,
    Q38_JSON_STRING, Q38_JSON_ARRAY, Q38_JSON_OBJECT
} Qwen38JsonKind;
struct Qwen38Json {
    Qwen38JsonKind kind;
    union {
        bool boolean;
        int64_t integer;
        double number;
        Qwen38Bytes string;
        struct { const Qwen38Json *items; size_t count; } array;
        struct { const Qwen38JsonMember *items; size_t count; } object;
    } as;
};
typedef enum { Q38_OMITTED, Q38_FALSE, Q38_TRUE } Qwen38Option;
typedef enum { Q38_EFFORT_OMITTED, Q38_EFFORT_LOW, Q38_EFFORT_MEDIUM,
    Q38_EFFORT_XHIGH } Qwen38Effort;
typedef struct {
    Qwen38Option enable_thinking, preserve_thinking;
    Qwen38Effort reasoning_effort;
    bool add_generation_prompt;
} Qwen38ChatOptions;
Qwen38Status qwen38_chat_render(const Qwen38Tokenizer *,
    const Qwen38Json *messages, const Qwen38Json *tools,
    Qwen38ChatOptions, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen38Error *);
#endif
