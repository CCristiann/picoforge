/* pf_api.h — the wire formats of pf-server: OpenAI Chat Completions and
 * Anthropic Messages, both turned into one normalized request, and one reply
 * turned back into either (a JSON body or Server-Sent Events).
 *
 * Normalized means what qwen4_chat_render accepts: roles system/user/
 * assistant/tool, string contents, tool-call arguments as JSON objects, tool
 * definitions as {type, function: {name, description, parameters}}. Fields
 * the template has no use for (ids, cache hints, names) are dropped here, so
 * the same conversation renders to the same bytes whichever client sent it. */
#ifndef PICOFORGE_PF_API_H
#define PICOFORGE_PF_API_H
#include "pf_json.h"
#include "pf_tools.h"

typedef enum { PF_OPENAI, PF_ANTHROPIC } PfProtocol;

typedef struct {
    PfProtocol protocol;
    Qwen4Json messages;               /* normalized array */
    Qwen4Json tools;                  /* normalized array, count 0: none */
    Qwen4Option thinking;             /* QW4_OMITTED: the server's default */
    Qwen4Effort effort;               /* QW4_EFFORT_OMITTED: the server's default */
    uint32_t max_tokens;              /* 0: the room left in the context */
    float temperature, top_p, min_p;  /* < 0: the server's default */
    int top_k;                        /* < 0: the server's default */
    bool stream, stream_usage;
    const char *model;                /* echoed back */
} PfChatRequest;

/* false: a client error, the reason in err (answered with HTTP 400). */
bool pf_openai_request(PfArena *, const Qwen4Json *body, PfChatRequest *, char *err, size_t errlen);
bool pf_anthropic_request(PfArena *, const Qwen4Json *body, PfChatRequest *, char *err, size_t errlen);

/* ------------------------------------------------------------ the reply
 * The generated text is "reasoning</think>content<tool_call>...": the
 * splitter routes the bytes as they are decoded, holding back only what could
 * still become one of the two tags. */
typedef enum { PF_REASONING, PF_CONTENT } PfPart;
typedef struct {
    int mode;                         /* 0 reasoning, 1 content, 2 tool calls */
    bool content_started;             /* leading blank lines of the content are dropped */
    Buf hold, reasoning, content, calls;
    void (*delta)(void *ctx, PfPart, const char *, size_t); void *ctx;
} PfSplit;
void pf_split_init(PfSplit *, bool thinking, void (*delta)(void *, PfPart, const char *, size_t), void *ctx);
void pf_split_feed(PfSplit *, const char *, size_t);
void pf_split_finish(PfSplit *);      /* flushes what was held back */
void pf_split_free(PfSplit *);

#define PF_MAX_CALLS 16
typedef struct { char id[40]; const char *name; Qwen4Json args; } PfToolCall;
typedef struct {
    const char *reasoning, *content;  /* trimmed */
    PfToolCall calls[PF_MAX_CALLS]; size_t ncalls;
    const char *finish;               /* "stop", "length" or "tool_calls" */
    size_t prompt, cached, generated; /* tokens: the prompt, its part reused, the reply */
} PfReply;
/* The reply's parts from a finished splitter. Parameter values become JSON
 * typed by the tool's schema: a "string" parameter stays a string, anything
 * else is parsed as JSON when it parses. */
void pf_reply_build(PfArena *, const PfSplit *, const Qwen4Json *tools, PfReply *);
/* The reply as a normalized assistant message (no reasoning): the same tree
 * a client's copy of it normalizes to, which is how the cache recognizes it. */
Qwen4Json pf_reply_message(PfArena *, const PfReply *);
/* The message as the client sent it, without its reasoning. */
Qwen4Json pf_message_without_reasoning(PfArena *, const Qwen4Json *message);

/* ------------------------------------------------------------ output */
typedef struct {
    int fd; PfProtocol protocol; bool stream, stream_usage, gone;
    char id[48]; const char *model; long created;
    int block, blocks;                /* Anthropic stream: the open block (-1 none, else PfPart), blocks sent */
    size_t prompt, cached;
} PfOut;
void pf_out_begin(PfOut *, size_t prompt_tokens, size_t cached_tokens);   /* stream headers + the first event */
void pf_out_delta(PfOut *, PfPart, const char *, size_t);
void pf_out_end(PfOut *, const PfReply *);
/* An error body in the protocol's shape (type: e.g. "invalid_request_error"). */
void pf_error_body(Buf *, PfProtocol, const char *type, const char *code, const char *message);
void pf_random_id(char *out, size_t cap, const char *prefix);
#endif
