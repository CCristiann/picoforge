/* pf_session.h — one model and one conversation, for pf-agent and, later,
 * pf-server.
 *
 * The conversation is a token transcript that only grows: each turn renders
 * through the chat template and appends only its own suffix, so the model
 * always sees exactly what it generated and a turn prefills only what is new
 * (src/qwen38_repl.c). Nothing here prints: the reply streams through a
 * callback and messages go to `note`. Saved sessions hold the transcript and
 * the whole graph state, so a resume reads a file instead of prefilling. */
#ifndef PICOFORGE_PF_SESSION_H
#define PICOFORGE_PF_SESSION_H
#include "pf_util.h"
#include "pf_tools.h"
#include "qwen38_run.h"
#include <limits.h>

typedef struct { const char *role, *content; } PfMsg;     /* "user", "assistant", "tool" */

typedef struct {
    void (*text)(void *ctx, const uint8_t *, size_t);       /* reply bytes as they are decoded */
    void (*progress)(void *ctx, size_t prefilled, size_t prompt, size_t generated);   /* optional */
    void *ctx;
} PfStream;

typedef struct {
    size_t prefilled, generated;      /* tokens */
    double prefill_s, decode_s;
    bool replayed, cancelled, length; /* graph rebuilt; stopped by the user; hit the token limit */
} PfReplyStats;

typedef struct {
    const char *model, *metallib;     /* absolute paths */
    uint32_t context;
    bool mtp; float temperature; int power;
} PfSessionOptions;

typedef struct {
    Q38Run run;
    int32_t *ids, *generated; size_t n;   /* the transcript is ids[0..n) */
    int32_t im_end;                       /* the end-of-turn token */
    bool thinking;                        /* reasoning before each reply */
    uint32_t max_new;                     /* tokens per reply, 0: the room left */
    char *system;                         /* the system prompt of the first turn (owned) */
    Qwen38Json tools; bool has_tools;     /* tool definitions for the template */
    char *model_path; uint64_t model_bytes;
    char workspace[PATH_MAX];             /* stored with a saved session */
    char name[65], title[120];            /* the saved session we continue; its first request */
    const char *script;                   /* --script: canned replies (tests) */
    const char *ids_path;                 /* --token-ids: transcript dump (tests) */
    void (*note)(void *ctx, const char *message); void *note_ctx;
} PfSession;

void pf_session_open(PfSession *, const PfSessionOptions *);   /* dies if the model cannot load */
void pf_session_close(PfSession *);
void pf_session_set_system(PfSession *, const char *text);
void pf_session_set_tools(PfSession *, const PfToolSpec *(*spec)(size_t), size_t count);
void pf_session_reset(PfSession *);       /* a new conversation */

/* Render msgs as the next turn and encode it after the transcript; returns the
 * number of new ids, 0 on error (noted: does not fit, bad text). */
size_t pf_session_append(PfSession *, const PfMsg *, size_t n);
/* The reply to the `added` ids just appended. It joins the transcript; returns
 * its text (special tokens dropped), or NULL if stopped before the first token. */
char *pf_session_reply(PfSession *, size_t added, const PfStream *, PfReplyStats *);
const char *pf_session_answer(const char *reply);   /* the text after the reasoning */

/* Saved sessions in $PICOFORGE_SESSIONS or ~/.picoforge/sessions. */
bool pf_session_save(PfSession *, const char *name);       /* NULL/"": the current name, else a timestamp */
bool pf_session_resume(PfSession *, const char *name);     /* also sets workspace, title, thinking */
void pf_session_list(Buf *out);
/* A saved session's model, context and MTP setting, before anything is loaded. */
bool pf_session_peek(const char *name, char model[PATH_MAX], uint32_t *context, bool *mtp);
void pf_session_dump(const PfSession *);   /* --token-ids */
#endif
