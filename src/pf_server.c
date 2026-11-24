/* pf_server.c — pf-server: the picoforge engine behind an HTTP API, for coding
 * agents and chat clients (opencode, Claude Code, Pi, ...).
 *
 * One model, one graph, one request at a time. Clients speak OpenAI Chat
 * Completions (/v1/chat/completions) or Anthropic Messages (/v1/messages);
 * pf_api turns both into the same normalized conversation, the chat template
 * renders it, and the engine continues from whatever part of it the graph
 * already holds.
 *
 * Why that last part matters: these APIs are stateless, so an agent resends
 * the whole conversation on every step, and a prefill of 30K tokens costs
 * more than a minute here. The graph keeps the tokens of the last request
 * (prompt + reply); a new request that extends them prefills only the new
 * part. Two things break a naive token match, and both are handled:
 *
 *   1. The client's copy of our reply is not our reply. Clients drop the
 *      reasoning, re-serialize tool arguments, trim text: re-rendered, the
 *      assistant message gives other tokens than the ones the model sampled.
 *      So when the conversation is "the last prompt + our reply + new
 *      messages" and the client's copy says the same thing as ours (compared
 *      as normalized trees, reasoning aside), we keep our exact tokens and
 *      render only the new messages after them (prompt_after_reply).
 *      The last few exchanges are remembered (MAX_EXCHANGES), not only the
 *      last one, so a request from another conversation in between does not
 *      lose them.
 *   2. The recurrent layers (Gated DeltaNet) cannot be rewound: a graph that
 *      went past the common prefix starts over. That is the case for an
 *      unrelated request in between (a title generator, a second session) and
 *      for a retried request. So states are checkpointed on disk (pf_cache):
 *      the live graph before a request throws it away, the system block of a
 *      long cold prompt (shared by every session of the same agent), and the
 *      end of a prompt that took a long prefill. A request then starts from
 *      the longest of: the live graph, a checkpoint, nothing.
 */
#include "pf_models.h"
#include "pf_api.h"
#include "pf_cache.h"
#include "pf_http.h"
#include "pf_session.h"
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_BODY (64u << 20)
#define DEFAULT_MAX_TOKENS 32768u    /* a reply when the client sets no limit (or less: the room left) */
#define MAX_EXCHANGES 8              /* past exchanges prompt_after_reply can recognize */

/* One answered request, as prompt_after_reply needs it: the prompt as the
 * client's messages render, how many messages, our reply in the canonical
 * form a client's copy of it is compared in, and the exact tokens of both. */
typedef struct { Buf prompt, reply; size_t count; int32_t *ids; size_t n; uint64_t serial; } Exchange;

typedef struct {
    PfSession s;                      /* the model, the graph and the live tokens s.ids[0..s.n) */
    int32_t *next;                    /* the prompt being built, swapped with s.ids */
    char model_id[128];
    /* Defaults for what a request leaves out. */
    bool thinking; Qwen4Effort effort;
    float temperature, top_p, min_p; int top_k; uint32_t max_tokens;
    const char *api_key;
    Exchange past[MAX_EXCHANGES]; uint64_t serial;
    PfCache cache;
    /* Per request. */
    int fd; PfOut out; PfSplit split; double gone_check;
    size_t base, total; double prefill_start, prefill_note;   /* prefill progress: tokens before this call, of total */
} Server;

static volatile sig_atomic_t quitting;
static void on_signal(int sig) { (void)sig; quitting = 1; qw4_stopped = 1; }

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...) {
    char when[16]; time_t now = time(NULL); struct tm tm;
    strftime(when, sizeof when, "%H:%M:%S", localtime_r(&now, &tm));
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "pf-server %s ", when); vfprintf(stderr, fmt, ap);
    va_end(ap);
}

/* ------------------------------------------------------------ rendering */

typedef struct { Qwen4ChatOptions o; const Qwen4Json *tools; } Render;

/* msgs[0..n) through the template; NULL with the reason in err. */
static char *render(const Server *sv, const Qwen4Json *msgs, size_t n, const Render *r, bool generation, size_t *len, char *err, size_t cap) {
    Qwen4Json list = {.kind = QW4_JSON_ARRAY, .as.array = {msgs, n}};
    Qwen4ChatOptions o = r->o; o.add_generation_prompt = generation;
    const Qwen4Json *tools = r->tools && r->tools->as.array.count ? r->tools : NULL;
    Qwen4Error e = {0};
    Qwen4Status rc = qwen4_chat_render(sv->s.run.tok, &list, tools, o, NULL, 0, len, &e);
    if (rc == QW4_RANGE)   /* past the template's 4 MiB of text: far past any context, and said so the same way */
        snprintf(err, cap, "This model's maximum context length is %u tokens. However, your messages are longer "
                 "than the 4 MiB of text a prompt may have.", sv->s.run.capacity);
    else if (rc) snprintf(err, cap, "%s", e.detail);
    if (rc) return NULL;
    char *text = malloc(*len+1);
    if (!text) die("pf-server: out of memory");
    qw4_check(qwen4_chat_render(sv->s.run.tok, &list, tools, o, (uint8_t *)text, *len, len, &e), &e);
    text[*len] = 0;
    return text;
}
/* A message as the cache compares it: alone after a placeholder user turn,
 * without reasoning, with fixed options (the bytes, not the meaning, decide). */
static char *canonical(const Server *sv, PfArena *a, const Qwen4Json *message, size_t *len) {
    Qwen4Json pair[2], content = pf_jstr(a, ".", 1);
    pair[0] = pf_jobject(a, 2);
    pf_jset(a, &pair[0], "role", pf_jstr(a, "user", 4)); pf_jset(a, &pair[0], "content", content);
    pair[1] = pf_message_without_reasoning(a, message);
    Render r = {.o = {.enable_thinking = QW4_FALSE}}; char err[256];
    return render(sv, pair, 2, &r, false, len, err, sizeof err);
}
static size_t encode(const Server *sv, const char *text, size_t n, int32_t *out, size_t cap, bool *full) {
    size_t count = 0; Qwen4Error e;
    Qwen4Status rc = qwen4_encode(sv->s.run.tok, (Qwen4Bytes){(const uint8_t *)text, n}, out, cap, &count, &e);
    if (rc == QW4_RANGE) count = cap;      /* past the tokenizer's 4 MiB input: more than fits, whatever the count */
    *full = rc == QW4_CAPACITY || rc == QW4_RANGE || count >= cap;
    if (rc && rc != QW4_CAPACITY && rc != QW4_RANGE) die("cannot encode a rendered prompt: %s", e.detail);
    return count;
}

/* ------------------------------------------------------------ the prompt */

/* Case 1 of the header: a past prompt, our reply to it, then new messages.
 * The prompt length in sv->next, 0 if no remembered exchange fits. */
static size_t after_exchange(Server *sv, PfArena *a, const Exchange *x, const PfChatRequest *rq, const Render *r, bool *full) {
    const Qwen4Json *m = rq->messages.as.array.items; size_t n = rq->messages.as.array.count, k = x->count;
    if (!x->ids || n <= k+1 || !pf_json_is(pf_json_get(&m[k], "role"), "assistant")) return 0;
    for (size_t i = k+1; i < n; i++) if (pf_json_is(pf_json_get(&m[i], "role"), "system")) return 0;
    char err[256]; size_t len, clen;
    char *before = render(sv, m, k, r, true, &len, err, sizeof err);
    bool same = before && len == x->prompt.n && !memcmp(before, x->prompt.p, len);
    free(before);
    char *theirs = same ? canonical(sv, a, &m[k], &clen) : NULL;
    same = theirs && clen == x->reply.n && !memcmp(theirs, x->reply.p, clen);
    free(theirs);
    if (!same) return 0;
    /* The new messages after a placeholder user turn, the placeholder's bytes cut
     * off (the template wants a user query; a tool result is not one). */
    Qwen4Json *tail = pf_alloc(a, (n-k)*sizeof *tail);
    tail[0] = pf_jobject(a, 2);
    pf_jset(a, &tail[0], "role", pf_jstr(a, "user", 4)); pf_jset(a, &tail[0], "content", pf_jstr(a, ".", 1));
    memcpy(tail+1, m+k+1, (n-k-1)*sizeof *tail);
    size_t plen, tlen, total = 0;
    char *placeholder = render(sv, tail, 1, r, false, &plen, err, sizeof err), *text = render(sv, tail, n-k, r, true, &tlen, err, sizeof err);
    bool ok = placeholder && text && plen <= tlen && !memcmp(placeholder, text, plen);
    if (ok) {
        /* Close our reply: it ends with <|im_end|> unless it was cut short. */
        Buf b = {0}; uint32_t cap = sv->s.run.capacity;
        buf_str(&b, x->n && x->ids[x->n-1] == sv->s.im_end ? "\n" : "<|im_end|>\n");
        buf_add(&b, text+plen, tlen-plen);
        memcpy(sv->next, x->ids, x->n*sizeof *x->ids);
        total = x->n + encode(sv, b.p, b.n, sv->next+x->n, cap-x->n, full);
        buf_free(&b);
    }
    free(placeholder); free(text);
    return ok ? total : 0;
}
static size_t prompt_after_reply(Server *sv, PfArena *a, const PfChatRequest *rq, const Render *r, bool *full) {
    const Exchange *order[MAX_EXCHANGES]; size_t n = 0;
    for (size_t i = 0; i < MAX_EXCHANGES; i++) if (sv->past[i].ids) order[n++] = &sv->past[i];
    for (size_t i = 1; i < n; i++)                     /* newest first: the likely one */
        for (size_t j = i; j && order[j]->serial > order[j-1]->serial; j--) { const Exchange *t = order[j]; order[j] = order[j-1]; order[j-1] = t; }
    for (size_t i = 0; i < n; i++) {
        size_t total = after_exchange(sv, a, order[i], rq, r, full);
        if (total) return total;
    }
    return 0;
}
/* Remember this exchange, replacing the oldest (or one with the same prompt:
 * a retried request supersedes its earlier answer). */
static void remember(Server *sv, Buf *prompt, const char *canon, size_t clen, size_t count) {
    Exchange *x = &sv->past[0];
    for (size_t i = 0; i < MAX_EXCHANGES; i++) {
        Exchange *y = &sv->past[i];
        if (y->ids && y->prompt.n == prompt->n && !memcmp(y->prompt.p, prompt->p, prompt->n)) { x = y; break; }
        if (!y->ids || y->serial < x->serial) x = y;
    }
    buf_free(&x->prompt); buf_free(&x->reply);
    x->prompt = *prompt; *prompt = (Buf){0};
    buf_add(&x->reply, canon, clen);
    x->ids = realloc(x->ids, (sv->s.n ? sv->s.n : 1)*sizeof *x->ids);
    if (!x->ids) die("pf-server: out of memory");
    memcpy(x->ids, sv->s.ids, sv->s.n*sizeof *x->ids);
    x->n = sv->s.n; x->count = count; x->serial = ++sv->serial;
}
/* The request as tokens in sv->next; *reuse: how many of them the graph's
 * tokens already match. 0 with err set on a client error. */
static size_t build_prompt(Server *sv, PfArena *a, const PfChatRequest *rq, const Render *r, size_t *reuse,
                           Buf *rendered, char *err, size_t cap) {
    size_t len; bool full = false;
    char *text = render(sv, rq->messages.as.array.items, rq->messages.as.array.count, r, true, &len, err, cap);
    if (!text) return 0;
    buf_add(rendered, text, len);
    size_t total = prompt_after_reply(sv, a, rq, r, &full), cap_tokens = sv->s.run.capacity;
    if (!total) total = encode(sv, text, len, sv->next, cap_tokens, &full);
    size_t i = 0;
    while (i < total && i < sv->s.n && sv->next[i] == sv->s.ids[i]) i++;
    *reuse = i;
    free(text);
    if (full || total+1 >= cap_tokens) {
        snprintf(err, cap, "This model's maximum context length is %u tokens. However, your messages resulted in %s%zu tokens.",
                 sv->s.run.capacity, full ? "more than " : "", total);
        return 0;
    }
    return total;
}

/* ------------------------------------------------------------ one generation */

static void on_text(void *ctx, const uint8_t *p, size_t n) { Server *sv = ctx; pf_split_feed(&sv->split, (const char *)p, n); }
static void on_delta(void *ctx, PfPart part, const char *p, size_t n) { Server *sv = ctx; pf_out_delta(&sv->out, part, p, n); }
/* After each prefill chunk and token: a client that hung up stops the work. */
/* A long prefill shows nothing to the client: say how far it is, in the log
 * and as an SSE keep-alive (ignored by parsers, but proof of life). */
static void prefill_note(Server *sv, size_t done, double now) {
    if (now-sv->prefill_note < 5) return;
    sv->prefill_note = now;
    double rate = now > sv->prefill_start ? (double)(done-sv->base)/(now-sv->prefill_start) : 0;
    logf_("prefill %zu/%zu tokens (%.0f t/s, about %.0f s left)\n", done, sv->total, rate,
          rate > 0 ? (double)(sv->total-done)/rate : 0.0);
    pf_out_keepalive(&sv->out);
}
static void on_progress(void *ctx, size_t prefilled, size_t prompt, size_t generated) {
    (void)prompt;
    Server *sv = ctx; double now = qw4_now();
    if (!generated) prefill_note(sv, sv->base+prefilled, now);
    if (now-sv->gone_check < 0.25) return;
    sv->gone_check = now;
    if (sv->out.gone || pf_http_gone(sv->fd)) { sv->out.gone = true; qw4_stopped = 1; }
}
static void sampler(Server *sv, const PfChatRequest *rq) {
    float t = rq->temperature >= 0 ? rq->temperature : sv->temperature, p = rq->top_p > 0 ? rq->top_p : sv->top_p;
    float mp = rq->min_p >= 0 ? rq->min_p : sv->min_p; int k = rq->top_k >= 0 ? rq->top_k : sv->top_k;
    qw4_sampler_init(&sv->s.run.sampler, t > 10 ? 10 : t, (uint32_t)k, p > 1 ? 1 : p, mp, (uint64_t)time(NULL) ^ (uint64_t)arc4random());
}
/* --script (tests): the next canned reply instead of a generation, appended to
 * the tokens as if sampled (the graph catches up on the next real request). */
static void scripted(Server *sv, size_t total, size_t *got) {
    PfSession *s = &sv->s;
    const char *end = strstr(s->script, "\n=====\n"), *start = s->script;
    size_t len = end ? (size_t)(end-start) : strlen(start);
    s->script += len + (end ? 7 : 0);
    if (len && start[len-1] == '\n') len--;
    Buf b = {0}; buf_add(&b, start, len); buf_str(&b, "<|im_end|>");
    bool full;
    *got = encode(sv, b.p, b.n, s->ids+total, s->run.capacity-total, &full);
    pf_split_feed(&sv->split, start, len);
    buf_free(&b);
}

/* Where the request starts (case 2 of the header): the live graph if the new
 * prompt extends it, else the longest checkpoint, else nothing. A live graph
 * about to be thrown away is checkpointed first. Runs before s.ids becomes
 * the new prompt (sv->next). Returns how many prompt tokens the graph holds
 * or will continue from. */
static size_t start_point(Server *sv, size_t reuse, size_t total) {
    PfSession *s = &sv->s; Qwen4Error e;
    size_t pos = (size_t)qwen4_gpu_forward_info(s->run.graph).next_pos;
    if (pos > reuse) pf_cache_save(&sv->cache, s, pos);
    size_t live = pos <= reuse ? pos : 0, loaded = pf_cache_load(&sv->cache, s, sv->next, total-1, live);
    if (loaded) return loaded;
    if (pos > reuse) { qw4_check(qwen4_gpu_forward_reset(s->run.graph, &e), &e); s->run.last_count = 0; return 0; }
    return reuse;
}
/* The tokens of the system block (system prompt + tool definitions), when the
 * prompt starts with one: the part every session of an agent shares. */
static size_t system_end(Server *sv, const Buf *rendered, size_t total) {
    const char *user = strstr(rendered->p, "<|im_start|>user");
    if (!user || user == rendered->p) return 0;
    bool full; size_t n = encode(sv, rendered->p, (size_t)(user-rendered->p), sv->next, sv->s.run.capacity, &full);
    return !full && n < total && !memcmp(sv->next, sv->s.ids, n*sizeof *sv->next) ? n : 0;
}
/* Prefill up to `upto`, then save the state there. False if stopped. */
static bool checkpoint(Server *sv, size_t *held, size_t upto, size_t *prefilled, double *seconds) {
    QW4Turn t; bool replayed;
    sv->base = *held;
    *prefilled += qw4_chat_prefill(&sv->s.run, sv->s.ids, *held, upto, on_progress, sv, &t, &replayed);
    *seconds += t.prefill_end-t.prefill_start;
    if (qw4_stopped) return false;
    *held = upto;
    pf_cache_save(&sv->cache, &sv->s, upto);
    return true;
}

static void chat(Server *sv, PfArena *a, const PfChatRequest *rq) {
    PfSession *s = &sv->s; char err[512]; size_t reuse = 0;
    bool thinking = rq->thinking == QW4_OMITTED ? sv->thinking : rq->thinking == QW4_TRUE;
    Render r = {.o = {.enable_thinking = thinking ? QW4_TRUE : QW4_FALSE, .reasoning_effort = rq->effort ? rq->effort : sv->effort},
                .tools = &rq->tools};
    Buf rendered = {0}, body = {0};
    size_t total = build_prompt(sv, a, rq, &r, &reuse, &rendered, err, sizeof err);
    if (!total) {
        pf_error_body(&body, rq->protocol, "invalid_request_error", strstr(err, "maximum context length") ? "context_length_exceeded" : NULL, err);
        pf_http_respond(sv->fd, 400, "application/json", body.p, body.n);
        logf_("400 %s\n", err);
        buf_free(&rendered); buf_free(&body);
        return;
    }
    /* Keep at least one token to append, or there are no logits to sample from. */
    if (reuse >= total) reuse = total-1;
    double start = qw4_now();
    bool script = s->script && *s->script;
    size_t held = script ? reuse : start_point(sv, reuse, total);
    int32_t *swap = s->ids; s->ids = sv->next; sv->next = swap;
    s->n = held;
    uint32_t room = s->run.capacity-(uint32_t)total-1, max_new = rq->max_tokens ? rq->max_tokens : sv->max_tokens;
    if (!max_new || max_new > room) max_new = room;

    sv->out = (PfOut){.fd = sv->fd, .protocol = rq->protocol, .stream = rq->stream, .stream_usage = rq->stream_usage,
                      .model = rq->model ? rq->model : sv->model_id};
    pf_split_init(&sv->split, thinking, on_delta, sv);
    sampler(sv, rq);
    QW4Turn t = {.reason = "stop"}; bool replayed = false; size_t prefilled = 0, got = 0; double prefill_s = 0;
    pf_out_begin(&sv->out, total, held);
    sv->gone_check = sv->prefill_start = sv->prefill_note = start; sv->total = total; sv->base = held;
    if (total-held > 2048) logf_("prefill of %zu new tokens (%zu reused) ...\n", total-held, held);
    if (script) { scripted(sv, total, &got); prefilled = total-reuse; }   /* what a generation would have prefilled */
    else {
        /* Checkpoints on the way: the system block of a cold prompt, the end of a long prefill. */
        size_t sys = system_end(sv, &rendered, total), min = sv->cache.min_tokens;
        bool going = !sv->cache.on || sys < min || sys <= held || pf_cache_has(&sv->cache, s, s->ids, sys) ||
                     checkpoint(sv, &held, sys, &prefilled, &prefill_s);
        if (going && sv->cache.on && total-1 >= held+min) going = checkpoint(sv, &held, total-1, &prefilled, &prefill_s);
        if (going) {
            sv->base = held;
            prefilled += qw4_chat_continue(&s->run, s->ids, held, total, max_new, on_text, sv, on_progress, s->generated, &t, &replayed);
            prefill_s += t.prefill_end-t.prefill_start;
            got = t.got;
            memcpy(s->ids+total, s->generated, got*sizeof *s->ids);
        } else t.reason = "cancelled";
    }
    s->n = total+got;                     /* even when stopped: the graph holds a prefix of these */
    pf_split_finish(&sv->split);
    pf_session_dump(s);

    PfReply reply = {.finish = !strcmp(t.reason, "length") ? "length" : "stop", .prompt = total,
                     .cached = total-prefilled, .generated = got};
    pf_reply_build(a, &sv->split, &rq->tools, &reply);
    if (!sv->out.gone) pf_out_end(&sv->out, &reply);
    /* Remember the exchange for prompt_after_reply (a cut reply too: the
     * client may send back what it received). */
    Qwen4Json mine = pf_reply_message(a, &reply); size_t len;
    char *canon = canonical(sv, a, &mine, &len);
    if (canon) remember(sv, &rendered, canon, len, rq->messages.as.array.count);
    free(canon);

    double decode_s = got ? t.end-t.decode_start : 0;
    logf_("%s %zu tokens (%zu reused%s), prefill %.1f t/s, %zu generated at %.1f t/s, %s, %.1f s\n",
          rq->protocol == PF_ANTHROPIC ? "messages" : "chat", total, reply.cached, replayed ? ", graph rebuilt" : "",
          prefill_s > 0 ? (double)prefilled/prefill_s : 0.0, got, decode_s > 0 && got > 1 ? (double)(got-1)/decode_s : 0.0,
          sv->out.gone ? "client gone" : reply.finish, qw4_now()-start);
    pf_split_free(&sv->split); buf_free(&rendered);
}

/* /v1/messages/count_tokens: the prompt as rendered, counted. */
static void count_tokens(Server *sv, const PfChatRequest *rq) {
    Render r = {.o = {.enable_thinking = (rq->thinking == QW4_OMITTED ? sv->thinking : rq->thinking == QW4_TRUE) ? QW4_TRUE : QW4_FALSE,
                      .reasoning_effort = rq->effort ? rq->effort : sv->effort}, .tools = &rq->tools};
    char err[512]; size_t len; bool full; Buf b = {0};
    char *text = render(sv, rq->messages.as.array.items, rq->messages.as.array.count, &r, true, &len, err, sizeof err);
    if (!text) {
        pf_error_body(&b, PF_ANTHROPIC, "invalid_request_error", NULL, err);
        pf_http_respond(sv->fd, 400, "application/json", b.p, b.n); buf_free(&b); return;
    }
    size_t n = encode(sv, text, len, sv->next, sv->s.run.capacity, &full);
    buf_fmt(&b, "{\"input_tokens\":%zu}", n);
    pf_http_respond(sv->fd, 200, "application/json", b.p, b.n);
    buf_free(&b); free(text);
}

/* ------------------------------------------------------------ routes */

static void respond_json(int fd, int status, const char *json) { pf_http_respond(fd, status, "application/json", json, strlen(json)); }
static void error(int fd, int status, PfProtocol p, const char *type, const char *message) {
    Buf b = {0}; pf_error_body(&b, p, type, NULL, message);
    pf_http_respond(fd, status, "application/json", b.p, b.n); buf_free(&b);
}
static void model_json(const Server *sv, Buf *b) {
    buf_str(b, "{\"id\":\""); json_escape(b, sv->model_id);
    buf_fmt(b, "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"picoforge\",\"context_length\":%u,"
               "\"max_model_len\":%u,\"type\":\"model\",\"display_name\":\"", sv->s.run.capacity, sv->s.run.capacity);
    json_escape(b, sv->model_id); buf_str(b, "\"}");
}
static void handle(Server *sv, const PfHttpRequest *hr) {
    const char *path = hr->path; char *query = strchr(hr->path, '?');
    if (query) *query = 0;
    bool post = !strcmp(hr->method, "POST"), get = !strcmp(hr->method, "GET");
    PfProtocol proto = !strncmp(path, "/v1/messages", 12) ? PF_ANTHROPIC : PF_OPENAI;
    if (!strcmp(hr->method, "OPTIONS")) { pf_http_respond(sv->fd, 204, NULL, "", 0); return; }
    if (get && (!strcmp(path, "/health") || !strcmp(path, "/"))) { respond_json(sv->fd, 200, "{\"status\":\"ok\"}"); return; }
    if (sv->api_key && strcmp(hr->auth, sv->api_key)) { error(sv->fd, 401, proto, "authentication_error", "invalid API key"); return; }
    if (get && !strcmp(path, "/v1/models")) {
        Buf b = {0}; buf_str(&b, "{\"object\":\"list\",\"data\":["); model_json(sv, &b); buf_str(&b, "]}");
        pf_http_respond(sv->fd, 200, "application/json", b.p, b.n); buf_free(&b); return;
    }
    if (get && !strncmp(path, "/v1/models/", 11)) {   /* one model whatever the name: the GGUF loaded is the model */
        Buf b = {0}; model_json(sv, &b); pf_http_respond(sv->fd, 200, "application/json", b.p, b.n); buf_free(&b); return;
    }
    bool chat_path = !strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions");
    bool messages = !strcmp(path, "/v1/messages"), counting = !strcmp(path, "/v1/messages/count_tokens");
    if (!chat_path && !messages && !counting) { error(sv->fd, 404, proto, "not_found_error", "no such endpoint"); return; }
    if (!post) { error(sv->fd, 405, proto, "invalid_request_error", "use POST"); return; }

    PfArena a = {0}; char err[512]; PfChatRequest rq;
    Qwen4Json *body = pf_json_parse(&a, hr->body, hr->body_n, err, sizeof err);
    bool ok = body && (proto == PF_OPENAI ? pf_openai_request(&a, body, &rq, err, sizeof err)
                                          : pf_anthropic_request(&a, body, &rq, err, sizeof err));
    if (!ok) { error(sv->fd, 400, proto, "invalid_request_error", err); logf_("400 %s\n", err); }
    else if (counting) count_tokens(sv, &rq);
    else chat(sv, &a, &rq);
    pf_arena_free(&a);
}

/* ------------------------------------------------------------ main */

static void usage(void) {
    fprintf(stderr,
        "usage: pf-server [-m MODEL] [options]\n"
        "  An OpenAI- and Anthropic-compatible HTTP server for the picoforge engine.\n"
        "  -m MODEL          a .gguf path or a short name (pf models); default: the only model\n"
        "  -c N              context in tokens (default 100000)\n"
        "  --host ADDR       listen address (default 127.0.0.1; 0.0.0.0 for the network)\n"
        "  --port N          (default 8080)\n"
        "  --api-key KEY     require 'Authorization: Bearer KEY' (or x-api-key)\n"
        "  --alias NAME      the model id clients see (default: the file name)\n"
        "  --no-think        no reasoning unless a request asks for it (--think: the default)\n"
        "  --effort E        default reasoning effort: low, medium (default) or high\n"
        "  --temp T --top-p P --top-k K --min-p M   default sampling (0.6, 0.95, 20, 0)\n"
        "  -n N              tokens per reply when the request sets no limit (default %u)\n"
        "  --no-mtp          no speculative decoding\n"
        "  --power N         GPU duty cycle in percent (default 100)\n"
        "  --cors            browser cross-origin headers\n"
        "  --cache-dir DIR   checkpoints of the graph state (default ~/.picoforge/kv-cache)\n"
        "  --cache-mb N      their disk budget (default 16384)\n"
        "  --cache-min N     the shortest prefix worth a checkpoint (default 4096 tokens)\n"
        "  --no-cache        no checkpoints\n", DEFAULT_MAX_TOKENS);
}
static uint32_t number(const char *s, uint32_t lo, uint32_t hi, const char *what) {
    char *end; errno = 0; unsigned long n = strtoul(s, &end, 10);
    if (errno || end == s || *end || n < lo || n > hi) die("pf-server: %s must be in [%u, %u]", what, lo, hi);
    return (uint32_t)n;
}
static float real(const char *s, float lo, float hi, const char *what) {
    char *end; float v = strtof(s, &end);
    if (*end || end == s || !(v >= lo && v <= hi)) die("pf-server: %s must be in [%g, %g]", what, (double)lo, (double)hi);
    return v;
}
static char *absolute(const char *path) {
    char cwd[PATH_MAX]; Buf b = {0};
    if (path[0] != '/' && getcwd(cwd, sizeof cwd)) buf_fmt(&b, "%s/", cwd);
    buf_str(&b, path); return b.p;
}
static void metallib_path(char out[PATH_MAX]) {   /* beside the real binary: the server runs from anywhere */
    char exe[PATH_MAX], resolved[PATH_MAX]; uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) || !realpath(exe, resolved)) die("pf-server: cannot find its own binary");
    *strrchr(resolved, '/') = 0;
    snprintf(out, PATH_MAX, "%s/picoforge.metallib", resolved);
}
static void note(void *ctx, const char *message) { (void)ctx; logf_("%s", message); }

int main(int argc, char **argv) {
    static Server sv = {.thinking = true, .effort = QW4_EFFORT_MEDIUM, .temperature = 0.6f, .top_p = 0.95f,
                        .top_k = 20, .max_tokens = DEFAULT_MAX_TOKENS};
    const char *model = NULL, *host = "127.0.0.1", *alias = NULL, *cache_dir = NULL; char *script = NULL;
    uint32_t cache_mb = 16384, cache_min = 4096; bool cache = true;
    int port = 8080;
    PfSessionOptions o = {.context = 100000, .mtp = true, .temperature = 0.6f, .power = 100};
    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i];
        if (!strcmp(opt, "-h") || !strcmp(opt, "--help")) { usage(); return 0; }
        else if (!strcmp(opt, "--no-think") || !strcmp(opt, "--think")) sv.thinking = !strcmp(opt, "--think");
        else if (!strcmp(opt, "--no-mtp")) o.mtp = false;
        else if (!strcmp(opt, "--cors")) pf_http_cors = true;
        else if (!strcmp(opt, "--no-cache")) cache = false;
        else if (i+1 == argc) { usage(); return 2; }
        else if (!strcmp(opt, "-m")) model = argv[++i];
        else if (!strcmp(opt, "-c")) o.context = number(argv[++i], 64, QWEN4_MAX_CONTEXT, "-c");
        else if (!strcmp(opt, "-n")) sv.max_tokens = number(argv[++i], 1, QWEN4_MAX_CONTEXT, "-n");
        else if (!strcmp(opt, "--host")) host = argv[++i];
        else if (!strcmp(opt, "--port")) port = (int)number(argv[++i], 1, 65535, "--port");
        else if (!strcmp(opt, "--api-key")) sv.api_key = argv[++i];
        else if (!strcmp(opt, "--alias")) alias = argv[++i];
        else if (!strcmp(opt, "--cache-dir")) cache_dir = argv[++i];
        else if (!strcmp(opt, "--cache-mb")) cache_mb = number(argv[++i], 1, 1u << 24, "--cache-mb");
        else if (!strcmp(opt, "--cache-min")) cache_min = number(argv[++i], 1, QWEN4_MAX_CONTEXT, "--cache-min");
        else if (!strcmp(opt, "--power")) o.power = (int)number(argv[++i], 1, 100, "--power");
        else if (!strcmp(opt, "--temp")) sv.temperature = real(argv[++i], 0, 10, "--temp");
        else if (!strcmp(opt, "--top-p")) sv.top_p = real(argv[++i], 0.01f, 1, "--top-p");
        else if (!strcmp(opt, "--top-k")) sv.top_k = (int)number(argv[++i], 0, 1000, "--top-k");
        else if (!strcmp(opt, "--min-p")) sv.min_p = real(argv[++i], 0, 1, "--min-p");
        else if (!strcmp(opt, "--effort")) {
            const char *e = argv[++i];
            sv.effort = !strcmp(e, "low") ? QW4_EFFORT_LOW : !strcmp(e, "medium") ? QW4_EFFORT_MEDIUM :
                        !strcmp(e, "high") ? QW4_EFFORT_XHIGH : (die("pf-server: --effort is low, medium or high"), QW4_EFFORT_MEDIUM);
        }
        else if (!strcmp(opt, "--script")) { FILE *f = fopen(argv[++i], "rb"); Buf b = {0}; char chunk[4096]; size_t got;   /* tests */
            if (!f) die("pf-server: cannot open %s", argv[i]);
            while ((got = fread(chunk, 1, sizeof chunk, f))) buf_add(&b, chunk, got);
            fclose(f); script = b.p ? b.p : strdup(""); }
        else if (!strcmp(opt, "--token-ids")) sv.s.ids_path = absolute(argv[++i]);   /* tests */
        else { usage(); return 2; }
    }
    char lib[PATH_MAX]; metallib_path(lib);
    char *chosen = pf_models_pick(model, argv[0]), *path = absolute(chosen); free(chosen);
    const char *base = strrchr(path, '/') ? strrchr(path, '/')+1 : path;
    snprintf(sv.model_id, sizeof sv.model_id, "%s", alias ? alias : base);
    char *dot = alias ? NULL : strstr(sv.model_id, ".gguf");
    if (dot) *dot = 0;

    int listener = pf_http_listen(host, port);    /* before loading: a busy port fails in a second, not a minute */
    if (listener < 0) return 1;
    signal(SIGPIPE, SIG_IGN);
    struct sigaction act = {0}; act.sa_handler = on_signal; sigemptyset(&act.sa_mask);
    sigaction(SIGINT, &act, NULL); sigaction(SIGTERM, &act, NULL);

    logf_("loading %s (context %u tokens) ...\n", path, o.context);
    o.model = path; o.metallib = lib;
    sv.s.note = note; sv.s.script = script;
    pf_session_open(&sv.s, &o);
    sv.next = malloc((size_t)sv.s.run.capacity*sizeof *sv.next);
    if (!sv.next) die("pf-server: out of memory");
    if (cache && !pf_cache_open(&sv.cache, cache_dir, (uint64_t)cache_mb << 20, cache_min)) logf_("kv cache: cannot use %s\n", sv.cache.dir);
    if (sv.cache.on) logf_("kv cache: %s (budget %u MiB, prefixes of %u+ tokens)\n", sv.cache.dir, cache_mb, cache_min);
    logf_("serving %s on http://%s:%d/v1 (context %u, thinking %s, MTP %s)\n", sv.model_id, host, port,
          sv.s.run.capacity, sv.thinking ? "on" : "off", o.mtp ? "on" : "off");

    while (!quitting) {
        int fd = accept(listener, NULL, NULL);
        if (fd < 0) { if (errno != EINTR) logf_("accept: %s\n", strerror(errno)); continue; }
        PfHttpRequest hr; int status;
        sv.fd = fd; qw4_stopped = 0;
        if (pf_http_read(fd, &hr, MAX_BODY, &status)) { handle(&sv, &hr); pf_http_request_free(&hr); }
        else if (status) error(fd, status, PF_OPENAI, "invalid_request_error", "malformed HTTP request");
        close(fd);
    }
    logf_("stopping\n");
    pf_cache_save(&sv.cache, &sv.s, (size_t)qwen4_gpu_forward_info(sv.s.run.graph).next_pos);   /* the conversation survives a restart */
    close(listener);
    pf_session_close(&sv.s);
    free(sv.next); free(path); free(script); free((char *)sv.s.ids_path);
    for (size_t i = 0; i < MAX_EXCHANGES; i++) { buf_free(&sv.past[i].prompt); buf_free(&sv.past[i].reply); free(sv.past[i].ids); }
    return 0;
}
