/* pf_api.c — see pf_api.h. Sections: shared normalization · OpenAI request ·
 * Anthropic request · the reply splitter · the reply · output. */
#include "pf_api.h"
#include "pf_http.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------ shared normalization */

static bool fail(char *err, size_t cap, const char *fmt, const char *arg) {
    snprintf(err, cap, fmt, arg ? arg : "");
    return false;
}
/* Client text that goes into the prompt never spells a special token: the
 * tokenizer would turn the text "<|im_end|>" (a file being read, a web page)
 * into the real end-of-turn token and let it close the message (the same rule
 * as pf-agent's clean_text). */
static Qwen4Json clean_str(PfArena *a, const char *s, size_t n) {
    Buf b = {0};
    for (size_t i = 0; i < n; i++) {
        buf_add(&b, s+i, 1);
        if (s[i] == '<' && i+1 < n && s[i+1] == '|') buf_add(&b, " ", 1);
    }
    Qwen4Json v = pf_jstr(a, b.p ? b.p : "", b.n);
    buf_free(&b);
    return v;
}
/* A message content as one string: a string, null, or text parts. */
static bool content_text(PfArena *a, const Qwen4Json *c, Qwen4Json *out, char *err, size_t cap) {
    if (!c || c->kind == QW4_JSON_NULL) { *out = pf_jstr(a, "", 0); return true; }
    if (c->kind == QW4_JSON_STRING) { *out = clean_str(a, (const char *)c->as.string.data, c->as.string.len); return true; }
    if (c->kind != QW4_JSON_ARRAY) return fail(err, cap, "a message content must be a string or an array of parts", NULL);
    Buf b = {0};
    for (size_t i = 0; i < c->as.array.count; i++) {
        const Qwen4Json *part = c->as.array.items+i, *type = pf_json_get(part, "type"), *text = pf_json_get(part, "text");
        if (pf_json_is(type, "text") || pf_json_is(type, "input_text") || pf_json_is(type, "output_text")) {
            if (!text || text->kind != QW4_JSON_STRING) { buf_free(&b); return fail(err, cap, "a text part needs a string \"text\"", NULL); }
            buf_add(&b, text->as.string.data, text->as.string.len);
        } else if (part->kind == QW4_JSON_STRING) buf_add(&b, part->as.string.data, part->as.string.len);
        else {
            const char *t = type && type->kind == QW4_JSON_STRING ? (const char *)type->as.string.data : "?";
            buf_free(&b);
            return fail(err, cap, strstr(t, "image") ? "this model reads text only (content part \"%s\")"
                                                      : "unsupported content part \"%s\"", pf_strdup(a, t, strlen(t)));
        }
    }
    *out = clean_str(a, b.p ? b.p : "", b.n);
    buf_free(&b);
    return true;
}
static Qwen4Json message(PfArena *a, const char *role, Qwen4Json content, const Qwen4Json *reasoning, const Qwen4Json *calls) {
    Qwen4Json m = pf_jobject(a, 4);
    pf_jset(a, &m, "role", pf_jstr(a, role, strlen(role)));
    pf_jset(a, &m, "content", content);
    if (reasoning && reasoning->as.string.len) pf_jset(a, &m, "reasoning_content", *reasoning);
    if (calls && calls->as.array.count) pf_jset(a, &m, "tool_calls", *calls);
    return m;
}
static Qwen4Json call(PfArena *a, const char *name, size_t n, Qwen4Json args) {
    Qwen4Json fn = pf_jobject(a, 2), wrap = pf_jobject(a, 1);
    pf_jset(a, &fn, "name", pf_jstr(a, name, n)); pf_jset(a, &fn, "arguments", args);
    pf_jset(a, &wrap, "function", fn);
    return wrap;
}
static bool tool_def(PfArena *a, const Qwen4Json *name, const Qwen4Json *desc, const Qwen4Json *params, Qwen4Json *out, char *err, size_t cap) {
    if (!name || name->kind != QW4_JSON_STRING || !name->as.string.len) return fail(err, cap, "a tool needs a name", NULL);
    if (params && params->kind != QW4_JSON_OBJECT && params->kind != QW4_JSON_NULL) return fail(err, cap, "tool parameters must be a JSON schema object", NULL);
    Qwen4Json fn = pf_jobject(a, 3), t = pf_jobject(a, 2);
    pf_jset(a, &fn, "name", *name);
    if (desc && desc->kind == QW4_JSON_STRING) pf_jset(a, &fn, "description", *desc);
    if (params && params->kind == QW4_JSON_OBJECT) pf_jset(a, &fn, "parameters", *params);
    pf_jset(a, &t, "type", pf_jstr(a, "function", 8)); pf_jset(a, &t, "function", fn);
    *out = t;
    return true;
}
/* Sampling and output fields both APIs share. */
static void common_fields(const Qwen4Json *body, PfChatRequest *r) {
    r->temperature = (float)pf_json_num(body, "temperature", -1);
    r->top_p = (float)pf_json_num(body, "top_p", -1);
    r->min_p = (float)pf_json_num(body, "min_p", -1);
    r->top_k = (int)pf_json_num(body, "top_k", -1);
    const Qwen4Json *stream = pf_json_get(body, "stream");
    r->stream = stream && stream->kind == QW4_JSON_BOOL && stream->as.boolean;
}
static uint32_t max_tokens(const Qwen4Json *body, const char *a, const char *b) {
    double n = pf_json_num(body, a, pf_json_num(body, b, 0));
    return n > 0 && n < 4e9 ? (uint32_t)n : 0;
}

/* ------------------------------------------------------------ OpenAI request */

static Qwen4Effort effort_of(const Qwen4Json *v, Qwen4Option *thinking) {
    if (pf_json_is(v, "none") || pf_json_is(v, "minimal")) { *thinking = QW4_FALSE; return QW4_EFFORT_OMITTED; }
    if (pf_json_is(v, "low")) return QW4_EFFORT_LOW;
    if (pf_json_is(v, "medium")) return QW4_EFFORT_MEDIUM;
    if (pf_json_is(v, "high") || pf_json_is(v, "xhigh") || pf_json_is(v, "max")) return QW4_EFFORT_XHIGH;
    return QW4_EFFORT_OMITTED;
}
/* {"id", "type", "function": {"name", "arguments": "<JSON text>"}} -> {"function": {"name", "arguments": {...}}} */
static bool openai_call(PfArena *a, const Qwen4Json *c, Qwen4Json *out, char *err, size_t cap) {
    const Qwen4Json *fn = pf_json_get(c, "function"), *name = pf_json_get(fn, "name"), *args = pf_json_get(fn, "arguments");
    if (!name || name->kind != QW4_JSON_STRING || !name->as.string.len) return fail(err, cap, "a tool call needs function.name", NULL);
    Qwen4Json parsed = pf_jobject(a, 0);
    if (args && args->kind == QW4_JSON_OBJECT) parsed = *args;
    else if (args && args->kind == QW4_JSON_STRING && args->as.string.len) {
        char why[128]; Qwen4Json *v = pf_json_parse(a, (const char *)args->as.string.data, args->as.string.len, why, sizeof why);
        if (!v || v->kind != QW4_JSON_OBJECT) return fail(err, cap, "tool call arguments must be a JSON object (%s)", v ? "not an object" : pf_strdup(a, why, strlen(why)));
        parsed = *v;
    }
    *out = call(a, (const char *)name->as.string.data, name->as.string.len, parsed);
    return true;
}
bool pf_openai_request(PfArena *a, const Qwen4Json *body, PfChatRequest *r, char *err, size_t cap) {
    *r = (PfChatRequest){.protocol = PF_OPENAI};
    const Qwen4Json *msgs = pf_json_get(body, "messages");
    if (!msgs || msgs->kind != QW4_JSON_ARRAY || !msgs->as.array.count) return fail(err, cap, "\"messages\" must be a non-empty array", NULL);
    size_t n = msgs->as.array.count;
    r->messages = pf_jarray(a, n);
    Buf system = {0}; bool leading = true;
    for (size_t i = 0; i < n; i++) {
        const Qwen4Json *m = msgs->as.array.items+i, *role = pf_json_get(m, "role");
        Qwen4Json text;
        if (!content_text(a, pf_json_get(m, "content"), &text, err, cap)) { buf_free(&system); return false; }
        bool sys = pf_json_is(role, "system") || pf_json_is(role, "developer");
        /* Leading system messages merge into the one system block; a later one
         * becomes a user message, so the prefix already computed stays valid. */
        if (sys && leading) {
            if (system.n) buf_str(&system, "\n\n");
            buf_add(&system, text.as.string.data, text.as.string.len);
            continue;
        }
        if (leading && system.n) pf_jpush(&r->messages, message(a, "system", pf_jstr(a, system.p, system.n), NULL, NULL));
        leading = false;
        if (sys || pf_json_is(role, "user")) pf_jpush(&r->messages, message(a, "user", text, NULL, NULL));
        else if (pf_json_is(role, "tool")) pf_jpush(&r->messages, message(a, "tool", text, NULL, NULL));
        else if (pf_json_is(role, "assistant")) {
            const Qwen4Json *rc = pf_json_get(m, "reasoning_content"), *calls = pf_json_get(m, "tool_calls");
            if (!rc || rc->kind != QW4_JSON_STRING) rc = pf_json_get(m, "reasoning");
            Qwen4Json reasoning = rc && rc->kind == QW4_JSON_STRING ? clean_str(a, (const char *)rc->as.string.data, rc->as.string.len) : pf_jstr(a, "", 0);
            Qwen4Json list = pf_jarray(a, calls && calls->kind == QW4_JSON_ARRAY ? calls->as.array.count : 0);
            for (size_t j = 0; calls && calls->kind == QW4_JSON_ARRAY && j < calls->as.array.count; j++) {
                Qwen4Json c;
                if (!openai_call(a, calls->as.array.items+j, &c, err, cap)) { buf_free(&system); return false; }
                pf_jpush(&list, c);
            }
            pf_jpush(&r->messages, message(a, "assistant", text, &reasoning, &list));
        } else { buf_free(&system); return fail(err, cap, "unknown message role", NULL); }
    }
    if (leading && system.n) pf_jpush(&r->messages, message(a, "system", pf_jstr(a, system.p, system.n), NULL, NULL));
    buf_free(&system);

    const Qwen4Json *tools = pf_json_get(body, "tools");
    bool none = pf_json_is(pf_json_get(body, "tool_choice"), "none");
    r->tools = pf_jarray(a, tools && tools->kind == QW4_JSON_ARRAY ? tools->as.array.count : 0);
    for (size_t i = 0; !none && tools && tools->kind == QW4_JSON_ARRAY && i < tools->as.array.count; i++) {
        const Qwen4Json *fn = pf_json_get(tools->as.array.items+i, "function");
        Qwen4Json t;
        if (!pf_json_is(pf_json_get(tools->as.array.items+i, "type"), "function")) continue;
        if (!tool_def(a, pf_json_get(fn, "name"), pf_json_get(fn, "description"), pf_json_get(fn, "parameters"), &t, err, cap)) return false;
        pf_jpush(&r->tools, t);
    }
    common_fields(body, r);
    r->max_tokens = max_tokens(body, "max_completion_tokens", "max_tokens");
    r->effort = effort_of(pf_json_get(body, "reasoning_effort"), &r->thinking);
    const Qwen4Json *reasoning = pf_json_get(body, "reasoning");     /* {"effort": ...} (OpenRouter style) */
    if (!r->effort && r->thinking == QW4_OMITTED) r->effort = effort_of(pf_json_get(reasoning, "effort"), &r->thinking);
    const Qwen4Json *kw = pf_json_get(pf_json_get(body, "chat_template_kwargs"), "enable_thinking");
    if (kw && kw->kind == QW4_JSON_BOOL) r->thinking = kw->as.boolean ? QW4_TRUE : QW4_FALSE;
    const Qwen4Json *opts = pf_json_get(body, "stream_options");
    const Qwen4Json *usage = pf_json_get(opts, "include_usage");
    r->stream_usage = usage && usage->kind == QW4_JSON_BOOL && usage->as.boolean;
    r->model = pf_json_str(a, body, "model");
    return true;
}

/* ------------------------------------------------------------ Anthropic request */

/* A tool_result's content: a string or text blocks. */
static bool result_text(PfArena *a, const Qwen4Json *block, Qwen4Json *out, char *err, size_t cap) {
    if (!content_text(a, pf_json_get(block, "content"), out, err, cap)) return false;
    const Qwen4Json *is_error = pf_json_get(block, "is_error");
    if (is_error && is_error->kind == QW4_JSON_BOOL && is_error->as.boolean) {
        Buf b = {0}; buf_str(&b, "Error: "); buf_add(&b, out->as.string.data, out->as.string.len);
        *out = pf_jstr(a, b.p, b.n); buf_free(&b);
    }
    return true;
}
static bool anthropic_message(PfArena *a, const Qwen4Json *m, Qwen4Json *list, char *err, size_t cap) {
    const Qwen4Json *role = pf_json_get(m, "role"), *content = pf_json_get(m, "content");
    bool user = pf_json_is(role, "user");
    if (!user && !pf_json_is(role, "assistant")) return fail(err, cap, "a message role is \"user\" or \"assistant\"", NULL);
    if (!content || content->kind != QW4_JSON_ARRAY) {
        Qwen4Json text;
        if (!content_text(a, content, &text, err, cap)) return false;
        pf_jpush(list, message(a, user ? "user" : "assistant", text, NULL, NULL));
        return true;
    }
    size_t nb = content->as.array.count;
    Buf text = {0}, thinking = {0}; Qwen4Json calls = pf_jarray(a, nb);
    for (size_t i = 0; i < nb; i++) {
        const Qwen4Json *b = content->as.array.items+i, *type = pf_json_get(b, "type");
        if (pf_json_is(type, "text")) {
            const Qwen4Json *t = pf_json_get(b, "text");
            if (t && t->kind == QW4_JSON_STRING) buf_add(&text, t->as.string.data, t->as.string.len);
        } else if (user && pf_json_is(type, "tool_result")) {
            if (text.n) { pf_jpush(list, message(a, "user", clean_str(a, text.p, text.n), NULL, NULL)); text.n = 0; }
            Qwen4Json r;
            if (!result_text(a, b, &r, err, cap)) { buf_free(&text); buf_free(&thinking); return false; }
            pf_jpush(list, message(a, "tool", r, NULL, NULL));
        } else if (!user && pf_json_is(type, "thinking")) {
            const Qwen4Json *t = pf_json_get(b, "thinking");
            if (t && t->kind == QW4_JSON_STRING) buf_add(&thinking, t->as.string.data, t->as.string.len);
        } else if (!user && pf_json_is(type, "tool_use")) {
            const Qwen4Json *name = pf_json_get(b, "name"), *input = pf_json_get(b, "input");
            if (!name || name->kind != QW4_JSON_STRING || !name->as.string.len) { buf_free(&text); buf_free(&thinking); return fail(err, cap, "a tool_use block needs a name", NULL); }
            pf_jpush(&calls, call(a, (const char *)name->as.string.data, name->as.string.len,
                                  input && input->kind == QW4_JSON_OBJECT ? *input : pf_jobject(a, 0)));
        } else if (pf_json_is(type, "image") || pf_json_is(type, "document")) {
            buf_free(&text); buf_free(&thinking);
            return fail(err, cap, "this model reads text only", NULL);
        }   /* redacted_thinking and unknown blocks carry nothing the template can show */
    }
    if (user && text.n) pf_jpush(list, message(a, "user", clean_str(a, text.p, text.n), NULL, NULL));
    if (!user) {
        Qwen4Json r = clean_str(a, thinking.p ? thinking.p : "", thinking.n);
        pf_jpush(list, message(a, "assistant", clean_str(a, text.p ? text.p : "", text.n), &r, &calls));
    }
    buf_free(&text); buf_free(&thinking);
    return true;
}
bool pf_anthropic_request(PfArena *a, const Qwen4Json *body, PfChatRequest *r, char *err, size_t cap) {
    *r = (PfChatRequest){.protocol = PF_ANTHROPIC};
    const Qwen4Json *msgs = pf_json_get(body, "messages");
    if (!msgs || msgs->kind != QW4_JSON_ARRAY || !msgs->as.array.count) return fail(err, cap, "\"messages\" must be a non-empty array", NULL);
    size_t slots = 1;
    for (size_t i = 0; i < msgs->as.array.count; i++) {
        const Qwen4Json *c = pf_json_get(msgs->as.array.items+i, "content");
        slots += 1 + (c && c->kind == QW4_JSON_ARRAY ? c->as.array.count : 0);
    }
    r->messages = pf_jarray(a, slots);
    const Qwen4Json *system = pf_json_get(body, "system");
    if (system && system->kind != QW4_JSON_NULL) {
        Qwen4Json text;
        if (!content_text(a, system, &text, err, cap)) return false;
        if (text.as.string.len) pf_jpush(&r->messages, message(a, "system", text, NULL, NULL));
    }
    for (size_t i = 0; i < msgs->as.array.count; i++)
        if (!anthropic_message(a, msgs->as.array.items+i, &r->messages, err, cap)) return false;

    const Qwen4Json *tools = pf_json_get(body, "tools"), *choice = pf_json_get(body, "tool_choice");
    bool none = pf_json_is(pf_json_get(choice, "type"), "none");
    r->tools = pf_jarray(a, tools && tools->kind == QW4_JSON_ARRAY ? tools->as.array.count : 0);
    for (size_t i = 0; !none && tools && tools->kind == QW4_JSON_ARRAY && i < tools->as.array.count; i++) {
        const Qwen4Json *t = tools->as.array.items+i, *schema = pf_json_get(t, "input_schema");
        if (!schema) continue;     /* server tools (web search, ...) have no schema: not ours to run */
        Qwen4Json def;
        if (!tool_def(a, pf_json_get(t, "name"), pf_json_get(t, "description"), schema, &def, err, cap)) return false;
        pf_jpush(&r->tools, def);
    }
    common_fields(body, r);
    r->stream_usage = true;
    r->max_tokens = max_tokens(body, "max_tokens", "max_tokens");
    const Qwen4Json *thinking = pf_json_get(pf_json_get(body, "thinking"), "type");
    if (pf_json_is(thinking, "disabled")) r->thinking = QW4_FALSE;
    else if (pf_json_is(thinking, "enabled") || pf_json_is(thinking, "adaptive")) r->thinking = QW4_TRUE;
    r->effort = effort_of(pf_json_get(pf_json_get(body, "output_config"), "effort"), &r->thinking);
    r->model = pf_json_str(a, body, "model");
    return true;
}

/* ------------------------------------------------------------ the reply splitter */

static const char think_end[] = "</think>", call_start[] = "<tool_call>";

/* How many bytes at the end of s could be the start of tag. */
static size_t partial(const char *s, size_t n, const char *tag) {
    size_t t = strlen(tag);
    for (size_t k = t-1 < n ? t-1 : n; k; k--) if (!memcmp(s+n-k, tag, k)) return k;
    return 0;
}
static void out(PfSplit *s, PfPart part, const char *p, size_t n) {
    if (part == PF_CONTENT && !s->content_started) {
        while (n && (*p == '\n' || *p == ' ' || *p == '\r' || *p == '\t')) { p++; n--; }
        if (!n) return;
        s->content_started = true;
    }
    if (!n) return;
    buf_add(part == PF_REASONING ? &s->reasoning : &s->content, p, n);
    if (s->delta) s->delta(s->ctx, part, p, n);
}
void pf_split_init(PfSplit *s, bool thinking, void (*delta)(void *, PfPart, const char *, size_t), void *ctx) {
    *s = (PfSplit){.mode = thinking ? 0 : 1, .delta = delta, .ctx = ctx};
}
void pf_split_feed(PfSplit *s, const char *p, size_t n) {
    buf_add(&s->hold, p, n);
    for (;;) {
        if (s->mode == 2) { buf_add(&s->calls, s->hold.p, s->hold.n); s->hold.n = 0; return; }
        const char *tag = s->mode == 0 ? think_end : call_start;
        char *at = s->hold.n ? strstr(s->hold.p, tag) : NULL;
        if (!at) {
            size_t keep = partial(s->hold.p, s->hold.n, tag), give = s->hold.n-keep;
            out(s, s->mode == 0 ? PF_REASONING : PF_CONTENT, s->hold.p, give);
            memmove(s->hold.p, s->hold.p+give, keep); s->hold.n = keep; s->hold.p[keep] = 0;
            return;
        }
        size_t before = (size_t)(at-s->hold.p), skip = s->mode == 0 ? strlen(tag) : 0;   /* the call keeps its tag: the parser wants it */
        out(s, s->mode == 0 ? PF_REASONING : PF_CONTENT, s->hold.p, before);
        memmove(s->hold.p, at+skip, s->hold.n-before-skip+1); s->hold.n -= before+skip;
        s->mode++;
    }
}
void pf_split_finish(PfSplit *s) {
    if (!s->hold.n) return;
    if (s->mode == 2) buf_add(&s->calls, s->hold.p, s->hold.n);
    else out(s, s->mode == 0 ? PF_REASONING : PF_CONTENT, s->hold.p, s->hold.n);
    s->hold.n = 0;
}
void pf_split_free(PfSplit *s) { buf_free(&s->hold); buf_free(&s->reasoning); buf_free(&s->content); buf_free(&s->calls); }

/* ------------------------------------------------------------ the reply */

static const char *trimmed(PfArena *a, const Buf *b) {
    const char *p = b->p ? b->p : ""; size_t n = b->n;
    while (n && strchr(" \t\r\n", *p)) { p++; n--; }
    while (n && strchr(" \t\r\n", p[n-1])) n--;
    return pf_strdup(a, p, n);
}
/* The JSON-schema type of parameter `key` of tool `name`, or NULL. */
static const Qwen4Json *param_type(const Qwen4Json *tools, const char *name, const char *key) {
    for (size_t i = 0; tools && i < tools->as.array.count; i++) {
        const Qwen4Json *fn = pf_json_get(tools->as.array.items+i, "function");
        if (pf_json_is(pf_json_get(fn, "name"), name))
            return pf_json_get(pf_json_get(pf_json_get(pf_json_get(fn, "parameters"), "properties"), key), "type");
    }
    return NULL;
}
static bool is_string_type(const Qwen4Json *t) {
    if (pf_json_is(t, "string")) return true;
    for (size_t i = 0; t && t->kind == QW4_JSON_ARRAY && i < t->as.array.count; i++)
        if (pf_json_is(t->as.array.items+i, "string")) return true;
    return false;
}
void pf_reply_build(PfArena *a, const PfSplit *s, const Qwen4Json *tools, PfReply *r) {
    r->reasoning = trimmed(a, &s->reasoning);
    PfCall calls[PF_MAX_CALLS];
    size_t n = s->calls.n ? pf_parse_calls(s->calls.p, calls, PF_MAX_CALLS) : 0;
    if (s->calls.n && !n) {   /* "<tool_call>" but nothing parsable: keep the text, lose nothing */
        Buf all = {0}; buf_add(&all, s->content.p, s->content.n); buf_add(&all, s->calls.p, s->calls.n);
        r->content = trimmed(a, &all); buf_free(&all);
    } else r->content = trimmed(a, &s->content);
    r->ncalls = n;
    for (size_t i = 0; i < n; i++) {
        PfToolCall *c = &r->calls[i];
        pf_random_id(c->id, sizeof c->id, "call_");
        c->name = pf_strdup(a, calls[i].name, strlen(calls[i].name));
        c->args = pf_jobject(a, calls[i].n);
        for (size_t j = 0; j < calls[i].n; j++) {
            const char *v = calls[i].value[j]; char why[8];
            Qwen4Json *parsed = is_string_type(param_type(tools, c->name, calls[i].key[j])) ? NULL
                              : pf_json_parse(a, v, strlen(v), why, sizeof why);
            pf_jset(a, &c->args, calls[i].key[j], parsed ? *parsed : pf_jstr(a, v, strlen(v)));
        }
        pf_call_free(&calls[i]);
    }
    if (n) r->finish = "tool_calls";
}
Qwen4Json pf_reply_message(PfArena *a, const PfReply *r) {
    Qwen4Json list = pf_jarray(a, r->ncalls);
    for (size_t i = 0; i < r->ncalls; i++) pf_jpush(&list, call(a, r->calls[i].name, strlen(r->calls[i].name), r->calls[i].args));
    return message(a, "assistant", pf_jstr(a, r->content, strlen(r->content)), NULL, &list);
}
Qwen4Json pf_message_without_reasoning(PfArena *a, const Qwen4Json *m) {
    Qwen4Json copy = pf_jobject(a, m->as.object.count);
    for (size_t i = 0; i < m->as.object.count; i++) {
        Qwen4Bytes k = m->as.object.items[i].key;
        if (k.len == 17 && !memcmp(k.data, "reasoning_content", 17)) continue;
        pf_jset(a, &copy, pf_strdup(a, (const char *)k.data, k.len), *m->as.object.items[i].value);
    }
    return copy;
}

/* ------------------------------------------------------------ output */

void pf_random_id(char *out, size_t cap, const char *prefix) {
    static const char hex[] = "0123456789abcdef";
    size_t n = (size_t)snprintf(out, cap, "%s", prefix);
    for (; n+1 < cap && n < strlen(prefix)+24; n++) out[n] = hex[arc4random_uniform(16)];
    out[n] = 0;
}
void pf_error_body(Buf *b, PfProtocol p, const char *type, const char *code, const char *message) {
    if (p == PF_ANTHROPIC) { buf_fmt(b, "{\"type\":\"error\",\"error\":{\"type\":\"%s\",\"message\":\"", type); json_escape(b, message); buf_str(b, "\"}}"); return; }
    buf_str(b, "{\"error\":{\"message\":\""); json_escape(b, message);
    buf_fmt(b, "\",\"type\":\"%s\",\"code\":", type);
    if (code) buf_fmt(b, "\"%s\"}}", code); else buf_str(b, "null}}");
}
static void send_buf(PfOut *o, Buf *b) {
    if (!o->gone && !pf_http_send(o->fd, b->p, b->n)) o->gone = true;
    b->n = 0;
}
static void sse(PfOut *o, const char *event, Buf *json) {
    Buf b = {0};
    if (event) buf_fmt(&b, "event: %s\n", event);
    buf_str(&b, "data: "); buf_add(&b, json->p, json->n); buf_str(&b, "\n\n");
    send_buf(o, &b); buf_free(&b); json->n = 0;
}
static void openai_chunk_head(const PfOut *o, Buf *b) {
    buf_fmt(b, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"", o->id, o->created);
    json_escape(b, o->model); buf_str(b, "\",\"choices\":[");
}
static void calls_json(Buf *b, const PfReply *r, bool openai) {
    for (size_t i = 0; i < r->ncalls; i++) {
        Buf args = {0}; pf_json_write(&args, &r->calls[i].args);
        if (i) buf_str(b, ",");
        if (openai) {
            buf_fmt(b, "{\"index\":%zu,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":\"", i, r->calls[i].id);
            json_escape(b, r->calls[i].name); buf_str(b, "\",\"arguments\":\""); json_escape(b, args.p); buf_str(b, "\"}}");
        } else {
            buf_fmt(b, "{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":\"", r->calls[i].id);
            json_escape(b, r->calls[i].name); buf_str(b, "\",\"input\":"); buf_add(b, args.p, args.n); buf_str(b, "}");
        }
        buf_free(&args);
    }
}
static void anthropic_usage(Buf *b, const PfOut *o, size_t generated) {
    buf_fmt(b, "{\"input_tokens\":%zu,\"cache_read_input_tokens\":%zu,\"cache_creation_input_tokens\":0,\"output_tokens\":%zu}",
            o->prompt-o->cached, o->cached, generated);
}
void pf_out_begin(PfOut *o, size_t prompt, size_t cached) {
    o->prompt = prompt; o->cached = cached; o->created = (long)time(NULL); o->block = -1;
    pf_random_id(o->id, sizeof o->id, o->protocol == PF_ANTHROPIC ? "msg_" : "chatcmpl-");
    if (!o->stream) return;
    if (!pf_http_sse_begin(o->fd)) { o->gone = true; return; }
    Buf b = {0};
    if (o->protocol == PF_OPENAI) {
        openai_chunk_head(o, &b); buf_str(&b, "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}");
        sse(o, NULL, &b);
    } else {
        buf_fmt(&b, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"", o->id);
        json_escape(&b, o->model); buf_str(&b, "\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":");
        anthropic_usage(&b, o, 0); buf_str(&b, "}}");
        sse(o, "message_start", &b);
    }
    buf_free(&b);
}
/* Anthropic: close the open content block, if any. */
static void close_block(PfOut *o, Buf *b) {
    if (o->block < 0) return;
    if (o->block == PF_REASONING) {   /* clients echo the signature back; ours is empty, nothing to verify */
        buf_fmt(b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"\"}}", o->blocks-1);
        sse(o, "content_block_delta", b);
    }
    buf_fmt(b, "{\"type\":\"content_block_stop\",\"index\":%d}", o->blocks-1);
    sse(o, "content_block_stop", b);
    o->block = -1;
}
void pf_out_delta(PfOut *o, PfPart part, const char *p, size_t n) {
    if (!o->stream || o->gone || !n) return;
    char *text = strndup(p, n); Buf b = {0};
    if (!text) die("pf-server: out of memory");
    if (o->protocol == PF_OPENAI) {
        openai_chunk_head(o, &b);
        buf_fmt(&b, "{\"index\":0,\"delta\":{\"%s\":\"", part == PF_REASONING ? "reasoning_content" : "content");
        json_escape(&b, text); buf_str(&b, "\"},\"finish_reason\":null}]}");
        sse(o, NULL, &b);
    } else {
        if (o->block != (int)part) {
            close_block(o, &b);
            buf_fmt(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":%s}", o->blocks,
                    part == PF_REASONING ? "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}" : "{\"type\":\"text\",\"text\":\"\"}");
            sse(o, "content_block_start", &b);
            o->block = (int)part; o->blocks++;
        }
        buf_fmt(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"%s\",\"%s\":\"", o->blocks-1,
                part == PF_REASONING ? "thinking_delta" : "text_delta", part == PF_REASONING ? "thinking" : "text");
        json_escape(&b, text); buf_str(&b, "\"}}");
        sse(o, "content_block_delta", &b);
    }
    buf_free(&b); free(text);
}
static const char *anthropic_stop(const char *finish) {
    return !strcmp(finish, "tool_calls") ? "tool_use" : !strcmp(finish, "length") ? "max_tokens" : "end_turn";
}
static void end_stream(PfOut *o, const PfReply *r) {
    Buf b = {0};
    if (o->protocol == PF_OPENAI) {
        if (r->ncalls) {
            openai_chunk_head(o, &b); buf_str(&b, "{\"index\":0,\"delta\":{\"tool_calls\":[");
            calls_json(&b, r, true); buf_str(&b, "]},\"finish_reason\":null}]}");
            sse(o, NULL, &b);
        }
        openai_chunk_head(o, &b); buf_fmt(&b, "{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}", r->finish);
        sse(o, NULL, &b);
        if (o->stream_usage) {
            buf_fmt(&b, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"", o->id, o->created);
            json_escape(&b, o->model);
            buf_fmt(&b, "\",\"choices\":[],\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,"
                        "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}}", r->prompt, r->generated, r->prompt+r->generated, r->cached);
            sse(o, NULL, &b);
        }
        buf_str(&b, "data: [DONE]\n\n"); send_buf(o, &b);
    } else {
        close_block(o, &b);
        for (size_t i = 0; i < r->ncalls; i++) {
            Buf args = {0}; pf_json_write(&args, &r->calls[i].args);
            buf_fmt(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":\"", o->blocks, r->calls[i].id);
            json_escape(&b, r->calls[i].name); buf_str(&b, "\",\"input\":{}}}");
            sse(o, "content_block_start", &b);
            buf_fmt(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"", o->blocks);
            json_escape(&b, args.p); buf_str(&b, "\"}}");
            sse(o, "content_block_delta", &b);
            buf_fmt(&b, "{\"type\":\"content_block_stop\",\"index\":%d}", o->blocks++);
            sse(o, "content_block_stop", &b);
            buf_free(&args);
        }
        buf_fmt(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},\"usage\":", anthropic_stop(r->finish));
        anthropic_usage(&b, o, r->generated); buf_str(&b, "}");
        sse(o, "message_delta", &b);
        buf_str(&b, "{\"type\":\"message_stop\"}"); sse(o, "message_stop", &b);
    }
    buf_free(&b);
}
void pf_out_end(PfOut *o, const PfReply *r) {
    if (o->stream) { end_stream(o, r); return; }
    Buf b = {0};
    if (o->protocol == PF_OPENAI) {
        buf_fmt(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"", o->id, o->created);
        json_escape(&b, o->model);
        buf_str(&b, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (*r->content || !r->ncalls) { buf_str(&b, "\""); json_escape(&b, r->content); buf_str(&b, "\""); } else buf_str(&b, "null");
        if (*r->reasoning) { buf_str(&b, ",\"reasoning_content\":\""); json_escape(&b, r->reasoning); buf_str(&b, "\""); }
        if (r->ncalls) { buf_str(&b, ",\"tool_calls\":["); calls_json(&b, r, true); buf_str(&b, "]"); }
        buf_fmt(&b, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,"
                    "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}}", r->finish, r->prompt, r->generated, r->prompt+r->generated, r->cached);
    } else {
        buf_fmt(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"", o->id);
        json_escape(&b, o->model); buf_str(&b, "\",\"content\":[");
        bool any = false;
        if (*r->reasoning) { buf_str(&b, "{\"type\":\"thinking\",\"thinking\":\""); json_escape(&b, r->reasoning); buf_str(&b, "\",\"signature\":\"\"}"); any = true; }
        if (*r->content) { buf_str(&b, any ? ",{\"type\":\"text\",\"text\":\"" : "{\"type\":\"text\",\"text\":\""); json_escape(&b, r->content); buf_str(&b, "\"}"); any = true; }
        if (r->ncalls) { if (any) buf_str(&b, ","); calls_json(&b, r, false); }
        buf_fmt(&b, "],\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":", anthropic_stop(r->finish));
        anthropic_usage(&b, o, r->generated); buf_str(&b, "}");
    }
    if (!pf_http_respond(o->fd, 200, "application/json", b.p, b.n)) o->gone = true;
    buf_free(&b);
}
