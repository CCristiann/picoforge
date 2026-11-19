/* pf_session.c — see pf_session.h. Sections: messages · JSON for the
 * template · turns · replies · saved sessions · lifetime. */
#include "pf_session.h"
#include "common.h"
#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_CALLS 16       /* messages in one appended turn (tool results of one reply) */

/* ------------------------------------------------------------ messages */

static void note(const PfSession *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void note(const PfSession *s, const char *fmt, ...) {
    char text[1024]; va_list ap;
    va_start(ap, fmt); vsnprintf(text, sizeof text, fmt, ap); va_end(ap);
    if (s && s->note) s->note(s->note_ctx, text);
    else fprintf(stderr, "pf-agent: %s", text);
}

/* ------------------------------------------------------------ JSON for the template
 * qwen38_chat_render takes borrowed JSON trees, not text. The tool
 * definitions are built once and live as long as the session. */
static Qwen38Json jstr(const char *s) {
    return (Qwen38Json){.kind = Q38_JSON_STRING, .as.string = {(const uint8_t *)s, strlen(s)}};
}
static Qwen38Json jobject(size_t cap) {
    Qwen38JsonMember *m = calloc(cap, sizeof *m);
    if (!m) die("pf-agent: out of memory");
    return (Qwen38Json){.kind = Q38_JSON_OBJECT, .as.object = {m, 0}};
}
static void jset(Qwen38Json *object, const char *key, Qwen38Json value) {
    Qwen38Json *boxed = malloc(sizeof *boxed);
    if (!boxed) die("pf-agent: out of memory");
    *boxed = value;
    Qwen38JsonMember *m = (Qwen38JsonMember *)object->as.object.items;   /* ours: built here */
    m[object->as.object.count++] = (Qwen38JsonMember){{(const uint8_t *)key, strlen(key)}, boxed};
}

/* [{"type":"function","function":{"name","description","parameters":{JSON schema}}}, ...] */
void pf_session_set_tools(PfSession *s, const PfToolSpec *(*spec)(size_t), size_t count) {
    Qwen38Json *list = calloc(count ? count : 1, sizeof *list);
    if (!list) die("pf-agent: out of memory");
    for (size_t i = 0; i < count; i++) {
        const PfToolSpec *t = spec(i);
        Qwen38Json props = jobject(6), *required = calloc(6, sizeof *required);
        size_t nreq = 0;
        if (!required) die("pf-agent: out of memory");
        for (const PfParam *p = t->params; p < t->params+6 && p->name; p++) {
            Qwen38Json prop = jobject(2);
            jset(&prop, "type", jstr(p->type)); jset(&prop, "description", jstr(p->description));
            jset(&props, p->name, prop);
            if (p->required) required[nreq++] = jstr(p->name);
        }
        Qwen38Json schema = jobject(3), fn = jobject(3);
        jset(&schema, "type", jstr("object")); jset(&schema, "properties", props);
        jset(&schema, "required", (Qwen38Json){.kind = Q38_JSON_ARRAY, .as.array = {required, nreq}});
        jset(&fn, "name", jstr(t->name)); jset(&fn, "description", jstr(t->description));
        jset(&fn, "parameters", schema);
        list[i] = jobject(2); jset(&list[i], "type", jstr("function")); jset(&list[i], "function", fn);
    }
    s->tools = (Qwen38Json){.kind = Q38_JSON_ARRAY, .as.array = {list, count}};
    s->has_tools = count > 0;
}

/* ------------------------------------------------------------ turns */
static uint8_t *render(const PfSession *s, const Qwen38Json *list, size_t n, bool with_tools, bool generation, size_t *len) {
    Qwen38Json messages = {.kind = Q38_JSON_ARRAY, .as.array = {list, n}};
    /* Effort "medium" adds no reasoning instruction to the system block (the
     * template's default, xhigh, asks for long reasoning that eats the context). */
    Qwen38ChatOptions o = {.enable_thinking = s->thinking ? Q38_TRUE : Q38_FALSE,
        .reasoning_effort = Q38_EFFORT_MEDIUM, .add_generation_prompt = generation};
    Qwen38Error e;
    if (qwen38_chat_render(s->run.tok, &messages, with_tools && s->has_tools ? &s->tools : NULL, o, NULL, 0, len, &e)) {
        note(s, "%s\n", e.detail); return NULL;
    }
    uint8_t *out = malloc(*len+1);
    if (!out) die("pf-agent: out of memory");
    q38_check(qwen38_chat_render(s->run.tok, &messages, with_tools && s->has_tools ? &s->tools : NULL, o, out, *len, len, &e), &e);
    return out;
}
/* Render msgs as the next turn and encode it after the transcript, at
 * ids[n..); returns the number of new ids (0: error, already reported). The
 * first turn carries the system prompt and the tool definitions. A later turn
 * renders [placeholder user, msgs...] and drops the placeholder's bytes: the
 * template refuses a list with no user query, and tool results are not one. */
size_t pf_session_append(PfSession *s, const PfMsg *msgs, size_t n) {
    Qwen38Json list[MAX_CALLS+1], values[MAX_CALLS+1][2]; Qwen38JsonMember fields[MAX_CALLS+1][2];
    const bool first = !s->n;
    for (size_t i = 0; i <= n; i++) {
        PfMsg m = i ? msgs[i-1] : first ? (PfMsg){"system", s->system} : (PfMsg){"user", "."};
        values[i][0] = jstr(m.role); values[i][1] = jstr(m.content);
        fields[i][0] = (Qwen38JsonMember){{(const uint8_t *)"role", 4}, &values[i][0]};
        fields[i][1] = (Qwen38JsonMember){{(const uint8_t *)"content", 7}, &values[i][1]};
        list[i] = (Qwen38Json){.kind = Q38_JSON_OBJECT, .as.object = {fields[i], 2}};
    }
    size_t len, skip = 0, count = 0;
    uint8_t *text = render(s, list, n+1, first, true, &len);
    if (!text) return 0;
    if (!first) {
        uint8_t *prefix = render(s, list, 1, false, false, &skip);
        if (!prefix || skip > len || memcmp(prefix, text, skip)) die("pf-agent: placeholder turn is not a prefix");
        free(prefix);
    }
    /* Close the previous reply: it ends with <|im_end|> unless it was cut. */
    Buf b = {0};
    if (!first) buf_str(&b, s->ids[s->n-1] == s->im_end ? "\n" : "<|im_end|>\n");
    buf_add(&b, text+skip, len-skip); free(text);
    Qwen38Error e; uint32_t cap = s->run.capacity;
    Qwen38Status rc = qwen38_encode(s->run.tok, (Qwen38Bytes){(const uint8_t *)b.p, b.n}, s->ids+s->n, cap-s->n, &count, &e);
    buf_free(&b);
    if (rc && rc != Q38_CAPACITY) { note(s, "cannot encode the turn: %s\n", e.detail); return 0; }
    if (rc || count >= cap-s->n) {
        note(s, "the context is full (%zu of %u tokens): /reset, or restart with a larger -c\n", s->n, cap);
        return 0;
    }
    return count;
}

/* The test oracle (--token-ids): transcript ids, its text, and the graph position. */
void pf_session_dump(const PfSession *s) {
    if (!s->ids_path) return;
    FILE *f = fopen(s->ids_path, "w");
    if (!f) die("pf-agent: cannot open %s", s->ids_path);
    size_t len = 0; Qwen38Error e; Buf json = {0};
    q38_check(qwen38_decode_utf8(s->run.tok, s->ids, s->n, false, NULL, 0, &len, &e), &e);
    char *text = malloc(len+1);
    if (!text) die("pf-agent: out of memory");
    q38_check(qwen38_decode_utf8(s->run.tok, s->ids, s->n, false, (uint8_t *)text, len, &len, &e), &e);
    text[len] = 0; json_escape(&json, text); free(text);
    fputs("{\"ids\":[", f);
    for (size_t i = 0; i < s->n; i++) fprintf(f, "%s%d", i ? "," : "", s->ids[i]);
    fprintf(f, "],\"kv\":%llu,\"text\":\"%s\"}\n", (unsigned long long)qwen38_gpu_forward_info(s->run.graph).next_pos, json.p);
    buf_free(&json);
    if (fclose(f)) die("pf-agent: writing %s failed", s->ids_path);
}
/* --script (tests): the next canned reply, up to a line "=====", encoded and
 * appended as if the model had generated it. NULL when the script is used up. */
static char *scripted_reply(PfSession *s, size_t total, const PfStream *stream) {
    if (!s->script || !*s->script) return NULL;
    const char *end = strstr(s->script, "\n=====\n");
    size_t len = end ? (size_t)(end-s->script) : strlen(s->script);
    const char *start = s->script;
    s->script += len + (end ? 7 : 0);
    if (len && start[len-1] == '\n') len--;      /* the file's last newline is not part of the reply */
    Buf b = {0}; buf_add(&b, start, len);
    char *reply = strdup(b.p);
    buf_str(&b, "<|im_end|>");
    size_t count = 0; Qwen38Error e;
    q38_check(qwen38_encode(s->run.tok, (Qwen38Bytes){(const uint8_t *)b.p, b.n}, s->ids+total, s->run.capacity-total, &count, &e), &e);
    buf_free(&b);
    s->n = total+count;
    if (stream && stream->text) { stream->text(stream->ctx, (const uint8_t *)reply, strlen(reply)); stream->text(stream->ctx, (const uint8_t *)"\n", 1); }
    return reply;
}
/* The text after the reasoning: only there do tool calls count. */
const char *pf_session_answer(const char *reply) {
    const char *p = reply, *close;
    while ((close = strstr(p, "</think>"))) p = close+8;
    return p;
}


/* ------------------------------------------------------------ replies */

char *pf_session_reply(PfSession *s, size_t added, const PfStream *stream, PfReplyStats *stats) {
    size_t total = s->n+added;
    PfReplyStats none; if (!stats) stats = &none;
    *stats = (PfReplyStats){0};
    char *canned = scripted_reply(s, total, stream);
    if (canned) { pf_session_dump(s); return canned; }
    Q38Turn t; bool replayed;
    size_t prefilled = q38_chat_continue(&s->run, s->ids, s->n, total, s->max_new,
        stream ? stream->text : NULL, stream ? stream->ctx : NULL, s->generated, &t, &replayed);
    bool cancelled = !strcmp(t.reason, "cancelled");
    memcpy(s->ids+total, s->generated, t.got*sizeof *s->ids);
    if (t.got || !cancelled) s->n = total+t.got;   /* stopped before the first token: drop the turn (as the REPL) */
    *stats = (PfReplyStats){.prefilled = prefilled, .generated = t.got, .prefill_s = t.prefill_end-t.prefill_start,
        .decode_s = t.end-t.decode_start, .replayed = replayed, .cancelled = cancelled, .length = !strcmp(t.reason, "length")};
    pf_session_dump(s);
    if (cancelled) return NULL;
    size_t len = 0; Qwen38Error e;
    q38_check(qwen38_decode_utf8(s->run.tok, s->generated, t.got, true, NULL, 0, &len, &e), &e);
    char *text = malloc(len+1);
    if (!text) die("pf-agent: out of memory");
    q38_check(qwen38_decode_utf8(s->run.tok, s->generated, t.got, true, (uint8_t *)text, len, &len, &e), &e);
    text[len] = 0;
    return text;
}

/* ------------------------------------------------------------ saved sessions
 * A session file holds what it takes to continue exactly where we stopped:
 * the transcript ids and the whole graph state (qwen38_gpu_state_write: KV
 * caches, recurrent states, ...), so a resume costs one file read instead of
 * a prefill of the whole conversation. Files live in $PICOFORGE_SESSIONS or
 * ~/.picoforge/sessions: NAME.pfs, plus NAME.txt, the transcript as text for
 * people to read (never loaded). They can hold private code and output. */

typedef struct {
    char magic[8];                      /* "PFAGENT" */
    uint32_t version, capacity, thinking, mtp;
    uint64_t n, model_bytes, last_count; int64_t saved;
    char model[PATH_MAX], workspace[PATH_MAX], title[120];
} SessionHeader;
static bool session_dir(char out[PATH_MAX]) {
    const char *dir = getenv("PICOFORGE_SESSIONS"), *home = getenv("HOME");
    if (dir && *dir) snprintf(out, PATH_MAX, "%s", dir);
    else if (home && *home) {
        snprintf(out, PATH_MAX, "%s/.picoforge", home); mkdir(out, 0700);
        snprintf(out, PATH_MAX, "%s/.picoforge/sessions", home);
    } else return false;
    return !mkdir(out, 0700) || errno == EEXIST;
}
/* NAME -> DIR/NAME.ext; names are [A-Za-z0-9._-], at most 64, not starting with '.'. */
static bool session_path(const PfSession *s, const char *name, const char *ext, char out[PATH_MAX]) {
    char dir[PATH_MAX]; size_t n = strlen(name);
    if (!n || n > 64 || name[0] == '.' || strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") != n) {
        note(s, "a session name has 1-64 letters, digits, '.', '_' or '-'\n"); return false;
    }
    if (!session_dir(dir)) { note(s, "cannot create the sessions directory\n"); return false; }
    snprintf(out, PATH_MAX, "%s/%s.%s", dir, name, ext);
    return true;
}
static FILE *session_open(const PfSession *s, const char *name, SessionHeader *h) {
    char path[PATH_MAX]; FILE *f;
    if (!session_path(s, name, "pfs", path)) return NULL;
    if (!(f = fopen(path, "rb"))) { note(s, "no session %s (/sessions lists them)\n", name); return NULL; }
    if (fread(h, sizeof *h, 1, f) != 1 || memcmp(h->magic, "PFAGENT", 8) || h->version != 1 || h->n > h->capacity) {
        note(s, "%s is not a pf-agent session file\n", path); fclose(f); return NULL;
    }
    h->model[PATH_MAX-1] = h->workspace[PATH_MAX-1] = h->title[sizeof h->title-1] = 0;
    return f;
}
/* /save [NAME]: the session we continue, else a new name from the clock. The
 * file is written next to its final name and renamed: never half a session. */
bool pf_session_save(PfSession *s, const char *name) {
    char stamp[32], path[PATH_MAX], tmp[PATH_MAX+16], text[PATH_MAX];
    time_t now = time(NULL); struct tm tm;
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime_r(&now, &tm));
    if (!name || !*name) name = s->name[0] ? s->name : stamp;
    if (!session_path(s, name, "pfs", path) || !session_path(s, name, "txt", text)) return false;
    if (!s->n) { note(s, "nothing to save yet\n"); return false; }
    SessionHeader h = {.magic = "PFAGENT", .version = 1, .capacity = s->run.capacity, .thinking = s->thinking,
        .mtp = s->run.mtp, .n = s->n, .model_bytes = s->model_bytes, .last_count = s->run.last_count, .saved = now};
    snprintf(h.model, sizeof h.model, "%s", s->model_path); snprintf(h.workspace, sizeof h.workspace, "%s", s->workspace);
    snprintf(h.title, sizeof h.title, "%s", s->title);
    snprintf(tmp, sizeof tmp, "%s.XXXXXX", path);
    int fd = mkstemp(tmp); FILE *f = fd < 0 ? NULL : fdopen(fd, "wb");
    if (!f) { note(s, "cannot write %s: %s\n", path, strerror(errno)); if (fd >= 0) { close(fd); unlink(tmp); } return false; }
    setvbuf(f, NULL, _IOFBF, 8u << 20);
    double start = q38_now(); Qwen38Error e = {0};
    bool ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(s->ids, sizeof *s->ids, s->n, f) == s->n &&
              !qwen38_gpu_state_write(s->run.graph, f, &e) && !fflush(f) && !fsync(fileno(f));
    long long bytes = ftello(f);
    ok = !fclose(f) && ok && !rename(tmp, path);
    if (!ok) { note(s, "saving failed: %s\n", e.detail[0] ? e.detail : strerror(errno)); unlink(tmp); return false; }
    size_t len = 0; FILE *t = fopen(text, "w");
    if (t && !qwen38_decode_utf8(s->run.tok, s->ids, s->n, false, NULL, 0, &len, &e)) {
        uint8_t *chars = malloc(len+1);
        if (chars && !qwen38_decode_utf8(s->run.tok, s->ids, s->n, false, chars, len, &len, &e)) fwrite(chars, 1, len, t);
        free(chars);
    }
    if (t) fclose(t);
    snprintf(s->name, sizeof s->name, "%s", name);
    note(s, "Saved session %s: %zu tokens, %.1f MiB in %.1f s (%s).\n", name, s->n, (double)bytes/1048576.0, q38_now()-start, path);
    return true;
}
/* /resume NAME: transcript and graph state back. The model must be the same
 * file; a different context size rebuilds the graph. The caller moves to the
 * session's workspace (s->workspace). */
bool pf_session_resume(PfSession *s, const char *name) {
    SessionHeader h; FILE *f = session_open(s, name, &h);
    if (!f) return false;
    bool ok = false; Qwen38Error e;
    if (h.model_bytes != s->model_bytes) note(s, "%s was saved with another model (%s)\n", name, h.model);
    else if (h.mtp != (uint32_t)s->run.mtp) note(s, "%s was saved with MTP %s: restart with%s --no-mtp\n", name, h.mtp ? "on" : "off", h.mtp ? "out" : "");
    else {
        if (h.capacity != s->run.capacity) {
            q38_run_graph(&s->run, h.capacity);
            int32_t *ids = realloc(s->ids, h.capacity*sizeof *ids), *gen = realloc(s->generated, h.capacity*sizeof *gen);
            if (!ids || !gen) die("pf-agent: out of memory");
            s->ids = ids; s->generated = gen;
        }
        setvbuf(f, NULL, _IOFBF, 8u << 20);
        double start = q38_now();
        ok = fread(s->ids, sizeof *s->ids, h.n, f) == h.n && !qwen38_gpu_state_read(s->run.graph, f, &e);
        if (!ok) note(s, "cannot load %s: %s\n", name, ferror(f) || feof(f) ? "file truncated" : e.detail);
        else note(s, "Resumed session %s: %llu tokens in %.1f s.\n", name, (unsigned long long)h.n, q38_now()-start);
    }
    fclose(f);
    if (!ok) { pf_session_reset(s); return false; }   /* never half a state: start clean */
    s->n = (size_t)h.n; s->run.last_count = (size_t)h.last_count; s->thinking = h.thinking;
    snprintf(s->name, sizeof s->name, "%s", name); snprintf(s->title, sizeof s->title, "%s", h.title);
    snprintf(s->workspace, sizeof s->workspace, "%s", h.workspace);
    return true;
}

typedef struct { SessionHeader h; char name[65]; } Listed;
static int by_saved(const void *x, const void *y) {
    int64_t a = ((const Listed *)x)->h.saved, b = ((const Listed *)y)->h.saved;
    return a < b ? 1 : a > b ? -1 : 0;
}
/* Newest first. A session's name is its file name. */
void pf_session_list(Buf *out) {
    char dir[PATH_MAX]; DIR *d = session_dir(dir) ? opendir(dir) : NULL; struct dirent *e;
    Listed *all = NULL; size_t n = 0;
    while (d && (e = readdir(d))) {
        size_t len = strlen(e->d_name); Listed item; FILE *f;
        if (len < 5 || len > 68 || strcmp(e->d_name+len-4, ".pfs")) continue;
        snprintf(item.name, sizeof item.name, "%.*s", (int)(len-4), e->d_name);
        if (!(f = session_open(NULL, item.name, &item.h))) continue;
        fclose(f);
        Listed *grown = realloc(all, (n+1)*sizeof *all);
        if (!grown) die("pf-agent: out of memory");
        all = grown; all[n++] = item;
    }
    if (d) closedir(d);
    if (!n) buf_fmt(out, "No saved sessions in %s.\n", dir);
    if (n) qsort(all, n, sizeof *all, by_saved);
    for (size_t i = 0; i < n; i++) {
        char when[32]; struct tm tm; time_t t = (time_t)all[i].h.saved;
        strftime(when, sizeof when, "%Y-%m-%d %H:%M", localtime_r(&t, &tm));
        buf_fmt(out, "%-24s %s  %6llu tokens  %s\n    %s\n", all[i].name, when, (unsigned long long)all[i].h.n,
                all[i].h.workspace, all[i].h.title);
    }
    free(all);
}

bool pf_session_peek(const char *name, char model[PATH_MAX], uint32_t *context, bool *mtp) {
    SessionHeader h; FILE *f = session_open(NULL, name, &h);
    if (!f) return false;
    fclose(f);
    snprintf(model, PATH_MAX, "%s", h.model); *context = h.capacity; *mtp = h.mtp != 0;
    return true;
}

/* ------------------------------------------------------------ lifetime */

void pf_session_open(PfSession *s, const PfSessionOptions *o) {
    Q38Run *r = &s->run; Qwen38Error e; size_t one = 0; int32_t probe[2]; struct stat st;
    if (stat(o->model, &st)) die("pf-agent: %s: %s", o->model, strerror(errno));
    s->model_path = strdup(o->model); s->model_bytes = (uint64_t)st.st_size;
    r->chunk = 1024; r->mtp = o->mtp; r->power = o->power;
    q38_policy_init(&r->policy, getenv("PICOFORGE_MTP_DRAFT"), getenv("PICOFORGE_MTP_COST"));
    q38_sampler_init(&r->sampler, o->temperature, 20, 0.95f, 0.0f, (uint64_t)time(NULL));   /* as the chat; 0: greedy */
    q38_run_open(r, o->model);
    if (o->context > r->c->max_positions) die("pf-agent: a context of %u exceeds the model's %u positions", o->context, r->c->max_positions);
    if (qwen38_encode(r->tok, (Qwen38Bytes){(const uint8_t *)"<|im_end|>", 10}, probe, 2, &one, &e) || one != 1)
        die("pf-agent: the model needs <|im_end|> as a single token");
    s->im_end = probe[0];
    q38_run_gpu(r, o->model, o->metallib, o->context);
    q38_run_graph(r, o->context);
    s->ids = malloc((size_t)o->context*sizeof *s->ids); s->generated = malloc((size_t)o->context*sizeof *s->generated);
    if (!s->ids || !s->generated || !s->model_path) die("pf-agent: out of memory");
}
void pf_session_close(PfSession *s) {
    q38_run_close(&s->run);
    free(s->ids); free(s->generated); free(s->system); free(s->model_path);
    /* the tool JSON tree lives as long as the process */
}
void pf_session_set_system(PfSession *s, const char *text) {
    free(s->system); s->system = strdup(text ? text : "");
    if (!s->system) die("pf-agent: out of memory");
}
void pf_session_reset(PfSession *s) {
    Qwen38Error e; q38_check(qwen38_gpu_forward_reset(s->run.graph, &e), &e);
    s->n = 0; s->run.last_count = 0; s->name[0] = s->title[0] = 0;
}
