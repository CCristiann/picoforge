/* pf_agent.c — pf-agent, a coding agent in the terminal on the picoforge engine.
 *
 * The model runs in this process (pf_session: transcript, template, replies,
 * saved sessions); it asks for tools in Qwen3.8's native format and pf_tools
 * runs them (files in the workspace, bash jobs, the web). This file is only the
 * interface.
 *
 * On a terminal two threads share the work (the shape of ds4-agent, ds4
 * b3f5018, MIT): the worker owns the model, the session and the tools; the UI
 * thread owns the terminal (pf_term: scrolling output, an input line that
 * stays editable, a status footer with the context bar and the speeds). The
 * worker never writes the terminal: its text goes through a queue and a wake
 * pipe, and its questions (approvals) wait for the UI's answer. Without a
 * terminal (pipes, tests, --non-interactive) one thread reads lines.
 */
#include "model_select.h"
#include "pf_session.h"
#include "pf_term.h"
#include "pf_tools.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_CALLS 16       /* tool calls run from one reply */
#define MAX_ROUNDS 64      /* tool rounds in one user turn before we stop and ask */

typedef enum { ST_IDLE, ST_LOADING, ST_PREFILL, ST_DECODE, ST_TOOL, ST_COMPACT, ST_ASK } State;

typedef struct {
    PfSession s;
    PfTools *tools;
    PfTerm *term;              /* NULL: plain lines (pipes, tests, --non-interactive) */
    bool color, hints;
    FILE *trace;               /* --trace */
    /* Shared by the worker and the UI thread, under mu. */
    pthread_mutex_t mu; pthread_cond_t cond; int wake[2];
    Buf out;                   /* text for the terminal */
    char *cmd;                 /* the next request for the worker */
    bool busy, quit;
    char *question; int answer;               /* an approval: answer -1 while waiting */
    int power_request;                        /* /power while busy, applied by the worker */
    State state; char tool[48];
    size_t prefilled, prompt, generated, used;
    double t0, t_decode, prefill_tps, decode_tps, last_prefill_tps, last_decode_tps, last_wake;
} Agent;

/* ------------------------------------------------------------ output */

static void wake(Agent *a) { char c = 1; ssize_t ignored = write(a->wake[1], &c, 1); (void)ignored; }
static void emit(Agent *a, const char *s, size_t n) {
    if (!a->term) { fwrite(s, 1, n, stdout); fflush(stdout); return; }
    pthread_mutex_lock(&a->mu); buf_add(&a->out, s, n); pthread_mutex_unlock(&a->mu);
    wake(a);
}
static void emitf(Agent *a, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void emitf(Agent *a, const char *fmt, ...) {
    char text[4096]; va_list ap;
    va_start(ap, fmt); int n = vsnprintf(text, sizeof text, fmt, ap); va_end(ap);
    if (n > 0) emit(a, text, (size_t)n < sizeof text ? (size_t)n : sizeof text-1);
}
static const char *ansi(const Agent *a, const char *code) { return a->color ? code : ""; }
static void set_state(Agent *a, State st, const char *tool) {
    pthread_mutex_lock(&a->mu);
    a->state = st; snprintf(a->tool, sizeof a->tool, "%s", tool ? tool : "");
    a->used = a->s.n;
    pthread_mutex_unlock(&a->mu); wake(a);
}
static void note(void *ctx, const char *message) { emit(ctx, message, strlen(message)); }

static void trace(Agent *a, const char *kind, const char *text) {
    if (!a->trace) return;
    char stamp[32]; time_t now = time(NULL); struct tm tm;
    strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", localtime_r(&now, &tm));
    fprintf(a->trace, "===== %s %s\n%s\n", stamp, kind, text ? text : "");
    fflush(a->trace);
}

/* ------------------------------------------------------------ approvals */

/* y = this once, a = always (rest of the session), anything else = no. On a
 * terminal the worker waits for the UI's answer; on plain lines stdin answers
 * if it is a terminal, else the answer is no (use --yes). */
static PfAnswer ask_user(void *ctx, const char *what) {
    Agent *a = ctx; int answer = 'n';
    trace(a, "approval asked", what);
    if (a->term) {
        pthread_mutex_lock(&a->mu);
        a->question = strdup(what); a->answer = -1; a->state = ST_ASK;
        pthread_mutex_unlock(&a->mu); wake(a);
        pthread_mutex_lock(&a->mu);
        while (a->answer < 0 && !qw4_stopped && !a->quit) pthread_cond_wait(&a->cond, &a->mu);
        answer = a->answer < 0 ? 'n' : a->answer;
        free(a->question); a->question = NULL; a->state = ST_TOOL;
        pthread_mutex_unlock(&a->mu); wake(a);
    } else if (isatty(STDIN_FILENO)) {
        printf("\n%s%s%s\nAllow? [y]es / [n]o / [a]lways: ", ansi(a, "\x1b[1m"), what, ansi(a, "\x1b[0m")); fflush(stdout);
        char line[64];
        if (fgets(line, sizeof line, stdin)) answer = tolower((unsigned char)line[0]); else clearerr(stdin);
    }
    PfAnswer r = answer == 'a' ? PF_ALWAYS : answer == 'y' || answer == 's' ? PF_ALLOW : PF_DENY;
    trace(a, "approval", r == PF_DENY ? "no" : r == PF_ALWAYS ? "always" : "yes");
    return r;
}

/* ------------------------------------------------------------ the system prompt */

static char *system_prompt(const Agent *a) {
    char date[32]; time_t now = time(NULL); struct tm tm;
    strftime(date, sizeof date, "%Y-%m-%d", localtime_r(&now, &tm));
    Buf b = {0}, project = {0};
    buf_fmt(&b, "You are pf-agent, a coding agent running locally on the user's Mac (macOS) through the "
        "picoforge engine. You work in the workspace %s; tool paths are relative to it. Today is %s.\n\n"
        "Work in small, verified steps: look before you change (list, search, read), make focused edits, "
        "and check them with the project's own commands through bash. Read a file before editing it: edit "
        "needs the exact current text. bash, write and edit need the user's approval; if they decline, ask "
        "how to proceed. Never invent file contents, command output or web pages: use the tools. When the "
        "task is done, reply with a short summary of what changed and how you verified it.", pf_tools_root(a->tools), date);
    if (pf_tools_read_text(a->tools, "AGENTS.md", 16384, &project)) {
        buf_str(&b, "\n\n# Project instructions (AGENTS.md)\n\n"); buf_add(&b, project.p, project.n);
    }
    buf_free(&project);
    return b.p;
}

/* ------------------------------------------------------------ showing a reply
 * The reply streams as it is decoded. Reasoning (<think>...</think>) is grey;
 * the XML of a tool call is hidden (each call is shown as one line once the
 * reply is done). A tag can arrive split across tokens, so a possible prefix
 * of one waits. */

typedef struct { Agent *a; Buf pend; bool think, hidden, newline; } View;
static const char *const tags[] = {"<think>", "</think>", "<tool_call>", "</tool_call>"};

static bool tag_prefix(const char *p, size_t n) {
    for (size_t i = 0; i < 4; i++) if (n < strlen(tags[i]) && !memcmp(p, tags[i], n)) return true;
    return false;
}
static void view_put(View *v, const char *s, size_t n) {
    if (!n || v->hidden) return;
    emit(v->a, s, n); v->newline = s[n-1] == '\n';
}
static void view_text(void *ctx, const uint8_t *bytes, size_t n) {
    View *v = ctx;
    buf_add(&v->pend, bytes, n);
    size_t i = 0;
    while (i < v->pend.n) {
        const char *p = v->pend.p+i; size_t left = v->pend.n-i; int tag = -1;
        for (int k = 0; k < 4; k++) if (left >= strlen(tags[k]) && !memcmp(p, tags[k], strlen(tags[k]))) tag = k;
        if (tag < 0 && *p == '<' && tag_prefix(p, left)) break;               /* wait for more */
        if (tag < 0) {
            size_t run = 1;
            while (run < left && p[run] != '<') run++;
            view_put(v, p, run); i += run; continue;
        }
        i += strlen(tags[tag]);
        if (tag == 0) { v->think = true; view_put(v, ansi(v->a, "\x1b[90m"), strlen(ansi(v->a, "\x1b[90m"))); }
        else if (tag == 1) {
            v->think = false; view_put(v, ansi(v->a, "\x1b[0m"), strlen(ansi(v->a, "\x1b[0m")));
            if (!v->newline) view_put(v, "\n", 1);
        }
        else if (tag == 2) { view_put(v, "\n...", 4); v->hidden = true; }
        else v->hidden = false;
    }
    memmove(v->pend.p, v->pend.p+i, v->pend.n-i+1); v->pend.n -= i;
}
static void view_end(View *v) {
    if (!v->hidden) view_put(v, v->pend.p, v->pend.n);
    if (v->think) emit(v->a, ansi(v->a, "\x1b[0m"), strlen(ansi(v->a, "\x1b[0m")));
    emit(v->a, "\n", 1); buf_free(&v->pend);
}

/* Prefill and decode counters for the footer. */
static void progress(void *ctx, size_t prefilled, size_t prompt, size_t generated) {
    View *v = ctx; Agent *a = v->a; double now = qw4_now();
    pthread_mutex_lock(&a->mu);
    if (!generated) {
        a->state = a->state == ST_LOADING ? ST_LOADING : ST_PREFILL; a->prefilled = prefilled; a->prompt = prompt;
        a->prefill_tps = now > a->t0 ? (double)prefilled/(now-a->t0) : 0;
    } else {
        if (a->state != ST_DECODE) { a->state = ST_DECODE; a->t_decode = now; }
        a->generated = generated;
        a->decode_tps = generated > 1 && now > a->t_decode ? (double)(generated-1)/(now-a->t_decode) : 0;
    }
    if (a->power_request) { a->s.run.power = a->power_request; a->s.run.power_prefill_avg = a->s.run.power_decode_avg = 0; a->power_request = 0; }
    bool tick = now-a->last_wake > 0.1;
    if (tick) a->last_wake = now;
    pthread_mutex_unlock(&a->mu);
    if (tick) wake(a);
}

/* One line per call: "-> name key=value ...", long values summarized. */
static void show_call(Agent *a, const PfCall *c) {
    Buf b = {0};
    buf_fmt(&b, "%s-> %s%s", ansi(a, "\x1b[36m"), c->name, ansi(a, "\x1b[0m"));
    for (size_t k = 0; k < c->n; k++) {
        const char *v = c->value[k], *nl = strchr(v, '\n');
        if (nl) buf_fmt(&b, " %s=<%ld lines>", c->key[k], count_lines(v, strlen(v)));
        else buf_fmt(&b, " %s=%.*s%s", c->key[k], 80, v, strlen(v) > 80 ? "..." : "");
    }
    buf_str(&b, "\n"); emit(a, b.p, b.n); buf_free(&b);
}
/* The first lines of a result, grey. */
static void show_result(Agent *a, const Buf *out) {
    const char *p = out->p; Buf b = {0};
    buf_str(&b, ansi(a, "\x1b[90m"));
    for (int shown = 0; *p && shown < 6; shown++) {
        const char *nl = strchr(p, '\n'); size_t len = nl ? (size_t)(nl-p) : strlen(p);
        buf_fmt(&b, "   %.*s\n", (int)(len < 160 ? len : 160), p);
        p += len + (nl != NULL);
    }
    if (*p) buf_fmt(&b, "   ... (%ld more lines)\n", count_lines(p, strlen(p)) - (out->p[out->n-1] == '\n'));
    buf_str(&b, ansi(a, "\x1b[0m")); emit(a, b.p, b.n); buf_free(&b);
}

static char *reply(Agent *a, size_t added) {
    pthread_mutex_lock(&a->mu);
    a->state = ST_PREFILL; a->prefilled = a->generated = 0; a->prompt = added; a->t0 = qw4_now();
    a->prefill_tps = a->decode_tps = 0;
    pthread_mutex_unlock(&a->mu); wake(a);
    View v = {.a = a, .newline = true};
    PfStream stream = {view_text, progress, &v}; PfReplyStats st;
    char *text = pf_session_reply(&a->s, added, &stream, &st);
    view_end(&v);
    double ptps = st.prefill_s > 0 ? (double)st.prefilled/st.prefill_s : 0, dtps = st.decode_s > 0 && st.generated > 1 ? (double)(st.generated-1)/st.decode_s : 0;
    pthread_mutex_lock(&a->mu);
    if (st.prefilled) a->last_prefill_tps = ptps;
    if (st.generated > 1) a->last_decode_tps = dtps;
    a->used = a->s.n;
    pthread_mutex_unlock(&a->mu);
    trace(a, "reply", text ? text : "(stopped)");
    if (!st.generated && !st.prefilled && text) return text;   /* a canned reply (--script) */
    char line[256];
    snprintf(line, sizeof line, "[prefill %zu tokens %.1f t/s, reply %zu tokens %.1f t/s, context %zu/%u%s%s]",
        st.prefilled, ptps, st.generated, dtps, a->s.n, a->s.run.capacity, st.replayed ? ", replayed" : "", st.cancelled ? ", interrupted" : "");
    if (a->term) emitf(a, "%s%s%s\n", ansi(a, "\x1b[90m"), line, ansi(a, "\x1b[0m"));
    else fprintf(stderr, "%s%s%s\n", ansi(a, "\x1b[90m"), line, ansi(a, "\x1b[0m"));
    if (st.length) emitf(a, "pf-agent: the reply hit the token limit\n");
    return text;
}

static void compact(Agent *a) {
    set_state(a, ST_COMPACT, NULL);
    emitf(a, "%s[compacting the context: the model summarizes the work so far]%s\n", ansi(a, "\x1b[90m"), ansi(a, "\x1b[0m"));
    View v = {.a = a, .newline = true};
    PfStream stream = {view_text, NULL, &v};
    pf_session_compact(&a->s, &stream);
    view_end(&v);
    trace(a, "compaction", "");
}

/* One user message, then reply -> tools -> results until the model answers
 * without a tool call, the user interrupts, or MAX_ROUNDS is reached. */
static void agent_turn(Agent *a, const char *text) {
    Buf msg = {0}; buf_str(&msg, text);
    if (a->hints) buf_str(&msg, "\n\n[pf-agent: hints are on. When you make a non-obvious programming choice, "
                               "add one short sentence on why.]");
    PfMsg user = {"user", msg.p};
    if (a->s.n == a->s.primed_n) {                  /* a session is named by its first request */
        Buf t = {0}; size_t len = strcspn(text, "\n");
        buf_add(&t, text, len < sizeof a->s.title-4 ? len : sizeof a->s.title-4); clean_text(&t);
        snprintf(a->s.title, sizeof a->s.title, "%s", t.p); buf_free(&t);
    }
    trace(a, "user", msg.p);
    if (pf_session_should_compact(&a->s, msg.n/2+1024)) compact(a);
    size_t added = pf_session_append(&a->s, &user, 1);
    buf_free(&msg);
    for (int round = 0; added && round < MAX_ROUNDS; round++) {
        char *answer = reply(a, added);
        if (!answer) return;
        PfCall calls[MAX_CALLS];
        size_t n = pf_parse_calls(pf_session_answer(answer), calls, MAX_CALLS);
        free(answer);
        if (!n || qw4_stopped) { for (size_t i = 0; i < n; i++) pf_call_free(&calls[i]); return; }
        Buf outs[MAX_CALLS] = {{0}}; PfMsg results[MAX_CALLS]; size_t bytes = 0;
        for (size_t i = 0; i < n; i++) {
            show_call(a, &calls[i]);
            set_state(a, ST_TOOL, calls[i].name);
            Buf args = {0};
            for (size_t k = 0; k < calls[i].n; k++) buf_fmt(&args, "%s=%s\n", calls[i].key[k], calls[i].value[k]);
            trace(a, calls[i].name, args.p); buf_free(&args);
            if (qw4_stopped) buf_str(&outs[i], "Tool error: interrupted by the user before it ran.\n");
            else pf_tools_run(a->tools, &calls[i], &outs[i]);
            trace(a, "result", outs[i].p);
            show_result(a, &outs[i]);
            results[i] = (PfMsg){"tool", outs[i].p}; bytes += outs[i].n;
            pf_call_free(&calls[i]);
        }
        if (pf_session_should_compact(&a->s, bytes/2+1024)) compact(a);   /* tokens are rarely fewer than bytes/3 */
        added = pf_session_append(&a->s, results, n);
        for (size_t i = 0; i < n; i++) buf_free(&outs[i]);
        if (qw4_stopped) { added = 0; emitf(a, "pf-agent: interrupted; the results are kept for your next message\n"); }
    }
    if (added) emitf(a, "pf-agent: %d tool rounds without an answer, stopping here\n", MAX_ROUNDS);
}

/* ------------------------------------------------------------ commands */

static void help(Agent *a) {
    emitf(a, "pf-agent: describe a task; the agent reads, edits and runs commands in this workspace.\n"
         "  /auto              Approve every bash/write/edit for the rest of the session.\n"
         "  /think [on|off]    Reasoning before each reply (/nothink = off).\n"
         "  /hints [on|off]    Short explanations of the agent's programming choices.\n"
         "  /power [N]         GPU duty cycle 1..100 (100: full speed).\n"
         "  /new, /reset       Start a new conversation.   /compact  Summarize it now to free context.\n"
         "  /history [N]       Your last N requests (default 10).\n"
         "  /save [NAME]       Save the session (transcript and model state).   /sessions  List them.\n"
         "  /resume NAME       Continue one.   /del NAME  Delete it.   /strip NAME  Keep only its transcript.\n"
         "  /help, /quit       This help; leave (Ctrl+D too).\n"
         "  Ctrl+C stops the reply or the running tool.%s\n",
         a->term ? " Alt+Enter or Ctrl+J: a new line; pasted text keeps its lines." : " End a line with \\ to continue it.");
}
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *end = s+strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = 0;
    return s;
}
static bool is_cmd(const char *cmd, const char *name) {   /* "/name" or "/name args" */
    size_t n = strlen(name);
    return !strncmp(cmd, name, n) && (!cmd[n] || cmd[n] == ' ');
}
/* "on"/"off" after a command, or a toggle when absent. */
static bool switch_arg(const char *arg, bool now) {
    return !*arg ? !now : !strcmp(arg, "on") || !strcmp(arg, "1") || !strcmp(arg, "yes");
}

/* After a resume: the session's workspace, and a system prompt for it. */
static void resumed(Agent *a) {
    if (strcmp(a->s.workspace, pf_tools_root(a->tools))) {
        if (chdir(a->s.workspace) || !pf_tools_set_root(a->tools, a->s.workspace))
            emitf(a, "pf-agent: the session's workspace %s is gone; staying in %s\n", a->s.workspace, pf_tools_root(a->tools));
        else { char *sys = system_prompt(a); pf_session_set_system(&a->s, sys); free(sys); }
    }
    snprintf(a->s.workspace, sizeof a->s.workspace, "%s", pf_tools_root(a->tools));
    emitf(a, "Workspace %s, thinking %s. First request: %s\n", a->s.workspace, a->s.thinking ? "on" : "off", a->s.title);
}
static void prime(Agent *a) {
    set_state(a, ST_LOADING, NULL);
    View v = {.a = a, .newline = true};
    PfStream stream = {NULL, progress, &v};
    pthread_mutex_lock(&a->mu); a->t0 = qw4_now(); pthread_mutex_unlock(&a->mu);
    pf_session_prime(&a->s, &stream);
    pf_session_dump(&a->s);
}

/* A command or a message, on the worker (or the only thread). */
static void run(Agent *a, char *line) {
    char *cmd = trim(line), *arg = "";
    if (!*cmd) return;
    if (*cmd == '/') { char *space = strchr(cmd, ' '); if (space) arg = trim(space+1); }
    trace(a, "input", cmd);
    if (!strcmp(cmd, "/help")) help(a);
    else if (!strcmp(cmd, "/auto")) { pf_tools_approve_all(a->tools); emitf(a, "Every bash/write/edit is approved for this session.\n"); }
    else if (is_cmd(cmd, "/think") || !strcmp(cmd, "/nothink")) {
        a->s.thinking = !strcmp(cmd, "/nothink") ? false : switch_arg(arg, a->s.thinking);
        emitf(a, "Thinking %s.\n", a->s.thinking ? "on" : "off");
    }
    else if (is_cmd(cmd, "/hints")) { a->hints = switch_arg(arg, a->hints); emitf(a, "Hints %s.\n", a->hints ? "on" : "off"); }
    else if (is_cmd(cmd, "/power")) {
        char *end; long n = *arg ? strtol(arg, &end, 10) : a->s.run.power;
        if (*arg && (*end || n < 1 || n > 100)) emitf(a, "pf-agent: /power needs N in [1,100]\n");
        else { a->s.run.power = (int)n; a->s.run.power_prefill_avg = a->s.run.power_decode_avg = 0; emitf(a, "Power: %ld%%.\n", n); }
    }
    else if (!strcmp(cmd, "/new") || !strcmp(cmd, "/reset")) { prime(a); emitf(a, "New conversation.\n"); }
    else if (!strcmp(cmd, "/compact")) compact(a);
    else if (is_cmd(cmd, "/history")) {
        Buf h = {0}; long n = *arg ? strtol(arg, NULL, 10) : 10;
        pf_session_history(&a->s, n > 0 ? (size_t)n : 10, &h); emit(a, h.p, h.n); buf_free(&h);
    }
    else if (is_cmd(cmd, "/save")) pf_session_save(&a->s, arg);
    else if (!strcmp(cmd, "/sessions") || !strcmp(cmd, "/list")) { Buf l = {0}; pf_session_list(&l); emit(a, l.p, l.n); buf_free(&l); }
    else if ((is_cmd(cmd, "/resume") || is_cmd(cmd, "/switch")) && *arg) {
        set_state(a, ST_LOADING, NULL);
        if (pf_session_resume(&a->s, arg)) resumed(a);
        pf_session_dump(&a->s);
    }
    else if (is_cmd(cmd, "/del") && *arg) pf_session_delete(&a->s, arg);
    else if (is_cmd(cmd, "/strip") && *arg) pf_session_strip(&a->s, arg);
    else if (*cmd == '/' && !strchr(cmd, '\n') && (!strchr(cmd, ' ') || strchr(cmd, ' ') > cmd+12)) emitf(a, "pf-agent: unknown command %s (try /help)\n", cmd);
    else agent_turn(a, cmd);
    set_state(a, ST_IDLE, NULL);
}

/* ------------------------------------------------------------ the worker */

static void *worker(void *arg) {
    Agent *a = arg;
    for (;;) {
        pthread_mutex_lock(&a->mu);
        while (!a->cmd && !a->quit) pthread_cond_wait(&a->cond, &a->mu);
        if (!a->cmd) { pthread_mutex_unlock(&a->mu); return NULL; }
        char *cmd = a->cmd; a->cmd = NULL; a->busy = true;
        pthread_mutex_unlock(&a->mu);
        qw4_stopped = 0;
        if (!strcmp(cmd, "\x01prime")) prime(a); else run(a, cmd);
        free(cmd);
        pthread_mutex_lock(&a->mu); a->busy = false; a->state = ST_IDLE; a->used = a->s.n; pthread_mutex_unlock(&a->mu);
        wake(a);
    }
}
static bool submit(Agent *a, const char *text) {
    pthread_mutex_lock(&a->mu);
    bool ok = !a->busy && !a->cmd;
    if (ok) { a->cmd = strdup(text); a->busy = true; pthread_cond_signal(&a->cond); }
    pthread_mutex_unlock(&a->mu);
    return ok;
}

/* ------------------------------------------------------------ the footer */

static void compact_count(char *out, size_t cap, size_t n) {
    if (n < 1000) snprintf(out, cap, "%zu", n);
    else if (n < 100000) snprintf(out, cap, "%.1fK", (double)n/1000.0);
    else snprintf(out, cap, "%zuK", n/1000);
}
/* " decoding 37.2 t/s 412 tok  ▕██████░░░░▏ 23.4K/100K 23%  last prefill 845 t/s · decode 38.1 t/s  MTP · think · power 100%" */
static void footer(Agent *a, Buf *b) {
    static const char *const names[] = {"ready", "loading", "prefill", "decoding", "tool", "compacting", "waiting for you"};
    pthread_mutex_lock(&a->mu);
    State st = a->state; size_t used = a->used, prefilled = a->prefilled, prompt = a->prompt, generated = a->generated;
    double ptps = a->prefill_tps, dtps = a->decode_tps, lp = a->last_prefill_tps, ld = a->last_decode_tps;
    char tool[48]; snprintf(tool, sizeof tool, "%s", a->tool);
    pthread_mutex_unlock(&a->mu);
    const uint32_t cap = a->s.run.capacity; char u[16], c[16], p1[16], p2[16];
    compact_count(u, sizeof u, used); compact_count(c, sizeof c, cap);
    buf_str(b, "\x1b[48;5;236;38;5;252m ");
    buf_fmt(b, "%s%s\x1b[38;5;252m", st == ST_IDLE ? "\x1b[38;5;114m" : st == ST_ASK ? "\x1b[38;5;214m" : "\x1b[38;5;81m", names[st]);
    if (st == ST_PREFILL || st == ST_LOADING) {
        compact_count(p1, sizeof p1, prefilled); compact_count(p2, sizeof p2, prompt);
        if (prompt) buf_fmt(b, " %s/%s %.0f t/s", p1, p2, ptps);
    } else if (st == ST_DECODE) buf_fmt(b, " %.1f t/s %zu tok", dtps, generated);
    else if (st == ST_TOOL && *tool) buf_fmt(b, " %s", tool);
    const int cells = 20; double frac = cap ? (double)used/cap : 0;
    int full = (int)(frac*cells+0.5); if (full > cells) full = cells;
    buf_str(b, "  \x1b[38;5;245m\xe2\x96\x95");
    for (int i = 0; i < cells; i++)
        buf_str(b, i >= full ? "\x1b[38;5;240m\xe2\x96\x91" : frac > 0.85 ? "\x1b[38;5;203m\xe2\x96\x88" : frac > 0.6 ? "\x1b[38;5;214m\xe2\x96\x88" : "\x1b[38;5;177m\xe2\x96\x88");
    buf_fmt(b, "\x1b[38;5;245m\xe2\x96\x8f\x1b[38;5;252m %s/%s %.0f%%", u, c, frac*100);
    if (lp > 0 || ld > 0) buf_fmt(b, "  \x1b[38;5;245mlast\x1b[38;5;252m prefill %.0f t/s \xc2\xb7 decode %.1f t/s", lp, ld);
    buf_fmt(b, "  \x1b[38;5;245m%s%s \xc2\xb7 power %d%%%s%s%s ",
        a->s.run.mtp ? "MTP \xc2\xb7 " : "", a->s.thinking ? "think" : "no think", a->s.run.power,
        a->hints ? " \xc2\xb7 hints" : "", a->s.name[0] ? " \xc2\xb7 " : "", a->s.name);
    buf_str(b, "                                                                                                    ");
}

/* ------------------------------------------------------------ the UI thread */

static volatile sig_atomic_t resized;
static void on_winch(int signum) { (void)signum; resized = 1; }

static void ui(Agent *a, const char *first, bool resumed_session) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, a)) die("pf-agent: cannot start the worker thread");
    struct sigaction action = {0}; action.sa_handler = on_winch; sigemptyset(&action.sa_mask);
    sigaction(SIGWINCH, &action, NULL);
    help(a);
    if (!resumed_session) submit(a, "\x01prime");
    char *queued = first ? strdup(first) : NULL;   /* -p: the first message, once the model is ready */
    bool asking = false, quitting = false; char *shown = NULL;
    for (;;) {
        struct pollfd fds[2] = {{pf_term_fd(a->term), POLLIN, 0}, {a->wake[0], POLLIN, 0}};
        poll(fds, 2, pf_term_pending(a->term) ? 0 : 200);
        if (fds[1].revents & POLLIN) { char sink[256]; while (read(a->wake[0], sink, sizeof sink) > 0) {} }
        if (resized) { resized = 0; pf_term_resized(a->term); }
        pthread_mutex_lock(&a->mu);
        Buf out = a->out; a->out = (Buf){0};
        bool busy = a->busy, ask = a->question != NULL;
        char question[1024]; snprintf(question, sizeof question, "%s", ask ? a->question : "");
        pthread_mutex_unlock(&a->mu);
        if (out.n) pf_term_output(a->term, out.p, out.n);
        buf_free(&out);
        if (ask != asking) {
            asking = ask;
            if (ask) { Buf q = {0}; buf_fmt(&q, "\n\x1b[1m%s\x1b[0m\n", question); pf_term_output(a->term, q.p, q.n); buf_free(&q); }
            pf_term_prompt(a->term, ask ? "Allow? [y]es / [n]o / [a]lways: " : "pf> ");
        }
        if (!busy && queued) { submit(a, queued); free(queued); queued = NULL; continue; }
        if (quitting && !busy) break;
        Buf status = {0}; footer(a, &status);                 /* redrawn only when it changes */
        if (!shown || strcmp(shown, status.p)) { pf_term_status(a->term, status.p); free(shown); shown = strdup(status.p); }
        buf_free(&status);
        if (!(fds[0].revents & POLLIN) && !pf_term_pending(a->term)) continue;
        char *line = NULL; PfTermEvent ev = pf_term_read(a->term, &line);
        if (ev == PF_TERM_SUBMIT && asking) {
            pthread_mutex_lock(&a->mu); a->answer = tolower((unsigned char)*trim(line)); if (!a->answer) a->answer = 'n';
            pthread_cond_broadcast(&a->cond); pthread_mutex_unlock(&a->mu);
        } else if (ev == PF_TERM_SUBMIT) {
            char *cmd = trim(line);
            if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) { quitting = true; if (busy) qw4_stopped = 1; }
            else if (!*cmd) {}
            else if (busy && is_cmd(cmd, "/power")) {
                long n = strtol(cmd+6, NULL, 10);
                if (n >= 1 && n <= 100) { pthread_mutex_lock(&a->mu); a->power_request = (int)n; pthread_mutex_unlock(&a->mu); emitf(a, "Power: %ld%% (from the next step).\n", n); }
            }
            else if (busy) emitf(a, "%s[busy: wait for the reply, or Ctrl+C to stop it]%s\n", ansi(a, "\x1b[33m"), ansi(a, "\x1b[0m"));
            else {
                Buf echo = {0}; buf_fmt(&echo, "\x1b[1mpf>\x1b[0m %s\n", cmd);
                pf_term_output(a->term, echo.p, echo.n); buf_free(&echo);
                submit(a, cmd);
            }
        } else if (ev == PF_TERM_INTERRUPT) {
            if (asking || busy) {
                qw4_stopped = 1;
                pthread_mutex_lock(&a->mu); if (asking) a->answer = 'n'; pthread_cond_broadcast(&a->cond); pthread_mutex_unlock(&a->mu);
            } else pf_term_clear(a->term);
        } else if (ev == PF_TERM_EOF) { quitting = true; if (busy) qw4_stopped = 1; }
        free(line);
    }
    pthread_mutex_lock(&a->mu); a->quit = true; pthread_cond_broadcast(&a->cond); pthread_mutex_unlock(&a->mu);
    pthread_join(thread, NULL);
    pthread_mutex_lock(&a->mu); Buf out = a->out; a->out = (Buf){0}; pthread_mutex_unlock(&a->mu);
    if (out.n) pf_term_output(a->term, out.p, out.n);
    buf_free(&out); free(queued); free(shown);
}

/* ------------------------------------------------------------ plain lines */

static volatile sig_atomic_t at_prompt;
static void interrupt(int signum) {
    (void)signum; qw4_stopped = 1;
    if (at_prompt) { ssize_t ignored = write(STDOUT_FILENO, "\npf> ", 5); (void)ignored; }
}
/* One message from stdin; a line ending in '\' continues on the next. NULL at EOF. */
static char *read_message(Buf *msg) {
    char *line = NULL; size_t cap = 0; msg->n = 0; buf_add(msg, "", 0);
    bool tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    for (;;) {
        if (tty) { fputs(msg->n ? "... " : "pf> ", stdout); fflush(stdout); }
        at_prompt = 1; errno = 0;
        ssize_t got = getline(&line, &cap, stdin);
        at_prompt = 0;
        if (got < 0 && errno == EINTR) { clearerr(stdin); msg->n = 0; msg->p[0] = 0; continue; }
        if (got < 0) { free(line); return msg->n ? msg->p : NULL; }
        size_t len = (size_t)got;
        while (len && (line[len-1] == '\n' || line[len-1] == '\r')) len--;
        bool more = len && line[len-1] == '\\';
        buf_add(msg, line, more ? len-1 : len);
        if (!more) { free(line); return msg->p; }
        buf_add(msg, "\n", 1);
    }
}
static void lines(Agent *a, const char *first, bool once) {
    struct sigaction action = {0}, old; action.sa_handler = interrupt; action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask); sigaction(SIGINT, &action, &old);
    if (first) { char *f = strdup(first); qw4_stopped = 0; run(a, f); free(f); }
    Buf msg = {0};
    for (char *line; !once && (line = read_message(&msg));) {
        char *cmd = trim(line);
        if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) break;
        qw4_stopped = 0; run(a, cmd);
    }
    buf_free(&msg); sigaction(SIGINT, &old, NULL);
}

/* ------------------------------------------------------------ main */

static void usage(void) {
    fputs("usage: pf-agent [-m MODEL] [-c CTX] [-n N] [-w DIR] [options]\n"
          "  -m MODEL          a .gguf path or a short name, as for pf\n"
          "  -c CTX            context in tokens (default 100000)\n"
          "  -n N              tokens per reply (default: the room left)\n"
          "  -w DIR            the workspace (default: the current directory)\n"
          "  -p TEXT           a first message; --prompt-file FILE reads it from FILE\n"
          "  --non-interactive no terminal interface: -p runs one turn, else stdin lines\n"
          "  --yes             approve every bash/write/edit\n"
          "  --no-think        reply without reasoning      --no-mtp  no multi-token prediction\n"
          "  --power N         GPU duty cycle in percent    --temp T  sampling temperature (0.6; 0 greedy)\n"
          "  --resume NAME     continue a saved session (its model, context and workspace)\n"
          "  --trace FILE      log messages, replies, tool calls and results\n"
          "       pf-agent --run [--yes] < TEXT    run the tool calls in TEXT, no model\n"
          "       pf-agent --parse < TEXT          print the tool calls in TEXT as JSON lines\n", stderr);
}

static char *read_stdin(void) {
    Buf text = {0}; char chunk[65536]; size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, stdin)) > 0) buf_add(&text, chunk, got);
    buf_add(&text, "", 0);
    return text.p;
}
/* --run: execute the calls written in stdin, in order (tests, debugging). */
static int run_stdin(PfTools *tools) {
    char *text = read_stdin(); PfCall calls[64];
    size_t n = pf_parse_calls(text, calls, 64);
    for (size_t i = 0; i < n; i++) {
        Buf out = {0}; pf_tools_run(tools, &calls[i], &out);
        printf("== %s\n%s%s", calls[i].name, out.p, out.n && out.p[out.n-1] == '\n' ? "" : "\n");
        buf_free(&out); pf_call_free(&calls[i]);
    }
    free(text);
    return 0;
}
/* --parse: the calls in stdin as JSON lines {"name":...,"args":{...}}. */
static int parse_stdin(void) {
    char *text = read_stdin(); Buf json = {0}; PfCall calls[64];
    size_t n = pf_parse_calls(text, calls, 64);
    for (size_t i = 0; i < n; i++) {
        buf_str(&json, "{\"name\":\""); json_escape(&json, calls[i].name); buf_str(&json, "\",\"args\":{");
        for (size_t k = 0; k < calls[i].n; k++) {
            buf_str(&json, k ? ",\"" : "\""); json_escape(&json, calls[i].key[k]);
            buf_str(&json, "\":\""); json_escape(&json, calls[i].value[k]); buf_str(&json, "\"");
        }
        buf_str(&json, "}}\n"); pf_call_free(&calls[i]);
    }
    if (json.n) fwrite(json.p, 1, json.n, stdout);
    free(text); buf_free(&json);
    return 0;
}

static uint32_t number(const char *s, uint32_t lo, uint32_t hi, const char *what) {
    char *end; errno = 0; unsigned long n = strtoul(s, &end, 10);
    if (errno || end == s || *end || n < lo || n > hi) die("pf-agent: %s must be in [%u, %u]", what, lo, hi);
    return (uint32_t)n;
}
/* picoforge.metallib beside the real binary, so the agent runs from any directory. */
static void metallib_path(char out[PATH_MAX]) {
    char exe[PATH_MAX], real[PATH_MAX]; uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) || !realpath(exe, real)) die("pf-agent: cannot find its own binary");
    *strrchr(real, '/') = 0;
    snprintf(out, PATH_MAX, "%s/picoforge.metallib", real);
}
static char *absolute(const char *path) {   /* before -w changes directory */
    char cwd[PATH_MAX]; Buf b = {0};
    if (path[0] != '/' && getcwd(cwd, sizeof cwd)) buf_fmt(&b, "%s/", cwd);
    buf_str(&b, path); return b.p;
}

int main(int argc, char **argv) {
    char root[PATH_MAX];
    if (argc > 1 && (!strcmp(argv[1], "--parse") || !strcmp(argv[1], "--run"))) {
        if (!strcmp(argv[1], "--parse")) return parse_stdin();
        if (!realpath(".", root)) die("pf-agent: cannot resolve the current directory");
        PfTools *tools = pf_tools_create(root, argc > 2 && !strcmp(argv[2], "--yes"), NULL, NULL);
        int rc = run_stdin(tools); pf_tools_destroy(tools); return rc;
    }
    static Agent a = {.s = {.thinking = true}};
    const char *model = NULL, *workspace = NULL, *resume = NULL, *trace_path = NULL;
    char *script = NULL, *ids_path = NULL, *first = NULL;
    PfSessionOptions o = {.context = 100000, .mtp = true, .temperature = 0.6f, .power = 100};
    bool yes = false, plain = false;
    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i];
        if (!strcmp(opt, "--yes")) yes = true;
        else if (!strcmp(opt, "--no-think") || !strcmp(opt, "--think")) a.s.thinking = !strcmp(opt, "--think");
        else if (!strcmp(opt, "--no-mtp")) o.mtp = false;
        else if (!strcmp(opt, "--non-interactive")) plain = true;
        else if (!strcmp(opt, "-h") || !strcmp(opt, "--help")) { usage(); return 0; }
        else if (i+1 == argc) { usage(); return 2; }
        else if (!strcmp(opt, "-m")) model = argv[++i];
        else if (!strcmp(opt, "-c")) o.context = number(argv[++i], 64, QWEN4_MAX_CONTEXT, "-c");
        else if (!strcmp(opt, "-n")) a.s.max_new = number(argv[++i], 0, 1u << 20, "-n");
        else if (!strcmp(opt, "-w")) workspace = argv[++i];
        else if (!strcmp(opt, "-p")) { free(first); first = strdup(argv[++i]); }
        else if (!strcmp(opt, "--prompt-file")) { size_t len; free(first); first = slurp(argv[++i], &len); }
        else if (!strcmp(opt, "--power")) o.power = (int)number(argv[++i], 1, 100, "--power");
        else if (!strcmp(opt, "--temp")) {
            char *end; o.temperature = strtof(argv[++i], &end);
            if (*end || !(o.temperature >= 0 && o.temperature <= 10)) die("pf-agent: --temp must be in [0, 10]");
        }
        else if (!strcmp(opt, "--script")) { size_t len; script = slurp(argv[++i], &len); }
        else if (!strcmp(opt, "--token-ids")) ids_path = absolute(argv[++i]);
        else if (!strcmp(opt, "--resume")) resume = argv[++i];
        else if (!strcmp(opt, "--trace")) trace_path = argv[++i];
        else { usage(); return 2; }
    }
    char saved_model[PATH_MAX];               /* --resume: the session's model, context and MTP setting */
    if (resume) {
        if (!pf_session_peek(resume, saved_model, &o.context, &o.mtp)) return 1;
        if (!model) model = saved_model;
    }
    if (trace_path && !(a.trace = fopen(trace_path, "a"))) die("pf-agent: cannot open %s: %s", trace_path, strerror(errno));
    char lib[PATH_MAX]; metallib_path(lib);
    char *chosen = model_select(model, argv[0]), *path = absolute(chosen); free(chosen);
    if (workspace && chdir(workspace)) die("pf-agent: cannot enter %s: %s", workspace, strerror(errno));
    if (!realpath(".", root)) die("pf-agent: cannot resolve the workspace");
    pthread_mutex_init(&a.mu, NULL); pthread_cond_init(&a.cond, NULL);
    if (pipe(a.wake)) die("pf-agent: cannot create a pipe");
    fcntl(a.wake[0], F_SETFL, O_NONBLOCK); fcntl(a.wake[1], F_SETFL, O_NONBLOCK);
    a.tools = pf_tools_create(root, yes, ask_user, &a);
    a.s.script = script; a.s.ids_path = ids_path; a.s.note = note; a.s.note_ctx = &a;
    snprintf(a.s.workspace, sizeof a.s.workspace, "%s", root);

    o.model = path; o.metallib = lib;
    fprintf(stderr, "pf-agent: loading %s (context %u tokens) ...\n", path, o.context);
    pf_session_open(&a.s, &o);
    char *sys = system_prompt(&a); pf_session_set_system(&a.s, sys); free(sys);
    pf_session_set_tools(&a.s, pf_tool_spec, pf_tool_count());
    if (!plain) a.term = pf_term_open();
    a.color = a.term || isatty(STDOUT_FILENO);
    if (!a.term) fprintf(stderr, "pf-agent: %s, context %u tokens, workspace %s\n", path, o.context, root);
    if (resume) {
        if (!pf_session_resume(&a.s, resume)) { pf_term_close(a.term); return 1; }
        resumed(&a); a.used = a.s.n;
    }
    if (a.term) { ui(&a, first, resume != NULL); pf_term_close(a.term); }
    else {
        if (!resume) prime(&a);
        lines(&a, first, plain && first);
    }

    pf_tools_destroy(a.tools);
    pf_session_close(&a.s); free(path); free(script); free(ids_path); free(first);
    if (a.trace) fclose(a.trace);
    return 0;
}
