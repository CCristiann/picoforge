/* pf_agent.c — pf-agent, a coding agent in the terminal on the picoforge engine.
 *
 * The model runs in this process (pf_session: transcript, template, replies,
 * saved sessions); it asks for tools in Qwen3.8's native format and pf_tools
 * runs them (files in the workspace, bash jobs, the web). This file is only the
 * interface: the prompt, the commands, how replies and tool calls are shown,
 * and the user's approvals.
 */
#include "model_select.h"
#include "pf_session.h"
#include "pf_tools.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_CALLS 16       /* tool calls run from one reply */
#define MAX_ROUNDS 64      /* tool rounds in one user turn before we stop and ask */

typedef struct {
    PfSession s;
    PfTools *tools;
    bool ask;              /* stdin is a terminal we can ask on */
} Agent;

/* ------------------------------------------------------------ approvals and notes */

/* y = this once, a = always (rest of the session), anything else = no. */
static PfAnswer ask_user(void *ctx, const char *what) {
    const Agent *a = ctx;
    if (!a->ask) return PF_DENY;
    fprintf(stdout, "\n\x1b[1m%s\x1b[0m\nAllow? [y]es / [n]o / [a]lways: ", what); fflush(stdout);
    char line[64];
    if (!fgets(line, sizeof line, stdin)) { clearerr(stdin); return PF_DENY; }
    if (line[0] == 'a' || line[0] == 'A') return PF_ALWAYS;
    return line[0] == 'y' || line[0] == 'Y' || line[0] == 's' || line[0] == 'S' ? PF_ALLOW : PF_DENY;
}
static void note(void *ctx, const char *message) { (void)ctx; fputs(message, stdout); fflush(stdout); }

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

/* ------------------------------------------------------------ showing a turn */

/* The reply streams through Q38Printer (grey reasoning), but the XML of a
 * tool call is hidden: each call is shown as one line once the reply is
 * done. A tag can arrive split across tokens, so a possible prefix waits. */
typedef struct { Q38Printer out; Buf pend; bool hidden; } View;
static void view_text(void *ctx, const uint8_t *s, size_t n) {
    View *v = ctx;
    buf_add(&v->pend, s, n);
    for (;;) {
        const char *tag = v->hidden ? "</tool_call>" : "<tool_call>";
        char *hit = strstr(v->pend.p, tag);
        size_t keep = hit ? (size_t)(hit-v->pend.p) : v->pend.n > 12 ? v->pend.n-12 : 0;
        if (!v->hidden) q38_print_text(&v->out, (const uint8_t *)v->pend.p, keep);
        size_t drop = hit ? keep+strlen(tag) : keep;
        memmove(v->pend.p, v->pend.p+drop, v->pend.n-drop+1); v->pend.n -= drop;
        if (!hit) return;
        if (!v->hidden) q38_print_text(&v->out, (const uint8_t *)"\n...", 4);
        v->hidden = !v->hidden;
    }
}
static void view_end(View *v) {
    if (!v->hidden && v->pend.n) q38_print_text(&v->out, (const uint8_t *)v->pend.p, v->pend.n);
    q38_print_end(&v->out); fputc('\n', stdout); buf_free(&v->pend);
}

/* One line per call: "-> name key=value ...", long values summarized. */
static void show_call(const PfCall *c) {
    printf("\x1b[36m-> %s\x1b[0m", c->name);
    for (size_t k = 0; k < c->n; k++) {
        const char *v = c->value[k], *nl = strchr(v, '\n');
        if (nl) printf(" %s=<%ld lines>", c->key[k], count_lines(v, strlen(v)));
        else printf(" %s=%.*s%s", c->key[k], 80, v, strlen(v) > 80 ? "..." : "");
    }
    putchar('\n');
}
/* The first lines of a result, grey. */
static void show_result(const Buf *out) {
    const char *p = out->p; int shown = 0;
    fputs("\x1b[90m", stdout);
    for (; *p && shown < 6; shown++) {
        const char *nl = strchr(p, '\n'); size_t len = nl ? (size_t)(nl-p) : strlen(p);
        printf("   %.*s\n", (int)(len < 160 ? len : 160), p);
        p += len + (nl != NULL);
    }
    if (*p) printf("   ... (%ld more lines)\n", count_lines(p, strlen(p)) - (out->p[out->n-1] == '\n'));
    fputs("\x1b[0m", stdout); fflush(stdout);
}

static char *reply(Agent *a, size_t added) {
    View v = {.out = {.color = isatty(STDOUT_FILENO)}}; q38_printer_shade(&v.out, a->s.thinking);
    PfStream stream = {view_text, &v}; PfReplyStats st;
    char *text = pf_session_reply(&a->s, added, &stream, &st);
    view_end(&v);
    if (!st.generated && !st.prefilled && text) return text;   /* a canned reply (--script) */
    fprintf(stderr, "\x1b[90m[prefill %zu tokens %.1f t/s, reply %zu tokens %.1f t/s, context %zu/%u%s%s]\x1b[0m\n",
        st.prefilled, st.prefill_s > 0 ? (double)st.prefilled/st.prefill_s : 0.0, st.generated,
        st.decode_s > 0 && st.generated > 1 ? (double)(st.generated-1)/st.decode_s : 0.0,
        a->s.n, a->s.run.capacity, st.replayed ? ", replayed" : "", st.cancelled ? ", interrupted" : "");
    if (st.length) fprintf(stderr, "pf-agent: the reply hit the token limit\n");
    return text;
}

/* One user message, then reply -> tools -> results until the model answers
 * without a tool call, the user interrupts, or MAX_ROUNDS is reached. */
static void agent_turn(Agent *a, const char *text) {
    PfMsg user = {"user", text};
    if (!a->s.n) {                                  /* a session is named by its first request */
        Buf t = {0}; size_t len = strcspn(text, "\n");
        buf_add(&t, text, len < sizeof a->s.title-4 ? len : sizeof a->s.title-4); clean_text(&t);
        snprintf(a->s.title, sizeof a->s.title, "%s", t.p); buf_free(&t);
    }
    size_t added = pf_session_append(&a->s, &user, 1);
    for (int round = 0; added && round < MAX_ROUNDS; round++) {
        char *answer = reply(a, added);
        if (!answer) return;
        PfCall calls[MAX_CALLS];
        size_t n = pf_parse_calls(pf_session_answer(answer), calls, MAX_CALLS);
        free(answer);
        if (!n || q38_stopped) { for (size_t i = 0; i < n; i++) pf_call_free(&calls[i]); return; }
        Buf outs[MAX_CALLS] = {{0}}; PfMsg results[MAX_CALLS];
        for (size_t i = 0; i < n; i++) {
            show_call(&calls[i]);
            if (q38_stopped) buf_str(&outs[i], "Tool error: interrupted by the user before it ran.\n");
            else pf_tools_run(a->tools, &calls[i], &outs[i]);
            show_result(&outs[i]);
            results[i] = (PfMsg){"tool", outs[i].p};
            pf_call_free(&calls[i]);
        }
        added = pf_session_append(&a->s, results, n);
        for (size_t i = 0; i < n; i++) buf_free(&outs[i]);
        if (q38_stopped) { added = 0; fprintf(stderr, "pf-agent: interrupted; the results are kept for your next message\n"); }
    }
    if (added) fprintf(stderr, "pf-agent: %d tool rounds without an answer, stopping here\n", MAX_ROUNDS);
}

/* ------------------------------------------------------------ sessions */

/* After a resume: the session's workspace, and a system prompt for it. */
static void resumed(Agent *a) {
    if (strcmp(a->s.workspace, pf_tools_root(a->tools))) {
        if (chdir(a->s.workspace) || !pf_tools_set_root(a->tools, a->s.workspace))
            fprintf(stderr, "pf-agent: the session's workspace %s is gone; staying in %s\n", a->s.workspace, pf_tools_root(a->tools));
        else { char *sys = system_prompt(a); pf_session_set_system(&a->s, sys); free(sys); }
    }
    snprintf(a->s.workspace, sizeof a->s.workspace, "%s", pf_tools_root(a->tools));
    printf("Workspace %s, thinking %s. First request: %s\n", a->s.workspace, a->s.thinking ? "on" : "off", a->s.title);
}

/* ------------------------------------------------------------ the prompt */

static volatile sig_atomic_t at_prompt;
static void interrupt(int signum) {
    (void)signum; q38_stopped = 1;
    if (at_prompt) { ssize_t ignored = write(STDOUT_FILENO, "\npf> ", 5); (void)ignored; }
}
static void help(void) {
    puts("pf-agent: describe a task; the agent reads, edits and runs commands in this workspace.\n"
         "  /auto          Approve every bash/write/edit for the rest of the session.\n"
         "  /think, /nothink  Reasoning on or off.\n"
         "  /reset         Start a new conversation.\n"
         "  /save [NAME]   Save the session (transcript and model state) to resume it later.\n"
         "  /sessions      List the saved sessions.  /resume NAME  Continue one.\n"
         "  /help, /quit   This help; leave (Ctrl+D too).\n"
         "  Ctrl+C         Stop the reply or the running tool.\n"
         "End a line with \\ to continue the message on the next line.");
}
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *end = s+strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = 0;
    return s;
}
static bool interactive(void) { return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO); }
/* One message from stdin; a line ending in '\' continues on the next. NULL at EOF. */
static char *read_message(Buf *msg) {
    char *line = NULL; size_t cap = 0; msg->n = 0; buf_add(msg, "", 0);
    for (;;) {
        if (interactive()) { fputs(msg->n ? "... " : "pf> ", stdout); fflush(stdout); }
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

static void conversation(Agent *a) {
    if (interactive()) help();
    Buf msg = {0};
    for (char *line; (line = read_message(&msg));) {
        char *cmd = trim(line);
        if (!*cmd) continue;
        if (!strcmp(cmd, "/help")) help();
        else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) break;
        else if (!strcmp(cmd, "/auto")) { pf_tools_approve_all(a->tools); puts("Every bash/write/edit is approved for this session."); }
        else if (!strcmp(cmd, "/think") || !strcmp(cmd, "/nothink")) {
            a->s.thinking = !strcmp(cmd, "/think"); printf("Thinking %s.\n", a->s.thinking ? "on" : "off");
        }
        else if (!strcmp(cmd, "/reset")) { pf_session_reset(&a->s); pf_session_dump(&a->s); puts("New conversation."); }
        else if (!strncmp(cmd, "/save", 5) && (!cmd[5] || cmd[5] == ' ')) pf_session_save(&a->s, trim(cmd+5));
        else if (!strcmp(cmd, "/sessions")) { Buf list = {0}; pf_session_list(&list); fputs(list.p, stdout); buf_free(&list); }
        else if (!strncmp(cmd, "/resume ", 8)) { if (pf_session_resume(&a->s, trim(cmd+8))) resumed(a); pf_session_dump(&a->s); }
        else if (*cmd == '/' && !strchr(cmd, ' ') && !strchr(cmd, '\n')) fprintf(stderr, "pf-agent: unknown command %s (try /help)\n", cmd);
        else agent_turn(a, cmd);
        fflush(stdout);
    }
    buf_free(&msg);
    if (interactive()) fputc('\n', stdout);
}

/* ------------------------------------------------------------ main */

static void usage(void) {
    fputs("usage: pf-agent [-m MODEL] [-c CTX] [-n N] [-w DIR] [--yes] [--no-think] [--no-mtp] [--power N]\n"
          "  -m MODEL    a .gguf path or a short name, as for picoforge\n"
          "  -c CTX      context in tokens (default 100000)\n"
          "  -n N        tokens per reply (default: the room left)\n"
          "  -w DIR      the workspace (default: the current directory)\n"
          "  --yes       approve every bash/write/edit\n"
          "  --no-think  reply without reasoning\n"
          "  --power N   GPU duty cycle in percent, as for picoforge\n"
          "  --temp T    sampling temperature (default 0.6; 0: greedy)\n"
          "  --resume NAME  continue a saved session (its model, context and workspace)\n"
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
    Agent a = {.s = {.thinking = true}};
    const char *model = NULL, *workspace = NULL, *resume = NULL; char *script = NULL, *ids_path = NULL;
    PfSessionOptions o = {.context = 100000, .mtp = true, .temperature = 0.6f, .power = 100};
    bool yes = false;
    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i];
        if (!strcmp(opt, "--yes")) yes = true;
        else if (!strcmp(opt, "--no-think")) a.s.thinking = false;
        else if (!strcmp(opt, "--no-mtp")) o.mtp = false;
        else if (!strcmp(opt, "-h") || !strcmp(opt, "--help")) { usage(); return 0; }
        else if (i+1 == argc) { usage(); return 2; }
        else if (!strcmp(opt, "-m")) model = argv[++i];
        else if (!strcmp(opt, "-c")) o.context = number(argv[++i], 64, QWEN38_MAX_CONTEXT, "-c");
        else if (!strcmp(opt, "-n")) a.s.max_new = number(argv[++i], 0, 1u << 20, "-n");
        else if (!strcmp(opt, "-w")) workspace = argv[++i];
        else if (!strcmp(opt, "--power")) o.power = (int)number(argv[++i], 1, 100, "--power");
        else if (!strcmp(opt, "--temp")) {
            char *end; o.temperature = strtof(argv[++i], &end);
            if (*end || !(o.temperature >= 0 && o.temperature <= 10)) die("pf-agent: --temp must be in [0, 10]");
        }
        else if (!strcmp(opt, "--script")) { size_t len; script = slurp(argv[++i], &len); }
        else if (!strcmp(opt, "--token-ids")) ids_path = absolute(argv[++i]);
        else if (!strcmp(opt, "--resume")) resume = argv[++i];
        else { usage(); return 2; }
    }
    char saved_model[PATH_MAX];               /* --resume: the session's model, context and MTP setting */
    if (resume) {
        if (!pf_session_peek(resume, saved_model, &o.context, &o.mtp)) return 1;
        if (!model) model = saved_model;
    }
    char lib[PATH_MAX]; metallib_path(lib);
    char *chosen = model_select(model, argv[0]), *path = absolute(chosen); free(chosen);
    if (workspace && chdir(workspace)) die("pf-agent: cannot enter %s: %s", workspace, strerror(errno));
    if (!realpath(".", root)) die("pf-agent: cannot resolve the workspace");
    a.ask = isatty(STDIN_FILENO);
    a.tools = pf_tools_create(root, yes, ask_user, &a);
    a.s.script = script; a.s.ids_path = ids_path; a.s.note = note;
    snprintf(a.s.workspace, sizeof a.s.workspace, "%s", root);
    struct sigaction action = {0}, old; action.sa_handler = interrupt; action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &old)) die("pf-agent: cannot install the Ctrl+C handler");

    o.model = path; o.metallib = lib;
    pf_session_open(&a.s, &o);
    char *sys = system_prompt(&a); pf_session_set_system(&a.s, sys); free(sys);
    pf_session_set_tools(&a.s, pf_tool_spec, pf_tool_count());
    fprintf(stderr, "pf-agent: %s, context %u tokens, workspace %s\n", path, o.context, root);
    if (resume) { if (!pf_session_resume(&a.s, resume)) return 1; resumed(&a); }

    conversation(&a);

    pf_tools_destroy(a.tools);
    sigaction(SIGINT, &old, NULL);
    pf_session_close(&a.s); free(path); free(script); free(ids_path);
    return 0;
}
