/* pf_tools.c — see pf_tools.h. Sections: arguments · workspace paths · pager
 * and approvals · file tools · shell jobs · the tool-call parser · tool table. */
#include "pf_tools.h"
#include "pf_web.h"
#include "common.h"
#include "qwen38_run.h"     /* q38_now, q38_stopped */
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <regex.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define OUT_LIMIT 12000    /* bytes of one tool result (~3-4K tokens); the rest waits for `more` */
#define MAX_PARAMS PF_MAX_PARAMS
#define MAX_JOBS 16
typedef struct {
    int id; pid_t pid; char *command;     /* id 0: a free slot */
    char log[PATH_MAX]; off_t shown;      /* stdout+stderr, and how much the model has seen */
    bool done; int status; double started;
} Job;
struct PfTools {
    char root[PATH_MAX];   /* the workspace */
    bool yes;              /* approve every side effect */
    PfAsk ask; void *ask_ctx;
    Buf more;              /* the rest of the last truncated tool output */
    Job jobs[MAX_JOBS]; int last_job;     /* shell jobs; ids grow from 1 */
    char job_dir[PATH_MAX];               /* their logs, a private temporary directory */
};

#define FILE_LIMIT (32u<<20)   /* read/edit refuse larger files */
#define SEARCH_MAX 300
typedef struct { const PfTools *a; regex_t re; Buf *out; size_t matches, files; } Search;

static const char *skip_space(const char *p) { while (isspace((unsigned char)*p)) p++; return p; }

/* ------------------------------------------------------------ arguments */

static const char *arg(const PfCall *c, const char *key) {
    for (size_t i = 0; i < c->n; i++) if (!strcmp(c->key[i], key)) return c->value[i];
    return NULL;
}
/* Optional integer argument in [lo, hi]; false (with a message) if malformed. */
static bool arg_long(const PfCall *c, const char *key, long fallback, long lo, long hi, long *out, Buf *err) {
    const char *v = arg(c, key); *out = fallback;
    if (!v || !*v) return true;
    char *end; errno = 0; long n = strtol(v, &end, 10);
    while (isspace((unsigned char)*end)) end++;
    if (errno || end == v || *end || n < lo || n > hi) {
        buf_fmt(err, "Tool error: %s must be an integer in [%ld, %ld]\n", key, lo, hi);
        return false;
    }
    *out = n; return true;
}
static bool arg_bool(const PfCall *c, const char *key) {
    const char *v = arg(c, key);
    return v && (!strcmp(v, "true") || !strcmp(v, "1") || !strcmp(v, "yes"));
}
/* ------------------------------------------------------------ workspace paths
 * Every file tool resolves its path with realpath(), which follows symlinks,
 * and refuses anything outside the workspace root: "../x", an absolute path
 * elsewhere, or a link inside the workspace pointing out of it. */
static bool inside(const PfTools *a, const char *path) {
    size_t n = strlen(a->root);
    return !strncmp(path, a->root, n) && (path[n] == 0 || path[n] == '/');
}
static const char *rel(const PfTools *a, const char *path) {   /* for messages */
    size_t n = strlen(a->root);
    if (!strncmp(path, a->root, n) && !path[n]) return ".";
    if (!strncmp(path, a->root, n) && path[n] == '/') return path+n+1;
    return path;
}
/* path (relative to the workspace, or absolute) -> out. may_be_new: the file
 * may not exist yet, but its directory must, inside the workspace. */
static bool resolve(const PfTools *a, const char *path, bool may_be_new, char out[PATH_MAX], Buf *err) {
    char full[PATH_MAX];
    if (!path || !*path) { buf_str(err, "Tool error: missing path\n"); return false; }
    int len = path[0] == '/' ? snprintf(full, sizeof full, "%s", path)
                             : snprintf(full, sizeof full, "%s/%s", a->root, path);
    if (len < 0 || (size_t)len >= sizeof full) { buf_str(err, "Tool error: path too long\n"); return false; }
    if (realpath(full, out)) {
        if (inside(a, out)) return true;
        buf_fmt(err, "Tool error: %s is outside the workspace %s\n", path, a->root);
        return false;
    }
    if (errno != ENOENT || !may_be_new) { buf_fmt(err, "Tool error: %s: %s\n", path, strerror(errno)); return false; }
    char *slash = strrchr(full, '/'), dir[PATH_MAX];
    const char *base = slash+1;
    if (!*base || !strcmp(base, ".") || !strcmp(base, "..")) { buf_fmt(err, "Tool error: bad file name %s\n", path); return false; }
    *slash = 0;
    if (!realpath(*full ? full : "/", dir)) {
        buf_fmt(err, "Tool error: the directory of %s does not exist (create it with bash: mkdir -p)\n", path);
        return false;
    }
    len = snprintf(out, PATH_MAX, "%s/%s", dir, base);
    if (len < 0 || len >= PATH_MAX) { buf_str(err, "Tool error: path too long\n"); return false; }
    if (inside(a, dir)) return true;
    buf_fmt(err, "Tool error: %s is outside the workspace %s\n", path, a->root);
    return false;
}
/* ------------------------------------------------------------ pager and approvals */

/* Keep one tool result within OUT_LIMIT: cut at a line end (or a UTF-8
 * boundary) and keep the rest for the `more` tool. */
static void page(PfTools *a, Buf *out) {
    clean_text(out);
    if (out->n <= OUT_LIMIT) return;
    size_t cut = OUT_LIMIT;
    while (cut > OUT_LIMIT/2 && out->p[cut-1] != '\n') cut--;
    if (out->p[cut-1] != '\n')
        for (cut = OUT_LIMIT; ((unsigned char)out->p[cut] & 0xC0) == 0x80; cut--) {}
    buf_free(&a->more); buf_add(&a->more, out->p+cut, out->n-cut);
    out->n = cut; out->p[cut] = 0;
    buf_fmt(out, "\n[output truncated: %zu more bytes, call more to continue]", a->more.n);
}
/* Ask before a side effect; PF_ALWAYS stops asking for the rest of the session.
 * Without a callback the answer is no (pf-agent --run without --yes). */
static bool approve(PfTools *a, const char *what) {
    if (a->yes) return true;
    PfAnswer answer = a->ask ? a->ask(a->ask_ctx, what) : PF_DENY;
    if (answer == PF_ALWAYS) a->yes = true;
    return answer != PF_DENY;
}
static const char declined[] = "Tool error: the user declined this action. Ask them how to proceed.\n";
/* ------------------------------------------------------------ file tools */

/* Whole text file into out; refuses directories, huge and binary files. */
static bool load_text(const PfTools *a, const char *path, Buf *out, Buf *err) {
    struct stat st;
    if (stat(path, &st)) { buf_fmt(err, "Tool error: %s: %s\n", rel(a, path), strerror(errno)); return false; }
    if (S_ISDIR(st.st_mode)) { buf_fmt(err, "Tool error: %s is a directory (use list)\n", rel(a, path)); return false; }
    if ((uint64_t)st.st_size > FILE_LIMIT) { buf_fmt(err, "Tool error: %s is larger than 32 MiB\n", rel(a, path)); return false; }
    FILE *f = fopen(path, "rb");
    if (!f) { buf_fmt(err, "Tool error: %s: %s\n", rel(a, path), strerror(errno)); return false; }
    char chunk[65536]; size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, f)) > 0 && out->n <= FILE_LIMIT) buf_add(out, chunk, got);
    bool bad = ferror(f); fclose(f);
    if (bad) { buf_fmt(err, "Tool error: cannot read %s\n", rel(a, path)); return false; }
    if (memchr(out->p ? out->p : "", 0, out->n < 8192 ? out->n : 8192)) {
        buf_fmt(err, "Tool error: %s looks binary\n", rel(a, path)); return false;
    }
    return true;
}
/* Write a file atomically: a temporary file in the same directory, flushed to
 * disk, then rename() over the target. A crash leaves the old or the new
 * content, never half of one. An existing file keeps its permissions. */
static bool write_atomic(const PfTools *a, const char *path, const char *data, size_t n, Buf *err) {
    char tmp[PATH_MAX]; struct stat st;
    mode_t mode = stat(path, &st) ? 0644 : (mode_t)(st.st_mode & 07777);
    int len = snprintf(tmp, sizeof tmp, "%s.pf-agent-XXXXXX", path);
    if (len < 0 || (size_t)len >= sizeof tmp) { buf_str(err, "Tool error: path too long\n"); return false; }
    int fd = mkstemp(tmp);
    if (fd < 0) { buf_fmt(err, "Tool error: cannot create a file next to %s: %s\n", rel(a, path), strerror(errno)); return false; }
    size_t done = 0;
    while (done < n) {
        ssize_t w = write(fd, data+done, n-done);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        done += (size_t)w;
    }
    bool ok = done == n && !fchmod(fd, mode) && !fsync(fd);
    ok = !close(fd) && ok && !rename(tmp, path);
    if (!ok) { buf_fmt(err, "Tool error: writing %s failed: %s\n", rel(a, path), strerror(errno)); unlink(tmp); }
    return ok;
}
static int by_name(const void *x, const void *y) { return strcmp(*(char *const *)x, *(char *const *)y); }
/* list path=. : one directory, sorted; "name/" for directories, sizes for files. */
static void tool_list(PfTools *a, const PfCall *c, Buf *out) {
    char dir[PATH_MAX]; const char *p = arg(c, "path");
    if (!resolve(a, p && *p ? p : ".", false, dir, out)) return;
    DIR *d = opendir(dir);
    if (!d) { buf_fmt(out, "Tool error: %s: %s\n", rel(a, dir), strerror(errno)); return; }
    char **names = NULL; size_t n = 0, cap = 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        if (n == cap) {
            cap = cap ? cap*2 : 64; char **grown = realloc(names, cap*sizeof *names);
            if (!grown) die("pf-agent: out of memory");
            names = grown;
        }
        if (!(names[n++] = strdup(e->d_name))) die("pf-agent: out of memory");
    }
    closedir(d);
    qsort(names, n, sizeof *names, by_name);
    buf_fmt(out, "%s/ (%zu entries)\n", rel(a, dir), n);
    for (size_t i = 0; i < n; i++) {
        char full[PATH_MAX]; struct stat st;
        snprintf(full, sizeof full, "%s/%s", dir, names[i]);
        if (lstat(full, &st)) buf_fmt(out, "%s ?\n", names[i]);
        else if (S_ISDIR(st.st_mode)) buf_fmt(out, "%s/\n", names[i]);
        else if (S_ISLNK(st.st_mode)) buf_fmt(out, "%s@\n", names[i]);
        else buf_fmt(out, "%s  %lld bytes\n", names[i], (long long)st.st_size);
        free(names[i]);
    }
    free(names);
}
/* read path [start] [end] : numbered lines (1-based, inclusive range). */
static void tool_read(PfTools *a, const PfCall *c, Buf *out) {
    char path[PATH_MAX]; long start, end; Buf text = {0};
    if (!resolve(a, arg(c, "path"), false, path, out)) return;
    if (!arg_long(c, "start", 1, 1, LONG_MAX, &start, out) || !arg_long(c, "end", LONG_MAX, 1, LONG_MAX, &end, out)) return;
    if (!load_text(a, path, &text, out)) { buf_free(&text); return; }
    long lines = 0;
    for (size_t i = 0; i < text.n; i++) lines += text.p[i] == '\n';
    if (text.n && text.p[text.n-1] != '\n') lines++;
    if (start > lines && lines) { buf_fmt(out, "Tool error: %s has only %ld lines\n", rel(a, path), lines); buf_free(&text); return; }
    buf_fmt(out, "%s (%ld lines)\n", rel(a, path), lines);
    const char *p = text.p ? text.p : "";
    for (long line = 1; *p && line <= end; line++) {
        const char *nl = strchr(p, '\n'); size_t len = nl ? (size_t)(nl-p) : strlen(p);
        if (line >= start) { buf_fmt(out, "%6ld\t", line); buf_add(out, p, len); buf_add(out, "\n", 1); }
        p += len + (nl != NULL);
    }
    buf_free(&text);
}
/* more : the next part of the last truncated output. */
static void tool_more(PfTools *a, const PfCall *c, Buf *out) {
    (void)c;
    if (!a->more.n) { buf_str(out, "Nothing more: the last output was complete.\n"); return; }
    buf_add(out, a->more.p, a->more.n); buf_free(&a->more);
}
static bool skipped_dir(const char *name) {
    static const char *const skip[] = {"node_modules", "build", "dist", "target", "venv", "__pycache__", NULL};
    if (name[0] == '.') return true;
    for (size_t i = 0; skip[i]; i++) if (!strcmp(name, skip[i])) return true;
    return false;
}
static void search_file(Search *s, const char *path) {
    struct stat st;
    if (stat(path, &st) || !S_ISREG(st.st_mode) || st.st_size > (4 << 20)) return;
    Buf text = {0}, ignored = {0};
    if (!load_text(s->a, path, &text, &ignored) || !text.n) { buf_free(&text); buf_free(&ignored); return; }
    s->files++;
    char *p = text.p;
    for (long line = 1; *p && s->matches < SEARCH_MAX; line++) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        if (!regexec(&s->re, p, 0, NULL, 0)) {
            size_t len = strlen(p); while (len && isspace((unsigned char)p[len-1])) len--;
            buf_fmt(s->out, "%s:%ld: ", rel(s->a, path), line);
            buf_add(s->out, p, len < 240 ? len : 240); buf_str(s->out, len > 240 ? " ...\n" : "\n");
            s->matches++;
        }
        if (!nl) break;
        p = nl+1;
    }
    buf_free(&text); buf_free(&ignored);
}
static void search_dir(Search *s, const char *dir, int depth) {
    DIR *d = depth < 32 ? opendir(dir) : NULL; struct dirent *e;
    if (!d) return;
    while ((e = readdir(d)) && s->matches < SEARCH_MAX) {
        char full[PATH_MAX]; struct stat st;
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        int len = snprintf(full, sizeof full, "%s/%s", dir, e->d_name);
        if (len < 0 || len >= PATH_MAX || lstat(full, &st)) continue;   /* lstat: never follow links out */
        if (S_ISDIR(st.st_mode)) { if (!skipped_dir(e->d_name)) search_dir(s, full, depth+1); }
        else if (S_ISREG(st.st_mode)) search_file(s, full);
    }
    closedir(d);
}
static void tool_search(PfTools *a, const PfCall *c, Buf *out) {
    const char *pattern = arg(c, "pattern"), *p = arg(c, "path");
    char path[PATH_MAX]; struct stat st; Buf re = {0};
    if (!pattern || !*pattern) { buf_str(out, "Tool error: missing pattern\n"); return; }
    if (!resolve(a, p && *p ? p : ".", false, path, out)) return;
    if (arg_bool(c, "regex")) buf_str(&re, pattern);
    else for (const char *q = pattern; *q; q++) {                 /* literal text: escape it */
        if (strchr("\\^$.|?*+()[]{}", *q)) buf_add(&re, "\\", 1);
        buf_add(&re, q, 1);
    }
    Search s = {.a = a, .out = out};
    int rc = regcomp(&s.re, re.p, REG_EXTENDED | REG_NOSUB | (arg_bool(c, "ignore_case") ? REG_ICASE : 0));
    buf_free(&re);
    if (rc) { char msg[128]; regerror(rc, &s.re, msg, sizeof msg); buf_fmt(out, "Tool error: bad regex: %s\n", msg); return; }
    if (!stat(path, &st) && S_ISDIR(st.st_mode)) search_dir(&s, path, 0); else search_file(&s, path);
    regfree(&s.re);
    if (!s.matches) buf_fmt(out, "No matches in %zu files.\n", s.files);
    else if (s.matches >= SEARCH_MAX) buf_fmt(out, "[stopped at %d matches: narrow the pattern or the path]\n", SEARCH_MAX);
}
/* write path content : create or replace a whole file, atomically. */
static void tool_write(PfTools *a, const PfCall *c, Buf *out) {
    char path[PATH_MAX], what[PATH_MAX+96]; struct stat st;
    const char *content = arg(c, "content");
    if (!content) { buf_str(out, "Tool error: missing content\n"); return; }
    if (!resolve(a, arg(c, "path"), true, path, out)) return;
    bool exists = !stat(path, &st);
    if (exists && !S_ISREG(st.st_mode)) { buf_fmt(out, "Tool error: %s is not a regular file\n", rel(a, path)); return; }
    size_t n = strlen(content);
    if (exists) snprintf(what, sizeof what, "write %s (%zu bytes, replaces %lld bytes)", rel(a, path), n, (long long)st.st_size);
    else snprintf(what, sizeof what, "write %s (%zu bytes, new file)", rel(a, path), n);
    if (!approve(a, what)) { buf_str(out, declined); return; }
    if (write_atomic(a, path, content, n, out)) buf_fmt(out, "Wrote %zu bytes to %s.\n", n, rel(a, path));
}
/* Up to `max` lines of s, each prefixed, for an approval preview. */
static void preview(Buf *b, const char *prefix, const char *s, int max) {
    for (int i = 0; *s && i < max; i++) {
        const char *nl = strchr(s, '\n'); size_t len = nl ? (size_t)(nl-s) : strlen(s);
        buf_str(b, prefix); buf_add(b, s, len < 160 ? len : 160); buf_add(b, "\n", 1);
        s += len + (nl != NULL);
        if (i+1 == max && *s) buf_fmt(b, "%s...\n", prefix);
    }
}
/* edit path old new : replace the one exact occurrence of `old`. */
static void tool_edit(PfTools *a, const PfCall *c, Buf *out) {
    const char *old = arg(c, "old"), *new = arg(c, "new");
    char path[PATH_MAX]; Buf text = {0}, next = {0}, what = {0};
    if (!old || !*old || !new) { buf_str(out, "Tool error: edit needs path, old (not empty) and new\n"); return; }
    if (!resolve(a, arg(c, "path"), false, path, out) || !load_text(a, path, &text, out)) { buf_free(&text); return; }
    size_t olen = strlen(old), count = 0; const char *hit = NULL;
    for (const char *p = text.p; p && (p = memmem(p, text.n-(size_t)(p-text.p), old, olen)); p += olen)
        if (!count++) hit = p;
    if (count != 1) {
        if (!count) buf_fmt(out, "Tool error: old text not found in %s (it must match exactly, spaces included; read the file again)\n", rel(a, path));
        else buf_fmt(out, "Tool error: old text found %zu times in %s; include more surrounding lines to make it unique\n", count, rel(a, path));
        buf_free(&text); return;
    }
    size_t at = (size_t)(hit-text.p);
    long line = count_lines(text.p, at);
    buf_add(&next, text.p, at); buf_str(&next, new); buf_add(&next, hit+olen, text.n-at-olen);
    buf_fmt(&what, "edit %s at line %ld\n", rel(a, path), line);
    preview(&what, "\x1b[31m- ", old, 12); buf_str(&what, "\x1b[0m"); preview(&what, "\x1b[32m+ ", new, 12); buf_str(&what, "\x1b[0m");
    if (!approve(a, what.p)) buf_str(out, declined);
    else if (write_atomic(a, path, next.p, next.n, out))
        buf_fmt(out, "Edited %s: lines %ld-%ld replaced by %ld lines.\n", rel(a, path), line,
                line+count_lines(old, olen)-1, count_lines(new, strlen(new)));
    buf_free(&text); buf_free(&next); buf_free(&what);
}
/* ------------------------------------------------------------ shell jobs
 * bash runs `/bin/bash -c COMMAND` through pf_spawn (pf_util.c), stdout and
 * stderr into a log file; past its timeout a command keeps running as a job. */
static bool job_reap(Job *j) {
    int status;
    if (!j->done && waitpid(j->pid, &status, WNOHANG) == j->pid) { j->done = true; j->status = status; }
    return j->done;
}
/* Wait up to `seconds` for the job to end; Ctrl+C stops waiting. */
static bool job_wait(Job *j, double seconds) {
    for (double end = q38_now()+seconds; !job_reap(j) && q38_now() < end && !q38_stopped;) pause_ms(20);
    return j->done;
}
/* SIGTERM to the whole group, SIGKILL if it is still there after 2 s. */
static void job_kill(Job *j) {
    if (job_reap(j)) return;
    killpg(j->pid, SIGTERM);
    for (int i = 0; i < 100 && !job_reap(j); i++) pause_ms(20);
    if (!j->done) { killpg(j->pid, SIGKILL); waitpid(j->pid, &j->status, 0); j->done = true; }
}
/* The log the model has not seen yet; past 1 MiB only the end. */
static void job_output(Job *j, Buf *out) {
    FILE *f = fopen(j->log, "rb"); struct stat st;
    if (!f || fstat(fileno(f), &st)) { if (f) fclose(f); return; }
    off_t from = j->shown, size = st.st_size;
    if (size-from > (1 << 20)) { buf_fmt(out, "[%lld bytes skipped]\n", (long long)(size-from-(1 << 20))); from = size-(1 << 20); }
    char chunk[65536]; size_t got;
    if (!fseeko(f, from, SEEK_SET))
        while (from < size && (got = fread(chunk, 1, sizeof chunk, f)) > 0) { buf_add(out, chunk, got); from += (off_t)got; }
    fclose(f); j->shown = from;
}
static void job_state(const Job *j, Buf *out) {
    if (!j->done) buf_fmt(out, "[job %d running for %.0f s: %s]\n", j->id, q38_now()-j->started, j->command);
    else if (WIFEXITED(j->status)) buf_fmt(out, "[exit code %d]\n", WEXITSTATUS(j->status));
    else buf_fmt(out, "[killed by signal %d]\n", WIFSIGNALED(j->status) ? WTERMSIG(j->status) : 0);
}
static void job_free(Job *j) { unlink(j->log); free(j->command); *j = (Job){0}; }
static Job *job_find(PfTools *a, const PfCall *c, Buf *out) {
    long id;
    if (!arg_long(c, "id", 0, 1, INT_MAX, &id, out)) return NULL;
    for (int i = 0; i < MAX_JOBS; i++) if (a->jobs[i].id && a->jobs[i].id == id) return &a->jobs[i];
    buf_fmt(out, "Tool error: no job %ld (bash_status with no id lists them)\n", id);
    return NULL;
}
/* bash command [timeout=120] : run it; past the timeout it keeps running as a job. */
static void tool_bash(PfTools *a, const PfCall *c, Buf *out) {
    const char *command = arg(c, "command"); long timeout; Job *j = NULL;
    if (!command || !*command) { buf_str(out, "Tool error: missing command\n"); return; }
    if (!arg_long(c, "timeout", 120, 1, 600, &timeout, out)) return;
    for (int i = 0; i < MAX_JOBS && !j; i++) if (!a->jobs[i].id) j = &a->jobs[i];
    if (!j) { buf_str(out, "Tool error: 16 jobs are running; stop one with bash_stop first\n"); return; }
    Buf what = {0}; buf_fmt(&what, "bash: %s", command);
    bool ok = approve(a, what.p); buf_free(&what);
    if (!ok) { buf_str(out, declined); return; }
    if (!a->job_dir[0]) {
        const char *tmp = getenv("TMPDIR");
        snprintf(a->job_dir, sizeof a->job_dir, "%s/pf-agent.XXXXXX", tmp && *tmp ? tmp : "/tmp");
        if (!mkdtemp(a->job_dir)) { a->job_dir[0] = 0; buf_fmt(out, "Tool error: no temporary directory: %s\n", strerror(errno)); return; }
    }
    *j = (Job){.id = ++a->last_job, .started = q38_now()};
    snprintf(j->log, sizeof j->log, "%s/job%d.log", a->job_dir, j->id);
    int fd = open(j->log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    char *argv[] = {"/bin/bash", "-c", (char *)command, NULL};
    j->pid = fd < 0 ? -1 : pf_spawn(argv, fd, fd);
    if (fd >= 0) close(fd);
    if (j->pid < 0) { buf_fmt(out, "Tool error: cannot start bash: %s\n", strerror(errno)); job_free(j); return; }
    if (!(j->command = strdup(command))) die("pf-agent: out of memory");
    bool ended = job_wait(j, (double)timeout);
    if (!ended && q38_stopped) { job_kill(j); ended = true; buf_str(out, "[interrupted by the user]\n"); }
    job_output(j, out);
    if (out->n && out->p[out->n-1] != '\n') buf_str(out, "\n");
    if (ended) { job_state(j, out); job_free(j); return; }
    buf_fmt(out, "[still running after %ld s as job %d: bash_status id=%d shows new output, bash_stop id=%d ends it]\n",
            timeout, j->id, j->id, j->id);
}
/* bash_status [id] : one job's new output and state, or the list of jobs. */
static void tool_bash_status(PfTools *a, const PfCall *c, Buf *out) {
    if (!arg(c, "id") || !*arg(c, "id")) {
        size_t n = 0;
        for (int i = 0; i < MAX_JOBS; i++) if (a->jobs[i].id) { job_reap(&a->jobs[i]); buf_fmt(out, "job %d: ", a->jobs[i].id); job_state(&a->jobs[i], out); n++; }
        if (!n) buf_str(out, "No background jobs.\n");
        return;
    }
    Job *j = job_find(a, c, out);
    if (!j) return;
    job_reap(j); job_output(j, out);
    if (out->n && out->p[out->n-1] != '\n') buf_str(out, "\n");
    job_state(j, out);
    if (j->done) job_free(j);
}
/* bash_stop id : end a job and its children. */
static void tool_bash_stop(PfTools *a, const PfCall *c, Buf *out) {
    Job *j = job_find(a, c, out);
    if (!j) return;
    job_kill(j); job_output(j, out);
    if (out->n && out->p[out->n-1] != '\n') buf_str(out, "\n");
    job_state(j, out); job_free(j);
}
/* At exit no job may outlive the agent, and the logs go. */
static void jobs_cleanup(PfTools *a) {
    for (int i = 0; i < MAX_JOBS; i++) if (a->jobs[i].id) { job_kill(&a->jobs[i]); job_free(&a->jobs[i]); }
    if (a->job_dir[0]) rmdir(a->job_dir);
}
/* ------------------------------------------------------------ tool-call parser
 * Inside <tool_call> ... </tool_call>:
 *   <function=NAME>
 *   <parameter=KEY>
 *   VALUE
 *   </parameter>
 *   </function>
 * The newline after an opening tag and the one before a closing tag belong to
 * the format, everything between is the value. A value may itself contain
 * "</parameter>" (say, when editing this file), so a closing tag counts only
 * when the next parameter or </function> follows it. */
static char *dup_range(const char *s, const char *e) {
    char *p = strndup(s, (size_t)(e-s));
    if (!p) die("pf-agent: out of memory");
    return p;
}
static const char *param_end(const char *p) {
    for (; (p = strstr(p, "</parameter>")); p++) {
        const char *next = skip_space(p+12);
        if (!strncmp(next, "<parameter=", 11) || !strncmp(next, "</function>", 11)) return p;
    }
    return NULL;
}
void pf_call_free(PfCall *c) {
    free(c->name);
    for (size_t i = 0; i < c->n; i++) { free(c->key[i]); free(c->value[i]); }
    *c = (PfCall){0};
}
/* Up to max calls in text; returns how many. Incomplete calls are skipped. */
size_t pf_parse_calls(const char *text, PfCall *calls, size_t max) {
    size_t n = 0;
    for (const char *p = text; n < max && (p = strstr(p, "<tool_call>")); ) {
        const char *q = skip_space(p+11), *gt;
        p += 11;
        if (strncmp(q, "<function=", 10) || !(gt = strchr(q+10, '>')) || gt == q+10) continue;
        PfCall c = {.name = dup_range(q+10, gt)};
        for (q = skip_space(gt+1); !strncmp(q, "<parameter=", 11) && c.n < MAX_PARAMS; ) {
            const char *key = q+11, *kend = strchr(key, '>'), *v, *vend;
            if (!kend || !(vend = param_end(kend))) break;
            v = kend+1; if (*v == '\n') v++;
            const char *cut = vend > v && vend[-1] == '\n' ? vend-1 : vend;
            c.key[c.n] = dup_range(key, kend); c.value[c.n] = dup_range(v, cut < v ? v : cut); c.n++;
            q = skip_space(vend+12);
        }
        if (strncmp(q, "</function>", 11)) { pf_call_free(&c); continue; }
        calls[n++] = c; p = q+11;
    }
    return n;
}

/* ------------------------------------------------------------ web tools */

static void tool_google_search(PfTools *a, const PfCall *c, Buf *out) { (void)a; pf_web_search(arg(c, "query"), out); }
static void tool_visit_page(PfTools *a, const PfCall *c, Buf *out) { (void)a; pf_web_visit(arg(c, "url"), out); }

/* ------------------------------------------------------------ tool table */

typedef struct { PfToolSpec spec; void (*run)(PfTools *, const PfCall *, Buf *); } Tool;
static const Tool tools[] = {
    {{"list", "List one directory of the workspace: subdirectories end with '/', files show their size.",
      {{"path", "string", "Directory, relative to the workspace root. Default: the root.", false}}}, tool_list},
    {{"read", "Read a text file with line numbers. Long output is cut: call more to continue.",
      {{"path", "string", "File, relative to the workspace root.", true},
       {"start", "integer", "First line to show, 1-based. Default 1.", false},
       {"end", "integer", "Last line to show, inclusive. Default: the end of the file.", false}}}, tool_read},
    {{"more", "Continue the last output that was cut (read, search, bash, visit_page, ...).", {{0}}}, tool_more},
    {{"search", "Search text in the workspace files, recursively (skips hidden directories, build "
      "outputs and binary files). Prints file:line: text.",
      {{"pattern", "string", "The text to find, or a POSIX extended regex when regex is true.", true},
       {"path", "string", "File or directory to search. Default: the root.", false},
       {"regex", "boolean", "Treat pattern as a regex. Default false.", false},
       {"ignore_case", "boolean", "Case-insensitive match. Default false.", false}}}, tool_search},
    {{"write", "Create a file or replace its whole content, atomically. The directory must exist. "
      "For changes to an existing file prefer edit.",
      {{"path", "string", "File, relative to the workspace root.", true},
       {"content", "string", "The complete new content of the file.", true}}}, tool_write},
    {{"edit", "Replace one exact occurrence of old text with new text in a file. old must match "
      "exactly once, whitespace included: read the file first.",
      {{"path", "string", "File, relative to the workspace root.", true},
       {"old", "string", "The exact text to replace (unique in the file).", true},
       {"new", "string", "The replacement text.", true}}}, tool_edit},
    {{"bash", "Run a shell command with /bin/bash -c in the workspace root; returns stdout and stderr "
      "with the exit code. A command still running after timeout seconds keeps running as a background job: "
      "check it with bash_status, end it with bash_stop. Not interactive: stdin is empty.",
      {{"command", "string", "The command line.", true},
       {"timeout", "integer", "Seconds to wait before it becomes a background job (1-600, default 120).", false}}}, tool_bash},
    {{"bash_status", "New output and state of a background job, or the list of jobs.",
      {{"id", "integer", "The job id. Default: list all jobs.", false}}}, tool_bash_status},
    {{"bash_stop", "Stop a background job and its child processes.",
      {{"id", "integer", "The job id.", true}}}, tool_bash_stop},
    {{"google_search", "Search the web (DuckDuckGo). Returns titles, URLs and snippets; "
      "read a result with visit_page.",
      {{"query", "string", "The search query.", true}}}, tool_google_search},
    {{"visit_page", "Fetch a web page (http or https) and return it as text, with its links: "
      "documentation, issues, READMEs, changelogs. Pages that need JavaScript come back nearly empty.",
      {{"url", "string", "The full URL, starting with http:// or https://.", true}}}, tool_visit_page},
};
size_t pf_tool_count(void) { return sizeof tools/sizeof *tools; }
const PfToolSpec *pf_tool_spec(size_t i) { return i < pf_tool_count() ? &tools[i].spec : NULL; }

void pf_tools_run(PfTools *a, const PfCall *c, Buf *out) {
    for (size_t i = 0; i < pf_tool_count(); i++) {
        if (strcmp(c->name, tools[i].spec.name)) continue;
        tools[i].run(a, c, out);
        if (!out->n) buf_str(out, "(no output)\n");
        page(a, out);
        return;
    }
    buf_fmt(out, "Tool error: unknown tool %s. Available:", c->name);
    for (size_t i = 0; i < pf_tool_count(); i++) buf_fmt(out, " %s", tools[i].spec.name);
    buf_str(out, "\n"); page(a, out);
}

PfTools *pf_tools_create(const char *root, bool yes, PfAsk ask, void *ask_ctx) {
    PfTools *a = calloc(1, sizeof *a);
    if (!a) die("pf-agent: out of memory");
    snprintf(a->root, sizeof a->root, "%s", root);
    a->yes = yes; a->ask = ask; a->ask_ctx = ask_ctx;
    return a;
}
void pf_tools_destroy(PfTools *a) {
    if (!a) return;
    jobs_cleanup(a); buf_free(&a->more); free(a);
}
const char *pf_tools_root(const PfTools *a) { return a->root; }
bool pf_tools_set_root(PfTools *a, const char *dir) {
    char real[PATH_MAX];
    if (!realpath(dir, real)) return false;
    snprintf(a->root, sizeof a->root, "%s", real);
    return true;
}
void pf_tools_approve_all(PfTools *a) { a->yes = true; }
bool pf_tools_read_text(const PfTools *a, const char *path, size_t limit, Buf *out) {
    Buf err = {0}; char full[PATH_MAX];
    bool ok = resolve(a, path, false, full, &err) && load_text(a, full, out, &err) && out->n <= limit;
    buf_free(&err);
    if (ok) clean_text(out);
    return ok;
}
