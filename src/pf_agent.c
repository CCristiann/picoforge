/* pf_agent.c — pf-agent, a small coding agent on the picoforge engine.
 *
 * The model runs inside this process: no server, no framework. The
 * conversation is a token transcript that only grows (as in the REPL,
 * src/qwen38_repl.c), so the model always sees exactly what it generated and
 * each turn prefills only its own suffix. The model asks for a tool in
 * Qwen3.8's native format (the chat template teaches it):
 *
 *   <tool_call>
 *   <function=read>
 *   <parameter=path>
 *   src/main.c
 *   </parameter>
 *   </function>
 *   </tool_call>
 *
 * We run the tool and send its output back as a `tool` message. Tools:
 *   list, read, more, search, write, edit   files, confined to the workspace
 *   bash, bash_status, bash_stop            shell commands and background jobs
 *   google_search, visit_page               the web, through /usr/bin/curl
 * Side effects (bash, write, edit) ask the user first.
 *
 * Sections: text buffers and hygiene · workspace paths · the pager and
 * approvals · file tools · the tool-call parser · tool table · main.
 */
#include "model_select.h"
#include "qwen38_run.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <regex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------ text buffers */

typedef struct { char *p; size_t n, cap; } Buf;   /* always NUL-terminated */

static void buf_add(Buf *b, const void *s, size_t n) {
    if (b->n+n+1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n+n+1) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) die("pf-agent: out of memory");
        b->p = p; b->cap = cap;
    }
    if (n) memcpy(b->p+b->n, s, n);
    b->n += n; b->p[b->n] = 0;
}
static void buf_str(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }
static void buf_fmt(Buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void buf_fmt(Buf *b, const char *fmt, ...) {
    char small[512]; va_list ap;
    va_start(ap, fmt); int len = vsnprintf(small, sizeof small, fmt, ap); va_end(ap);
    if (len < 0) return;
    if ((size_t)len < sizeof small) { buf_add(b, small, (size_t)len); return; }
    char *big = malloc((size_t)len+1);
    if (!big) die("pf-agent: out of memory");
    va_start(ap, fmt); vsnprintf(big, (size_t)len+1, fmt, ap); va_end(ap);
    buf_add(b, big, (size_t)len); free(big);
}
static void buf_free(Buf *b) { free(b->p); *b = (Buf){0}; }

/* Length of the valid UTF-8 sequence at s, 0 if invalid (overlong forms,
 * surrogates and code points past U+10FFFF are invalid). */
static size_t utf8_len(const unsigned char *s, size_t n) {
    unsigned c = s[0], lo = 0x80, hi = 0xBF; size_t len;
    if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return 0;
    if (n < len || s[1] < lo || s[1] > hi) return 0;
    for (size_t k = 2; k < len; k++) if ((s[k] & 0xC0) != 0x80) return 0;
    return len;
}

/* Tool output goes back into the prompt, so it must be valid UTF-8 (the
 * template renderer refuses anything else) and must never spell a special
 * token: the tokenizer turns the text "<|im_end|>" into the real end-of-turn
 * token, so a file or web page containing it could close the tool message and
 * fake a new turn (prompt injection at the token level). Invalid bytes and
 * control characters (terminal colors too) become '?', and "<|" becomes "< |". */
static void clean_text(Buf *b) {
    Buf out = {0}; const unsigned char *s = (const unsigned char *)b->p;
    for (size_t i = 0; i < b->n;) {
        unsigned c = s[i];
        if (c == '<' && i+1 < b->n && s[i+1] == '|') { buf_str(&out, "< |"); i += 2; }
        else if (c >= 0x80) {
            size_t len = utf8_len(s+i, b->n-i);
            if (len) buf_add(&out, s+i, len); else buf_add(&out, "?", 1);
            i += len ? len : 1;
        } else {
            if ((c >= 0x20 && c != 0x7F) || c == '\n' || c == '\t') buf_add(&out, s+i, 1);
            else if (c != '\r') buf_add(&out, "?", 1);
            i++;
        }
    }
    buf_add(&out, "", 0);
    buf_free(b); *b = out;
}

/* JSON string body for the test-facing outputs (--parse). */
static void json_escape(Buf *b, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { buf_add(b, "\\", 1); buf_add(b, s, 1); }
        else if (c == '\n') buf_str(b, "\\n");
        else if (c == '\t') buf_str(b, "\\t");
        else if (c < 0x20) buf_fmt(b, "\\u%04x", c);
        else buf_add(b, s, 1);
    }
}

/* ------------------------------------------------------------ the agent */

#define OUT_LIMIT 12000    /* bytes of one tool result (~3-4K tokens); the rest waits for `more` */
#define MAX_PARAMS 8

typedef struct { char *name; size_t n; char *key[MAX_PARAMS], *value[MAX_PARAMS]; } Call;

typedef struct {
    char root[PATH_MAX];   /* the workspace (realpath of the start directory) */
    bool yes;              /* approve every side effect: --yes, or "a" at a prompt */
    bool ask;              /* stdin is a terminal we can ask on */
    Buf more;              /* the rest of the last truncated tool output */
    /* The model and the conversation (absent in --run / --parse). */
    Q38Run run;
    int32_t *ids, *generated; size_t n;   /* the transcript is ids[0..n) */
    int32_t im_end;        /* the end-of-turn token */
    bool thinking;         /* reasoning before each reply */
    uint32_t max_new;      /* tokens per reply, 0: the room left */
    char *system;          /* the system prompt */
    Qwen38Json tool_defs;  /* the tools, as the template wants them */
    const char *script;    /* --script: the next canned reply (tests) */
    const char *ids_path;  /* --token-ids: transcript dump after each reply (tests) */
} Agent;

static const char *arg(const Call *c, const char *key) {
    for (size_t i = 0; i < c->n; i++) if (!strcmp(c->key[i], key)) return c->value[i];
    return NULL;
}
/* Optional integer argument in [lo, hi]; false (with a message) if malformed. */
static bool arg_long(const Call *c, const char *key, long fallback, long lo, long hi, long *out, Buf *err) {
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
static bool arg_bool(const Call *c, const char *key) {
    const char *v = arg(c, key);
    return v && (!strcmp(v, "true") || !strcmp(v, "1") || !strcmp(v, "yes"));
}

/* ------------------------------------------------------------ workspace paths
 * Every file tool resolves its path with realpath(), which follows symlinks,
 * and refuses anything outside the workspace root: "../x", an absolute path
 * elsewhere, or a link inside the workspace pointing out of it. */

static bool inside(const Agent *a, const char *path) {
    size_t n = strlen(a->root);
    return !strncmp(path, a->root, n) && (path[n] == 0 || path[n] == '/');
}
static const char *rel(const Agent *a, const char *path) {   /* for messages */
    size_t n = strlen(a->root);
    if (!strncmp(path, a->root, n) && !path[n]) return ".";
    if (!strncmp(path, a->root, n) && path[n] == '/') return path+n+1;
    return path;
}
/* path (relative to the workspace, or absolute) -> out. may_be_new: the file
 * may not exist yet, but its directory must, inside the workspace. */
static bool resolve(const Agent *a, const char *path, bool may_be_new, char out[PATH_MAX], Buf *err) {
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
static void page(Agent *a, Buf *out) {
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

/* Ask before a side effect: y = this once, a = always (rest of the session),
 * anything else = no. Without a terminal the answer is no (use --yes). */
static bool approve(Agent *a, const char *what) {
    if (a->yes) return true;
    if (!a->ask) return false;
    fprintf(stdout, "\n\x1b[1m%s\x1b[0m\nAllow? [y]es / [n]o / [a]lways: ", what); fflush(stdout);
    char line[64];
    if (!fgets(line, sizeof line, stdin)) { clearerr(stdin); return false; }
    if (line[0] == 'a' || line[0] == 'A') { a->yes = true; return true; }
    return line[0] == 'y' || line[0] == 'Y' || line[0] == 's' || line[0] == 'S';
}
static const char declined[] = "Tool error: the user declined this action. Ask them how to proceed.\n";

/* ------------------------------------------------------------ file tools */

#define FILE_LIMIT (32u<<20)   /* read/edit refuse larger files */

/* Whole text file into out; refuses directories, huge and binary files. */
static bool load_text(const Agent *a, const char *path, Buf *out, Buf *err) {
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
static bool write_atomic(const Agent *a, const char *path, const char *data, size_t n, Buf *err) {
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
static void tool_list(Agent *a, const Call *c, Buf *out) {
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
static void tool_read(Agent *a, const Call *c, Buf *out) {
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
static void tool_more(Agent *a, const Call *c, Buf *out) {
    (void)c;
    if (!a->more.n) { buf_str(out, "Nothing more: the last output was complete.\n"); return; }
    buf_add(out, a->more.p, a->more.n); buf_free(&a->more);
}

/* search pattern [path] [regex] [ignore_case] : matching lines, "file:line: text".
 * Skips hidden directories (.git), build outputs and binary or huge files. */
typedef struct { const Agent *a; regex_t re; Buf *out; size_t matches, files; } Search;
#define SEARCH_MAX 300
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
static void tool_search(Agent *a, const Call *c, Buf *out) {
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
static void tool_write(Agent *a, const Call *c, Buf *out) {
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
static long count_lines(const char *s, size_t n) {
    long lines = 1;
    for (size_t i = 0; i < n; i++) lines += s[i] == '\n';
    return lines;
}

/* edit path old new : replace the one exact occurrence of `old`. */
static void tool_edit(Agent *a, const Call *c, Buf *out) {
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
static const char *skip_space(const char *p) { while (isspace((unsigned char)*p)) p++; return p; }
static const char *param_end(const char *p) {
    for (; (p = strstr(p, "</parameter>")); p++) {
        const char *next = skip_space(p+12);
        if (!strncmp(next, "<parameter=", 11) || !strncmp(next, "</function>", 11)) return p;
    }
    return NULL;
}
static void call_free(Call *c) {
    free(c->name);
    for (size_t i = 0; i < c->n; i++) { free(c->key[i]); free(c->value[i]); }
    *c = (Call){0};
}
/* Up to max calls in text; returns how many. Incomplete calls are skipped. */
static size_t parse_calls(const char *text, Call *calls, size_t max) {
    size_t n = 0;
    for (const char *p = text; n < max && (p = strstr(p, "<tool_call>")); ) {
        const char *q = skip_space(p+11), *gt;
        p += 11;
        if (strncmp(q, "<function=", 10) || !(gt = strchr(q+10, '>')) || gt == q+10) continue;
        Call c = {.name = dup_range(q+10, gt)};
        for (q = skip_space(gt+1); !strncmp(q, "<parameter=", 11) && c.n < MAX_PARAMS; ) {
            const char *key = q+11, *kend = strchr(key, '>'), *v, *vend;
            if (!kend || !(vend = param_end(kend))) break;
            v = kend+1; if (*v == '\n') v++;
            const char *cut = vend > v && vend[-1] == '\n' ? vend-1 : vend;
            c.key[c.n] = dup_range(key, kend); c.value[c.n] = dup_range(v, cut < v ? v : cut); c.n++;
            q = skip_space(vend+12);
        }
        if (strncmp(q, "</function>", 11)) { call_free(&c); continue; }
        calls[n++] = c; p = q+11;
    }
    return n;
}

/* ------------------------------------------------------------ tool table */

/* Each tool with the description and parameters the model reads (they go
 * into the system prompt once, as JSON schema, through the template). */
typedef struct { const char *name, *type, *description; bool required; } Param;
typedef struct {
    const char *name; void (*run)(Agent *, const Call *, Buf *);
    const char *description; Param params[4];
} Tool;
static const Tool tools[] = {
    {"list", tool_list, "List one directory of the workspace: subdirectories end with '/', files show their size.",
     {{"path", "string", "Directory, relative to the workspace root. Default: the root.", false}}},
    {"read", tool_read, "Read a text file with line numbers. Long output is cut: call more to continue.",
     {{"path", "string", "File, relative to the workspace root.", true},
      {"start", "integer", "First line to show, 1-based. Default 1.", false},
      {"end", "integer", "Last line to show, inclusive. Default: the end of the file.", false}}},
    {"more", tool_more, "Continue the last output that was cut (read, search, bash, visit_page, ...).", {{0}}},
    {"search", tool_search, "Search text in the workspace files, recursively (skips hidden directories, build "
     "outputs and binary files). Prints file:line: text.",
     {{"pattern", "string", "The text to find, or a POSIX extended regex when regex is true.", true},
      {"path", "string", "File or directory to search. Default: the root.", false},
      {"regex", "boolean", "Treat pattern as a regex. Default false.", false},
      {"ignore_case", "boolean", "Case-insensitive match. Default false.", false}}},
    {"write", tool_write, "Create a file or replace its whole content, atomically. The directory must exist. "
     "For changes to an existing file prefer edit.",
     {{"path", "string", "File, relative to the workspace root.", true},
      {"content", "string", "The complete new content of the file.", true}}},
    {"edit", tool_edit, "Replace one exact occurrence of old text with new text in a file. old must match "
     "exactly once, whitespace included: read the file first.",
     {{"path", "string", "File, relative to the workspace root.", true},
      {"old", "string", "The exact text to replace (unique in the file).", true},
      {"new", "string", "The replacement text.", true}}},
};

/* One call -> its result text, cleaned and paged. */
static void run_tool(Agent *a, const Call *c, Buf *out) {
    for (size_t i = 0; i < sizeof tools/sizeof *tools; i++) {
        if (strcmp(c->name, tools[i].name)) continue;
        tools[i].run(a, c, out);
        if (!out->n) buf_str(out, "(no output)\n");
        page(a, out);
        return;
    }
    buf_fmt(out, "Tool error: unknown tool %s. Available:", c->name);
    for (size_t i = 0; i < sizeof tools/sizeof *tools; i++) buf_fmt(out, " %s", tools[i].name);
    buf_str(out, "\n"); page(a, out);
}

/* ------------------------------------------------------------ main */

/* ------------------------------------------------------------ JSON for the template
 * qwen38_chat_render takes borrowed JSON trees, not text. The tool
 * definitions are built once and live as long as the process. */

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
static Qwen38Json tool_definitions(void) {
    size_t n = sizeof tools/sizeof *tools;
    Qwen38Json *list = calloc(n, sizeof *list);
    if (!list) die("pf-agent: out of memory");
    for (size_t i = 0; i < n; i++) {
        Qwen38Json props = jobject(4), *required = calloc(4, sizeof *required);
        size_t nreq = 0;
        if (!required) die("pf-agent: out of memory");
        for (const Param *p = tools[i].params; p < tools[i].params+4 && p->name; p++) {
            Qwen38Json prop = jobject(2);
            jset(&prop, "type", jstr(p->type)); jset(&prop, "description", jstr(p->description));
            jset(&props, p->name, prop);
            if (p->required) required[nreq++] = jstr(p->name);
        }
        Qwen38Json schema = jobject(3), fn = jobject(3);
        jset(&schema, "type", jstr("object")); jset(&schema, "properties", props);
        jset(&schema, "required", (Qwen38Json){.kind = Q38_JSON_ARRAY, .as.array = {required, nreq}});
        jset(&fn, "name", jstr(tools[i].name)); jset(&fn, "description", jstr(tools[i].description));
        jset(&fn, "parameters", schema);
        list[i] = jobject(2); jset(&list[i], "type", jstr("function")); jset(&list[i], "function", fn);
    }
    return (Qwen38Json){.kind = Q38_JSON_ARRAY, .as.array = {list, n}};
}

/* ------------------------------------------------------------ the conversation */

#define MAX_CALLS 16       /* tool calls run from one reply */
#define MAX_ROUNDS 64      /* tool rounds in one user turn before we stop and ask */

static char *system_prompt(const Agent *a) {
    char date[32]; time_t now = time(NULL); struct tm tm;
    strftime(date, sizeof date, "%Y-%m-%d", localtime_r(&now, &tm));
    Buf b = {0};
    buf_fmt(&b, "You are pf-agent, a coding agent running locally on the user's Mac (macOS) through the "
        "picoforge engine. You work in the workspace %s; tool paths are relative to it. Today is %s.\n\n"
        "Work in small, verified steps: look before you change (list, search, read), make focused edits, "
        "and check them with the project's own commands through bash. Read a file before editing it: edit "
        "needs the exact current text. bash, write and edit need the user's approval; if they decline, ask "
        "how to proceed. Never invent file contents, command output or web pages: use the tools. When the "
        "task is done, reply with a short summary of what changed and how you verified it.", a->root, date);
    char path[PATH_MAX]; Buf text = {0}, err = {0};
    snprintf(path, sizeof path, "%s/AGENTS.md", a->root);
    if (!access(path, R_OK) && load_text(a, path, &text, &err) && text.n <= 16384) {
        clean_text(&text);
        buf_str(&b, "\n\n# Project instructions (AGENTS.md)\n\n"); buf_add(&b, text.p, text.n);
    }
    buf_free(&text); buf_free(&err);
    return b.p;
}

typedef struct { const char *role, *content; } Msg;

static uint8_t *render(const Agent *a, const Qwen38Json *list, size_t n, bool with_tools, bool generation, size_t *len) {
    Qwen38Json messages = {.kind = Q38_JSON_ARRAY, .as.array = {list, n}};
    /* Effort "medium" adds no reasoning instruction to the system block (the
     * template's default, xhigh, asks for long reasoning that eats the context). */
    Qwen38ChatOptions o = {.enable_thinking = a->thinking ? Q38_TRUE : Q38_FALSE,
        .reasoning_effort = Q38_EFFORT_MEDIUM, .add_generation_prompt = generation};
    Qwen38Error e;
    if (qwen38_chat_render(a->run.tok, &messages, with_tools ? &a->tool_defs : NULL, o, NULL, 0, len, &e)) {
        fprintf(stderr, "pf-agent: %s\n", e.detail); return NULL;
    }
    uint8_t *out = malloc(*len+1);
    if (!out) die("pf-agent: out of memory");
    q38_check(qwen38_chat_render(a->run.tok, &messages, with_tools ? &a->tool_defs : NULL, o, out, *len, len, &e), &e);
    return out;
}

/* Render msgs as the next turn and encode it after the transcript, at
 * ids[n..); returns the number of new ids (0: error, already reported). The
 * first turn carries the system prompt and the tool definitions. A later turn
 * renders [placeholder user, msgs...] and drops the placeholder's bytes: the
 * template refuses a list with no user query, and tool results are not one. */
static size_t append_turn(Agent *a, const Msg *msgs, size_t n) {
    Qwen38Json list[MAX_CALLS+1], values[MAX_CALLS+1][2]; Qwen38JsonMember fields[MAX_CALLS+1][2];
    const bool first = !a->n;
    for (size_t i = 0; i <= n; i++) {
        Msg m = i ? msgs[i-1] : first ? (Msg){"system", a->system} : (Msg){"user", "."};
        values[i][0] = jstr(m.role); values[i][1] = jstr(m.content);
        fields[i][0] = (Qwen38JsonMember){{(const uint8_t *)"role", 4}, &values[i][0]};
        fields[i][1] = (Qwen38JsonMember){{(const uint8_t *)"content", 7}, &values[i][1]};
        list[i] = (Qwen38Json){.kind = Q38_JSON_OBJECT, .as.object = {fields[i], 2}};
    }
    size_t len, skip = 0, count = 0;
    uint8_t *text = render(a, list, n+1, first, true, &len);
    if (!text) return 0;
    if (!first) {
        uint8_t *prefix = render(a, list, 1, false, false, &skip);
        if (!prefix || skip > len || memcmp(prefix, text, skip)) die("pf-agent: placeholder turn is not a prefix");
        free(prefix);
    }
    /* Close the previous reply: it ends with <|im_end|> unless it was cut. */
    Buf b = {0};
    if (!first) buf_str(&b, a->ids[a->n-1] == a->im_end ? "\n" : "<|im_end|>\n");
    buf_add(&b, text+skip, len-skip); free(text);
    Qwen38Error e; uint32_t cap = a->run.capacity;
    Qwen38Status rc = qwen38_encode(a->run.tok, (Qwen38Bytes){(const uint8_t *)b.p, b.n}, a->ids+a->n, cap-a->n, &count, &e);
    buf_free(&b);
    if (rc && rc != Q38_CAPACITY) { fprintf(stderr, "pf-agent: cannot encode the turn: %s\n", e.detail); return 0; }
    if (rc || count >= cap-a->n) {
        fprintf(stderr, "pf-agent: the context is full (%zu of %u tokens): /reset, or restart with a larger -c\n", a->n, cap);
        return 0;
    }
    return count;
}

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
    if (!v->hidden) q38_print_text(&v->out, (const uint8_t *)v->pend.p, v->pend.n);
    q38_print_end(&v->out); fputc('\n', stdout); buf_free(&v->pend);
}

/* The test oracle (--token-ids): transcript ids, its text, and the graph position. */
static void dump_ids(const Agent *a) {
    if (!a->ids_path) return;
    FILE *f = fopen(a->ids_path, "w");
    if (!f) die("pf-agent: cannot open %s", a->ids_path);
    size_t len = 0; Qwen38Error e; Buf json = {0};
    q38_check(qwen38_decode_utf8(a->run.tok, a->ids, a->n, false, NULL, 0, &len, &e), &e);
    char *text = malloc(len+1);
    if (!text) die("pf-agent: out of memory");
    q38_check(qwen38_decode_utf8(a->run.tok, a->ids, a->n, false, (uint8_t *)text, len, &len, &e), &e);
    text[len] = 0; json_escape(&json, text); free(text);
    fputs("{\"ids\":[", f);
    for (size_t i = 0; i < a->n; i++) fprintf(f, "%s%d", i ? "," : "", a->ids[i]);
    fprintf(f, "],\"kv\":%llu,\"text\":\"%s\"}\n", (unsigned long long)qwen38_gpu_forward_info(a->run.graph).next_pos, json.p);
    buf_free(&json);
    if (fclose(f)) die("pf-agent: writing %s failed", a->ids_path);
}

/* --script (tests): the next canned reply, up to a line "=====", encoded and
 * appended as if the model had generated it. NULL when the script is used up. */
static char *scripted_reply(Agent *a, size_t total) {
    if (!a->script || !*a->script) return NULL;
    const char *end = strstr(a->script, "\n=====\n");
    size_t len = end ? (size_t)(end-a->script) : strlen(a->script);
    const char *start = a->script;
    a->script += len + (end ? 7 : 0);
    if (len && start[len-1] == '\n') len--;      /* the file's last newline is not part of the reply */
    Buf b = {0}; buf_add(&b, start, len);
    char *reply = strdup(b.p);
    buf_str(&b, "<|im_end|>");
    size_t count = 0; Qwen38Error e;
    q38_check(qwen38_encode(a->run.tok, (Qwen38Bytes){(const uint8_t *)b.p, b.n}, a->ids+total, a->run.capacity-total, &count, &e), &e);
    buf_free(&b);
    a->n = total+count;
    printf("%s\n", reply);
    return reply;
}

/* The model's reply to the `added` ids just appended. The reply joins the
 * transcript; returns its text (special tokens dropped), or NULL when the user
 * stopped it before any tool could run. */
static char *reply(Agent *a, size_t added) {
    size_t total = a->n+added;
    char *canned = scripted_reply(a, total);
    if (canned) { dump_ids(a); return canned; }
    View v = {.out = {.color = isatty(STDOUT_FILENO)}}; q38_printer_shade(&v.out, a->thinking);
    Q38Turn t; bool replayed;
    size_t prefilled = q38_chat_continue(&a->run, a->ids, a->n, total, a->max_new, view_text, &v, a->generated, &t, &replayed);
    view_end(&v);
    bool cancelled = !strcmp(t.reason, "cancelled");
    memcpy(a->ids+total, a->generated, t.got*sizeof *a->ids);
    if (t.got || !cancelled) a->n = total+t.got;   /* stopped before the first token: drop the turn (as the REPL) */
    double prefill = t.prefill_end-t.prefill_start, decode = t.end-t.decode_start;
    fprintf(stderr, "\x1b[90m[prefill %zu tokens %.1f t/s, reply %zu tokens %.1f t/s, context %zu/%u%s%s]\x1b[0m\n",
        prefilled, prefill > 0 ? (double)prefilled/prefill : 0.0, t.got, decode > 0 && t.got > 1 ? (double)(t.got-1)/decode : 0.0,
        a->n, a->run.capacity, replayed ? ", replayed" : "", cancelled ? ", interrupted" : "");
    dump_ids(a);
    if (cancelled) return NULL;
    if (!strcmp(t.reason, "length")) fprintf(stderr, "pf-agent: the reply hit the token limit\n");
    size_t len = 0; Qwen38Error e;
    q38_check(qwen38_decode_utf8(a->run.tok, a->generated, t.got, true, NULL, 0, &len, &e), &e);
    char *text = malloc(len+1);
    if (!text) die("pf-agent: out of memory");
    q38_check(qwen38_decode_utf8(a->run.tok, a->generated, t.got, true, (uint8_t *)text, len, &len, &e), &e);
    text[len] = 0;
    return text;
}

/* One line per call: "-> name key=value ...", long values summarized. */
static void show_call(const Call *c) {
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

/* The text after the reasoning: only there do tool calls count. */
static const char *answer_part(const char *reply) {
    const char *p = reply, *close;
    while ((close = strstr(p, "</think>"))) p = close+8;
    return p;
}

/* One user message, then reply -> tools -> results until the model answers
 * without a tool call, the user interrupts, or MAX_ROUNDS is reached. */
static void agent_turn(Agent *a, const char *text) {
    Msg user = {"user", text};
    size_t added = append_turn(a, &user, 1);
    for (int round = 0; added && round < MAX_ROUNDS; round++) {
        char *answer = reply(a, added);
        if (!answer) return;
        Call calls[MAX_CALLS];
        size_t n = parse_calls(answer_part(answer), calls, MAX_CALLS);
        free(answer);
        if (!n || q38_stopped) { for (size_t i = 0; i < n; i++) call_free(&calls[i]); return; }
        Buf outs[MAX_CALLS] = {{0}}; Msg results[MAX_CALLS];
        for (size_t i = 0; i < n; i++) {
            show_call(&calls[i]);
            if (q38_stopped) buf_str(&outs[i], "Tool error: interrupted by the user before it ran.\n");
            else run_tool(a, &calls[i], &outs[i]);
            show_result(&outs[i]);
            results[i] = (Msg){"tool", outs[i].p};
            call_free(&calls[i]);
        }
        added = append_turn(a, results, n);
        for (size_t i = 0; i < n; i++) buf_free(&outs[i]);
        if (q38_stopped) { added = 0; fprintf(stderr, "pf-agent: interrupted; the results are kept for your next message\n"); }
    }
    if (added) fprintf(stderr, "pf-agent: %d tool rounds without an answer, stopping here\n", MAX_ROUNDS);
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
        else if (!strcmp(cmd, "/auto")) { a->yes = true; puts("Every bash/write/edit is approved for this session."); }
        else if (!strcmp(cmd, "/think") || !strcmp(cmd, "/nothink")) {
            a->thinking = !strcmp(cmd, "/think"); printf("Thinking %s.\n", a->thinking ? "on" : "off");
        }
        else if (!strcmp(cmd, "/reset")) {
            Qwen38Error e; q38_check(qwen38_gpu_forward_reset(a->run.graph, &e), &e);
            a->n = 0; a->run.last_count = 0; buf_free(&a->more); dump_ids(a); puts("New conversation.");
        }
        else if (*cmd == '/' && !strchr(cmd, ' ') && !strchr(cmd, '\n')) fprintf(stderr, "pf-agent: unknown command %s (try /help)\n", cmd);
        else agent_turn(a, cmd);
        fflush(stdout);
    }
    buf_free(&msg);
    if (interactive()) fputc('\n', stdout);
}

static void usage(void) {
    fputs("usage: pf-agent [-m MODEL] [-c CTX] [-n N] [-w DIR] [--yes] [--no-think] [--no-mtp] [--power N]\n"
          "  -m MODEL    a .gguf path or a short name, as for picoforge\n"
          "  -c CTX      context in tokens (default 32768)\n"
          "  -n N        tokens per reply (default: the room left)\n"
          "  -w DIR      the workspace (default: the current directory)\n"
          "  --yes       approve every bash/write/edit\n"
          "  --no-think  reply without reasoning\n"
          "  --power N   GPU duty cycle in percent, as for picoforge\n"
          "       pf-agent --run [--yes] < TEXT    run the tool calls in TEXT, no model\n"
          "       pf-agent --parse < TEXT          print the tool calls in TEXT as JSON lines\n", stderr);
}

static char *read_stdin(void) {
    Buf text = {0}; char chunk[65536]; size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, stdin)) > 0) buf_add(&text, chunk, got);
    buf_add(&text, "", 0);
    return text.p;
}

/* --run: execute the calls written in stdin, in order, as the model would
 * have them run (tests, debugging). Approvals: --yes, else declined. */
static int run_stdin(Agent *a) {
    char *text = read_stdin(); Call calls[64];
    size_t n = parse_calls(text, calls, 64);
    for (size_t i = 0; i < n; i++) {
        Buf out = {0}; run_tool(a, &calls[i], &out);
        printf("== %s\n%s%s", calls[i].name, out.p, out.n && out.p[out.n-1] == '\n' ? "" : "\n");
        buf_free(&out); call_free(&calls[i]);
    }
    free(text);
    return 0;
}

/* --parse: the calls in stdin as JSON lines {"name":...,"args":{...}}. */
static int parse_stdin(void) {
    char *text = read_stdin(); Buf json = {0}; Call calls[64];
    size_t n = parse_calls(text, calls, 64);
    for (size_t i = 0; i < n; i++) {
        buf_str(&json, "{\"name\":\""); json_escape(&json, calls[i].name); buf_str(&json, "\",\"args\":{");
        for (size_t k = 0; k < calls[i].n; k++) {
            buf_str(&json, k ? ",\"" : "\""); json_escape(&json, calls[i].key[k]);
            buf_str(&json, "\":\""); json_escape(&json, calls[i].value[k]); buf_str(&json, "\"");
        }
        buf_str(&json, "}}\n"); call_free(&calls[i]);
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
    Agent a = {.thinking = true};
    if (argc > 1 && (!strcmp(argv[1], "--parse") || !strcmp(argv[1], "--run"))) {
        if (!realpath(".", a.root)) die("pf-agent: cannot resolve the current directory");
        if (!strcmp(argv[1], "--parse")) return parse_stdin();
        a.yes = argc > 2 && !strcmp(argv[2], "--yes");
        int rc = run_stdin(&a); buf_free(&a.more); return rc;
    }
    const char *model = NULL, *workspace = NULL; char *script = NULL, *ids_path = NULL;
    uint32_t ctx = 32768, power = 100; bool mtp = true;
    for (int i = 1; i < argc; i++) {
        const char *o = argv[i];
        if (!strcmp(o, "--yes")) a.yes = true;
        else if (!strcmp(o, "--no-think")) a.thinking = false;
        else if (!strcmp(o, "--no-mtp")) mtp = false;
        else if (!strcmp(o, "-h") || !strcmp(o, "--help")) { usage(); return 0; }
        else if (i+1 == argc) { usage(); return 2; }
        else if (!strcmp(o, "-m")) model = argv[++i];
        else if (!strcmp(o, "-c")) ctx = number(argv[++i], 64, 32768, "-c");
        else if (!strcmp(o, "-n")) a.max_new = number(argv[++i], 0, 1u << 20, "-n");
        else if (!strcmp(o, "-w")) workspace = argv[++i];
        else if (!strcmp(o, "--power")) power = number(argv[++i], 1, 100, "--power");
        else if (!strcmp(o, "--script")) { size_t len; script = slurp(argv[++i], &len); }
        else if (!strcmp(o, "--token-ids")) ids_path = absolute(argv[++i]);
        else { usage(); return 2; }
    }
    char lib[PATH_MAX]; metallib_path(lib);
    char *chosen = model_select(model, argv[0]), *path = absolute(chosen); free(chosen);
    if (workspace && chdir(workspace)) die("pf-agent: cannot enter %s: %s", workspace, strerror(errno));
    if (!realpath(".", a.root)) die("pf-agent: cannot resolve the workspace");
    a.ask = isatty(STDIN_FILENO); a.script = script; a.ids_path = ids_path;
    struct sigaction action = {0}, old; action.sa_handler = interrupt; action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &old)) die("pf-agent: cannot install the Ctrl+C handler");

    Q38Run *r = &a.run; Qwen38Error e; size_t one = 0; int32_t probe[2];
    r->chunk = 1024; r->mtp = mtp; r->power = (int)power;
    q38_policy_init(&r->policy, getenv("PICOFORGE_MTP_DRAFT"), getenv("PICOFORGE_MTP_COST"));
    q38_sampler_init(&r->sampler, 0.6f, 20, 0.95f, 0.0f, (uint64_t)time(NULL));   /* as the chat */
    q38_run_open(r, path);
    if (ctx > r->c->max_positions) die("pf-agent: -c %u exceeds the model's %u positions", ctx, r->c->max_positions);
    if (qwen38_encode(r->tok, (Qwen38Bytes){(const uint8_t *)"<|im_end|>", 10}, probe, 2, &one, &e) || one != 1)
        die("pf-agent: the model needs <|im_end|> as a single token");
    a.im_end = probe[0];
    q38_run_gpu(r, path, lib);
    q38_run_graph(r, ctx);
    a.ids = malloc((size_t)ctx*sizeof *a.ids); a.generated = malloc((size_t)ctx*sizeof *a.generated);
    if (!a.ids || !a.generated) die("pf-agent: out of memory");
    a.system = system_prompt(&a); a.tool_defs = tool_definitions();
    fprintf(stderr, "pf-agent: %s, context %u tokens, workspace %s\n", path, ctx, a.root);

    conversation(&a);

    sigaction(SIGINT, &old, NULL);
    q38_run_close(r); free(path); free(a.ids); free(a.generated); free(a.system);
    free(script); free(ids_path); buf_free(&a.more);
    return 0;
}
