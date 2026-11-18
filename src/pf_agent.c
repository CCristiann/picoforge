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
#include "qwen38_run.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <regex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
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

typedef struct { const char *name; void (*run)(Agent *, const Call *, Buf *); } Tool;
static const Tool tools[] = {
    {"list", tool_list}, {"read", tool_read}, {"more", tool_more}, {"search", tool_search},
    {"write", tool_write}, {"edit", tool_edit},
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

static void usage(void) {
    fputs("usage: pf-agent --run [--yes] < TEXT    run the tool calls in TEXT, no model\n"
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

int main(int argc, char **argv) {
    Agent a = {0};
    if (!realpath(".", a.root)) die("pf-agent: cannot resolve the current directory");
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--yes")) a.yes = true;
    if (argc > 1 && !strcmp(argv[1], "--parse")) return parse_stdin();
    if (argc > 1 && !strcmp(argv[1], "--run")) { int rc = run_stdin(&a); buf_free(&a.more); return rc; }
    usage();
    return 2;
}
