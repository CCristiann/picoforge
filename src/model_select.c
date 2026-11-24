/* model_select.c — see model_select.h. Plain C: a directory scan, a name match, a terminal prompt. */
#define _DARWIN_C_SOURCE   /* strcasestr, realpath */
#include "model_select.h"
#include "common.h"

#include <dirent.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { char *name, *path; const char *handle; double gib; } Model;
typedef struct { Model *at; size_t n, cap; char dir[PATH_MAX]; } Models;

/* $PICOFORGE_MODELS, else models/ beside the real binary (so it works from any working directory). */
static void models_dir(const char *argv0, char *out) {
    const char *env = getenv("PICOFORGE_MODELS");
    if (env && *env) { snprintf(out, PATH_MAX, "%s", env); return; }
    char exe[PATH_MAX], real[PATH_MAX]; uint32_t size = sizeof exe;
    const char *bin = _NSGetExecutablePath(exe, &size) == 0 && realpath(exe, real) ? real : argv0;
    const char *slash = strrchr(bin, '/');
    snprintf(out, PATH_MAX, "%.*smodels", slash ? (int)(slash - bin + 1) : 0, bin);
}

static void add(Models *m, const char *path, const char *file) {
    size_t len = strlen(file);
    struct stat st;
    if (len <= 5 || strcmp(file + len - 5, ".gguf") || stat(path, &st) || !S_ISREG(st.st_mode)) return;
    if (m->n == m->cap) {
        m->cap = m->cap ? 2 * m->cap : 8;
        if (!(m->at = realloc(m->at, m->cap * sizeof *m->at))) die("model list allocation");
    }
    Model *x = &m->at[m->n++];
    x->path = strdup(path); x->name = strndup(file, len - 5); x->gib = (double)st.st_size / (1u << 30);
    if (!x->path || !x->name) die("model list allocation");
}
/* The .gguf files of `dir` (depth 1 = also one level of subdirectories). */
static void scan(Models *m, const char *dir, int depth) {
    DIR *d = opendir(dir);
    if (!d) return;
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char path[PATH_MAX]; struct stat st;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        if (depth && !stat(path, &st) && S_ISDIR(st.st_mode)) scan(m, path, depth - 1);
        else add(m, path, e->d_name);
    }
    closedir(d);
}
static int by_name(const void *a, const void *b) { return strcasecmp(((const Model *)a)->name, ((const Model *)b)->name); }
/* The handle of a model is its name without the words every model shares, keeping at least one
 * word of each: Qwen3.8-Flash-Next-Q2 and Qwen3.8-Flash-Next-Q2-Code5 are "q2" and "q2-code5". A
 * lone model keeps its last word. Words are the parts between '-'. */
static void handles(Models *m) {
    size_t cut = 0;
    if (m->n == 1) { const char *dash = strrchr(m->at[0].name, '-'); cut = dash ? (size_t)(dash - m->at[0].name + 1) : 0; }
    for (size_t p = 1; m->n > 1; p++) {
        int shared = 1;
        for (size_t i = 0; i < m->n && shared; i++)
            shared = strlen(m->at[i].name) > p && !strncasecmp(m->at[i].name, m->at[0].name, p);
        if (!shared) break;
        if (m->at[0].name[p - 1] == '-') cut = p;
    }
    for (size_t i = 0; i < m->n; i++) m->at[i].handle = m->at[i].name + cut;
}
static Models find(const char *argv0) {
    Models m = {0};
    models_dir(argv0, m.dir);
    scan(&m, m.dir, 1);
    if (m.n) { qsort(m.at, m.n, sizeof *m.at, by_name); handles(&m); }
    return m;
}
static void print(FILE *f, const Models *m) {
    int width = 0;
    for (size_t i = 0; i < m->n; i++) if ((int)strlen(m->at[i].handle) > width) width = (int)strlen(m->at[i].handle);
    for (size_t i = 0; i < m->n; i++) {
        char handle[PATH_MAX]; snprintf(handle, sizeof handle, "%s", m->at[i].handle);
        for (char *c = handle; *c; c++) if (*c >= 'A' && *c <= 'Z') *c = (char)(*c - 'A' + 'a');
        fprintf(f, "  %zu) -m %-*s %7.1f GiB  %s\n", i + 1, width, handle, m->at[i].gib, m->at[i].path);
    }
}
static char *pick(Models *m, size_t i) {
    char *path = m->at[i].path; m->at[i].path = NULL;
    for (size_t j = 0; j < m->n; j++) { free(m->at[j].name); free(m->at[j].path); }
    free(m->at);
    return path;
}
static _Noreturn void fail(const Models *m, const char *why) {
    fprintf(stderr, "picoforge: %s\n", why);
    if (m->n) { fprintf(stderr, "models in %s:\n", m->dir); print(stderr, m); }
    else fprintf(stderr, "no .gguf model in %s (set PICOFORGE_MODELS or pass -m FILE.gguf)\n", m->dir);
    exit(2);
}

/* `query` is one or more whole words of `name`: "q4" is in ...-Q2-Code5-Q4 but "4" and "q" are not. */
static int has_words(const char *name, const char *query) {
    size_t n = strlen(query);
    for (const char *at = name; (at = strcasestr(at, query)); at++)
        if ((at == name || at[-1] == '-') && (at[n] == 0 || at[n] == '-')) return 1;
    return 0;
}
/* Four rules, from strict to loose; the first that matches anything decides, and it must match one
 * model only: the full name, the handle, whole words of the name, any part of the name. So "q2"
 * is the Q2 even beside Q2-Code5, and "code" picks Q2-Code5 while it is the only coding model. */
static char *by_short_name(Models *m, const char *query) {
    for (int rule = 0; rule < 4; rule++) {
        size_t found = 0, hits = 0;
        for (size_t i = 0; i < m->n; i++) {
            const char *name = m->at[i].name;
            int match = rule == 0 ? !strcasecmp(name, query) : rule == 1 ? !strcasecmp(m->at[i].handle, query)
                      : rule == 2 ? has_words(name, query) : strcasestr(name, query) != NULL;
            if (match) { found = i; hits++; }
        }
        if (hits == 1) {
            if (rule >= 2) fprintf(stderr, "picoforge: -m %s is %s\n", query, m->at[found].name);
            return pick(m, found);
        }
        if (hits) {
            char why[PATH_MAX + 64];
            snprintf(why, sizeof why, "\"%s\" matches %zu models: use more of the name", query, hits);
            fail(m, why);
        }
    }
    char why[PATH_MAX + 64];
    snprintf(why, sizeof why, "no model matches \"%s\"", query);
    fail(m, why);
}
/* Several models and nobody said which: ask, but only when a person is at the terminal. */
static char *ask(Models *m) {
    if (!isatty(STDIN_FILENO) || !isatty(STDERR_FILENO)) fail(m, "several models: choose one with -m NAME");
    fprintf(stderr, "Which model?\n"); print(stderr, m);
    for (;;) {
        fprintf(stderr, "number or name: ");
        char line[256];
        if (!fgets(line, sizeof line, stdin)) exit(2);
        line[strcspn(line, "\r\n")] = 0;
        char *end; long k = strtol(line, &end, 10);
        if (*line && !*end && k >= 1 && (size_t)k <= m->n) return pick(m, (size_t)k - 1);
        if (*line && *end) return by_short_name(m, line);
    }
}

char *model_select(const char *arg, const char *argv0) {
    if (!arg || !*arg) arg = getenv("PICOFORGE_MODEL");
    struct stat st;
    if (arg && *arg && !stat(arg, &st) && S_ISREG(st.st_mode)) {
        char *path = strdup(arg);
        if (!path) die("model path allocation");
        return path;
    }
    Models m = find(argv0);
    if (arg && *arg) {
        if (strchr(arg, '/')) { char why[PATH_MAX + 32]; snprintf(why, sizeof why, "no such model file: %s", arg); fail(&m, why); }
        return by_short_name(&m, arg);
    }
    if (m.n == 1) return pick(&m, 0);
    if (!m.n) fail(&m, "no model given");
    return ask(&m);
}

int model_list(const char *argv0) {
    Models m = find(argv0);
    if (!m.n) fail(&m, "no models");
    printf("models in %s (-m takes the name shown, any part of the file name that is unique, or a path):\n", m.dir);
    print(stdout, &m);
    free(pick(&m, 0));
    return 0;
}
