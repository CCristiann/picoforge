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

typedef struct { char *name, *path; double gib; } Model;
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
static Models find(const char *argv0) {
    Models m = {0};
    models_dir(argv0, m.dir);
    scan(&m, m.dir, 1);
    if (m.n) qsort(m.at, m.n, sizeof *m.at, by_name);
    return m;
}
/* The shortest handle worth typing: the last '-' part of the name ("q4") when it picks this model alone. */
static void short_name(const Models *m, size_t i, char *out, size_t cap) {
    const char *dash = strrchr(m->at[i].name, '-'), *tail = dash ? dash + 1 : m->at[i].name;
    size_t hits = 0;
    for (size_t j = 0; j < m->n; j++) hits += strcasestr(m->at[j].name, tail) != NULL;
    snprintf(out, cap, "%s", hits == 1 ? tail : m->at[i].name);
    for (char *c = out; hits == 1 && *c; c++) if (*c >= 'A' && *c <= 'Z') *c = (char)(*c - 'A' + 'a');
}
static void print(FILE *f, const Models *m) {
    for (size_t i = 0; i < m->n; i++) {
        char handle[64]; short_name(m, i, handle, sizeof handle);
        fprintf(f, "  %zu) -m %-8s %-26s %7.1f GiB  %s\n", i + 1, handle, m->at[i].name, m->at[i].gib, m->at[i].path);
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

/* An exact stem wins over substrings, so "Qwen3.8-Flash-Next-Q2" never collides with a longer name. */
static char *by_short_name(Models *m, const char *name) {
    size_t found = SIZE_MAX, hits = 0;
    for (size_t i = 0; i < m->n; i++) if (!strcasecmp(m->at[i].name, name)) return pick(m, i);
    for (size_t i = 0; i < m->n; i++) if (strcasestr(m->at[i].name, name)) { found = i; hits++; }
    if (hits == 1) return pick(m, found);
    char why[PATH_MAX + 64];
    snprintf(why, sizeof why, hits ? "\"%s\" matches more than one model" : "no model matches \"%s\"", name);
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
    printf("models in %s (-m takes the short name, any unique part of the name, or a path):\n", m.dir);
    print(stdout, &m);
    free(pick(&m, 0));
    return 0;
}
