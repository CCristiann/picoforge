/* pf_cache.c — see pf_cache.h. */
#include "pf_cache.h"
#include "common.h"
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
    char magic[8];                    /* "PFKVC01" */
    uint32_t capacity, mtp;
    uint64_t n, model_id, last_count;
} Header;

/* The model and the hash of the token ids name the file: one prefix of one model is one file,
 * and models sharing the directory never replace each other's entries. */
static void entry_path(const PfCache *c, const PfSession *s, const int32_t *ids, size_t n, char out[PATH_MAX]) {
    snprintf(out, PATH_MAX, "%s/%016llx-%016llx-%zu.pfc", c->dir, (unsigned long long)s->model_id,
             (unsigned long long)fnv1a(ids, n*sizeof *ids), n);
}

bool pf_cache_open(PfCache *c, const char *dir, uint64_t budget, size_t min_tokens) {
    *c = (PfCache){.budget = budget, .min_tokens = min_tokens};
    const char *env = getenv("PICOFORGE_KV_CACHE"), *home = getenv("HOME");
    if (dir) snprintf(c->dir, sizeof c->dir, "%s", dir);
    else if (env && *env) snprintf(c->dir, sizeof c->dir, "%s", env);
    else if (home && *home) {
        snprintf(c->dir, sizeof c->dir, "%s/.picoforge", home); mkdir(c->dir, 0700);
        snprintf(c->dir, sizeof c->dir, "%s/.picoforge/kv-cache", home);
    } else return false;
    c->on = !mkdir(c->dir, 0700) || errno == EEXIST;
    return c->on;
}

/* The header and ids of an entry that fits this graph, or NULL. */
static FILE *entry_open(const PfSession *s, const char *path, Header *h, int32_t **ids) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    *ids = NULL;
    if (fread(h, sizeof *h, 1, f) != 1 || memcmp(h->magic, "PFKVC01", 8) || h->capacity != s->run.capacity ||
        h->mtp != (uint32_t)s->run.mtp || h->model_id != s->model_id || !h->n || h->n > h->capacity) { fclose(f); return NULL; }
    *ids = malloc(h->n*sizeof **ids);
    if (!*ids) die("out of memory");
    if (fread(*ids, sizeof **ids, h->n, f) != h->n) { free(*ids); *ids = NULL; fclose(f); return NULL; }
    return f;
}

/* Oldest first (by mtime: a load touches its file) until the directory fits. */
typedef struct { char name[96]; off_t size; time_t when; } Item;
static int by_age(const void *x, const void *y) {
    time_t a = ((const Item *)x)->when, b = ((const Item *)y)->when;
    return a < b ? -1 : a > b;
}
static void evict(const PfCache *c) {
    DIR *d = opendir(c->dir); struct dirent *e; Item *all = NULL; size_t n = 0; uint64_t total = 0;
    while (d && (e = readdir(d))) {
        size_t len = strlen(e->d_name); char path[PATH_MAX]; struct stat st;
        if (len < 5 || len >= sizeof all->name || strcmp(e->d_name+len-4, ".pfc")) continue;
        snprintf(path, sizeof path, "%s/%s", c->dir, e->d_name);
        if (stat(path, &st)) continue;
        Item *grown = realloc(all, (n+1)*sizeof *all);
        if (!grown) die("out of memory");
        all = grown; snprintf(all[n].name, sizeof all[n].name, "%s", e->d_name);
        all[n].size = st.st_size; all[n].when = st.st_mtime; total += (uint64_t)st.st_size; n++;
    }
    if (d) closedir(d);
    qsort(all, n, sizeof *all, by_age);
    for (size_t i = 0; i+1 < n && total > c->budget; i++) {   /* the newest entry stays, whatever its size */
        char path[PATH_MAX]; snprintf(path, sizeof path, "%s/%s", c->dir, all[i].name);
        if (!unlink(path)) total -= (uint64_t)all[i].size;
    }
    free(all);
}

bool pf_cache_save(const PfCache *c, PfSession *s, size_t n) {
    if (!c->on || n < c->min_tokens || pf_cache_has(c, s, s->ids, n)) return false;
    char path[PATH_MAX], tmp[PATH_MAX+16]; entry_path(c, s, s->ids, n, path);
    snprintf(tmp, sizeof tmp, "%s.XXXXXX", path);
    int fd = mkstemp(tmp); FILE *f = fd < 0 ? NULL : fdopen(fd, "wb");
    if (!f) { if (fd >= 0) { close(fd); unlink(tmp); } return false; }
    setvbuf(f, NULL, _IOFBF, 8u << 20);
    Header h = {.magic = "PFKVC01", .capacity = s->run.capacity, .mtp = s->run.mtp, .n = n,
                .model_id = s->model_id, .last_count = s->run.last_count};
    Qwen4Error e = {0}; double start = qw4_now();
    bool ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(s->ids, sizeof *s->ids, n, f) == n &&
              !qwen4_gpu_state_write(s->run.graph, f, &e) && !fflush(f);
    long long bytes = ftello(f);
    ok = !fclose(f) && ok && !rename(tmp, path);   /* never half an entry under its real name */
    if (!ok) { unlink(tmp); if (s->note) s->note(s->note_ctx, "kv cache: saving failed\n"); return false; }
    char msg[256]; snprintf(msg, sizeof msg, "kv cache: saved %zu tokens (%.0f MiB in %.2f s)\n", n, (double)bytes/1048576.0, qw4_now()-start);
    if (s->note) s->note(s->note_ctx, msg);
    evict(c);
    return true;
}

bool pf_cache_has(const PfCache *c, const PfSession *s, const int32_t *ids, size_t n) {
    if (!c->on) return false;
    char path[PATH_MAX]; Header h; int32_t *saved;
    entry_path(c, s, ids, n, path);
    FILE *f = entry_open(s, path, &h, &saved);
    bool same = f && h.n == n && !memcmp(saved, ids, n*sizeof *ids);
    if (f) { fclose(f); free(saved); }
    return same;
}

size_t pf_cache_load(const PfCache *c, PfSession *s, const int32_t *ids, size_t limit, size_t better_than) {
    if (!c->on) return 0;
    DIR *d = opendir(c->dir); struct dirent *e; char best[PATH_MAX] = ""; size_t best_n = better_than;
    char model[20]; snprintf(model, sizeof model, "%016llx-", (unsigned long long)s->model_id);
    while (d && (e = readdir(d))) {
        size_t len = strlen(e->d_name), n; char path[PATH_MAX]; Header h; int32_t *saved;
        /* The name carries the model and the length: most entries are ruled out without opening them. */
        const char *dash = strrchr(e->d_name, '-');
        if (len < 5 || strcmp(e->d_name+len-4, ".pfc") || strncmp(e->d_name, model, 17) || !dash ||
            sscanf(dash+1, "%zu", &n) != 1 || n <= best_n || n > limit) continue;
        snprintf(path, sizeof path, "%s/%s", c->dir, e->d_name);
        FILE *f = entry_open(s, path, &h, &saved);
        if (f && h.n == n && !memcmp(saved, ids, n*sizeof *ids)) { snprintf(best, sizeof best, "%s", path); best_n = n; }
        if (f) { fclose(f); free(saved); }
    }
    if (d) closedir(d);
    if (!best[0]) return 0;
    Header h; int32_t *saved; Qwen4Error err = {0}; double start = qw4_now();
    FILE *f = entry_open(s, best, &h, &saved);
    if (!f) return 0;
    free(saved);
    setvbuf(f, NULL, _IOFBF, 8u << 20);
    bool ok = !qwen4_gpu_state_read(s->run.graph, f, &err);   /* on failure the graph is reset */
    fclose(f);
    char msg[256];
    if (!ok) {
        s->run.last_count = 0;
        snprintf(msg, sizeof msg, "kv cache: cannot load %s: %s\n", best, err.detail);
        if (s->note) s->note(s->note_ctx, msg);
        unlink(best);
        return 0;
    }
    s->run.last_count = (size_t)h.last_count;
    utimes(best, NULL);                                       /* recently used: evicted last */
    snprintf(msg, sizeof msg, "kv cache: loaded %zu tokens in %.2f s\n", best_n, qw4_now()-start);
    if (s->note) s->note(s->note_ctx, msg);
    return best_n;
}
