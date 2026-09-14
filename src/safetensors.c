/* safetensors.c — map the checkpoint, parse its header, hand out tensors.
 *
 * Layout on disk:
 *
 *     [0, 8)        u64 little-endian: length N of the JSON header
 *     [8, 8+N)      the JSON header: name -> {dtype, shape, data_offsets}
 *     [8+N, EOF)    raw tensor bytes; data_offsets index into THIS region
 *
 * We mmap the whole file. Three reasons, in increasing order of importance:
 * fread would copy 1.4 GB before computing a single logit; mmap is lazy and
 * the pages stay in the OS page cache, so the second run starts warm; and on
 * unified memory the mapped pages are already GPU-addressable, so Phase 2 can
 * hand them to Metal with newBufferWithBytesNoCopy instead of keeping a second
 * copy of the model. On a 30B MoE that is the difference between fitting in
 * 64 GB and not fitting.
 */
#include "picoforge.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------- entries */
static DType parse_dtype(const char *s) {
    if (strcmp(s, "BF16") == 0) return DT_BF16;
    if (strcmp(s, "F32")  == 0) return DT_F32;
    if (strcmp(s, "F16")  == 0) return DT_F16;
    if (strcmp(s, "I8")   == 0) return DT_I8;       /* Phase 3: 8-bit codes     */
    if (strcmp(s, "U8")   == 0) return DT_U8;       /* Phase 3: packed nibbles  */
    die("safetensors: unsupported dtype \"%s\"", s);
    return DT_F32;                                     /* unreachable */
}

static size_t dtype_size(DType d) {
    return (d == DT_F32) ? 4u : (d == DT_I8 || d == DT_U8) ? 1u : 2u;
}

/* Walk the top-level object once. If `out` is NULL we only count, which is
 * how we learn how much to allocate before the second, filling pass. */
static int walk_header(const char *json, Tensor *out,
                       const unsigned char *data, size_t data_len) {
    const char *p = json_expect(json, '{');
    int n = 0;

    while (*p) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (*p == '}') break;

        char name[80];
        p = json_string(p, name, sizeof name);
        p = json_expect(p, ':');

        /* Not a tensor: safetensors stores free-form strings here. */
        if (strcmp(name, "__metadata__") == 0) { p = json_skip(p); continue; }

        if (!out) { p = json_skip(p); n++; continue; }

        Tensor *t = &out[n++];
        memcpy(t->name, name, sizeof t->name);

        long offsets[2] = {0, 0};
        int  n_off = 0;
        bool seen_dtype = false, seen_shape = false;

        p = json_expect(p, '{');
        for (;;) {
            while (isspace((unsigned char)*p) || *p == ',') p++;
            if (*p == '}') { p++; break; }

            char key[32];
            p = json_string(p, key, sizeof key);
            p = json_expect(p, ':');

            if (strcmp(key, "dtype") == 0) {
                char dt[16];
                p = json_string(p, dt, sizeof dt);
                t->dtype = parse_dtype(dt);
                seen_dtype = true;
            } else if (strcmp(key, "shape") == 0) {
                p = json_ints(p, t->shape, 4, &t->ndim);
                seen_shape = true;
            } else if (strcmp(key, "data_offsets") == 0) {
                p = json_ints(p, offsets, 2, &n_off);
            } else {
                p = json_skip(p);
            }
        }

        if (!seen_dtype || !seen_shape || n_off != 2)
            die("safetensors: entry \"%s\" is missing dtype/shape/data_offsets", t->name);

        t->nelem = 1;
        for (int i = 0; i < t->ndim; i++) {
            if (t->shape[i] <= 0) die("safetensors: \"%s\" has a non-positive dim", t->name);
            t->nelem *= (size_t)t->shape[i];
        }

        /* Trust nothing the file says about its own size. A bad offset here
         * would be a read straight out of the mapping — a segfault if we are
         * lucky, silent garbage weights if we are not. */
        size_t want = t->nelem * dtype_size(t->dtype);
        if (offsets[0] < 0 || offsets[1] < offsets[0])
            die("safetensors: \"%s\" has a reversed offset range [%ld,%ld]",
                t->name, offsets[0], offsets[1]);
        if ((size_t)(offsets[1] - offsets[0]) != want)
            die("safetensors: \"%s\" claims %ld bytes but its shape needs %zu",
                t->name, offsets[1] - offsets[0], want);
        /* Separate from the check above on purpose. The same rejection with
         * one message would report a shape mismatch for a file that is
         * simply truncated, and send the reader looking in the wrong place. */
        if ((size_t)offsets[1] > data_len)
            die("safetensors: \"%s\" ends at %ld but the file holds only %zu "
                "bytes of tensor data — truncated download?",
                t->name, offsets[1], data_len);

        t->data = data + offsets[0];
    }
    return n;
}

/* ---------------------------------------------------------------- open */

/* Map one safetensors file and append its tensors to st->tensors. */
static void open_file(const char *path, SafeTensors *st) {
    if (st->n_maps == ST_MAX_SHARDS) die("more than %d shards", ST_MAX_SHARDS);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat sb;
    if (fstat(fd, &sb) != 0) die("cannot stat %s", path);
    if (sb.st_size < 8) die("%s is too small to be a safetensors file", path);
    const size_t map_len = (size_t)sb.st_size;

    /* PROT_READ | MAP_PRIVATE: our view, copy-on-write, never written back. */
    void *map = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("mmap failed on %s", path);
    close(fd);                       /* the mapping keeps the file alive */
    st->maps[st->n_maps] = map;
    st->map_lens[st->n_maps++] = map_len;

    const unsigned char *base = map;

    /* u64 little-endian, assembled byte by byte rather than memcpy'd into a
     * uint64_t: the format is defined as little-endian, our CPU merely
     * happens to agree. Spelling it out costs nothing and states the rule. */
    uint64_t hdr_len = 0;
    for (int i = 7; i >= 0; i--) hdr_len = (hdr_len << 8) | base[i];
    if (hdr_len == 0 || hdr_len > map_len - 8)
        die("safetensors: header length %llu is impossible in %s",
            (unsigned long long)hdr_len, path);

    /* The header inside the mapping is NOT NUL-terminated: the first tensor's
     * bytes start immediately after its closing brace. Every str* call would
     * walk straight into the weights and read them as text. Copying it out
     * makes the rest of this file safe by construction. */
    char *json = malloc((size_t)hdr_len + 1);
    if (!json) die("out of memory for the %llu-byte header",
                   (unsigned long long)hdr_len);
    memcpy(json, base + 8, (size_t)hdr_len);
    json[hdr_len] = '\0';

    const unsigned char *data = base + 8 + hdr_len;
    size_t data_len = map_len - 8 - (size_t)hdr_len;

    const int n = walk_header(json, NULL, data, data_len);
    Tensor *grown = realloc(st->tensors, (size_t)(st->n_tensors + n) * sizeof *grown);
    if (!grown) die("out of memory for %d tensor records", st->n_tensors + n);
    st->tensors = grown;
    memset(st->tensors + st->n_tensors, 0, (size_t)n * sizeof *grown);
    (void)walk_header(json, st->tensors + st->n_tensors, data, data_len);
    st->n_tensors += n;
    free(json);
}

static int cmp_tensor_name(const void *a, const void *b) {
    return strcmp(((const Tensor *)a)->name, ((const Tensor *)b)->name);
}

void st_open(const char *model_dir, SafeTensors *st) {
    memset(st, 0, sizeof *st);
    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", model_dir);
    FILE *probe = fopen(path, "rb");

    int indexed = 0;
    if (!probe) {
        snprintf(path, sizeof path, "%s/model.safetensors", model_dir);
        open_file(path, st);
    } else {
        /* Sharded: the index's weight_map says which file holds each tensor.
         * Every file it names is opened once, in order of first mention, and
         * the index's count is held to what the files actually contain. */
        fclose(probe);
        char *index = slurp(path, NULL);
        const char *p = strstr(index, "\"weight_map\"");
        if (!p) die("%s has no weight_map", path);
        p = json_expect(p + strlen("\"weight_map\""), ':');
        p = json_expect(p, '{');
        char files[ST_MAX_SHARDS][128];
        int n_files = 0;
        for (;;) {
            while (isspace((unsigned char)*p) || *p == ',') p++;
            if (*p == '}') break;
            char name[128], file[128];
            p = json_string(p, name, sizeof name);
            p = json_expect(p, ':');
            while (isspace((unsigned char)*p)) p++;
            p = json_string(p, file, sizeof file);
            indexed++;
            bool seen = false;
            for (int i = 0; i < n_files; i++) seen |= strcmp(files[i], file) == 0;
            if (seen) continue;
            if (n_files == ST_MAX_SHARDS) die("%s names more than %d shards", path, ST_MAX_SHARDS);
            memcpy(files[n_files++], file, sizeof file);
        }
        free(index);
        for (int i = 0; i < n_files; i++) {
            snprintf(path, sizeof path, "%s/%s", model_dir, files[i]);
            open_file(path, st);
        }
        if (st->n_tensors != indexed)
            die("the index lists %d tensors, its %d shards hold %d", indexed, n_files, st->n_tensors);
    }
    st->map = st->maps[0];
    st->map_len = st->map_lens[0];

    /* Sorted once, so a lookup is a binary search: Qwen3-30B-A3B has ~19k
     * tensors, and binding all of them by linear search would be 1.8e8
     * string compares. Sorting also exposes a name two shards both claim. */
    qsort(st->tensors, (size_t)st->n_tensors, sizeof *st->tensors, cmp_tensor_name);
    for (int i = 1; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i - 1].name, st->tensors[i].name) == 0)
            die("checkpoint holds \"%s\" twice", st->tensors[i].name);
}

void st_close(SafeTensors *st) {
    for (int i = 0; i < st->n_maps; i++) munmap(st->maps[i], st->map_lens[i]);
    free(st->tensors);
    memset(st, 0, sizeof *st);
}

const Tensor *st_try(const SafeTensors *st, const char *name) {
    int lo = 0, hi = st->n_tensors - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const int c = strcmp(st->tensors[mid].name, name);
        if (c == 0) return &st->tensors[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return NULL;
}

const Tensor *st_find(const SafeTensors *st, const char *name) {
    const Tensor *t = st_try(st, name);
    if (!t) die("checkpoint has no tensor named \"%s\"", name);
    return t;
}

void tensor_to_f32(const Tensor *t, float *out) {
    switch (t->dtype) {
    case DT_BF16: {
        const uint16_t *src = t->data;
        for (size_t i = 0; i < t->nelem; i++) out[i] = bf16_to_f32(src[i]);
        break;
    }
    case DT_F32:
        memcpy(out, t->data, t->nelem * sizeof(float));
        break;
    default:
        die("tensor_to_f32: \"%s\" has an unhandled dtype", t->name);
    }
}

void st_summary(const SafeTensors *st) {
    size_t on_disk = 0, bytes = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        on_disk += st->tensors[i].nelem;
        bytes   += st->tensors[i].nelem * dtype_size(st->tensors[i].dtype);
    }

    /* Qwen3-0.6B ships lm_head.weight as a byte-for-byte copy of the embedding
     * the config tells us to reuse; counting it would inflate "parameters" by
     * 155.6 M. Qwen3-30B-A3B ships a head that is its own matrix. Which one a
     * file holds is checked on the bytes, not assumed from the name. */
    size_t unique = on_disk;
    const Tensor *head = st_try(st, "lm_head.weight"), *emb = st_try(st, "model.embed_tokens.weight");
    const bool tied_copy = head && emb && head->dtype == emb->dtype && head->nelem == emb->nelem &&
                           memcmp(head->data, emb->data, head->nelem * dtype_size(head->dtype)) == 0;
    if (tied_copy) unique -= head->nelem;

    printf("\n=== checkpoint ===\n");
    double mapped = 0;
    for (int i = 0; i < st->n_maps; i++) mapped += (double)st->map_lens[i];
    printf("mapped                : %.2f GB%s\n", mapped / 1e9,
           st->n_maps > 1 ? " (sharded)" : "");
    if (st->n_maps > 1) printf("shards                : %d\n", st->n_maps);
    printf("tensors               : %d\n", st->n_tensors);
    printf("tensor bytes          : %.2f GB\n", (double)bytes / 1e9);
    printf("parameters on disk    : %.1f M\n", (double)on_disk / 1e6);
    if (tied_copy)
        printf("unique parameters     : %.1f M (lm_head.weight is a tied copy)\n",
               (double)unique / 1e6);
    printf("as fp32 in RAM        : %.2f GB (x2 vs bf16 — Phase 3's whole point)\n",
           (double)on_disk * 4.0 / 1e9);
}
