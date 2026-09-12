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

/* ---------------------------------------------------------------- JSON
 * Step 1.1 looked up keys it already knew. Here we must ENUMERATE 311 keys
 * we do not know, which needs one primitive we did not have: skipping a
 * complete JSON value, whatever its type, including nesting and escapes.
 */
static const char *json_skip(const char *p) {
    while (isspace((unsigned char)*p)) p++;
    if (*p == '"') {                                   /* string */
        p++;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        if (*p != '"') die("safetensors: unterminated string in header");
        return p + 1;
    }
    if (*p == '{' || *p == '[') {                      /* object or array */
        char open = *p, close = (*p == '{') ? '}' : ']';
        int depth = 0;
        while (*p) {
            if (*p == '"') { p = json_skip(p); continue; }  /* braces inside
                                                               strings must
                                                               not count */
            if (*p == open) depth++;
            else if (*p == close && --depth == 0) return p + 1;
            p++;
        }
        die("safetensors: unbalanced '%c' in header", open);
    }
    while (*p && *p != ',' && *p != '}' && *p != ']'   /* number or literal */
           && !isspace((unsigned char)*p)) p++;
    return p;
}

static const char *json_str_into(const char *p, char *out, size_t cap) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != '"') die("safetensors: expected a string in header");
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') die("safetensors: escapes in names are not supported");
        if (i + 1 >= cap) die("safetensors: name longer than %zu chars", cap - 1);
        out[i++] = *p++;
    }
    if (*p != '"') die("safetensors: unterminated name in header");
    out[i] = '\0';
    return p + 1;
}

static const char *json_ints_into(const char *p, long *out, int cap, int *n_out) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != '[') die("safetensors: expected an array in header");
    p++;
    int n = 0;
    for (;;) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (*p == ']') { p++; break; }
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) die("safetensors: malformed integer in header array");
        if (n >= cap) die("safetensors: array longer than %d entries", cap);
        out[n++] = v;
        p = end;
    }
    *n_out = n;
    return p;
}

static const char *expect(const char *p, char c) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != c) die("safetensors: expected '%c' in header, found '%c'", c, *p);
    return p + 1;
}

/* ------------------------------------------------------------- entries */
static DType parse_dtype(const char *s) {
    if (strcmp(s, "BF16") == 0) return DT_BF16;
    if (strcmp(s, "F32")  == 0) return DT_F32;
    if (strcmp(s, "F16")  == 0) return DT_F16;
    die("safetensors: unsupported dtype \"%s\"", s);
    return DT_F32;                                     /* unreachable */
}

static size_t dtype_size(DType d) { return (d == DT_F32) ? 4u : 2u; }

/* Walk the top-level object once. If `out` is NULL we only count, which is
 * how we learn how much to allocate before the second, filling pass. */
static int walk_header(const char *json, Tensor *out,
                       const unsigned char *data, size_t data_len) {
    const char *p = expect(json, '{');
    int n = 0;

    while (*p) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (*p == '}') break;

        char name[80];
        p = json_str_into(p, name, sizeof name);
        p = expect(p, ':');

        /* Not a tensor: safetensors stores free-form strings here. */
        if (strcmp(name, "__metadata__") == 0) { p = json_skip(p); continue; }

        if (!out) { p = json_skip(p); n++; continue; }

        Tensor *t = &out[n++];
        memcpy(t->name, name, sizeof t->name);

        long offsets[2] = {0, 0};
        int  n_off = 0;
        bool seen_dtype = false, seen_shape = false;

        p = expect(p, '{');
        for (;;) {
            while (isspace((unsigned char)*p) || *p == ',') p++;
            if (*p == '}') { p++; break; }

            char key[32];
            p = json_str_into(p, key, sizeof key);
            p = expect(p, ':');

            if (strcmp(key, "dtype") == 0) {
                char dt[16];
                p = json_str_into(p, dt, sizeof dt);
                t->dtype = parse_dtype(dt);
                seen_dtype = true;
            } else if (strcmp(key, "shape") == 0) {
                p = json_ints_into(p, t->shape, 4, &t->ndim);
                seen_shape = true;
            } else if (strcmp(key, "data_offsets") == 0) {
                p = json_ints_into(p, offsets, 2, &n_off);
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
void st_open(const char *model_dir, SafeTensors *st) {
    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors", model_dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat sb;
    if (fstat(fd, &sb) != 0) die("cannot stat %s", path);
    if (sb.st_size < 8) die("%s is too small to be a safetensors file", path);
    st->map_len = (size_t)sb.st_size;

    /* PROT_READ | MAP_PRIVATE: our view, copy-on-write, never written back. */
    st->map = mmap(NULL, st->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (st->map == MAP_FAILED) die("mmap failed on %s", path);
    close(fd);                       /* the mapping keeps the file alive */

    const unsigned char *base = st->map;

    /* u64 little-endian, assembled byte by byte rather than memcpy'd into a
     * uint64_t: the format is defined as little-endian, our CPU merely
     * happens to agree. Spelling it out costs nothing and states the rule. */
    uint64_t hdr_len = 0;
    for (int i = 7; i >= 0; i--) hdr_len = (hdr_len << 8) | base[i];
    if (hdr_len == 0 || hdr_len > st->map_len - 8)
        die("safetensors: header length %llu is impossible",
            (unsigned long long)hdr_len);

    /* The header inside the mapping is NOT NUL-terminated: the first tensor's
     * bytes start immediately after its closing brace. Every str* call would
     * walk straight into the weights and read them as text. Copying 35 KB out
     * makes the rest of this file safe by construction. */
    char *json = malloc((size_t)hdr_len + 1);
    if (!json) die("out of memory for the %llu-byte header",
                   (unsigned long long)hdr_len);
    memcpy(json, base + 8, (size_t)hdr_len);
    json[hdr_len] = '\0';

    const unsigned char *data = base + 8 + hdr_len;
    size_t data_len = st->map_len - 8 - (size_t)hdr_len;

    st->n_tensors = walk_header(json, NULL, data, data_len);
    st->tensors = calloc((size_t)st->n_tensors, sizeof *st->tensors);
    if (!st->tensors) die("out of memory for %d tensor records", st->n_tensors);
    (void)walk_header(json, st->tensors, data, data_len);

    free(json);
}

void st_close(SafeTensors *st) {
    if (st->map) munmap(st->map, st->map_len);
    free(st->tensors);
    st->map = NULL;
    st->tensors = NULL;
    st->n_tensors = 0;
}

const Tensor *st_find(const SafeTensors *st, const char *name) {
    for (int i = 0; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
    die("checkpoint has no tensor named \"%s\"", name);
    return NULL;                                       /* unreachable */
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

    /* tie_word_embeddings means lm_head.weight is a byte-for-byte copy of the
     * embedding matrix that the config tells us to ignore. Counting it would
     * inflate "parameters" by 155.6 M — a quarter of this model. */
    size_t unique = on_disk;
    bool tied_copy = false;
    for (int i = 0; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i].name, "lm_head.weight") == 0) {
            unique -= st->tensors[i].nelem;
            tied_copy = true;
        }

    printf("\n=== checkpoint ===\n");
    printf("mapped                : %.2f GB\n", (double)st->map_len / 1e9);
    printf("tensors               : %d\n", st->n_tensors);
    printf("tensor bytes          : %.2f GB\n", (double)bytes / 1e9);
    printf("parameters on disk    : %.1f M\n", (double)on_disk / 1e6);
    if (tied_copy)
        printf("unique parameters     : %.1f M (lm_head.weight is a tied copy)\n",
               (double)unique / 1e6);
    printf("as fp32 in RAM        : %.2f GB (x2 vs bf16 — Phase 3's whole point)\n",
           (double)on_disk * 4.0 / 1e9);
}
