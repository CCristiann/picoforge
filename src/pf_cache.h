/* pf_cache.h — saved graph states on disk, found again by token prefix.
 *
 * The graph cannot be rewound (the recurrent layers keep one state, not one
 * per position), so a request that does not extend the tokens in the graph
 * starts over. pf-server keeps checkpoints instead: an entry is a token
 * prefix plus the whole graph state after it (qwen4_gpu_state_write: KV
 * caches, recurrent states, ...). A new prompt loads the longest entry that
 * is a prefix of it and prefills only the rest.
 *
 * Entries live in one directory (default ~/.picoforge/kv-cache), one file
 * each, named by a hash of their tokens; the least recently used go when the
 * directory grows past its budget. They hold prompt text and model state:
 * private, and disposable (stop the server before deleting them). */
#ifndef PICOFORGE_PF_CACHE_H
#define PICOFORGE_PF_CACHE_H
#include "pf_session.h"

typedef struct {
    bool on;
    char dir[PATH_MAX];
    uint64_t budget;            /* bytes */
    size_t min_tokens;          /* shorter prefixes are cheaper to prefill than to keep */
} PfCache;

/* dir NULL: $PICOFORGE_KV_CACHE or ~/.picoforge/kv-cache. False (cache off) if it cannot be created. */
bool pf_cache_open(PfCache *, const char *dir, uint64_t budget_bytes, size_t min_tokens);
/* The graph's state, which holds s->ids[0..n) (n: the graph position), as an entry. */
bool pf_cache_save(const PfCache *, PfSession *, size_t n);
bool pf_cache_has(const PfCache *, const PfSession *, const int32_t *ids, size_t n);
/* The longest entry that is a prefix of ids[0..limit) and longer than
 * `better_than` tokens, loaded into the graph; its length. 0: nothing loaded
 * (the graph is untouched, or reset if the entry was damaged: then it is deleted). */
size_t pf_cache_load(const PfCache *, PfSession *, const int32_t *ids, size_t limit, size_t better_than);
#endif
