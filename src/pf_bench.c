/* pf_bench.c — pf-bench: `pf bench` as its own command (ds4-bench's place).
 * The median and p10/p90 of repeated runs, docs/BENCHMARKS.md. */
#include "qwen38_cli.h"
#include <stdlib.h>
#include "common.h"

int main(int argc, char **argv) {
    char **args = calloc((size_t)argc+2, sizeof *args);   /* pf-bench ARGS -> pf bench ARGS */
    if (!args) die("out of memory");
    args[0] = argv[0]; args[1] = "bench";
    for (int i = 1; i < argc; i++) args[i+1] = argv[i];
    int rc = qwen38_bench(argc+1, args);
    free(args);
    return rc;
}
