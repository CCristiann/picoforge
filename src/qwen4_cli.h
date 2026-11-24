#ifndef PICOFORGE_QWEN4_CLI_H
#define PICOFORGE_QWEN4_CLI_H
#include <stdbool.h>
int qwen4_cli(int argc,char **argv);
/* Short form: -m MODEL, -p, -c, -n and every long-form option (pf with no argument: the chat). */
int qwen4_short(int argc,char **argv);
/* pf bench / pf-bench [short-form options]: median/p10/p90 of repeated runs. */
int qwen4_bench(int argc,char **argv);
#endif
