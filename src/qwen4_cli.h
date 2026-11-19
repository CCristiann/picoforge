#ifndef PICOFORGE_QWEN4_CLI_H
#define PICOFORGE_QWEN4_CLI_H
#include <stdbool.h>
int qwen4_cli(int argc,char **argv);
/* Short form: -m/-p/-c/-n/--mtp/--thinking first, or no argument with $PICOFORGE_MODEL. */
bool qwen4_short_form(int argc,char **argv);
int qwen4_short(int argc,char **argv);
/* pf bench / pf-bench [short-form options]: median/p10/p90 of repeated runs. */
int qwen4_bench(int argc,char **argv);
#endif
