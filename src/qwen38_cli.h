#ifndef PICOFORGE_QWEN38_CLI_H
#define PICOFORGE_QWEN38_CLI_H
#include <stdbool.h>
int qwen38_cli(int argc,char **argv);
/* Short form: -m/-p/-c/-n/--mtp/--thinking first, or no argument with $PICOFORGE_MODEL. */
bool qwen38_short_form(int argc,char **argv);
int qwen38_short(int argc,char **argv);
/* picoforge bench [short-form options]: median/p10/p90 of repeated runs. */
int qwen38_bench(int argc,char **argv);
#endif
