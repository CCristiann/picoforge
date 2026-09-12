/* main.c — picoforge entry point.
 *
 * Usage: ./picoforge [model_dir]
 */
#include "picoforge.h"

#include <stdio.h>

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "models/Qwen3-0.6B";

    Qwen3Config cfg;
    config_load(model_dir, &cfg);
    config_print(&cfg);
    return 0;
}
