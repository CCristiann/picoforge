#include "gguf.h"
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    Gguf g;
    gguf_open(argv[1], &g);
    gguf_inspect(&g);
    gguf_close(&g);
    return 0;
}
