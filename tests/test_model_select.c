/* test_model_select.c — which model each -m name picks, on directories of empty .gguf files.
 * Every choice runs in a child process: a failed choice exits 2 with the list of models. */
#include "model_select.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
static char root[] = "/tmp/pf-model-select-XXXXXX";

/* A fresh models directory holding models/NAME/NAME.gguf for each name. */
static void models(const char *const *names) {
    static int k;
    char dir[256]; snprintf(dir, sizeof dir, "%s/%d", root, k++);
    mkdir(dir, 0700);
    for (; *names; names++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s", dir, *names); mkdir(path, 0700);
        snprintf(path, sizeof path, "%s/%s/%s.gguf", dir, *names, *names);
        FILE *f = fopen(path, "w"); if (f) fclose(f);
    }
    setenv("PICOFORGE_MODELS", dir, 1);
}
/* The stem `query` picks, or "" when the choice fails. */
static void expect(const char *query, const char *want) {
    int fd[2]; char got[256] = "";
    if (pipe(fd)) { perror("pipe"); exit(1); }
    pid_t pid = fork();
    if (!pid) {
        close(fd[0]); freopen("/dev/null", "w", stderr);
        char *path = model_select(query, "pf"), *base = strrchr(path, '/');
        dprintf(fd[1], "%.*s", (int)strlen(base + 1) - 5, base + 1);
        exit(0);
    }
    close(fd[1]);
    ssize_t n = read(fd[0], got, sizeof got - 1); got[n > 0 ? n : 0] = 0; close(fd[0]);
    int status; waitpid(pid, &status, 0);
    int ok = *want ? WIFEXITED(status) && !WEXITSTATUS(status) && !strcmp(got, want)
                   : WIFEXITED(status) && WEXITSTATUS(status) == 2;
    if (!ok) { failures++; printf("FAIL -m %s: got \"%s\" (status %d), want \"%s\"\n", query, got, status, want); }
}

int main(void) {
    if (!mkdtemp(root)) { perror("mkdtemp"); return 1; }
    const char *q2 = "Qwen3.8-Flash-Next-Q2", *q4 = "Qwen3.8-Flash-Next-Q4", *c5 = "Qwen3.8-Flash-Next-Q2-Code5";

    models((const char *[]){q2, c5, NULL});
    expect("q2", q2);                    /* the handle, though the word Q2 is in both names */
    expect("Q2", q2);
    expect("q2-code5", c5);
    expect("code5", c5);                 /* whole word */
    expect("code", c5);                  /* any unique part */
    expect("Qwen3.8-Flash-Next-Q2", q2); /* full name */
    expect("flash", "");                 /* in both: refused */
    expect("q4", "");                    /* in neither */

    models((const char *[]){q2, q4, "Qwen3.8-Flash-Next-Q2-CodeHot5-Q4", NULL});
    expect("q4", q4);                    /* the handle beats the word Q4 of the mixed name */
    expect("hot5", "Qwen3.8-Flash-Next-Q2-CodeHot5-Q4");

    models((const char *[]){q2, c5, "Qwen3.8-Flash-Next-Q2-Code10", NULL});
    expect("code", "");
    expect("code10", "Qwen3.8-Flash-Next-Q2-Code10");

    models((const char *[]){q2, NULL});
    expect("q2", q2);
    expect(NULL, q2);                    /* the only model */

    models((const char *[]){q2, "Qwen4-27B-Q4", NULL});   /* no shared word: handles are full names */
    expect("q4", "Qwen4-27B-Q4");
    expect("q2", q2);

    char cmd[64]; snprintf(cmd, sizeof cmd, "rm -rf %s", root); system(cmd);
    if (failures) { printf("%d model_select failures\n", failures); return 1; }
    printf("model_select: ok\n");
    return 0;
}
