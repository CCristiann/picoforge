/* pf_util.c — see pf_util.h. */
#include "pf_util.h"
#include "common.h"
#include "qwen4_run.h"     /* qw4_now, qw4_stopped */
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;


void buf_add(Buf *b, const void *s, size_t n) {
    if (b->n+n+1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n+n+1) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) die("out of memory");
        b->p = p; b->cap = cap;
    }
    if (n) memcpy(b->p+b->n, s, n);
    b->n += n; b->p[b->n] = 0;
}
void buf_str(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }
void buf_fmt(Buf *b, const char *fmt, ...) {
    char small[512]; va_list ap;
    va_start(ap, fmt); int len = vsnprintf(small, sizeof small, fmt, ap); va_end(ap);
    if (len < 0) return;
    if ((size_t)len < sizeof small) { buf_add(b, small, (size_t)len); return; }
    char *big = malloc((size_t)len+1);
    if (!big) die("out of memory");
    va_start(ap, fmt); vsnprintf(big, (size_t)len+1, fmt, ap); va_end(ap);
    buf_add(b, big, (size_t)len); free(big);
}
void buf_free(Buf *b) { free(b->p); *b = (Buf){0}; }
/* Length of the valid UTF-8 sequence at s, 0 if invalid (overlong forms,
 * surrogates and code points past U+10FFFF are invalid). */
size_t utf8_len(const unsigned char *s, size_t n) {
    unsigned c = s[0], lo = 0x80, hi = 0xBF; size_t len;
    if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return 0;
    if (n < len || s[1] < lo || s[1] > hi) return 0;
    for (size_t k = 2; k < len; k++) if ((s[k] & 0xC0) != 0x80) return 0;
    return len;
}
/* Tool output goes back into the prompt, so it must be valid UTF-8 (the
 * template renderer refuses anything else) and must never spell a special
 * token: the tokenizer turns the text "<|im_end|>" into the real end-of-turn
 * token, so a file or web page containing it could close the tool message and
 * fake a new turn (prompt injection at the token level). Invalid bytes and
 * control characters (terminal colors too) become '?', and "<|" becomes "< |". */
void clean_text(Buf *b) {
    Buf out = {0}; const unsigned char *s = (const unsigned char *)b->p;
    for (size_t i = 0; i < b->n;) {
        unsigned c = s[i];
        if (c == '<' && i+1 < b->n && s[i+1] == '|') { buf_str(&out, "< |"); i += 2; }
        else if (c >= 0x80) {
            size_t len = utf8_len(s+i, b->n-i);
            if (len) buf_add(&out, s+i, len); else buf_add(&out, "?", 1);
            i += len ? len : 1;
        } else {
            if ((c >= 0x20 && c != 0x7F) || c == '\n' || c == '\t') buf_add(&out, s+i, 1);
            else if (c != '\r') buf_add(&out, "?", 1);
            i++;
        }
    }
    buf_add(&out, "", 0);
    buf_free(b); *b = out;
}
/* JSON string body. */
void json_escape(Buf *b, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { buf_add(b, "\\", 1); buf_add(b, s, 1); }
        else if (c == '\n') buf_str(b, "\\n");
        else if (c == '\t') buf_str(b, "\\t");
        else if (c < 0x20) buf_fmt(b, "\\u%04x", c);
        else buf_add(b, s, 1);
    }
}
/* Leading and trailing spaces/newlines off, in place. */
char *trim_line(char *s) {
    while (isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n-1])) s[--n] = 0;
    return s;
}
uint64_t fnv1a(const void *data, size_t n) {
    uint64_t h = 1469598103934665603ull; const uint8_t *p = data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
long count_lines(const char *s, size_t n) {
    long lines = 1;
    for (size_t i = 0; i < n; i++) lines += s[i] == '\n';
    return lines;
}
void pause_ms(long ms) { struct timespec t = {ms/1000, (ms%1000)*1000000}; nanosleep(&t, NULL); }
void utf8_put(Buf *b, unsigned long cp) {
    char u[4]; size_t n;
    if (!cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = '?';
    if (cp < 0x80) { u[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { u[0] = (char)(0xC0 | cp >> 6); u[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
    else if (cp < 0x10000) { u[0] = (char)(0xE0 | cp >> 12); u[1] = (char)(0x80 | (cp >> 6 & 0x3F)); u[2] = (char)(0x80 | (cp & 0x3F)); n = 3; }
    else { u[0] = (char)(0xF0 | cp >> 18); u[1] = (char)(0x80 | (cp >> 12 & 0x3F)); u[2] = (char)(0x80 | (cp >> 6 & 0x3F)); u[3] = (char)(0x80 | (cp & 0x3F)); n = 4; }
    buf_add(b, u, n);
}
/* Children start with posix_spawn, not fork: the agent maps tens of GB of
 * weights and a fork would have to duplicate their page tables. Each child gets
 * its own process group, so Ctrl+C reaches the agent and not the command, and
 * killpg ends the whole tree. Only stdin (/dev/null), stdout and stderr are
 * inherited: every other descriptor (the model file, the terminal) is closed in
 * the child (POSIX_SPAWN_CLOEXEC_DEFAULT, Darwin). */
pid_t pf_spawn(char *const argv[], int out_fd, int err_fd) {
    posix_spawn_file_actions_t files; posix_spawnattr_t attr; pid_t pid;
    sigset_t none, defaults;
    sigemptyset(&none); sigemptyset(&defaults); sigaddset(&defaults, SIGINT); sigaddset(&defaults, SIGPIPE);
    posix_spawn_file_actions_init(&files);
    posix_spawn_file_actions_addopen(&files, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&files, out_fd, STDOUT_FILENO);
    if (err_fd >= 0) posix_spawn_file_actions_adddup2(&files, err_fd, STDERR_FILENO);
    else posix_spawn_file_actions_addopen(&files, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawnattr_init(&attr);
    posix_spawnattr_setsigmask(&attr, &none); posix_spawnattr_setsigdefault(&attr, &defaults);
    posix_spawnattr_setpgroup(&attr, 0);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK |
                                    POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT);
    int rc = posix_spawn(&pid, argv[0], &files, &attr, argv, environ);
    posix_spawn_file_actions_destroy(&files); posix_spawnattr_destroy(&attr);
    return rc ? -1 : pid;
}
/* Run argv; up to `limit` bytes of its stdout into out. Exit code, or -1 if it
 * could not start, ran past `seconds` or was interrupted (then it is killed). */
int pf_capture(char *const argv[], size_t limit, double seconds, Buf *out) {
    int fds[2];
    if (pipe(fds)) return -1;
    pid_t pid = pf_spawn(argv, fds[1], -1);
    close(fds[1]);
    if (pid < 0) { close(fds[0]); return -1; }
    bool killed = false; char chunk[65536];
    for (double end = qw4_now()+seconds;;) {
        int left = (int)((end-qw4_now())*1000);
        if (left <= 0 || qw4_stopped) { killpg(pid, SIGKILL); killed = true; break; }
        struct pollfd p = {fds[0], POLLIN, 0};
        if (poll(&p, 1, left < 100 ? left : 100) <= 0) continue;
        ssize_t got = read(fds[0], chunk, sizeof chunk);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        if (out->n < limit) buf_add(out, chunk, (size_t)got);
    }
    close(fds[0]);
    int status = 0; waitpid(pid, &status, 0);
    return killed || !WIFEXITED(status) ? -1 : WEXITSTATUS(status);
}
