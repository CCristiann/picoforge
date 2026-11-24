/* pf_util.h — small pieces shared by pf-agent and, later, pf-server: growable
 * text, UTF-8 and prompt hygiene, JSON string escapes, child processes. */
#ifndef PICOFORGE_PF_UTIL_H
#define PICOFORGE_PF_UTIL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

typedef struct { char *p; size_t n, cap; } Buf;   /* NUL-terminated once anything was added */
void buf_add(Buf *, const void *, size_t);
void buf_str(Buf *, const char *);
void buf_fmt(Buf *, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void buf_free(Buf *);

size_t utf8_len(const unsigned char *, size_t);   /* valid sequence length at s, 0 if invalid */
void utf8_put(Buf *, unsigned long codepoint);   /* invalid code points become '?' */
/* Text that goes back into a prompt: valid UTF-8, no control characters, and
 * never the spelling of a special token ("<|" becomes "< |"). */
void clean_text(Buf *);
void json_escape(Buf *, const char *);           /* a JSON string body */
char *trim_line(char *);                         /* spaces off both ends, in place */
long count_lines(const char *, size_t);
uint64_t fnv1a(const void *, size_t);           /* a 64-bit FNV-1a hash */

void pause_ms(long ms);
/* argv[0] (absolute) in its own process group, stdin /dev/null, stdout to
 * out_fd, stderr to err_fd (-1: /dev/null); nothing else of ours inherited. */
pid_t pf_spawn(char *const argv[], int out_fd, int err_fd);
/* Run argv, up to `limit` bytes of stdout into out; exit code, -1 if it could
 * not start, ran past `seconds` or qw4_stopped was raised (then it is killed). */
int pf_capture(char *const argv[], size_t limit, double seconds, Buf *out);
#endif
