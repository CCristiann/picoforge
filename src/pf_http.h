/* pf_http.h — the smallest HTTP/1.1 server pf-server needs.
 *
 * One connection at a time (one model, one graph: requests queue in the
 * listen backlog), one request per connection ("Connection: close"), bodies
 * by Content-Length only. Streaming replies are Server-Sent Events written
 * straight to the socket; a failed write or a closed peer means the client
 * went away, and the generation stops. */
#ifndef PICOFORGE_PF_HTTP_H
#define PICOFORGE_PF_HTTP_H
#include "pf_util.h"

typedef struct {
    char method[16], path[512];
    char auth[256];                  /* "Authorization: Bearer X" or "x-api-key: X": X */
    char *body; size_t body_n;       /* owned, NUL-terminated */
} PfHttpRequest;

int pf_http_listen(const char *host, int port);           /* the socket, or -1 (reason on stderr) */
/* Read one request (headers <= 64 KiB, body <= max_body). On false the status
 * to answer is in *status (400, 408, 413), or 0 if the peer is already gone. */
bool pf_http_read(int fd, PfHttpRequest *, size_t max_body, int *status);
void pf_http_request_free(PfHttpRequest *);

extern bool pf_http_cors;                                   /* --cors: browser cross-origin headers */
bool pf_http_respond(int fd, int status, const char *content_type, const char *body, size_t n);
bool pf_http_sse_begin(int fd);
bool pf_http_send(int fd, const char *data, size_t n);     /* false: the client is gone */
bool pf_http_gone(int fd);                                  /* the peer closed its side */
#endif
