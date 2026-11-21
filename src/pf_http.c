/* pf_http.c — see pf_http.h. */
#include "pf_http.h"
#include "common.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

#define HEADER_LIMIT (64u << 10)
#define READ_TIMEOUT_MS 30000      /* a client that stops sending mid-request */

bool pf_http_cors;

int pf_http_listen(const char *host, int port) {
    struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM, .ai_flags = AI_PASSIVE}, *res;
    char service[16]; snprintf(service, sizeof service, "%d", port);
    int rc = getaddrinfo(host, service, &hints, &res);
    if (rc) { fprintf(stderr, "pf-server: %s: %s\n", host, gai_strerror(rc)); return -1; }
    int fd = -1;
    for (struct addrinfo *ai = res; ai && fd < 0; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) || listen(fd, 64)) { close(fd); fd = -1; }
    }
    freeaddrinfo(res);
    if (fd < 0) fprintf(stderr, "pf-server: cannot listen on %s:%d: %s\n", host, port, strerror(errno));
    return fd;
}

/* recv with a deadline; 0 at end of stream, -1 on error or timeout. */
static ssize_t recv_some(int fd, char *p, size_t n) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int r = poll(&pfd, 1, READ_TIMEOUT_MS);
    if (r <= 0) return -1;
    ssize_t got;
    do got = recv(fd, p, n, 0); while (got < 0 && errno == EINTR);
    return got;
}
/* The value of header `name` in the header block, copied into out. */
static bool header(const char *head, const char *name, char *out, size_t cap) {
    size_t n = strlen(name);
    for (const char *line = strstr(head, "\r\n"); line; line = strstr(line, "\r\n")) {
        line += 2;
        if (!strncasecmp(line, name, n) && line[n] == ':') {
            const char *v = line+n+1, *end = strstr(v, "\r\n");
            while (*v == ' ' || *v == '\t') v++;
            size_t len = end ? (size_t)(end-v) : strlen(v);
            if (len >= cap) len = cap-1;
            memcpy(out, v, len); out[len] = 0;
            return true;
        }
    }
    return false;
}
bool pf_http_read(int fd, PfHttpRequest *r, size_t max_body, int *status) {
    *r = (PfHttpRequest){0}; *status = 400;
    Buf b = {0}; char chunk[16384]; char *end = NULL;
    while (!end) {
        ssize_t got = recv_some(fd, chunk, sizeof chunk);
        if (got <= 0) { *status = b.n ? 408 : 0; buf_free(&b); return false; }
        buf_add(&b, chunk, (size_t)got);
        end = strstr(b.p, "\r\n\r\n");
        if (!end && b.n > HEADER_LIMIT) { *status = 431; buf_free(&b); return false; }
    }
    *end = 0;
    if (sscanf(b.p, "%15s %511s", r->method, r->path) != 2) { buf_free(&b); return false; }
    char value[256];
    if (header(b.p, "Authorization", value, sizeof value)) {
        const char *v = strncasecmp(value, "Bearer ", 7) ? value : value+7;
        snprintf(r->auth, sizeof r->auth, "%s", v);
    } else if (header(b.p, "x-api-key", value, sizeof value)) snprintf(r->auth, sizeof r->auth, "%s", value);
    if (header(b.p, "Transfer-Encoding", value, sizeof value)) { *status = 411; buf_free(&b); return false; }
    size_t want = 0;
    if (header(b.p, "Content-Length", value, sizeof value)) {
        char *e; errno = 0; unsigned long long n = strtoull(value, &e, 10);
        if (errno || *e) { buf_free(&b); return false; }
        if (n > max_body) { *status = 413; buf_free(&b); return false; }
        want = (size_t)n;
    }
    size_t head = (size_t)(end+4-b.p), have = b.n-head;
    if (have > want) have = want;
    r->body = malloc(want+1);
    if (!r->body) die("pf-server: out of memory");
    memcpy(r->body, end+4, have);
    buf_free(&b);
    while (have < want) {
        ssize_t got = recv_some(fd, r->body+have, want-have);
        if (got <= 0) { *status = 408; pf_http_request_free(r); return false; }
        have += (size_t)got;
    }
    r->body[want] = 0; r->body_n = want;
    return true;
}
void pf_http_request_free(PfHttpRequest *r) { free(r->body); r->body = NULL; r->body_n = 0; }

bool pf_http_send(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t put = send(fd, p, n, 0);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return false;
        p += put; n -= (size_t)put;
    }
    return true;
}
static const char *reason(int status) {
    switch (status) {
    case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request";
    case 401: return "Unauthorized"; case 404: return "Not Found"; case 405: return "Method Not Allowed";
    case 408: return "Request Timeout"; case 411: return "Length Required"; case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large"; case 500: return "Internal Server Error";
    case 503: return "Service Unavailable"; default: return "Error";
    }
}
static void common_headers(Buf *h) {
    buf_str(h, "Connection: close\r\nCache-Control: no-cache\r\n");
    if (pf_http_cors) buf_str(h, "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                                 "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
}
bool pf_http_respond(int fd, int status, const char *type, const char *body, size_t n) {
    Buf h = {0};
    buf_fmt(&h, "HTTP/1.1 %d %s\r\n", status, reason(status));
    common_headers(&h);
    if (type) buf_fmt(&h, "Content-Type: %s\r\n", type);
    buf_fmt(&h, "Content-Length: %zu\r\n\r\n", n);
    bool ok = pf_http_send(fd, h.p, h.n) && pf_http_send(fd, body, n);
    buf_free(&h);
    return ok;
}
bool pf_http_sse_begin(int fd) {
    Buf h = {0};
    buf_str(&h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Accel-Buffering: no\r\n");
    common_headers(&h); buf_str(&h, "\r\n");
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);   /* tokens leave as they come */
    bool ok = pf_http_send(fd, h.p, h.n);
    buf_free(&h);
    return ok;
}
/* Readable with nothing to read means the peer closed (or reset) the connection.
 * A client that keeps sending would look alive; ours never do after the body. */
bool pf_http_gone(int fd) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    if (poll(&pfd, 1, 0) <= 0) return false;
    if (pfd.revents & (POLLHUP | POLLERR)) return true;
    char c; ssize_t got = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    return got == 0 || (got < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}
