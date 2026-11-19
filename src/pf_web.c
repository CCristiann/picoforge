/* pf_web.c — see pf_web.h. Sections: curl (the fallback) · HTML to text ·
 * HTTP and WebSocket to Chrome · CDP · Chrome · the tools.
 *
 * curl (/usr/bin/curl, part of macOS) runs with an argument vector, never
 * through a shell, so a query or a URL cannot become a command. It reads no
 * config file (-q), speaks only http(s), also across redirects, and has a time
 * and size limit. Its search reads DuckDuckGo's HTML page: no key, no account.
 * PF_AGENT_CURL replaces curl in the tests. */
#include "pf_web.h"
#include "common.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include "qwen4_run.h"     /* qw4_now, qw4_stopped */

typedef struct { int code; char url[2048], type[256]; } Fetched;
/* curl with the common safety flags, then args. The body goes to out; the
 * status line written after it (-w) is parsed into f and cut off. */
static bool fetch(char *const args[], Buf *out, Fetched *f, Buf *err) {
    const char *curl = getenv("PF_AGENT_CURL");
    char *argv[32] = {(char *)(curl && *curl ? curl : "/usr/bin/curl"), "-q", "-sS", "-L", "--max-time", "20",
        "--max-filesize", "8000000", "--proto", "=http,https", "--proto-redir", "=http,https",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
        "-w", "\n\x01%{http_code} %{url_effective} %{content_type}"};
    size_t n = 0;
    while (argv[n]) n++;                     /* the fixed flags above; the rest of argv is zero */
    for (size_t i = 0; args[i] && n < 31; i++) argv[n++] = args[i];
    argv[n] = NULL;
    int rc = pf_capture(argv, 9u << 20, 25, out);
    char *mark = NULL;                       /* the last "\n\x01"; the body may hold NUL bytes */
    for (char *p = out->p; p && (p = memmem(p, out->n-(size_t)(p-out->p), "\n\x01", 2)); p += 2) mark = p;
    *f = (Fetched){0};
    if (mark && sscanf(mark+2, "%d %2047s %255[^\n]", &f->code, f->url, f->type) >= 2) { *mark = 0; out->n = (size_t)(mark-out->p); }
    if (!rc && mark) return true;
    const char *why = rc == -1 ? "timed out or interrupted" : rc == 6 ? "cannot resolve the host" :
        rc == 7 ? "cannot connect" : rc == 28 ? "timed out" : rc == 63 ? "larger than 8 MB" :
        rc == 35 || rc == 60 ? "TLS error" : "failed";
    buf_fmt(err, "Tool error: the request %s (curl exit %d)\n", why, rc);
    return false;
}
/* "&amp;", "&#39;", "&#x2014;"... at s; appends the text, returns the length used (0: not an entity). */
static size_t entity(const char *s, const char *end, Buf *out) {
    static const struct { const char *name; unsigned long cp; } named[] = {
        {"amp", '&'}, {"lt", '<'}, {"gt", '>'}, {"quot", '"'}, {"apos", '\''}, {"nbsp", ' '}, {"ndash", 0x2013},
        {"mdash", 0x2014}, {"hellip", 0x2026}, {"lsquo", 0x2018}, {"rsquo", 0x2019}, {"ldquo", 0x201C},
        {"rdquo", 0x201D}, {"copy", 0xA9}, {"middot", 0xB7}, {"laquo", 0xAB}, {"raquo", 0xBB}};
    const char *semi = memchr(s, ';', (size_t)(end-s) < 12 ? (size_t)(end-s) : 12);
    if (!semi) return 0;
    size_t len = (size_t)(semi-s-1);
    if (s[1] == '#') {
        char *stop; unsigned long cp = s[2] == 'x' || s[2] == 'X' ? strtoul(s+3, &stop, 16) : strtoul(s+2, &stop, 10);
        if (stop != semi) return 0;
        utf8_put(out, cp); return len+2;
    }
    for (size_t i = 0; i < sizeof named/sizeof *named; i++)
        if (strlen(named[i].name) == len && !strncmp(s+1, named[i].name, len)) { utf8_put(out, named[i].cp); return len+2; }
    return 0;
}
static const char *find_ci(const char *s, const char *end, const char *needle) {
    size_t n = strlen(needle);
    for (; s+n <= end; s++) if (!strncasecmp(s, needle, n)) return s;
    return NULL;
}
/* href relative to the page URL -> absolute (into out); false for anchors,
 * mailto:, javascript: and the like. */
static bool url_join(const char *base, const char *href, Buf *out) {
    const char *scheme_end = strstr(base, "://");
    if (!strncmp(href, "http://", 7) || !strncmp(href, "https://", 8)) { buf_str(out, href); return true; }
    if (!scheme_end || !*href || *href == '#' || strchr(href, ':')) return false;
    const char *host = scheme_end+3, *path = strchr(host, '/');
    if (!strncmp(href, "//", 2)) buf_add(out, base, (size_t)(host-2-base));
    else if (*href == '/') buf_add(out, base, path ? (size_t)(path-base) : strlen(base));
    else { const char *dir = path ? strrchr(path, '/') : NULL; if (dir) buf_add(out, base, (size_t)(dir+1-base)); else { buf_str(out, base); buf_str(out, "/"); } }
    buf_str(out, href);
    char *hash = strchr(out->p, '#');
    if (hash) { out->n = (size_t)(hash-out->p); *hash = 0; }
    return true;
}
/* HTML -> readable text: tags dropped (block tags become line breaks, list
 * items "- ", headings "#"), script/style/head skipped, entities decoded,
 * spaces collapsed outside <pre>. With a base URL the links are listed at the
 * end, absolute, so the model can follow them with visit_page. */
static void newline(Buf *t) {
    while (t->n && t->p[t->n-1] == ' ') t->p[--t->n] = 0;
    if (t->n && !(t->n >= 2 && t->p[t->n-1] == '\n' && t->p[t->n-2] == '\n')) buf_add(t, "\n", 1);
}
void pf_html_to_text(const char *s, size_t n, const char *base, Buf *t) {
    static const char *const skip[] = {"script", "style", "noscript", "svg", "template", "head", "title", NULL};
    static const char *const block[] = {"p", "div", "br", "ul", "ol", "tr", "table", "section", "article", "header",
        "footer", "nav", "main", "blockquote", "pre", "hr", "dt", "dd", "figure", "aside", "details", "summary", "form", NULL};
    const char *end = s+n, *title = find_ci(s, end, "<title");
    Buf links = {0}; size_t nlinks = 0; bool pre = false, space = false;
    if (title && (title = memchr(title, '>', (size_t)(end-title)))) {
        const char *close = find_ci(title, end, "</title");
        if (close) { buf_str(t, "# "); pf_html_to_text(title+1, (size_t)(close-title-1), NULL, t); newline(t); newline(t); }
    }
    while (s < end) {
        /* A tag starts with a letter, '/', '!' or '?'; any other '<' is text, as in a browser. */
        bool tag = *s == '<' && s+1 < end && (isalpha((unsigned char)s[1]) || (s[1] && strchr("/!?", s[1])));
        if (!tag) {
            if (!pre && isspace((unsigned char)*s)) { space = true; s++; continue; }
            if (space && t->n && t->p[t->n-1] != '\n' && t->p[t->n-1] != ' ') buf_add(t, " ", 1);
            size_t used = *s == '&' ? entity(s, end, t) : 0;
            if (used) s += used; else buf_add(t, s++, 1);
            space = false;
            continue;
        }
        if (!strncmp(s, "<!--", 4)) { const char *close = strstr(s, "-->"); s = close && close < end ? close+3 : end; continue; }
        const char *gt = memchr(s, '>', (size_t)(end-s)), *p = s+1;
        if (!gt) break;
        bool closing = *p == '/'; char name[16]; size_t k = 0;
        if (closing) p++;
        while (p < gt && isalnum((unsigned char)*p) && k < sizeof name-1) name[k++] = (char)tolower((unsigned char)*p++);
        name[k] = 0; s = gt+1;
        bool skipped = false;
        for (size_t i = 0; skip[i] && !closing; i++) if (!strcmp(name, skip[i])) {
            char close[24]; snprintf(close, sizeof close, "</%s", name);
            const char *c = find_ci(s, end, close); const char *cgt = c ? memchr(c, '>', (size_t)(end-c)) : NULL;
            s = cgt ? cgt+1 : end; skipped = true;
        }
        if (skipped) continue;
        if (!strcmp(name, "pre")) pre = !closing;
        if (!strcmp(name, "li") && !closing) { newline(t); buf_str(t, "- "); }
        else if (name[0] == 'h' && name[1] >= '1' && name[1] <= '6' && !name[2]) {
            newline(t); newline(t);
            if (!closing) { for (char l = '1'; l <= name[1]; l++) buf_add(t, "#", 1); buf_add(t, " ", 1); }
        }
        else if (!strcmp(name, "td") || !strcmp(name, "th")) { if (!closing) buf_add(t, " ", 1); }
        else for (size_t i = 0; block[i]; i++) if (!strcmp(name, block[i])) { newline(t); break; }
        if (base && !closing && !strcmp(name, "a") && nlinks < 60) {
            const char *h = find_ci(p, gt, "href=");
            if (h && (h[5] == '"' || h[5] == '\'')) {
                const char *q = memchr(h+6, h[5], (size_t)(gt-h-6));
                Buf raw = {0}, href = {0}, url = {0};
                if (q) buf_add(&raw, h+6, (size_t)(q-h-6));
                for (const char *r = raw.p; r && *r;) {                       /* &amp; in attributes */
                    size_t used = *r == '&' ? entity(r, raw.p+raw.n, &href) : 0;
                    if (used) r += used; else buf_add(&href, r++, 1);
                }
                if (href.n && url_join(base, href.p, &url) && !strstr(links.p ? links.p : "", url.p)) {
                    buf_fmt(&links, "- %s\n", url.p); nlinks++;
                }
                buf_free(&raw); buf_free(&href); buf_free(&url);
            }
        }
        space = space || closing;
    }
    newline(t);
    if (links.n) { newline(t); buf_str(t, "Links:\n"); buf_add(t, links.p, links.n); }
    buf_free(&links);
}
static char *ddg_url(const char *href) {
    Buf raw = {0}, url = {0};
    for (const char *r = href; *r;) { size_t used = *r == '&' ? entity(r, r+strlen(r), &raw) : 0; if (used) r += used; else buf_add(&raw, r++, 1); }
    const char *u = strstr(raw.p ? raw.p : "", "uddg=");
    if (u) {                                         /* //duckduckgo.com/l/?uddg=<percent-encoded URL>&rut=... */
        for (u += 5; *u && *u != '&'; u++) {
            unsigned hex;
            if (*u == '%' && sscanf(u+1, "%2x", &hex) == 1) { char ch = (char)hex; buf_add(&url, &ch, 1); u += 2; }
            else buf_add(&url, u, 1);
        }
    } else {
        if (raw.p && !strncmp(raw.p, "//", 2)) buf_str(&url, "https:");
        buf_str(&url, raw.p ? raw.p : "");
    }
    buf_free(&raw); buf_add(&url, "", 0);
    return url.p;
}
static void curl_search(const char *query, Buf *out) {
    if (!query || !*query) { buf_str(out, "Tool error: missing query\n"); return; }
    Buf form = {0}, html = {0}; Fetched f; size_t n = 0;
    buf_fmt(&form, "q=%s", query);
    char *args[] = {"--data-urlencode", form.p, "https://html.duckduckgo.com/html/", NULL};
    if (fetch(args, &html, &f, out)) {
        for (const char *p = html.p; n < 10 && (p = strstr(p, "class=\"result__a\""));) {
            const char *open = p, *gt = strchr(p, '>'), *close = gt ? strstr(gt, "</a>") : NULL;
            while (open > html.p && *open != '<') open--;
            if (!close) break;
            const char *h = strstr(open, "href=\""), *hend = h ? strchr(h+6, '"') : NULL;
            const char *next = strstr(close, "class=\"result__a\""), *snip = strstr(close, "class=\"result__snippet\"");
            if (h && hend && h < gt) {
                Buf href = {0}, title = {0}, text = {0}; buf_add(&href, h+6, (size_t)(hend-h-6));
                char *url = ddg_url(href.p);
                pf_html_to_text(gt+1, (size_t)(close-gt-1), NULL, &title);
                if (snip && (!next || snip < next)) {
                    const char *sgt = strchr(snip, '>'), *send = sgt ? strstr(sgt, "</a>") : NULL;
                    if (send) pf_html_to_text(sgt+1, (size_t)(send-sgt-1), NULL, &text);
                }
                if (!strstr(url, "duckduckgo.com/y.js"))   /* an ad */
                    buf_fmt(out, "%zu. %s\n   %s\n   %s\n", ++n, trim_line(title.p), url, text.p ? trim_line(text.p) : "");
                free(url); buf_free(&href); buf_free(&title); buf_free(&text);
            }
            p = close;
        }
        if (!n && (strstr(html.p ? html.p : "", "anomaly") || strstr(html.p ? html.p : "", "challenge")))
            buf_str(out, "Tool error: DuckDuckGo asked for a human check; this tool does not pass it. Try again later, "
                         "or use visit_page on a URL you know.\n");
        else if (!n) buf_fmt(out, "No results for: %s\n", query);
    }
    buf_free(&form); buf_free(&html);
}
/* The page as text, or the raw body for text, JSON, XML. */
static void curl_visit(const char *url, Buf *out) {
    if (!url || (strncmp(url, "http://", 7) && strncmp(url, "https://", 8)) || strpbrk(url, " \t\r\n")) {
        buf_str(out, "Tool error: url must be one http:// or https:// address\n"); return;
    }
    Buf body = {0}; Fetched f; char *args[] = {(char *)url, NULL};
    if (fetch(args, &body, &f, out)) {
        const char *type = f.type;
        bool html = strstr(type, "html") || (!*type && find_ci(body.p ? body.p : "", body.p+body.n, "<html"));
        bool text = html || !strncmp(type, "text/", 5) || strstr(type, "json") || strstr(type, "xml") ||
                    strstr(type, "javascript") || (!*type && !memchr(body.p ? body.p : "", 0, body.n));
        if (f.code >= 400) buf_fmt(out, "Tool error: HTTP %d for %s\n", f.code, f.url);
        else buf_fmt(out, "%s\n\n", f.url);
        if (!text) buf_fmt(out, "Tool error: not a text page (%s, %zu bytes)\n", *type ? type : "unknown type", body.n);
        else if (html) pf_html_to_text(body.p ? body.p : "", body.n, f.url, out);
        else buf_add(out, body.p ? body.p : "", body.n);
    }
    buf_free(&body);
}

/* ------------------------------------------------------------ HTTP and WebSocket to Chrome
 * Only to 127.0.0.1, only to the port we started Chrome on. */

#define CDP_TIMEOUT 25.0           /* seconds for one page: navigation, rendering, extraction */
#define CDP_MAX_MESSAGE (16u<<20)

static int tcp_connect(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int one = 1; setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr)) { close(fd); return -1; }
    return fd;
}
static bool send_all(int fd, const void *data, size_t n) {
    const char *p = data;
    while (n) { ssize_t w = send(fd, p, n, 0); if (w < 0 && errno == EINTR) continue; if (w <= 0) return false; p += w; n -= (size_t)w; }
    return true;
}
/* Wait until fd is readable or the deadline passes (or the user interrupts). */
static bool readable(int fd, double deadline) {
    for (;;) {
        double left = deadline-qw4_now();
        if (left <= 0 || qw4_stopped) return false;
        struct pollfd p = {fd, POLLIN, 0};
        int r = poll(&p, 1, left > 0.2 ? 200 : (int)(left*1000)+1);
        if (r > 0) return true;
        if (r < 0 && errno != EINTR) return false;
    }
}
/* One HTTP request to Chrome's endpoint; the body, NULL if it does not answer. */
static char *http(int port, const char *method, const char *path) {
    int fd = tcp_connect(port);
    if (fd < 0) return NULL;
    Buf req = {0}, resp = {0};
    buf_fmt(&req, "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", method, path, port);
    bool ok = send_all(fd, req.p, req.n); char chunk[65536]; double deadline = qw4_now()+5;
    while (ok && readable(fd, deadline)) {
        ssize_t got = recv(fd, chunk, sizeof chunk, 0);
        if (got <= 0) break;
        buf_add(&resp, chunk, (size_t)got);
    }
    close(fd); buf_free(&req);
    char *body = resp.p ? strstr(resp.p, "\r\n\r\n") : NULL;
    char *out = body && !strncmp(resp.p, "HTTP/1.1 200", 12) ? strdup(body+4) : NULL;
    buf_free(&resp);
    return out;
}

/* The string value of "key" in a JSON text (first occurrence), unescaped. */
static bool json_get(const char *json, const char *key, Buf *out) {
    char pattern[64]; snprintf(pattern, sizeof pattern, "\"%s\":", key);
    const char *p = json ? strstr(json, pattern) : NULL;
    if (!p) return false;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p != '"') return false;
    for (p++; *p && *p != '"'; p++) {
        if (*p != '\\') { buf_add(out, p, 1); continue; }
        char c = *++p;
        if (c == 'n') buf_add(out, "\n", 1); else if (c == 't') buf_add(out, "\t", 1);
        else if (c == 'r') buf_add(out, "\r", 1); else if (c == 'b' || c == 'f') {}
        else if (c == 'u') {
            unsigned cp = 0, lo = 0;
            if (sscanf(p+1, "%4x", &cp) != 1) return false;
            p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u' && sscanf(p+3, "%4x", &lo) == 1 && lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000+((cp-0xD800) << 10)+(lo-0xDC00); p += 6;
            }
            utf8_put(out, cp);
        }
        else if (c) buf_add(out, &c, 1);
        else return false;
    }
    buf_add(out, "", 0);
    return *p == '"';
}

/* A WebSocket client, enough for CDP (RFC 6455): one text message per frame
 * or continued frames, client frames masked, ping answered. The server's
 * accept key is not checked: the peer is the Chrome we started on 127.0.0.1. */
typedef struct { int fd; Buf in; int next_id; } Ws;

static bool ws_open(Ws *ws, const char *url) {
    int port = 0; char path[512];
    if (sscanf(url, "ws://127.0.0.1:%d%511s", &port, path) != 2 && sscanf(url, "ws://localhost:%d%511s", &port, path) != 2) return false;
    *ws = (Ws){.fd = tcp_connect(port), .next_id = 1};
    if (ws->fd < 0) return false;
    unsigned char key[16]; arc4random_buf(key, sizeof key);
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char k64[25] = {0};
    for (int i = 0, o = 0; i < 16; i += 3) {
        unsigned v = (unsigned)key[i] << 16 | (i+1 < 16 ? (unsigned)key[i+1] << 8 : 0) | (i+2 < 16 ? key[i+2] : 0);
        k64[o++] = b64[v >> 18 & 63]; k64[o++] = b64[v >> 12 & 63];
        k64[o++] = i+1 < 16 ? b64[v >> 6 & 63] : '='; k64[o++] = i+2 < 16 ? b64[v & 63] : '=';
    }
    Buf req = {0}, resp = {0}; char chunk[4096];
    buf_fmt(&req, "GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n", path, port, k64);
    bool ok = send_all(ws->fd, req.p, req.n); double deadline = qw4_now()+5;
    while (ok && !(resp.p && strstr(resp.p, "\r\n\r\n")) && readable(ws->fd, deadline)) {
        ssize_t got = recv(ws->fd, chunk, sizeof chunk, 0);
        if (got <= 0) break;
        buf_add(&resp, chunk, (size_t)got);
    }
    char *end = resp.p ? strstr(resp.p, "\r\n\r\n") : NULL;
    ok = end && !strncmp(resp.p, "HTTP/1.1 101", 12);
    if (ok) buf_add(&ws->in, end+4, resp.n-(size_t)(end+4-resp.p));   /* a frame may follow at once */
    buf_free(&req); buf_free(&resp);
    if (!ok) { close(ws->fd); ws->fd = -1; }
    return ok;
}
static void ws_close(Ws *ws) { if (ws->fd >= 0) close(ws->fd); ws->fd = -1; buf_free(&ws->in); }
static bool ws_send(Ws *ws, int opcode, const char *data, size_t n) {
    unsigned char head[14]; size_t h = 0; unsigned char mask[4]; arc4random_buf(mask, 4);
    head[h++] = (unsigned char)(0x80 | opcode);
    if (n < 126) head[h++] = (unsigned char)(0x80 | n);
    else if (n < 65536) { head[h++] = 0x80 | 126; head[h++] = (unsigned char)(n >> 8); head[h++] = (unsigned char)n; }
    else { head[h++] = 0x80 | 127; for (int i = 7; i >= 0; i--) head[h++] = (unsigned char)((uint64_t)n >> (8*i)); }
    memcpy(head+h, mask, 4); h += 4;
    char *masked = malloc(n ? n : 1);
    if (!masked) die("pf-agent: out of memory");
    for (size_t i = 0; i < n; i++) masked[i] = (char)(data[i] ^ mask[i & 3]);
    bool ok = send_all(ws->fd, head, h) && send_all(ws->fd, masked, n);
    free(masked);
    return ok;
}
/* Bytes until ws->in holds at least n, or false at the deadline. */
static bool ws_fill(Ws *ws, size_t n, double deadline) {
    char chunk[65536];
    while (ws->in.n < n) {
        if (!readable(ws->fd, deadline)) return false;
        ssize_t got = recv(ws->fd, chunk, sizeof chunk, 0);
        if (got <= 0) return false;
        buf_add(&ws->in, chunk, (size_t)got);
    }
    return true;
}
/* The next complete text message. */
static bool ws_read(Ws *ws, Buf *msg, double deadline) {
    msg->n = 0;
    for (;;) {
        if (!ws_fill(ws, 2, deadline)) return false;
        const unsigned char *b = (const unsigned char *)ws->in.p;
        bool fin = b[0] & 0x80; int op = b[0] & 15; uint64_t len = b[1] & 127; size_t h = 2;
        if (len == 126) { if (!ws_fill(ws, 4, deadline)) return false; b = (const unsigned char *)ws->in.p; len = (uint64_t)b[2] << 8 | b[3]; h = 4; }
        else if (len == 127) {
            if (!ws_fill(ws, 10, deadline)) return false;
            b = (const unsigned char *)ws->in.p; len = 0; for (int i = 2; i < 10; i++) len = len << 8 | b[i]; h = 10;
        }
        if (b[1] & 0x80 || len > CDP_MAX_MESSAGE || !ws_fill(ws, h+(size_t)len, deadline)) return false;   /* servers never mask */
        const char *payload = ws->in.p+h;
        if (op == 9) ws_send(ws, 10, payload, (size_t)len);                 /* ping -> pong */
        else if (op == 8) return false;                                     /* closed */
        else if (op == 1 || op == 0) buf_add(msg, payload, (size_t)len);
        size_t used = h+(size_t)len;
        memmove(ws->in.p, ws->in.p+used, ws->in.n-used+1); ws->in.n -= used;
        if ((op == 1 || op == 0) && fin) { buf_add(msg, "", 0); return true; }
    }
}

/* ------------------------------------------------------------ CDP
 * A command is {"id":N,"method":...,"params":{...}}; its answer starts with the
 * same id. Events (no id) arriving meanwhile are skipped. */

/* The id an answer starts with ({"id": N, ...}), -1 for an event. */
static long answer_id(const char *msg) {
    const char *p = msg;
    while (*p == ' ' || *p == '{') p++;
    if (strncmp(p, "\"id\"", 4)) return -1;
    for (p += 4; *p == ' ' || *p == ':'; p++) {}
    return isdigit((unsigned char)*p) ? strtol(p, NULL, 10) : -1;
}
static bool cdp(Ws *ws, const char *method, const char *params, Buf *answer, double deadline) {
    Buf req = {0}, msg = {0}; int id = ws->next_id++;
    buf_fmt(&req, "{\"id\":%d,\"method\":\"%s\",\"params\":%s}", id, method, params ? params : "{}");
    bool ok = ws_send(ws, 1, req.p, req.n);
    while (ok && (ok = ws_read(ws, &msg, deadline))) if (answer_id(msg.p) == id) break;
    /* A quote inside a JSON string is escaped, so "error":{ can only be the key. */
    if (ok && strstr(msg.p, "\"error\":{")) ok = false;
    if (ok && answer) { buf_free(answer); *answer = msg; msg = (Buf){0}; }
    buf_free(&req); buf_free(&msg);
    return ok;
}
/* Runtime.evaluate of a JavaScript expression (a promise is awaited); its string value. */
static bool evaluate(Ws *ws, const char *expression, Buf *value, double deadline) {
    Buf params = {0}, answer = {0};
    buf_str(&params, "{\"expression\":\""); json_escape(&params, expression);
    buf_str(&params, "\",\"returnByValue\":true,\"awaitPromise\":true}");
    bool ok = cdp(ws, "Runtime.evaluate", params.p, &answer, deadline) && !strstr(answer.p, "\"exceptionDetails\"");
    value->n = 0;
    ok = ok && json_get(answer.p, "value", value);
    buf_free(&params); buf_free(&answer);
    return ok;
}

/* ------------------------------------------------------------ Chrome */

struct PfWeb {
    PfWebAsk ask; void *ask_ctx;
    int port;
    enum { CHROME_UNKNOWN, CHROME_READY, CHROME_NO } chrome;   /* NO: missing or declined, curl from now on */
    char why[160];                                             /* why not, for the fallback note */
};

static const char *chrome_binary(void) {
    static const char *const candidates[] = {
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", NULL};
    const char *env = getenv("PF_AGENT_CHROME");
    if (env && *env) return access(env, X_OK) ? NULL : env;
    for (size_t i = 0; candidates[i]; i++) if (!access(candidates[i], X_OK)) return candidates[i];
    return NULL;
}
static bool chrome_answers(const PfWeb *w) {
    char *version = http(w->port, "GET", "/json/version");
    bool ok = version && strstr(version, "webSocketDebuggerUrl");
    free(version);
    return ok;
}
/* Chrome on our port: already running (an earlier start), or started now
 * after the user agreed. False (with w->why) means: use curl. */
static bool chrome_ready(PfWeb *w) {
    if (w->chrome == CHROME_READY && chrome_answers(w)) return true;
    if (w->chrome == CHROME_NO) return false;
    if (chrome_answers(w)) { w->chrome = CHROME_READY; return true; }
    const char *exe = chrome_binary(); const char *home = getenv("HOME");
    if (!exe || !home) { w->chrome = CHROME_NO; snprintf(w->why, sizeof w->why, "Chrome is not installed"); return false; }
    char profile[1024], port[48], dir[1100];
    snprintf(profile, sizeof profile, "%s/.picoforge", home); mkdir(profile, 0700);
    snprintf(profile, sizeof profile, "%s/.picoforge/browser", home); mkdir(profile, 0700);
    Buf what = {0};
    buf_fmt(&what, "start Chrome for the web tools (a visible window, its own profile in %s; it stays open)", profile);
    bool yes = w->ask && w->ask(w->ask_ctx, what.p);
    buf_free(&what);
    if (!yes) { w->chrome = CHROME_NO; snprintf(w->why, sizeof w->why, "Chrome was not started (declined)"); return false; }
    snprintf(port, sizeof port, "--remote-debugging-port=%d", w->port);
    snprintf(dir, sizeof dir, "--user-data-dir=%s", profile);
    char *argv[] = {(char *)exe, port, dir, "--no-first-run", "--no-default-browser-check", "about:blank", NULL};
    int null = open("/dev/null", O_WRONLY);
    pid_t pid = null < 0 ? -1 : pf_spawn(argv, null, null);
    if (null >= 0) close(null);
    for (double end = qw4_now()+15; pid > 0 && qw4_now() < end && !qw4_stopped; pause_ms(200))
        if (chrome_answers(w)) { w->chrome = CHROME_READY; return true; }
    w->chrome = CHROME_NO; snprintf(w->why, sizeof w->why, "Chrome did not start");
    return false;
}

PfWeb *pf_web_create(PfWebAsk ask, void *ask_ctx) {
    PfWeb *w = calloc(1, sizeof *w);
    if (!w) die("pf-agent: out of memory");
    const char *port = getenv("PF_AGENT_CDP_PORT");
    w->ask = ask; w->ask_ctx = ask_ctx; w->port = port && *port ? atoi(port) : 9334;
    return w;
}
void pf_web_destroy(PfWeb *w) { free(w); }

/* Open a tab, load url, run the extractor `js` once the page is ready, close
 * the tab. The text the extractor returns, or "Tool error: ..." in out. */
static bool in_chrome(PfWeb *w, const char *url, const char *js, Buf *text, Buf *out) {
    double deadline = qw4_now()+CDP_TIMEOUT;
    Buf target = {0}, ws_url = {0}, id = {0}, params = {0}, state = {0};
    char *tab = http(w->port, "PUT", "/json/new?about:blank");
    bool ok = tab && json_get(tab, "webSocketDebuggerUrl", &ws_url) && json_get(tab, "id", &id);
    free(tab);
    Ws ws = {.fd = -1};
    ok = ok && ws_open(&ws, ws_url.p) && cdp(&ws, "Page.enable", NULL, NULL, deadline) && cdp(&ws, "Runtime.enable", NULL, NULL, deadline);
    buf_str(&params, "{\"url\":\""); json_escape(&params, url); buf_str(&params, "\"}");
    ok = ok && cdp(&ws, "Page.navigate", params.p, NULL, deadline);
    /* Ready: the document loaded, then a moment for scripts that render after load. */
    while (ok && (ok = evaluate(&ws, "document.readyState", &state, deadline)) && strcmp(state.p, "complete")) pause_ms(150);
    if (ok) pause_ms(600);
    ok = ok && evaluate(&ws, js, text, deadline);
    if (!ok) buf_str(out, qw4_stopped ? "Tool error: interrupted by the user\n" : "Tool error: Chrome did not load the page in time\n");
    ws_close(&ws);
    if (id.n) { char path[600]; snprintf(path, sizeof path, "/json/close/%s", id.p); free(http(w->port, "GET", path)); }
    buf_free(&target); buf_free(&ws_url); buf_free(&id); buf_free(&params); buf_free(&state);
    return ok;
}

/* ------------------------------------------------------------ the tools */

/* Google's results: each title (h3) inside a link, its URL (Google's /url
 * redirect unwrapped) and the text around it. The first line is the address
 * the page ended on, to tell results from a consent page or a human check. */
static const char search_js[] =
    "(() => {"
    " const lines = [location.href], seen = new Set();"
    " const real = h => { try { const u = new URL(h); return u.pathname === '/url' && u.searchParams.get('q') ? u.searchParams.get('q') : u.href; } catch (e) { return ''; } };"
    " for (const h3 of document.querySelectorAll('a h3')) {"
    "  const link = h3.closest('a'), url = real(link.href);"
    "  if (!/^https?:/.test(url) || seen.has(url) || /(^|\\.)google\\./.test(new URL(url).hostname)) continue;"
    "  seen.add(url);"
    "  let box = link;"
    "  for (let i = 0; i < 6 && box.parentElement && box.innerText.length < 160; i++) box = box.parentElement;"
    "  const title = h3.innerText.trim(), around = box.innerText.replace(title, '').replace(/\\s+/g, ' ').trim();"
    "  lines.push(seen.size + '. ' + title + '\\n   ' + url + '\\n   ' + around.slice(0, 320));"
    "  if (seen.size >= 10) break;"
    " }"
    " if (lines.length === 1) lines.push('', (document.body ? document.body.innerText : '').slice(0, 800));"
    " return lines.join('\\n');"
    "})()";

/* A page's rendered text (the main part when the page marks one), after
 * scrolling to the end so lazy content loads, then its links. */
static const char page_js[] =
    "(async () => {"
    " for (let i = 0, h = 0; i < 12; i++) {"
    "  const H = document.documentElement.scrollHeight; if (H === h) break; h = H;"
    "  window.scrollTo(0, H); await new Promise(r => setTimeout(r, 250));"
    " }"
    " window.scrollTo(0, 0);"
    " const root = document.querySelector('main, article, [role=main]') || document.body;"
    " const text = (root ? root.innerText : '').replace(/\\n{3,}/g, '\\n\\n').slice(0, 900000);"
    " const links = [], seen = new Set();"
    " for (const a of document.querySelectorAll('a[href]')) {"
    "  const t = a.innerText.replace(/\\s+/g, ' ').trim();"
    "  if (t.length < 2 || !/^https?:/.test(a.href) || seen.has(a.href)) continue;"
    "  seen.add(a.href); links.push('- ' + t.slice(0, 100) + ': ' + a.href.split('#')[0]);"
    "  if (links.length >= 60) break;"
    " }"
    " return location.href + '\\n\\n# ' + (document.title || location.href) + '\\n\\n' + text + (links.length ? '\\n\\nLinks:\\n' + links.join('\\n') : '');"
    "})()";

/* Google answers automated-looking traffic with a consent page or a human
 * check. We neither click nor solve them: the user can, in the visible window. */
static bool blocked(const char *page) {
    return strstr(page, "consent.google.") || strstr(page, "/sorry/") || strstr(page, "unusual traffic") ||
           strstr(page, "Before you continue to Google") || strstr(page, "Prima di continuare");
}

void pf_web_search(PfWeb *w, const char *query, Buf *out) {
    if (!query || !*query) { buf_str(out, "Tool error: missing query\n"); return; }
    if (!chrome_ready(w)) {
        buf_fmt(out, "[%s: DuckDuckGo results through curl]\n", w->why);
        curl_search(query, out); return;
    }
    Buf url = {0}, text = {0};
    buf_str(&url, "https://www.google.com/search?hl=en&q=");
    for (const unsigned char *q = (const unsigned char *)query; *q; q++) {   /* percent-encode */
        if (isalnum(*q) || strchr("-_.~", *q)) buf_add(&url, q, 1); else buf_fmt(&url, "%%%02X", *q);
    }
    if (in_chrome(w, url.p, search_js, &text, out)) {
        if (blocked(text.p)) buf_str(out, "Tool error: Google is showing a consent page or a human check in the Chrome window. "
                                          "Ask the user to complete it there, then search again.\n");
        else { const char *nl = strchr(text.p, '\n'); buf_str(out, nl ? nl+1 : ""); }
    }
    buf_free(&url); buf_free(&text);
}

void pf_web_visit(PfWeb *w, const char *url, Buf *out) {
    if (!url || (strncmp(url, "http://", 7) && strncmp(url, "https://", 8)) || strpbrk(url, " \t\r\n")) {
        buf_str(out, "Tool error: url must be one http:// or https:// address\n"); return;
    }
    if (!chrome_ready(w)) { buf_fmt(out, "[%s: the page through curl, without JavaScript]\n", w->why); curl_visit(url, out); return; }
    Buf text = {0};
    if (in_chrome(w, url, page_js, &text, out)) buf_add(out, text.p, text.n);
    buf_free(&text);
}
