/* pf_web.c — the web tools through /usr/bin/curl (part of macOS), run with an
 * argument vector, never through a shell, so a query or a URL cannot become a
 * command. curl reads no config file (-q), speaks only http(s), also across
 * redirects, and has a time and size limit. Search reads DuckDuckGo's HTML
 * page: no key, no account. PF_AGENT_CURL replaces curl in the tests. */
#include "pf_web.h"
#include "common.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

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
void pf_web_search(const char *query, Buf *out) {
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
void pf_web_visit(const char *url, Buf *out) {
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
