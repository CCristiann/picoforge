/* pf_web.h — the web tools' backend: a search and a page as text.
 *
 * Through Chrome, driven with the DevTools protocol (CDP): pages are rendered,
 * JavaScript included, and searches go to Google. Chrome runs visibly with its
 * own profile (~/.picoforge/browser) and starts only after `ask` says yes;
 * consent banners and human checks are left to the user in that window. If
 * Chrome is missing or declined, curl does the work: DuckDuckGo for searches,
 * plain HTML for pages. */
#ifndef PICOFORGE_PF_WEB_H
#define PICOFORGE_PF_WEB_H
#include "pf_util.h"

typedef bool (*PfWebAsk)(void *ctx, const char *what);
typedef struct PfWeb PfWeb;
PfWeb *pf_web_create(PfWebAsk ask, void *ask_ctx);
void pf_web_destroy(PfWeb *);                    /* Chrome stays open, as a browser the user started */
void pf_web_search(PfWeb *, const char *query, Buf *out);   /* results, or "Tool error: ..." */
void pf_web_visit(PfWeb *, const char *url, Buf *out);      /* the page as text with its links */
/* HTML -> readable text; with a base URL the links are listed at the end. */
void pf_html_to_text(const char *html, size_t n, const char *base, Buf *out);
#endif
