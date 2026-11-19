/* pf_web.h — the web tools' backend: a search and a page as text. */
#ifndef PICOFORGE_PF_WEB_H
#define PICOFORGE_PF_WEB_H
#include "pf_util.h"
/* The first results of a web search, or "Tool error: ..." in out. */
void pf_web_search(const char *query, Buf *out);
/* A page (http/https) as text with its links, or "Tool error: ..." in out. */
void pf_web_visit(const char *url, Buf *out);
/* HTML -> readable text; with a base URL the links are listed at the end. */
void pf_html_to_text(const char *html, size_t n, const char *base, Buf *out);
#endif
