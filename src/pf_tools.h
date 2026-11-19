/* pf_tools.h — tool calls in the model's format, and the tools pf-agent runs.
 *
 * Nothing here prints or reads the terminal: side effects ask through a
 * callback (pf-agent asks the user; a server could refuse, or not run tools at
 * all and only parse the calls). Every result is cleaned for the prompt and
 * cut at PF_OUT_LIMIT bytes, the rest kept for the `more` tool. */
#ifndef PICOFORGE_PF_TOOLS_H
#define PICOFORGE_PF_TOOLS_H
#include "pf_util.h"

/* <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call> */
#define PF_MAX_PARAMS 8
typedef struct { char *name; size_t n; char *key[PF_MAX_PARAMS], *value[PF_MAX_PARAMS]; } PfCall;
size_t pf_parse_calls(const char *text, PfCall *calls, size_t max);   /* complete calls only */
void pf_call_free(PfCall *);

/* What the model reads about a tool (JSON schema in the system prompt). */
typedef struct { const char *name, *type, *description; bool required; } PfParam;
typedef struct { const char *name, *description; PfParam params[6]; } PfToolSpec;
size_t pf_tool_count(void);
const PfToolSpec *pf_tool_spec(size_t i);

typedef enum { PF_DENY, PF_ALLOW, PF_ALWAYS } PfAnswer;
typedef PfAnswer (*PfAsk)(void *ctx, const char *what);   /* PF_ALWAYS: stop asking */

typedef struct PfTools PfTools;
/* root: the workspace (realpath); file tools never leave it. yes: approve all. */
PfTools *pf_tools_create(const char *root, bool yes, PfAsk ask, void *ask_ctx);
void pf_tools_destroy(PfTools *);              /* stops running jobs, removes their logs */
void pf_tools_run(PfTools *, const PfCall *, Buf *out);
const char *pf_tools_root(const PfTools *);
bool pf_tools_set_root(PfTools *, const char *dir);   /* a resumed session's workspace */
void pf_tools_approve_all(PfTools *);
/* A text file of the workspace (AGENTS.md), at most `limit` bytes; false if absent. */
bool pf_tools_read_text(const PfTools *, const char *path, size_t limit, Buf *out);
#endif
