/* pf_term.h — a terminal split in three: output that scrolls, an input line
 * that stays editable while the model works, and a status footer.
 *
 * The output rows are an ANSI scroll region, so text written there scrolls
 * without touching the two bottom rows. Raw mode: keys arrive one by one,
 * Ctrl+C is a key (no SIGINT), and bracketed paste keeps pasted newlines in the
 * input instead of submitting half a paste. One thread only: pf-agent's UI. */
#ifndef PICOFORGE_PF_TERM_H
#define PICOFORGE_PF_TERM_H
#include <stdbool.h>
#include <stddef.h>

typedef struct PfTerm PfTerm;
typedef enum { PF_TERM_NONE, PF_TERM_SUBMIT, PF_TERM_INTERRUPT, PF_TERM_EOF } PfTermEvent;

PfTerm *pf_term_open(void);                     /* NULL if stdin/stdout are not a terminal */
void pf_term_close(PfTerm *);                   /* the terminal as it was */
int pf_term_fd(const PfTerm *);                 /* stdin, to poll */
void pf_term_output(PfTerm *, const char *, size_t);   /* append to the scrolling rows */
void pf_term_status(PfTerm *, const char *line);       /* the footer (may hold colors) */
void pf_term_prompt(PfTerm *, const char *prompt);     /* "pf> ", or a question */
void pf_term_resized(PfTerm *);                 /* after SIGWINCH */
/* Read the keys available on stdin. SUBMIT: *line is the entered text (owned
 * by the caller); INTERRUPT: Ctrl+C; EOF: Ctrl+D on an empty line. */
PfTermEvent pf_term_read(PfTerm *, char **line);
bool pf_term_pending(const PfTerm *);           /* keys read but not handled yet: read again */
void pf_term_clear(PfTerm *);                   /* empty the input line */
#endif
