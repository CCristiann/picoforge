/* pf_term.c — see pf_term.h.
 *
 * Screen: rows 1..R-2 scroll (DECSTBM sets the region), row R-1 is the input,
 * row R the footer. The output cursor is kept with DECSC/DECRC (ESC 7 / ESC 8):
 * restore it, write, save it again, then draw the two bottom rows with
 * absolute moves and leave the visible cursor in the input. */
#include "pf_term.h"
#include "pf_util.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#define HISTORY 200

struct PfTerm {
    struct termios saved;
    int rows, cols;
    Buf input; size_t cursor;             /* the line being edited; cursor as a byte offset */
    Buf pending;                          /* bytes of an incomplete key sequence */
    bool paste;                           /* inside a bracketed paste: Enter is a newline */
    char prompt[96], *status;
    char *history[HISTORY]; size_t nhistory, browsing;   /* browsing == nhistory: the draft */
    char *draft;
};

static void set_line(PfTerm *t, const char *s);
static void put(const char *s, size_t n) {
    while (n) { ssize_t w = write(STDOUT_FILENO, s, n); if (w <= 0) return; s += w; n -= (size_t)w; }
}
static void measure(PfTerm *t) {
    struct winsize w;
    t->rows = !ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) && w.ws_row >= 5 ? w.ws_row : 24;
    t->cols = !ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) && w.ws_col >= 20 ? w.ws_col : 80;
}
/* The scroll region and the output cursor at its last row. */
static void layout(PfTerm *t) {
    Buf b = {0};
    buf_fmt(&b, "\x1b[%d;1H\x1b[2K\x1b[%d;1H\x1b[2K", t->rows-1, t->rows);   /* clear the bottom rows */
    buf_fmt(&b, "\x1b[1;%dr\x1b[%d;1H\x1b" "7", t->rows-2, t->rows-2);
    put(b.p, b.n); buf_free(&b);
}

/* Columns a UTF-8 string takes (one per code point; escapes take none). */
static size_t width(const char *s, size_t n) {
    size_t w = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\x1b') { while (i < n && !((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z'))) i++; continue; }
        if (((unsigned char)s[i] & 0xC0) != 0x80) w++;
    }
    return w;
}
/* s cut to `cols` columns, escapes kept. */
static void clip(Buf *out, const char *s, size_t cols) {
    size_t w = 0, n = strlen(s);
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\x1b') {
            size_t j = i;
            while (j < n && !((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z'))) j++;
            buf_add(out, s+i, j < n ? j-i+1 : n-i); i = j; continue;
        }
        if (((unsigned char)s[i] & 0xC0) != 0x80 && ++w > cols) break;
        buf_add(out, s+i, 1);
    }
}

/* The input row (prompt + the visible window of the line, newlines as '¶')
 * and the footer; the cursor ends in the input. */
static void draw(PfTerm *t) {
    Buf b = {0}, shown = {0};
    size_t at = 0;                                   /* display column of the cursor within shown */
    for (size_t i = 0; i < t->input.n; i++) {
        if (i == t->cursor) at = width(shown.p ? shown.p : "", shown.n);
        if (t->input.p[i] == '\n') buf_str(&shown, "\xc2\xb6"); else buf_add(&shown, t->input.p+i, 1);
    }
    if (t->cursor == t->input.n) at = width(shown.p ? shown.p : "", shown.n);
    size_t pw = width(t->prompt, strlen(t->prompt)), room = (size_t)t->cols > pw+2 ? (size_t)t->cols-pw-1 : 1;
    size_t skip = at >= room ? at-room+1 : 0, col = 0, i = 0;   /* columns scrolled off the left */
    while (i < shown.n && col < skip) {
        i++; col++;
        while (i < shown.n && ((unsigned char)shown.p[i] & 0xC0) == 0x80) i++;
    }
    buf_fmt(&b, "\x1b[%d;1H\x1b[2K\x1b[1m%s\x1b[0m", t->rows-1, t->prompt);
    clip(&b, shown.p ? shown.p+i : "", room);
    buf_fmt(&b, "\x1b[%d;1H\x1b[2K", t->rows);
    clip(&b, t->status ? t->status : "", (size_t)t->cols-1);
    buf_fmt(&b, "\x1b[0m\x1b[%d;%zuH", t->rows-1, pw+at-skip+1);
    put(b.p, b.n); buf_free(&b); buf_free(&shown);
}

PfTerm *pf_term_open(void) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return NULL;
    PfTerm *t = calloc(1, sizeof *t);
    if (!t) die("out of memory");
    if (tcgetattr(STDIN_FILENO, &t->saved)) { free(t); return NULL; }
    struct termios raw = t->saved;
    raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG | IEXTEN);
    raw.c_iflag &= ~(tcflag_t)(IXON | ICRNL);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    measure(t);
    snprintf(t->prompt, sizeof t->prompt, "pf> ");
    put("\n\n\x1b[?2004h", 11);                      /* room for the bottom rows; bracketed paste on */
    layout(t); draw(t);
    return t;
}
void pf_term_close(PfTerm *t) {
    if (!t) return;
    char end[64]; snprintf(end, sizeof end, "\x1b[r\x1b[?2004l\x1b[%d;1H\x1b[2K\x1b[%d;1H\x1b[2K", t->rows-1, t->rows);
    put(end, strlen(end));
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &t->saved);
    for (size_t i = 0; i < t->nhistory; i++) free(t->history[i]);
    buf_free(&t->input); buf_free(&t->pending); free(t->status); free(t->draft); free(t);
}
int pf_term_fd(const PfTerm *t) { (void)t; return STDIN_FILENO; }
bool pf_term_pending(const PfTerm *t) { return t->pending.n > 0; }
void pf_term_clear(PfTerm *t) { set_line(t, ""); draw(t); }

void pf_term_output(PfTerm *t, const char *s, size_t n) {
    if (!n) return;
    put("\x1b" "8", 2); put(s, n); put("\x1b" "7", 2);
    draw(t);
}
void pf_term_status(PfTerm *t, const char *line) {
    free(t->status); t->status = strdup(line ? line : "");
    draw(t);
}
void pf_term_prompt(PfTerm *t, const char *prompt) {
    snprintf(t->prompt, sizeof t->prompt, "%s", prompt);
    draw(t);
}
void pf_term_resized(PfTerm *t) { measure(t); layout(t); draw(t); }

/* ------------------------------------------------------------ editing */

static size_t prev_char(const PfTerm *t, size_t i) { while (i && ((unsigned char)t->input.p[--i] & 0xC0) == 0x80) {} return i; }
static size_t next_char(const PfTerm *t, size_t i) {
    if (i < t->input.n) i++;
    while (i < t->input.n && ((unsigned char)t->input.p[i] & 0xC0) == 0x80) i++;
    return i;
}
static void insert(PfTerm *t, const char *s, size_t n) {
    Buf b = {0};
    buf_add(&b, t->input.p ? t->input.p : "", t->cursor); buf_add(&b, s, n);
    buf_add(&b, t->input.p ? t->input.p+t->cursor : "", t->input.n-t->cursor);
    buf_free(&t->input); t->input = b; t->cursor += n;
}
static void erase(PfTerm *t, size_t from, size_t to) {   /* bytes [from, to) */
    if (to <= from) return;
    memmove(t->input.p+from, t->input.p+to, t->input.n-to+1);
    t->input.n -= to-from;
    if (t->cursor >= to) t->cursor -= to-from; else if (t->cursor > from) t->cursor = from;
}
static void set_line(PfTerm *t, const char *s) {
    buf_free(&t->input); buf_add(&t->input, s ? s : "", s ? strlen(s) : 0); t->cursor = t->input.n;
}
static void browse(PfTerm *t, int step) {
    if (!t->nhistory || (step < 0 && !t->browsing) || (step > 0 && t->browsing >= t->nhistory)) return;
    if (t->browsing == t->nhistory) { free(t->draft); t->draft = strdup(t->input.p ? t->input.p : ""); }
    t->browsing = step < 0 ? t->browsing-1 : t->browsing+1;
    set_line(t, t->browsing == t->nhistory ? t->draft : t->history[t->browsing]);
}
static char *submit(PfTerm *t) {
    char *line = strdup(t->input.p ? t->input.p : "");
    if (!line) die("out of memory");
    if (*line && (!t->nhistory || strcmp(t->history[t->nhistory-1], line))) {
        if (t->nhistory == HISTORY) { free(t->history[0]); memmove(t->history, t->history+1, (HISTORY-1)*sizeof *t->history); t->nhistory--; }
        t->history[t->nhistory++] = strdup(line);
    }
    t->browsing = t->nhistory;
    set_line(t, "");
    return line;
}

/* One key from p[0..n): its length, 0 if the sequence is incomplete. */
static size_t key(PfTerm *t, const char *p, size_t n, PfTermEvent *event, char **line) {
    unsigned char c = (unsigned char)p[0];
    if (c == 0x1b) {
        if (n < 2) return 0;
        if (p[1] == '\r') { insert(t, "\n", 1); return 2; }                 /* Alt+Enter: a newline */
        if (p[1] != '[') return 1;
        size_t j = 2;
        while (j < n && !((p[j] >= 'A' && p[j] <= 'Z') || (p[j] >= 'a' && p[j] <= 'z') || p[j] == '~')) j++;
        if (j == n) return n > 16 ? n : 0;
        const char *seq = p+2; size_t len = j-1;
        if (len == 4 && !memcmp(seq, "200~", 4)) t->paste = true;
        else if (len == 4 && !memcmp(seq, "201~", 4)) t->paste = false;
        else if (*seq == 'D') t->cursor = prev_char(t, t->cursor);
        else if (*seq == 'C') t->cursor = next_char(t, t->cursor);
        else if (*seq == 'A') browse(t, -1);
        else if (*seq == 'B') browse(t, +1);
        else if (*seq == 'H' || (len == 2 && seq[0] == '1')) t->cursor = 0;
        else if (*seq == 'F' || (len == 2 && seq[0] == '4')) t->cursor = t->input.n;
        else if (len == 2 && seq[0] == '3') erase(t, t->cursor, next_char(t, t->cursor));
        return j+1;
    }
    if (c == '\r') {
        if (t->paste) { insert(t, "\n", 1); return 1; }
        *event = PF_TERM_SUBMIT; *line = submit(t); return 1;
    }
    if (c == '\n') { insert(t, "\n", 1); return 1; }                        /* Ctrl+J: a newline */
    if (c == 0x7f || c == 0x08) { erase(t, prev_char(t, t->cursor), t->cursor); return 1; }
    if (c == 0x03) { *event = PF_TERM_INTERRUPT; return 1; }
    if (c == 0x04) { if (!t->input.n) *event = PF_TERM_EOF; else erase(t, t->cursor, next_char(t, t->cursor)); return 1; }
    if (c == 0x01) { t->cursor = 0; return 1; }
    if (c == 0x05) { t->cursor = t->input.n; return 1; }
    if (c == 0x15) { erase(t, 0, t->cursor); return 1; }
    if (c == 0x0b) { erase(t, t->cursor, t->input.n); return 1; }
    if (c == 0x17) {                                                       /* Ctrl+W: the previous word */
        size_t i = t->cursor;
        while (i && t->input.p[i-1] == ' ') i--;
        while (i && t->input.p[i-1] != ' ') i--;
        erase(t, i, t->cursor); return 1;
    }
    if (c == 0x0c) { layout(t); return 1; }
    if (c < 0x20) return 1;                                                /* other control keys: ignored */
    size_t len = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
    if (len > n) return 0;
    insert(t, p, len);
    return len;
}

PfTermEvent pf_term_read(PfTerm *t, char **line) {
    char chunk[4096]; ssize_t got = read(STDIN_FILENO, chunk, sizeof chunk);
    if (got > 0) buf_add(&t->pending, chunk, (size_t)got);
    PfTermEvent event = PF_TERM_NONE; size_t at = 0;
    *line = NULL;
    while (at < t->pending.n && event == PF_TERM_NONE) {
        size_t used = key(t, t->pending.p+at, t->pending.n-at, &event, line);
        if (!used) break;
        at += used;
    }
    if (at) { memmove(t->pending.p, t->pending.p+at, t->pending.n-at+1); t->pending.n -= at; }
    draw(t);
    return event;
}
