/* qw4q.h — picoforge weight encoders for the routed experts (IQ2_XXS, Q2_K).
 *
 * Rows of ncols floats (a multiple of 256) become GGML blocks. `importance`
 * holds ncols values (an imatrix row: mean squared input per column) shared by
 * every row. Compat mode reproduces DwarfStar b3f5018 gguf-tools/quants.c byte
 * for byte; refine mode (O1) searches against the real decoder levels.
 * Functions return 0, or -1 for a shape or argument they cannot handle.
 */
#ifndef QW4Q_H
#define QW4Q_H

#include <stddef.h>
#include <stdint.h>

typedef enum { QW4Q_COMPAT = 0, QW4Q_REFINE = 1 } qw4q_mode;

enum { QW4Q_Q2_K = 10, QW4Q_IQ2_XXS = 16 };   /* GGML type ids */

size_t qw4q_row_bytes(int type, int64_t ncols);
int qw4q_quantize_iq2_xxs(const float *src, int64_t nrows, int64_t ncols, const float *importance,
                          qw4q_mode mode, uint8_t *dst);
int qw4q_quantize_q2_k(const float *src, int64_t nrows, int64_t ncols, const float *importance, uint8_t *dst);
int qw4q_dequantize(int type, const uint8_t *src, int64_t nrows, int64_t ncols, float *dst);

#endif
