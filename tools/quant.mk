# Quantization pipeline (tools/quant/): the qw4q encoder library and its tests.
# CPU only, no model files. Same float flags as the DwarfStar goldens: -O3, no
# fast-math, so the compat encoders reproduce their rounding bit for bit.

QW4Q_CFLAGS := -O3 -std=c11 -Wall -Wextra -Werror -Isrc

build/libqw4q.dylib: tools/quant/qw4q.c tools/quant/qw4q.h src/qwen4_iq2_grid.inc
	@mkdir -p build
	$(CC) $(QW4Q_CFLAGS) -dynamiclib -fPIC -o $@ tools/quant/qw4q.c -lm -pthread

test-quant: build/libqw4q.dylib
	@$(PYTHON) tests/test_qw4q.py

.PHONY: test-quant
