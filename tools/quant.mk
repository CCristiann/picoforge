# Quantization pipeline (tools/quant/): the qw4q encoder library and its tests.
# CPU only, no model files. Same float flags as the DwarfStar goldens: -O3, no
# fast-math, so the compat encoders reproduce their rounding bit for bit.

QW4Q_CFLAGS := -O3 -std=c11 -Wall -Wextra -Werror -Isrc

# The DwarfStar translation unit keeps its own flags (no -Werror: not our code).
build/quant-ds4_quants.o: tools/quant/third_party/ds4_quants.c tools/quant/third_party/quants.h
	@mkdir -p build
	$(CC) -O3 -std=c11 -fPIC -c -o $@ $<

build/libqw4q.dylib: build/quant-ds4_quants.o tools/quant/qw4q.c tools/quant/third_party/ds4_quants.c tools/quant/third_party/quants.h tools/quant/qw4q.h src/qwen4_iq2_grid.inc
	@mkdir -p build
	$(CC) $(QW4Q_CFLAGS) -dynamiclib -fPIC -o $@ tools/quant/qw4q.c build/quant-ds4_quants.o -lm -pthread

test-quant: build/libqw4q.dylib
	@$(PYTHON) tests/test_qw4q.py
	@$(PYTHON) tests/test_quant_io.py

.PHONY: test-quant
