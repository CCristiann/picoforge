# picoforge — plain C11, no external dependencies (CLAUDE.md principle #3).
#
# The warning set is deliberately strict. -Wconversion in particular is
# noisy in most projects, but this engine is nothing but numeric code:
# a silent int/float/size_t narrowing here is a wrong logit later.

CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
LDFLAGS := -lm

SRC     := $(wildcard src/*.c)
OBJC    := $(wildcard src/*.m)
OBJ     := $(SRC:.c=.o) $(OBJC:.m=.o)
METALLIB := picoforge.metallib

# Objective-C appears exactly once, for Metal, and links the frameworks.
FRAMEWORKS := -framework Metal -framework Foundation

picoforge: $(OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(FRAMEWORKS)

src/%.o: src/%.c src/picoforge.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/%.o: src/%.m src/picoforge.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Kernels compile to AIR, then link into a library loaded at runtime. Keeping
# them out of the binary means a kernel can be edited and re-tested without
# relinking the engine.
$(METALLIB): src/kernels.metal
	@mkdir -p build
	# -std=metal4.0: TensorOps and the tensor types live in Metal 4.
	xcrun -sdk macosx metal -std=metal4.0 -c $< -o build/kernels.air
	xcrun -sdk macosx metallib build/kernels.air -o $@

# Differential test: the C primitives judged by the Python oracle.
tests/test_ops: tests/test_ops.c src/ops.o src/picoforge.h
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_ops.c src/ops.o $(LDFLAGS)

test: tests/test_ops
	@./tools/venv/bin/python tests/test_ops.py

# A local UI in front of the real binary: http://127.0.0.1:8000
ui: picoforge
	@./tools/venv/bin/python tools/ui/server.py

# Everything, in dependency order: primitives, then the tables they feed,
# then the pipelines built on those.
test-all: test test-nfc test-tokenizer test-encode test-forward test-generate test-quant-formats

# Phase 3: the quantisers keep the bounds formats.py promises.
test-quant-formats:
	@./tools/venv/bin/python tests/test_quant_formats.py

test-generate: picoforge
	@./tools/venv/bin/python tests/test_generate.py

test-nfc: picoforge
	@./tools/venv/bin/python tests/test_nfc.py

test-encode: picoforge
	@./tools/venv/bin/python tests/test_encode.py

test-tokenizer: picoforge
	@./tools/venv/bin/python tests/test_tokenizer.py

# Chain of oracles, link two: the C forward pass judged by the NumPy one.
test-forward: picoforge
	@./tools/venv/bin/python tests/test_forward.py

clean:
	rm -f src/*.o picoforge tests/test_ops $(METALLIB) build/*.air

.PHONY: clean test test-all ui test-forward test-tokenizer test-encode test-nfc test-generate test-quant-formats
