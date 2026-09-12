# picoforge — plain C11, no external dependencies (CLAUDE.md principle #3).
#
# The warning set is deliberately strict. -Wconversion in particular is
# noisy in most projects, but this engine is nothing but numeric code:
# a silent int/float/size_t narrowing here is a wrong logit later.

CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
LDFLAGS := -lm

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)

picoforge: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c src/picoforge.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Differential test: the C primitives judged by the Python oracle.
tests/test_ops: tests/test_ops.c src/ops.o src/picoforge.h
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_ops.c src/ops.o $(LDFLAGS)

test: tests/test_ops
	@./tools/venv/bin/python tests/test_ops.py

# Everything, in dependency order: primitives, then the tables they feed,
# then the pipelines built on those.
test-all: test test-nfc test-tokenizer test-encode test-forward test-generate

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
	rm -f src/*.o picoforge tests/test_ops

.PHONY: clean test test-all test-forward test-tokenizer test-encode test-nfc test-generate
