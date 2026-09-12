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

clean:
	rm -f src/*.o picoforge tests/test_ops

.PHONY: clean test
