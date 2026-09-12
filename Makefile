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

clean:
	rm -f src/*.o picoforge

.PHONY: clean
