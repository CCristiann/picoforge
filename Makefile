# picoforge — plain C11, no external dependencies (docs/DESIGN.md, principle 3).
#
# The warning set is deliberately strict. -Wconversion in particular is
# noisy in most projects, but this engine is nothing but numeric code:
# a silent int/float/size_t narrowing here is a wrong logit later.

CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
LDFLAGS := -lm
PYTHON  ?= ./tools/venv/bin/python

SRC     := $(filter-out src/pf_%.c,$(wildcard src/*.c))
OBJC    := $(wildcard src/*.m)
OBJ     := $(SRC:.c=.o) $(OBJC:.m=.m.o)
METALLIB := picoforge.metallib

# Objective-C contains Metal glue and links the platform frameworks.
FRAMEWORKS := -framework Metal -framework Foundation

picoforge: $(OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(FRAMEWORKS)

# pf-agent: the coding agent, linked with the same engine objects minus
# picoforge's main. pf_session/pf_tools/pf_web/pf_util are meant for a future
# pf-server too; pf_agent.c is only the terminal interface.
ENGINE_OBJ := $(filter-out src/main.o,$(OBJ))
PF_OBJ     := src/pf_util.o src/pf_web.o src/pf_tools.o src/pf_session.o
pf-agent: src/pf_agent.o $(PF_OBJ) $(ENGINE_OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ src/pf_agent.o $(PF_OBJ) $(ENGINE_OBJ) $(LDFLAGS) $(FRAMEWORKS)

# Every object depends on every header: a constant changed only in a header (a
# Metal wire struct, a tile size) must rebuild the host code that mirrors it.
HEADERS := $(wildcard src/*.h)

src/%.o: src/%.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

src/%.m.o: src/%.m $(HEADERS)
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Kernels compile to AIR, then link into a library loaded at runtime. Keeping
# them out of the binary means a kernel can be edited and re-tested without
# relinking the engine. The per-stage rules live in tools/qwen38*.mk.
# -std=metal4.0: TensorOps and the tensor types live in Metal 4.
$(METALLIB): build/qwen38-b/packed.air build/qwen38-c/recurrent.air build/qwen38-d/attention.air build/qwen38-f/moe.air build/qwen38-e/ple.air build/qwen38-g2/forward.air build/qwen38-r1/matvec.air build/qwen38-r1/moe_fused.air
	xcrun -sdk macosx metallib $^ -o $@

build/qwen38-e/ple.air: src/qwen38_ple.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -c $< -o $@

build/qwen38-g2/forward.air: src/qwen38_forward.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -c $< -o $@

# ------------------------------------------------------------ microbenchmark
# One matmul kernel in three implementations (naive, simdgroup, TensorOps) plus
# the dispatch cost. Model-agnostic: it measures the machine. See
# tools/microbench/main.c and docs/BENCHMARKS.md.
build/microbench.metallib: tools/microbench/kernels.metal
	@mkdir -p build
	xcrun -sdk macosx metal -std=metal4.0 -c $< -o build/microbench.air
	xcrun -sdk macosx metallib build/microbench.air -o $@

build/microbench: tools/microbench/main.c tools/microbench/microbench.c tools/microbench/metal.m tools/microbench/microbench.h src/util.c src/common.h build/microbench.metallib
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -Itools/microbench -o $@ $(filter %.c %.m,$^) $(LDFLAGS) $(FRAMEWORKS)

microbench: build/microbench

# ------------------------------------------------------------------- tests
# Offline checks: no checkpoint, no long GPU workload. The full Qwen3.8 gate
# is `make test-qwen38` (tools/qwen38.mk).
test-all: test-qwen38-tools test-gguf test-gpu-weights test-qwen38-contracts test-qwen38-expert-pool test-qwen38-sample test-pf-agent

# Phase 0 tools for the Qwen3.8 target do not link DwarfStar into our engine.
build/process_memory: tools/probe/process_memory.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $<

build/memory_budget: tools/probe/memory_budget.m
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -o $@ $< $(FRAMEWORKS)

test-qwen38-tools: build/process_memory build/memory_budget
	@$(PYTHON) tests/test_qwen38_inventory.py
	@$(PYTHON) tests/test_run_observed.py
	@$(PYTHON) tests/test_qwen38_grading.py
	@$(PYTHON) tests/test_qwen38_evidence.py

src/gguf.o src/gguf_reader.o src/gguf_plan.o src/gpu_weights.m.o src/main.o: src/gguf.h
src/main.o: src/qwen38_cli.h
src/qwen38_cli.m.o src/qwen38_repl.o src/qwen38_bench.o: src/qwen38_run.h src/qwen38_cli.h src/qwen38_forward_gpu.h src/qwen38_g_internal.h src/qwen38_text.h

build/test_gguf_reader: tests/test_gguf_reader.c src/gguf_reader.c src/util.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_reader.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gguf_inspect: tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/util.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gguf_plan: tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/util.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gpu_weights: tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/util.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -o $@ tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/util.c $(LDFLAGS) $(FRAMEWORKS)

test-gguf: build/test_gguf_reader build/test_gguf_inspect build/test_gguf_plan
	@$(PYTHON) tests/test_gguf.py

test-gpu-weights: build/test_gpu_weights
	@MTL_DEBUG_LAYER=1 ./build/test_gpu_weights
	@$(PYTHON) tests/test_gpu_weights.py

build/test_expert_pool: tests/test_qwen38_expert_pool.c src/qwen38_expert_pool.c src/qwen38_expert_pool.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Werror -g -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -o $@ tests/test_qwen38_expert_pool.c src/qwen38_expert_pool.c

test-qwen38-expert-pool: build/test_expert_pool
	@./build/test_expert_pool

build/test_sample: tests/test_qwen38_sample.c src/qwen38_sample.c src/qwen38_sample.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Werror -g -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -o $@ tests/test_qwen38_sample.c src/qwen38_sample.c -lm
test-qwen38-sample: build/test_sample
	@./build/test_sample

# pf-agent tools and parser, offline (no model); the loop on the synthetic
# checkpoint (GPU, under the lock) is test-pf-agent-model.
test-pf-agent: pf-agent
	@$(PYTHON) tests/test_pf_agent.py

test-pf-agent-model: pf-agent build/qwen38-g2/cli.gguf
	$(Q38_LOCK) env PF_AGENT_GGUF=build/qwen38-g2/cli.gguf $(Q38_TEST) tests/test_pf_agent.py -v
.PHONY: test-pf-agent test-pf-agent-model

clean:
	rm -f src/*.o picoforge $(METALLIB) build/*.air

.PHONY: clean microbench test-qwen38-expert-pool test-qwen38-sample test-all test-qwen38-tools test-gguf test-gpu-weights

# Contract declarations and CPU-only lock checks; no model or GPU workload.
# Metal-only headers (ruling R3: decode helpers of the fast kernels) are not C
# contracts; their .air rules list them as prerequisites instead.
Q38_METAL_HEADERS := src/qwen38_quant_decode.h
Q38_HEADERS := $(filter-out $(Q38_METAL_HEADERS),$(wildcard src/qwen38*.h))
Q38_PUBLIC_HEADERS := $(filter-out %_internal.h,$(Q38_HEADERS))
src/gpu_weights.m.o build/test_gpu_weights: $(Q38_HEADERS)
test-qwen38-contracts:
	@for header in $(Q38_PUBLIC_HEADERS); do $(CC) $(CFLAGS) -Werror -x c -fsyntax-only $$header || exit; done
	@printf '%s\n' $(foreach h,$(notdir $(Q38_PUBLIC_HEADERS)),'#include "$(h)"') | $(CC) $(CFLAGS) -Werror -Isrc -x objective-c -fsyntax-only -
	@$(PYTHON) tests/test_qwen38_gpu_lock.py
.PHONY: test-qwen38-contracts

include tools/qwen38.mk
