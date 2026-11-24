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

# The commands are pf (chat, one answer, models), pf-bench, pf-agent and pf-server; the
# kernels stay in picoforge.metallib.
all: pf pf-bench pf-agent pf-server

# src/qwen4_* and the gguf/gpu files are the engine; src/pf_* the commands around it. pf-bench is
# pf itself under another name (src/pf.c looks at argv[0]).
pf pf-bench: $(OBJ) src/pf_models.o $(METALLIB)
	$(CC) $(CFLAGS) -o $@ $(OBJ) src/pf_models.o $(LDFLAGS) $(FRAMEWORKS)

# pf-agent: the coding agent, linked with the same engine objects minus
# pf's main. pf_session/pf_tools/pf_web/pf_util are shared with pf-server;
# pf_agent.c is only the terminal interface.
ENGINE_OBJ := $(filter-out src/pf.o,$(OBJ))
PF_OBJ     := src/pf_models.o src/pf_util.o src/pf_web.o src/pf_tools.o src/pf_session.o src/pf_term.o
pf-agent: src/pf_agent.o $(PF_OBJ) $(ENGINE_OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ src/pf_agent.o $(PF_OBJ) $(ENGINE_OBJ) $(LDFLAGS) $(FRAMEWORKS)
# pf-server: the HTTP API (OpenAI and Anthropic formats) for coding agents.
SERVER_OBJ := src/pf_json.o src/pf_http.o src/pf_api.o src/pf_cache.o
pf-server: src/pf_server.o $(SERVER_OBJ) $(PF_OBJ) $(ENGINE_OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ src/pf_server.o $(SERVER_OBJ) $(PF_OBJ) $(ENGINE_OBJ) $(LDFLAGS) $(FRAMEWORKS)

# Every object depends on every header: a constant changed only in a header (a
# Metal wire struct, a tile size) must rebuild the host code that mirrors it.
HEADERS := $(wildcard src/*.h)

src/%.o: src/%.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

src/%.m.o: src/%.m $(HEADERS)
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Kernels compile to AIR, then link into a library loaded at runtime. Keeping
# them out of the binary means a kernel can be edited and re-tested without
# relinking the engine. The per-stage rules live in tools/qwen4*.mk.
# -std=metal4.0: TensorOps and the tensor types live in Metal 4.
$(METALLIB): build/qwen4-b/packed.air build/qwen4-c/recurrent.air build/qwen4-d/attention.air build/qwen4-f/moe.air build/qwen4-e/ple.air build/qwen4-g2/forward.air build/qwen4-r1/matvec.air build/qwen4-r1/moe_fused.air
	xcrun -sdk macosx metallib $^ -o $@

build/qwen4-e/ple.air: src/qwen4_ple.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -c $< -o $@

build/qwen4-g2/forward.air: src/qwen4_forward.metal
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

build/microbench: tools/microbench/main.c tools/microbench/microbench.c tools/microbench/metal.m tools/microbench/microbench.h src/common.c src/common.h build/microbench.metallib
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -Itools/microbench -o $@ $(filter %.c %.m,$^) $(LDFLAGS) $(FRAMEWORKS)

microbench: build/microbench

# ------------------------------------------------------------------- tests
# Offline checks: no checkpoint, no long GPU workload. The full Qwen3.8 gate
# is `make test-qwen4` (tools/qwen4.mk).
test-all: test-quant test-qwen4-tools test-gguf test-gpu-weights test-qwen4-contracts test-qwen4-expert-pool test-qwen4-sample test-pf-models test-pf-agent

# Phase 0 tools for the Qwen3.8 target do not link DwarfStar into our engine.
build/process_memory: tools/probe/process_memory.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $<

build/memory_budget: tools/probe/memory_budget.m
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -o $@ $< $(FRAMEWORKS)

test-qwen4-tools: build/process_memory build/memory_budget
	@$(PYTHON) tests/test_qwen4_inventory.py
	@$(PYTHON) tests/test_run_observed.py
	@$(PYTHON) tests/test_qwen4_grading.py
	@$(PYTHON) tests/test_qwen4_evidence.py

src/gguf.o src/gguf_reader.o src/gguf_plan.o src/gpu_weights.m.o src/pf.o: src/gguf.h
src/pf.o src/qwen4_run.m.o src/qwen4_repl.o src/qwen4_bench.o: src/qwen4_run.h src/qwen4_forward_gpu.h src/qwen4_graph_internal.h src/qwen4_text.h

build/test_gguf_reader: tests/test_gguf_reader.c src/gguf_reader.c src/common.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_reader.c src/gguf_reader.c src/common.c $(LDFLAGS)

build/test_gguf_inspect: tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/common.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/common.c $(LDFLAGS)

build/test_gguf_plan: tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/common.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/common.c $(LDFLAGS)

build/test_gpu_weights: tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/common.c src/gguf.h src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -o $@ tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/common.c $(LDFLAGS) $(FRAMEWORKS)

test-gguf: build/test_gguf_reader build/test_gguf_inspect build/test_gguf_plan
	@$(PYTHON) tests/test_gguf.py

test-gpu-weights: build/test_gpu_weights
	@MTL_DEBUG_LAYER=1 ./build/test_gpu_weights
	@$(PYTHON) tests/test_gpu_weights.py

build/test_expert_pool: tests/test_qwen4_expert_pool.c src/qwen4_expert_pool.c src/qwen4_expert_pool.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Werror -g -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -o $@ tests/test_qwen4_expert_pool.c src/qwen4_expert_pool.c

test-qwen4-expert-pool: build/test_expert_pool
	@./build/test_expert_pool

build/test_sample: tests/test_qwen4_sample.c src/qwen4_sample.c src/qwen4_sample.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Werror -g -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -o $@ tests/test_qwen4_sample.c src/qwen4_sample.c -lm
test-qwen4-sample: build/test_sample
	@./build/test_sample

build/test_pf_models: tests/test_pf_models.c src/pf_models.c src/pf_models.h src/common.c src/common.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Werror -g -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -o $@ tests/test_pf_models.c src/pf_models.c src/common.c $(LDFLAGS)
test-pf-models: build/test_pf_models
	@./build/test_pf_models

# pf-agent tools and parser, offline (no model); the loop on the synthetic
# checkpoint (GPU, under the lock) is test-pf-agent-model.
test-pf-agent: pf-agent
	@$(PYTHON) tests/test_pf_agent.py

test-pf-agent-model: pf-agent build/qwen4-g2/cli.gguf
	$(QW4_LOCK) env PF_AGENT_GGUF=build/qwen4-g2/cli.gguf $(QW4_TEST) tests/test_pf_agent.py -v
.PHONY: test-pf-agent test-pf-agent-model

# pf-server: both APIs and the prompt cache on the synthetic checkpoint (GPU, under the lock),
# then on its mixed-expert copy (the layout of the Code5/Code10 files).
test-pf-server-model: pf-server build/qwen4-g2/cli.gguf build/qwen4-g2/cli-mixed.gguf
	$(QW4_LOCK) env PF_SERVER_GGUF=build/qwen4-g2/cli.gguf $(QW4_TEST) tests/test_pf_server.py -v
	$(QW4_LOCK) env PF_SERVER_GGUF=build/qwen4-g2/cli-mixed.gguf $(QW4_TEST) tests/test_pf_server.py -v
.PHONY: test-pf-server-model

clean:
	rm -f src/*.o pf pf-bench pf-agent pf-server $(METALLIB) build/*.air

.PHONY: clean microbench test-qwen4-expert-pool test-qwen4-sample test-pf-models test-all test-qwen4-tools test-gguf test-gpu-weights

# Contract declarations and CPU-only lock checks; no model or GPU workload.
# Metal-only headers (ruling R3: decode helpers of the fast kernels) are not C
# contracts; their .air rules list them as prerequisites instead.
QW4_METAL_HEADERS := src/qwen4_quant_decode.h
QW4_HEADERS := $(filter-out $(QW4_METAL_HEADERS),$(wildcard src/qwen4*.h))
QW4_PUBLIC_HEADERS := $(filter-out %_internal.h,$(QW4_HEADERS))
src/gpu_weights.m.o build/test_gpu_weights: $(QW4_HEADERS)
test-qwen4-contracts:
	@for header in $(QW4_PUBLIC_HEADERS); do $(CC) $(CFLAGS) -Werror -x c -fsyntax-only $$header || exit; done
	@printf '%s\n' $(foreach h,$(notdir $(QW4_PUBLIC_HEADERS)),'#include "$(h)"') | $(CC) $(CFLAGS) -Werror -Isrc -x objective-c -fsyntax-only -
	@$(PYTHON) tests/test_qwen4_gpu_lock.py
.PHONY: test-qwen4-contracts

include tools/qwen4.mk
include tools/quant.mk
