# picoforge — plain C11, no external dependencies (docs/DESIGN.md, principle 3).
#
# The warning set is deliberately strict. -Wconversion in particular is
# noisy in most projects, but this engine is nothing but numeric code:
# a silent int/float/size_t narrowing here is a wrong logit later.

CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
LDFLAGS := -lm
PYTHON  ?= ./tools/venv/bin/python

SRC     := $(wildcard src/*.c)
OBJC    := $(wildcard src/*.m)
OBJ     := $(SRC:.c=.o) $(OBJC:.m=.m.o)
METALLIB := picoforge.metallib

# Objective-C contains Metal glue and links the platform frameworks.
FRAMEWORKS := -framework Metal -framework Foundation

picoforge: $(OBJ) $(METALLIB)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(FRAMEWORKS)

src/%.o: src/%.c src/picoforge.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/%.m.o: src/%.m src/picoforge.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Kernels compile to AIR, then link into a library loaded at runtime. Keeping
# them out of the binary means a kernel can be edited and re-tested without
# relinking the engine.
build/kernels.air: src/kernels.metal
	@mkdir -p build
	# -std=metal4.0: TensorOps and the tensor types live in Metal 4.
	xcrun -sdk macosx metal -std=metal4.0 -c $< -o build/kernels.air

$(METALLIB): build/kernels.air build/qwen38-b/packed.air build/qwen38-c/recurrent.air build/qwen38-d/attention.air build/qwen38-f/moe.air build/qwen38-e/ple.air build/qwen38-g2/forward.air build/qwen38-r1/matvec.air build/qwen38-r1/moe_fused.air
	xcrun -sdk macosx metallib $^ -o $@

build/qwen38-e/ple.air: src/qwen38_ple.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -c $< -o $@

# Differential test: the C primitives judged by the Python oracle.
tests/test_ops: tests/test_ops.c src/ops.o src/picoforge.h
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_ops.c src/ops.o $(LDFLAGS)

test: tests/test_ops
	@$(PYTHON) tests/test_ops.py

# A local UI in front of the real binary: http://127.0.0.1:8000
ui: picoforge
	@$(PYTHON) tools/ui/server.py

# Everything, in dependency order: primitives, then the tables they feed,
# then the pipelines built on those.
test-all: test test-expert-cost test-bench-speculate test-attention test-context test-nfc test-tokenizer test-encode test-forward test-generate \
          test-quant-formats test-quant-kernels test-forward-quant test-speculate test-spec-experts test-oracle-moe test-forward-moe test-forward-moe-quant test-shards test-routing-trace

build/test_attention: tests/test_attention.m src/metal.m.o src/util.o src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -o $@ tests/test_attention.m src/metal.m.o src/util.o $(LDFLAGS) $(FRAMEWORKS)

test-attention: build/test_attention $(METALLIB)
	@./build/test_attention

test-context: picoforge build/tiny-qwen3-moe
	@$(PYTHON) tests/test_context.py

test-bench-speculate:
	@$(PYTHON) tests/test_bench_speculate.py

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

build/test_gguf_reader: tests/test_gguf_reader.c src/gguf_reader.c src/util.c src/gguf.h src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_reader.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gguf_inspect: tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/util.c src/gguf.h src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/gguf_inspect.c src/gguf.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gguf_plan: tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/util.c src/gguf.h src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fsanitize=address,undefined -Isrc -o $@ tests/test_gguf_plan.c src/gguf_plan.c src/gguf_reader.c src/util.c $(LDFLAGS)

build/test_gpu_weights: tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/util.c src/gguf.h src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fobjc-arc -Isrc -o $@ tests/test_gpu_weights.m src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/util.c $(LDFLAGS) $(FRAMEWORKS)

test-gguf: build/test_gguf_reader build/test_gguf_inspect build/test_gguf_plan
	@$(PYTHON) tests/test_gguf.py

test-gpu-weights: build/test_gpu_weights
	@MTL_DEBUG_LAYER=1 ./build/test_gpu_weights
	@$(PYTHON) tests/test_gpu_weights.py

test-all: test-qwen38-tools test-gguf test-gpu-weights

.PHONY: test-qwen38-tools test-gguf test-gpu-weights

build/test_expert_cost: tests/test_expert_cost.c src/expert_cost.o src/config.o src/json.o src/util.o src/picoforge.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_expert_cost.c src/expert_cost.o src/config.o src/json.o src/util.o $(LDFLAGS)

test-expert-cost: build/test_expert_cost
	@./build/test_expert_cost

test-spec-experts: picoforge build/tiny-qwen3-moe
	@$(PYTHON) tests/test_spec_experts.py

# Phase 4: a tiny random Qwen3-MoE (same architecture and on-disk layout as
# Qwen3-30B-A3B, 3 layers), and the NumPy oracle verified against it.
build/tiny-qwen3-moe: tools/oracle/make_tiny_moe.py
	@$(PYTHON) tools/oracle/make_tiny_moe.py $@

test-oracle-moe: build/tiny-qwen3-moe
	@$(PYTHON) tools/oracle/verify.py build/tiny-qwen3-moe

# One Qwen3-30B-A3B layer at its real shapes, random weights, for timing.
build/synth-30b-layer: tools/synth/make_moe_layer.py
	@$(PYTHON) tools/synth/make_moe_layer.py $@

build/synth-30b-layer-q8_row: build/synth-30b-layer tools/quant/quantize.py tools/quant/formats.py
	@$(PYTHON) tools/quant/quantize.py build/synth-30b-layer q8_row

build/tiny-qwen3-moe-sharded: build/tiny-qwen3-moe tools/synth/shard_checkpoint.py
	@$(PYTHON) tools/synth/shard_checkpoint.py build/tiny-qwen3-moe $@ 3

test-shards: picoforge build/tiny-qwen3-moe-sharded
	@$(PYTHON) tests/test_shards.py

test-routing-trace: picoforge build/tiny-qwen3-moe
	@$(PYTHON) tests/test_routing_trace.py

test-forward-moe: picoforge build/tiny-qwen3-moe
	@$(PYTHON) tests/test_forward.py build/tiny-qwen3-moe

build/tiny-qwen3-moe-q8_row: build/tiny-qwen3-moe tools/quant/quantize.py tools/quant/formats.py
	@$(PYTHON) tools/quant/quantize.py build/tiny-qwen3-moe q8_row

test-forward-moe-quant: picoforge build/tiny-qwen3-moe-q8_row
	@$(PYTHON) tests/test_forward_quant.py build/tiny-qwen3-moe-q8_row

# Phase 4: speculative decoding emits exactly what plain greedy emits.
test-speculate: picoforge quant-models build/tiny-qwen3-moe
	@$(PYTHON) tests/test_speculate.py

# Phase 3: the quantisers keep the bounds formats.py promises.
test-quant-formats:
	@$(PYTHON) tests/test_quant_formats.py

# Quantised checkpoints, rebuilt when the quantiser changes. Weights stay out
# of git like the originals; these are derived from them in seconds.
QMODELS := models/Qwen3-0.6B-q8_row-embed models/Qwen3-0.6B-q4_g32 models/Qwen3-0.6B-q4_g32-embed
models/Qwen3-0.6B-%-embed: tools/quant/quantize.py tools/quant/formats.py
	@$(PYTHON) tools/quant/quantize.py models/Qwen3-0.6B $* --embed
models/Qwen3-0.6B-%: tools/quant/quantize.py tools/quant/formats.py
	@$(PYTHON) tools/quant/quantize.py models/Qwen3-0.6B $*
quant-models: $(QMODELS)

# Chain of oracles, Phase 3: the quantised engine against NumPy on the
# dequantised weights, and the quantised kernels against the CPU reference.
test-forward-quant: picoforge quant-models
	@$(PYTHON) tests/test_forward_quant.py $(QMODELS)

# The engine's exit status is the verdict; a pipe into sed would swallow it.
test-quant-kernels: picoforge
	@mkdir -p build
	@./picoforge models/Qwen3-0.6B --quant-check > build/quant_check.log; s=$$?; \
	 sed -n '/quantised kernels/,$$p' build/quant_check.log; exit $$s

test-generate: picoforge
	@$(PYTHON) tests/test_generate.py

test-nfc: picoforge
	@$(PYTHON) tests/test_nfc.py

test-encode: picoforge
	@$(PYTHON) tests/test_encode.py

test-tokenizer: picoforge
	@$(PYTHON) tests/test_tokenizer.py

# Chain of oracles, link two: the C forward pass judged by the NumPy one.
test-forward: picoforge
	@$(PYTHON) tests/test_forward.py

clean:
	rm -f src/*.o picoforge tests/test_ops $(METALLIB) build/*.air

.PHONY: clean test test-all test-expert-cost test-spec-experts test-bench-speculate test-attention test-context ui test-forward test-tokenizer test-encode test-nfc test-generate \
        test-quant-formats test-quant-kernels test-forward-quant quant-models

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

build/qwen38-g2/forward.air: src/qwen38_forward.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -c $< -o $@
