# Qwen3.8 integration targets. Preparation is explicit; tests are offline.
QW4_REFERENCE_ROOT ?= $(abspath ../picoforge)
QW4_CHECKPOINT ?= $(QW4_REFERENCE_ROOT)/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
QW4_DWARFSTAR ?= $(abspath ../ds4-qwen38-baseline)
QW4_SOURCE_DIR ?=
QW4_FLAGS := $(CFLAGS) -Werror -ffp-contract=off -fno-fast-math -Isrc
QW4_SAN := -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
# Synthetic oracles: no system message, and the pre-Perf-III GEMM admission (tiny shapes).
QW4_ENV := env PICOFORGE_SYSTEM= PICOFORGE_GEMM_MIN_TILES=0 PICOFORGE_GROUPED_MIN_PAIRS=256 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 UBSAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
QW4_TEST := $(QW4_ENV) $(PYTHON) -B tools/qwen4_test.py
QW4_LOCK := $(PYTHON) -B tools/qwen4_gpu_lock.py --timeout 60 --
QW4_SOURCE := build/qwen4-f/upstream/modeling_qwen4_exp.py
QW4_REF := build/qwen4-b/reference/reference
QW4_SLICES := $(foreach kind,0 1 8 10 16 30,--slices build/qwen4-b/slices-$(kind))
QW4_RECURRENT = src/qwen4_base.c src/qwen4_conv.c src/qwen4_gdn.c src/qwen4_config.c src/qwen4_weights.c src/qwen4_packed.c src/gguf_plan.c $(QW4_READER)
QW4_GPU = src/qwen4_recurrent_gpu.m src/qwen4_packed_gpu.m src/gpu_weights.m
QW4_READER := src/gguf.c src/gguf_reader.c src/common.c
QW4_STORAGE := src/gpu_weights.m src/gguf_plan.c $(QW4_READER)

$(filter src/qwen4%.o,$(OBJ)): CFLAGS += -ffp-contract=off -fno-fast-math
$(filter src/qwen4%.o,$(OBJ)): $(QW4_HEADERS) src/gguf.h src/qwen4_iq2_grid.inc

build/qwen4-b/packed.air: src/qwen4_packed.metal src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen4-c/recurrent.air: src/qwen4_recurrent.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

qwen4-test-assets:
	$(QW4_ENV) $(PYTHON) -B tools/qwen4_prepare_tests.py --dwarfstar "$(QW4_DWARFSTAR)" --checkpoint "$(QW4_CHECKPOINT)" $(if $(QW4_SOURCE_DIR),--source-dir "$(QW4_SOURCE_DIR)")

build/qwen4-b/test_packed: tests/test_qwen4_packed.c src/qwen4_packed.c $(QW4_READER) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

build/qwen4-b/test_packed_gpu: tests/test_qwen4_packed_gpu.m src/qwen4_packed_gpu.m src/qwen4_packed.c $(QW4_STORAGE) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-c/test_cpu: tests/test_qwen4_recurrent.c $(QW4_RECURRENT) $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

# ctypes library is unsanitized; the same sources run first in the sanitized harness.
build/qwen4-c/libcpu.dylib: $(QW4_RECURRENT) $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) -dynamiclib $(filter %.c,$^) -lm -o $@

build/qwen4-c/test_gpu: tests/test_qwen4_recurrent.m $(QW4_GPU) $(QW4_RECURRENT) $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-d/test_attention: tests/test_qwen4_attention.c src/qwen4_attention.c $(QW4_RECURRENT) $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

build/qwen4-f/test_moe: tests/test_qwen4_moe.c src/qwen4_moe.c $(QW4_RECURRENT) $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

build/qwen4-integration/test_links: tests/test_qwen4_integration.c src/qwen4_config.c src/qwen4_weights.c src/qwen4_packed.c src/qwen4_base.c src/qwen4_conv.c src/qwen4_gdn.c src/qwen4_attention.c src/qwen4_ngram.c src/qwen4_ple.c src/qwen4_moe.c src/gguf_plan.c $(QW4_READER) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

build/qwen4-a/small.gguf: tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py tests/fixtures/qwen4/text/config.json
	$(QW4_ENV) $(PYTHON) -B tools/synth/make_qwen4_small.py

test-qwen4-links: build/qwen4-integration/test_links build/qwen4-a/small.gguf
	$(QW4_ENV) ./build/qwen4-integration/test_links build/qwen4-a/small.gguf

test-qwen4-a:
	$(QW4_ENV) $(PYTHON) -B tools/synth/make_qwen4_small.py --verify --checkpoint "$(QW4_CHECKPOINT)" --reference-root "$(QW4_REFERENCE_ROOT)"

test-qwen4-b-cpu: build/qwen4-b/test_packed
	@mkdir -p build/qwen4-integration
	$(QW4_TEST) tests/test_qwen4_packed.py --reference $(QW4_REF) --native $< $(QW4_SLICES) --report "$$(mktemp -d build/qwen4-integration/b-cpu.XXXXXX)/report.json" -v

test-qwen4-c-cpu: build/qwen4-c/test_cpu build/qwen4-c/libcpu.dylib
	$(QW4_ENV) ./build/qwen4-c/test_cpu
	@mkdir -p build/qwen4-integration
	$(QW4_TEST) tests/test_qwen4_recurrent.py --source $(QW4_SOURCE) --cpu-lib build/qwen4-c/libcpu.dylib --report "$$(mktemp -d build/qwen4-integration/c-cpu.XXXXXX)/report.json" -v

test-qwen4-d-cpu: build/qwen4-d/test_attention
	$(QW4_TEST) tests/test_qwen4_attention.py --source $(QW4_SOURCE) --native $< AttentionOracleTests NativeAttentionTests -v

test-qwen4-e-cpu:
	$(QW4_TEST) tests/test_qwen4_ngram.py -v

test-qwen4-f-cpu: build/qwen4-f/test_moe
	@mkdir -p build/qwen4-integration
	$(QW4_ENV) ./$<
	$(QW4_LOCK) env QW4_F_NATIVE="$(abspath $<)" QW4_F_PACKED_REFERENCE="$(abspath $(QW4_REF))" QW4_F_REPORT="$$(mktemp -d build/qwen4-integration/f-cpu.XXXXXX)/report.json" $(QW4_TEST) tests/test_qwen4_moe.py -v

test-qwen4-a-gpu: $(METALLIB)
	$(QW4_ENV) $(PYTHON) -B tools/synth/make_qwen4_small.py --verify --gpu --checkpoint "$(QW4_CHECKPOINT)" --reference-root "$(QW4_REFERENCE_ROOT)"

test-qwen4-b-gpu: build/qwen4-b/test_packed_gpu $(METALLIB)
	@mkdir -p build/qwen4-integration
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 QW4_PACKED_LIBRARY="$(abspath $(METALLIB))" $(QW4_TEST) tests/test_qwen4_packed.py --reference $(QW4_REF) --native $< --metal $(QW4_SLICES) --report "$$(mktemp -d build/qwen4-integration/b-gpu.XXXXXX)/report.json" -v

test-qwen4-c-gpu: build/qwen4-c/test_gpu $(METALLIB)
	$(QW4_LOCK) $(QW4_ENV) env MTL_DEBUG_LAYER=1 ./$< "$(abspath $(METALLIB))"

# Recipes, rather than parallel prerequisites, keep aggregate runs bounded.
test-qwen4-cpu:
	$(MAKE) test-qwen4-contracts
	$(QW4_TEST) tests/test_qwen4_test.py -v
	$(MAKE) test-qwen4-a
	$(MAKE) test-qwen4-b-cpu
	$(MAKE) test-qwen4-c-cpu
	$(MAKE) test-qwen4-d-cpu
	$(MAKE) test-qwen4-e-cpu
	$(MAKE) test-qwen4-f-cpu
	$(MAKE) test-qwen4-links

test-qwen4-gpu:
	$(MAKE) test-qwen4-a-gpu
	$(MAKE) test-qwen4-b-gpu
	$(MAKE) test-qwen4-c-gpu
	$(MAKE) test-qwen4-d-gpu
	$(MAKE) test-qwen4-f-gpu
	$(MAKE) test-qwen4-g0

test-qwen4:
	$(MAKE) test-qwen4-cpu
	$(MAKE) test-qwen4-gpu
	$(MAKE) test-qwen4-compositions
	$(MAKE) test-qwen4-g1
	$(MAKE) test-qwen4-g2-parts
	$(MAKE) test-qwen4-g2
	$(MAKE) test-qwen4-g2-state
	$(MAKE) test-qwen4-cli
	$(MAKE) test-quant-engine
	$(MAKE) test-qwen4-repl
	$(MAKE) test-qwen4-variants

.PHONY: qwen4-test-assets test-qwen4 test-qwen4-cpu test-qwen4-gpu test-qwen4-links \
 test-qwen4-a test-qwen4-b-cpu test-qwen4-c-cpu test-qwen4-d-cpu \
 test-qwen4-e-cpu test-qwen4-f-cpu test-qwen4-a-gpu test-qwen4-b-gpu test-qwen4-c-gpu

include tools/qwen4_components.mk

include tools/qwen4_graph.mk

include tools/qwen4_perf.mk
