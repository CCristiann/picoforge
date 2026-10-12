# Coordinator-owned integration targets. Preparation is explicit; tests are offline.
Q38_REFERENCE_ROOT ?= $(abspath ../picoforge)
Q38_CHECKPOINT ?= $(Q38_REFERENCE_ROOT)/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
Q38_DWARFSTAR ?= $(abspath ../ds4-qwen38-baseline)
Q38_SOURCE_DIR ?=
Q38_FLAGS := $(CFLAGS) -Werror -ffp-contract=off -fno-fast-math -Isrc
Q38_SAN := -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
Q38_ENV := env HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 UBSAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
Q38_TEST := $(Q38_ENV) $(PYTHON) -B tools/qwen38_test.py
Q38_LOCK := $(PYTHON) -B tools/qwen38_gpu_lock.py --timeout 60 --
Q38_SOURCE := build/qwen38-f/upstream/modeling_qwen4_exp.py
Q38_REF := build/qwen38-b/reference/reference
Q38_SLICES := $(foreach kind,0 1 8 10 16 30,--slices build/qwen38-b/slices-$(kind))
Q38_RECURRENT = src/qwen38_base.c src/qwen38_conv.c src/qwen38_gdn.c src/qwen38_config.c src/qwen38_weights.c src/qwen38_packed.c src/gguf_plan.c $(Q38_READER)
Q38_GPU = src/qwen38_recurrent_gpu.m src/qwen38_packed_gpu.m src/gpu_weights.m
Q38_READER := src/gguf.c src/gguf_reader.c src/util.c
Q38_STORAGE := src/gpu_weights.m src/gguf_plan.c $(Q38_READER)

$(filter src/qwen38%.o,$(OBJ)): CFLAGS += -ffp-contract=off -fno-fast-math
$(filter src/qwen38%.o,$(OBJ)): $(Q38_HEADERS) src/gguf.h src/qwen38_iq2_grid.inc

build/qwen38-b/packed.air: src/qwen38_packed.metal src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen38-c/recurrent.air: src/qwen38_recurrent.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

qwen38-test-assets:
	$(Q38_ENV) $(PYTHON) -B tools/qwen38_prepare_tests.py --dwarfstar "$(Q38_DWARFSTAR)" --checkpoint "$(Q38_CHECKPOINT)" $(if $(Q38_SOURCE_DIR),--source-dir "$(Q38_SOURCE_DIR)")

build/qwen38-b/test_packed: tests/test_qwen38_packed.c src/qwen38_packed.c $(Q38_READER) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

build/qwen38-b/test_packed_gpu: tests/test_qwen38_packed_gpu.m src/qwen38_packed_gpu.m src/qwen38_packed.c $(Q38_STORAGE) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-c/test_cpu: tests/test_qwen38_recurrent.c $(Q38_RECURRENT) $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

# ctypes library is unsanitized; the same sources run first in the sanitized harness.
build/qwen38-c/libcpu.dylib: $(Q38_RECURRENT) $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -dynamiclib $(filter %.c,$^) -lm -o $@

build/qwen38-c/test_gpu: tests/test_qwen38_recurrent.m $(Q38_GPU) $(Q38_RECURRENT) $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-d/test_attention: tests/test_qwen38_attention.c src/qwen38_attention.c $(Q38_RECURRENT) $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

build/qwen38-f/test_moe: tests/test_qwen38_moe.c src/qwen38_moe.c $(Q38_RECURRENT) $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

build/qwen38-integration/test_links: tests/test_qwen38_integration.c src/qwen38_config.c src/qwen38_weights.c src/qwen38_packed.c src/qwen38_base.c src/qwen38_conv.c src/qwen38_gdn.c src/qwen38_attention.c src/qwen38_ngram.c src/qwen38_ple.c src/qwen38_moe.c src/gguf_plan.c $(Q38_READER) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

build/qwen38-a/small.gguf: tools/synth/make_qwen38_a.py tools/oracle/qwen38_config.py tests/fixtures/qwen38/a/config.json
	$(Q38_ENV) $(PYTHON) -B tools/synth/make_qwen38_a.py

test-qwen38-links: build/qwen38-integration/test_links build/qwen38-a/small.gguf
	$(Q38_ENV) ./build/qwen38-integration/test_links build/qwen38-a/small.gguf

test-qwen38-a:
	$(Q38_ENV) $(PYTHON) -B tools/synth/make_qwen38_a.py --verify --checkpoint "$(Q38_CHECKPOINT)" --reference-root "$(Q38_REFERENCE_ROOT)"

test-qwen38-b-cpu: build/qwen38-b/test_packed
	@mkdir -p build/qwen38-integration
	$(Q38_TEST) tests/test_qwen38_packed.py --reference $(Q38_REF) --native $< $(Q38_SLICES) --report "$$(mktemp -d build/qwen38-integration/b-cpu.XXXXXX)/report.json" -v

test-qwen38-c-cpu: build/qwen38-c/test_cpu build/qwen38-c/libcpu.dylib
	$(Q38_ENV) ./build/qwen38-c/test_cpu
	@mkdir -p build/qwen38-integration
	$(Q38_TEST) tests/test_qwen38_recurrent.py --source $(Q38_SOURCE) --cpu-lib build/qwen38-c/libcpu.dylib --report "$$(mktemp -d build/qwen38-integration/c-cpu.XXXXXX)/report.json" -v

test-qwen38-d-cpu: build/qwen38-d/test_attention
	$(Q38_TEST) tests/test_qwen38_attention.py --source $(Q38_SOURCE) --native $< AttentionOracleTests NativeAttentionTests -v

test-qwen38-e-cpu:
	$(Q38_TEST) tests/test_qwen38_ngram.py -v

test-qwen38-f-cpu: build/qwen38-f/test_moe
	@mkdir -p build/qwen38-integration
	$(Q38_ENV) ./$<
	$(Q38_LOCK) env Q38_F_NATIVE="$(abspath $<)" Q38_F_PACKED_REFERENCE="$(abspath $(Q38_REF))" Q38_F_REPORT="$$(mktemp -d build/qwen38-integration/f-cpu.XXXXXX)/report.json" $(Q38_TEST) tests/test_qwen38_moe.py -v

test-qwen38-a-gpu: $(METALLIB)
	$(Q38_ENV) $(PYTHON) -B tools/synth/make_qwen38_a.py --verify --gpu --checkpoint "$(Q38_CHECKPOINT)" --reference-root "$(Q38_REFERENCE_ROOT)"

test-qwen38-b-gpu: build/qwen38-b/test_packed_gpu $(METALLIB)
	@mkdir -p build/qwen38-integration
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" $(Q38_TEST) tests/test_qwen38_packed.py --reference $(Q38_REF) --native $< --metal $(Q38_SLICES) --report "$$(mktemp -d build/qwen38-integration/b-gpu.XXXXXX)/report.json" -v

test-qwen38-c-gpu: build/qwen38-c/test_gpu $(METALLIB)
	$(Q38_LOCK) $(Q38_ENV) env MTL_DEBUG_LAYER=1 ./$< "$(abspath $(METALLIB))"

# Recipes, rather than parallel prerequisites, keep aggregate runs bounded.
test-qwen38-cpu:
	$(MAKE) test-qwen38-contracts
	$(Q38_TEST) tests/test_qwen38_test.py -v
	$(MAKE) test-qwen38-a
	$(MAKE) test-qwen38-b-cpu
	$(MAKE) test-qwen38-c-cpu
	$(MAKE) test-qwen38-d-cpu
	$(MAKE) test-qwen38-e-cpu
	$(MAKE) test-qwen38-f-cpu
	$(MAKE) test-qwen38-links

test-qwen38-gpu:
	$(MAKE) test-qwen38-a-gpu
	$(MAKE) test-qwen38-b-gpu
	$(MAKE) test-qwen38-c-gpu
	$(MAKE) test-qwen38-d-gpu
	$(MAKE) test-qwen38-f-gpu

test-qwen38:
	$(MAKE) test-qwen38-cpu
	$(MAKE) test-qwen38-gpu
	$(MAKE) test-qwen38-compositions

.PHONY: qwen38-test-assets test-qwen38 test-qwen38-cpu test-qwen38-gpu test-qwen38-links \
 test-qwen38-a test-qwen38-b-cpu test-qwen38-c-cpu test-qwen38-d-cpu \
 test-qwen38-e-cpu test-qwen38-f-cpu test-qwen38-a-gpu test-qwen38-b-gpu test-qwen38-c-gpu

include tools/qwen38_v2.mk
