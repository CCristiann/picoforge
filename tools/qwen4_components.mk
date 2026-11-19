# Attention and MoE kernels, their GPU component tests and the real-checkpoint
# compositions. Included by tools/qwen4.mk. CPU selection does not run Metal classes.
build/qwen4-d/attention.air: src/qwen4_attention.metal src/qwen4_attention_internal.h
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen4-f/moe.air: src/qwen4_moe.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen4-d/test_attention_gpu: tests/test_qwen4_attention.m src/qwen4_attention.c src/qwen4_attention_gpu.m $(QW4_RECURRENT) $(QW4_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-f/test_moe_gpu: tests/test_qwen4_moe.m tests/test_qwen4_moe.c src/qwen4_moe.c src/qwen4_moe.m $(QW4_RECURRENT) $(QW4_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter-out tests/test_qwen4_moe.c,$(filter %.c %.m,$^)) $(FRAMEWORKS) -lm -o $@

build/qwen4-e/test_ngram: tests/test_qwen4_ngram.c src/qwen4_ngram.c src/qwen4_ple.c $(QW4_RECURRENT) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) $(filter %.c,$^) -lm -o $@

qwen4-composition-assets: build/qwen4-c/test_cpu build/qwen4-d/test_attention build/qwen4-f/test_moe
	$(QW4_LOCK) $(QW4_ENV) $(PYTHON) -B tools/qwen4_prepare_compositions.py --checkpoint "$(QW4_CHECKPOINT)"

test-qwen4-d-gpu: build/qwen4-d/test_attention_gpu build/qwen4-d/test_attention $(METALLIB)
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_attention.py --source $(QW4_SOURCE) --native build/qwen4-d/test_attention --metal $< --metal-library "$(abspath $(METALLIB))" --slice-fixture build/qwen4-d/slice-01 MetalAttentionTests -v

test-qwen4-f-gpu: build/qwen4-f/test_moe_gpu $(METALLIB)
	@mkdir -p build/qwen4-integration
	@report_dir=$$(mktemp -d build/qwen4-integration/f-gpu.XXXXXX); \
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 QW4_F_NATIVE="$(abspath $<)" QW4_F_METAL=1 QW4_F_LIBRARY="$(abspath $(METALLIB))" QW4_F_PACKED_REFERENCE="$(abspath $(QW4_REF))" QW4_F_REPORT="$$report_dir/report.json" QW4_F_ALLOCATIONS="$$report_dir/allocations.jsonl" $(QW4_TEST) tests/test_qwen4_moe.py \
	NativeForwardTests.test_float_composed_stages_chunks_and_shared NativeForwardTests.test_float_nonfinite_and_zero_input NativeForwardTests.test_packed_composition_and_640_768_padding NativeForwardTests.test_native_tied_block_and_near_tie_margins

test-qwen4-compositions: build/qwen4-d/test_attention build/qwen4-e/test_ngram build/qwen4-c/test_cpu build/qwen4-c/libcpu.dylib build/qwen4-c/test_gpu $(METALLIB)
	$(QW4_LOCK) $(QW4_ENV) env MTL_DEBUG_LAYER=1 $(PYTHON) -B tools/qwen4_compositions.py --checkpoint "$(QW4_CHECKPOINT)" --metallib "$(abspath $(METALLIB))"

.PHONY: qwen4-composition-assets test-qwen4-d-gpu test-qwen4-f-gpu test-qwen4-compositions

# Every harness using C composition also compiles the packed/reader dependencies.
build/qwen4-c/test_cpu build/qwen4-c/libcpu.dylib build/qwen4-c/test_gpu \
build/qwen4-d/test_attention build/qwen4-d/test_attention_gpu \
build/qwen4-e/test_ngram build/qwen4-f/test_moe build/qwen4-f/test_moe_gpu: src/gguf.h src/qwen4_iq2_grid.inc
