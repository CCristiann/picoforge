# Coordinator-owned v2 compositions. CPU selection does not run Metal classes.
build/qwen38-d/attention.air: src/qwen38_attention.metal src/qwen38_d_internal.h
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen38-f/moe.air: src/qwen38_moe.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

build/qwen38-d/test_attention_gpu: tests/test_qwen38_attention.m src/qwen38_attention.c src/qwen38_attention_gpu.m $(Q38_RECURRENT) $(Q38_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-f/test_moe_gpu: tests/test_qwen38_moe.m tests/test_qwen38_moe.c src/qwen38_moe.c src/qwen38_moe.m $(Q38_RECURRENT) $(Q38_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter-out tests/test_qwen38_moe.c,$(filter %.c %.m,$^)) $(FRAMEWORKS) -lm -o $@

build/qwen38-e/test_ngram: tests/test_qwen38_ngram.c src/qwen38_ngram.c src/qwen38_ple.c $(Q38_RECURRENT) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) $(filter %.c,$^) -lm -o $@

qwen38-composition-assets: build/qwen38-c/test_cpu build/qwen38-d/test_attention build/qwen38-f/test_moe
	$(Q38_LOCK) $(Q38_ENV) $(PYTHON) -B tools/qwen38_prepare_compositions.py --checkpoint "$(Q38_CHECKPOINT)"

test-qwen38-d-gpu: build/qwen38-d/test_attention_gpu build/qwen38-d/test_attention $(METALLIB)
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 $(Q38_TEST) tests/test_qwen38_attention.py --source $(Q38_SOURCE) --native build/qwen38-d/test_attention --metal $< --metal-library "$(abspath $(METALLIB))" --slice-fixture build/qwen38-d/slice-01 MetalAttentionTests -v

test-qwen38-f-gpu: build/qwen38-f/test_moe_gpu $(METALLIB)
	@mkdir -p build/qwen38-integration
	@report_dir=$$(mktemp -d build/qwen38-integration/f-gpu.XXXXXX); \
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 Q38_F_NATIVE="$(abspath $<)" Q38_F_METAL=1 Q38_F_LIBRARY="$(abspath $(METALLIB))" Q38_F_PACKED_REFERENCE="$(abspath $(Q38_REF))" Q38_F_REPORT="$$report_dir/report.json" Q38_F_ALLOCATIONS="$$report_dir/allocations.jsonl" $(Q38_TEST) tests/test_qwen38_moe.py \
	NativeForwardTests.test_float_composed_stages_chunks_and_shared NativeForwardTests.test_float_nonfinite_and_zero_input NativeForwardTests.test_packed_composition_and_640_768_padding NativeForwardTests.test_native_tied_block_and_near_tie_margins

test-qwen38-compositions: build/qwen38-d/test_attention build/qwen38-e/test_ngram build/qwen38-c/test_cpu build/qwen38-c/libcpu.dylib build/qwen38-c/test_gpu $(METALLIB)
	$(Q38_LOCK) $(Q38_ENV) env MTL_DEBUG_LAYER=1 $(PYTHON) -B tools/qwen38_compositions.py --checkpoint "$(Q38_CHECKPOINT)" --metallib "$(abspath $(METALLIB))"

.PHONY: qwen38-composition-assets test-qwen38-d-gpu test-qwen38-f-gpu test-qwen38-compositions

# Every harness using C composition also compiles the packed/reader dependencies.
build/qwen38-c/test_cpu build/qwen38-c/libcpu.dylib build/qwen38-c/test_gpu \
build/qwen38-d/test_attention build/qwen38-d/test_attention_gpu \
build/qwen38-e/test_ngram build/qwen38-f/test_moe build/qwen38-f/test_moe_gpu: src/gguf.h src/qwen38_iq2_grid.inc
