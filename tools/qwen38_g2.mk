# G2, ordinary graph on the GPU over the G0 session. Coordinator-owned.
Q38_G2_GPU := src/qwen38_gpu.m src/qwen38_packed_gpu.m src/qwen38_recurrent_gpu.m src/qwen38_moe.m \
 src/qwen38_attention_gpu.m src/gpu_weights.m

build/qwen38-g2/test_parts: tests/test_qwen38_g2_parts.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen38-g2-parts: build/qwen38-g2/test_parts build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./$< build/qwen38-g1/graph.gguf "$(abspath $(METALLIB))"

.PHONY: test-qwen38-g2-parts
