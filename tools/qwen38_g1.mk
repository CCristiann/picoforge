# G1, ordinary graph on the CPU. Coordinator-owned.
# The harness maps weights through the existing Metal store (no GPU work),
# so it runs under the common lock like the other device-opening tests.
Q38_G1_CPU := src/qwen38_forward.c src/qwen38_attention.c src/qwen38_ngram.c src/qwen38_ple.c \
 src/qwen38_moe.c $(Q38_RECURRENT)

build/qwen38-g1/graph.gguf: tools/synth/make_qwen38_graph.py tools/synth/make_qwen38_a.py tools/oracle/qwen38_config.py
	$(Q38_ENV) $(PYTHON) -B $<

build/qwen38-g1/test_forward: tests/test_qwen38_forward.c $(Q38_G1_CPU) src/gpu_weights.m src/metal.m $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen38-g1: build/qwen38-g1/test_forward build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_forward.py --native $< --gguf build/qwen38-g1/graph.gguf -v

.PHONY: test-qwen38-g1

# Unsanitized build of the same harness, for bounded smoke runs on the real
# checkpoint (the sanitized one remains the correctness gate).
build/qwen38-g1/forward_real: tests/test_qwen38_forward.c $(Q38_G1_CPU) src/gpu_weights.m src/metal.m $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@
