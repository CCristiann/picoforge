# G2, ordinary graph on the GPU over the G0 session. Coordinator-owned.
Q38_G2_GPU := src/qwen38_gpu.m src/qwen38_packed_gpu.m src/qwen38_recurrent_gpu.m src/qwen38_moe.m \
 src/qwen38_attention_gpu.m src/qwen38_ple_gpu.m src/qwen38_forward_gpu.m src/gpu_weights.m

build/qwen38-g2/test_parts: tests/test_qwen38_g2_parts.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen38-g2-parts: build/qwen38-g2/test_parts build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./$< build/qwen38-g1/graph.gguf "$(abspath $(METALLIB))"

.PHONY: test-qwen38-g2-parts

build/qwen38-g2/test_forward: tests/test_qwen38_forward_gpu.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-g2/forward_real: tests/test_qwen38_forward_gpu.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen38-g2: build/qwen38-g2/test_forward build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 $(Q38_TEST) tests/test_qwen38_forward.py --native $< --gguf build/qwen38-g1/graph.gguf -v

.PHONY: test-qwen38-g2

# State/fault and >32-token MoE microbatch gate (same sanitized harness).
test-qwen38-g2-state: build/qwen38-g2/test_forward build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 $(Q38_TEST) tests/test_qwen38_forward_gpu.py --native $< --gguf build/qwen38-g1/graph.gguf -v
.PHONY: test-qwen38-g2-state

build/qwen38-g2/cli.gguf: tools/synth/make_qwen38_cli.py tools/synth/make_qwen38_graph.py tools/synth/make_qwen38_a.py tools/oracle/qwen38_config.py tests/fixtures/qwen38/a/manifest.json
	$(Q38_ENV) $(PYTHON) -B $< --reference-root "$(Q38_REFERENCE_ROOT)"

test-qwen38-cli: picoforge build/qwen38-g1/test_forward build/qwen38-g2/cli.gguf
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_cli.py -v
.PHONY: test-qwen38-cli

# Explicit, heavyweight real-checkpoint gate. Never part of the tiny suite.
# G3_OUTPUT must name a fresh directory; the script retains failed evidence.
G3_OUTPUT ?= build/qwen38-g3
qwen38-g3: picoforge build/qwen38-g2/forward_real build/qwen38-g2/test_parts build/qwen38-g1/graph.gguf build/process_memory
	$(Q38_LOCK) $(Q38_ENV) $(PYTHON) -B tools/qwen38_g3.py --checkpoint "$(Q38_CHECKPOINT)" --dwarfstar "$(Q38_DWARFSTAR)" --output "$(G3_OUTPUT)"
.PHONY: qwen38-g3
