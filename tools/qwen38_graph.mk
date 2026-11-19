# The model graph, end to end: the GPU session lifecycle (G0), the graph on the
# CPU (G1) and on the GPU (G2), the CLI/REPL/MTP tests and the real-checkpoint G3
# gate. Included by tools/qwen38.mk.

# G0: GPU session lifecycle.
# The binary opens a Metal device, so it runs under the common GPU lock.
build/qwen38-g/test_gpu_lifecycle: tests/test_qwen38_gpu_lifecycle.m src/qwen38_gpu.m $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.m,$^) $(FRAMEWORKS) -o $@

test-qwen38-g0: build/qwen38-g/test_gpu_lifecycle
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./$<

.PHONY: test-qwen38-g0

# G1: ordinary graph on the CPU.
# The harness maps weights through the existing Metal store (no GPU work),
# so it runs under the common lock like the other device-opening tests.
Q38_G1_CPU := src/qwen38_forward.c src/qwen38_attention.c src/qwen38_ngram.c src/qwen38_ple.c \
 src/qwen38_moe.c $(Q38_RECURRENT)

build/qwen38-g1/graph.gguf: tools/synth/make_qwen38_graph.py tools/synth/make_qwen38_small.py tools/oracle/qwen38_config.py
	$(Q38_ENV) $(PYTHON) -B $<

build/qwen38-g1/test_forward: tests/test_qwen38_forward.c $(Q38_G1_CPU) src/gpu_weights.m tests/metal_device.m $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen38-g1: build/qwen38-g1/test_forward build/qwen38-g1/graph.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_forward.py --native $< --gguf build/qwen38-g1/graph.gguf -v

.PHONY: test-qwen38-g1

# Unsanitized build of the same harness, for bounded smoke runs on the real
# checkpoint (the sanitized one remains the correctness gate).
build/qwen38-g1/forward_real: tests/test_qwen38_forward.c $(Q38_G1_CPU) src/gpu_weights.m tests/metal_device.m $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

# G2: ordinary graph on the GPU over the G0 session.
Q38_G2_GPU := src/qwen38_gpu.m src/qwen38_packed_gpu.m src/qwen38_recurrent_gpu.m src/qwen38_moe.m \
 src/qwen38_attention_gpu.m src/qwen38_ple_gpu.m src/qwen38_forward_gpu.m src/qwen38_profile.m src/gpu_weights.m src/qwen38_expert_pool.c

build/qwen38-g2/test_parts: tests/test_qwen38_gpu_parts.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
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

build/qwen38-g2/cli.gguf: tools/synth/make_qwen38_cli.py tools/synth/make_qwen38_graph.py tools/synth/make_qwen38_small.py tools/oracle/qwen38_config.py tests/fixtures/qwen38/text/manifest.json
	$(Q38_ENV) $(PYTHON) -B $< --reference-root "$(Q38_REFERENCE_ROOT)"

test-qwen38-cli: pf build/qwen38-g1/test_forward build/qwen38-g2/cli.gguf
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_cli.py -v
.PHONY: test-qwen38-cli

# Upstream defaults that Flash-Next overrides (no PLE, SiLU GDN gate, QSA off the
# every-4th pattern), on the same tiny graph: CPU and GPU graph against the oracle,
# the GPU parts and state gates, and the CLI against the CPU graph.
build/qwen38-g1/variant.gguf: tools/synth/make_qwen38_graph.py tools/synth/make_qwen38_small.py tools/oracle/qwen38_config.py
	$(Q38_ENV) $(PYTHON) -B $< --variant --output $@

build/qwen38-g2/cli-variant.gguf: tools/synth/make_qwen38_cli.py tools/synth/make_qwen38_graph.py tools/oracle/qwen38_config.py tests/fixtures/qwen38/text/manifest.json
	$(Q38_ENV) $(PYTHON) -B $< --reference-root "$(Q38_REFERENCE_ROOT)" --variant --output $@

test-qwen38-variants: pf build/qwen38-g1/test_forward build/qwen38-g2/test_forward build/qwen38-g2/test_parts build/qwen38-g1/variant.gguf build/qwen38-g2/cli-variant.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_forward.py --native build/qwen38-g1/test_forward --gguf build/qwen38-g1/variant.gguf -v
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 $(Q38_TEST) tests/test_qwen38_forward.py --native build/qwen38-g2/test_forward --gguf build/qwen38-g1/variant.gguf -v
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 $(Q38_TEST) tests/test_qwen38_forward_gpu.py --native build/qwen38-g2/test_forward --gguf build/qwen38-g1/variant.gguf -v
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./build/qwen38-g2/test_parts build/qwen38-g1/variant.gguf "$(abspath $(METALLIB))"
	$(Q38_LOCK) env Q38_CLI_GGUF="$(abspath build/qwen38-g2/cli-variant.gguf)" $(Q38_TEST) tests/test_qwen38_cli.py -v
.PHONY: test-qwen38-variants

test-qwen38-repl: pf build/qwen38-g2/cli.gguf
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_repl.py -v
.PHONY: test-qwen38-repl

# Explicit, heavyweight real-checkpoint gate. Never part of the tiny suite.
# G3_OUTPUT must name a fresh directory; the script retains failed evidence.
G3_OUTPUT ?= build/qwen38-g3
qwen38-g3: pf build/qwen38-g2/forward_real build/qwen38-g2/test_parts build/qwen38-g1/graph.gguf build/process_memory
	$(Q38_LOCK) $(Q38_ENV) $(PYTHON) -B tools/qwen38_g3.py --checkpoint "$(Q38_CHECKPOINT)" --dwarfstar "$(Q38_DWARFSTAR)" --output "$(G3_OUTPUT)"
.PHONY: qwen38-g3

# Persistent engine tests take the existing cross-worktree lock once.
test-qwen38-session: pf build/qwen38-g2/cli.gguf
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_session.py -v

ui-qwen38: pf
	$(PYTHON) tools/ui/qwen38/server.py --checkpoint "$(Q38_CHECKPOINT)"

# The server's resident worker acquires the lock; do not wrap this target again.
test-qwen38-server: pf build/qwen38-g2/cli.gguf build/qwen38-mtp/cli.gguf
	$(Q38_ENV) $(PYTHON) -B tests/test_qwen38_server.py -v

build/qwen38-attention-bench: tools/probe/qwen38_attention_bench.m src/qwen38_attention_internal.h
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -fobjc-arc $< $(FRAMEWORKS) -lm -o $@

.PHONY: test-qwen38-session test-qwen38-server ui-qwen38

test-qwen38-native-tools: build/process_memory
	$(Q38_ENV) $(PYTHON) -B tests/test_qwen38_native_suite.py -v
	$(Q38_ENV) $(PYTHON) -B tests/test_qwen38_native_bounded.py -v
	$(Q38_ENV) $(PYTHON) -B tests/test_run_observed.py -v
.PHONY: test-qwen38-native-tools

build/qwen38-mtp/test_mtp: tests/test_qwen38_mtp.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-mtp/mtp_real: tests/test_qwen38_mtp.m $(Q38_G1_CPU) $(Q38_G2_GPU) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen38-mtp/cli.gguf: tools/synth/make_qwen38_cli.py tools/synth/make_qwen38_graph.py
	$(Q38_ENV) $(PYTHON) -B $< --reference-root "$(Q38_REFERENCE_ROOT)" --mtp --output $@

test-qwen38-mtp: build/qwen38-mtp/test_mtp build/qwen38-mtp/cli.gguf $(METALLIB)
	@mkdir -p build/qwen38-mtp/tiny build/qwen38-mtp/tiny33
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./$< build/qwen38-mtp/cli.gguf build/qwen38-mtp/tiny 3 1 12 45 19 71 83 120 64 20 49
	# R3.1: chunk 33 runs the GEMMs and 1-token appends the per-row kernels; the replay is bitwise under gemm
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 Q38_REF_KERNELS=gemm ./$< build/qwen38-mtp/cli.gguf build/qwen38-mtp/tiny33 33 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
.PHONY: test-qwen38-mtp

test-qwen38-speculative: pf build/qwen38-mtp/cli.gguf $(METALLIB)
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_speculative.py -v
.PHONY: test-qwen38-speculative
