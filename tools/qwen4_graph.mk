# The model graph, end to end: the GPU session lifecycle (G0), the graph on the
# CPU (G1) and on the GPU (G2), the CLI/REPL/MTP tests and the real-checkpoint G3
# gate. Included by tools/qwen4.mk.

# G0: GPU session lifecycle.
# The binary opens a Metal device, so it runs under the common GPU lock.
build/qwen4-g/test_gpu_lifecycle: tests/test_qwen4_gpu_lifecycle.m src/qwen4_gpu.m $(QW4_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.m,$^) $(FRAMEWORKS) -o $@

test-qwen4-g0: build/qwen4-g/test_gpu_lifecycle
	$(QW4_LOCK) $(QW4_ENV) MTL_DEBUG_LAYER=1 ./$<

.PHONY: test-qwen4-g0

# G1: ordinary graph on the CPU.
# The harness maps weights through the existing Metal store (no GPU work),
# so it runs under the common lock like the other device-opening tests.
QW4_G1_CPU := src/qwen4_forward.c src/qwen4_attention.c src/qwen4_ngram.c src/qwen4_ple.c \
 src/qwen4_moe.c $(QW4_RECURRENT)

build/qwen4-g1/graph.gguf: tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py
	$(QW4_ENV) $(PYTHON) -B $<

build/qwen4-g1/test_forward: tests/test_qwen4_forward.c $(QW4_G1_CPU) src/gpu_weights.m tests/metal_device.m $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen4-g1: build/qwen4-g1/test_forward build/qwen4-g1/graph.gguf $(METALLIB)
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_forward.py --native $< --gguf build/qwen4-g1/graph.gguf -v

.PHONY: test-qwen4-g1

# Unsanitized build of the same harness, for bounded smoke runs on the real
# checkpoint (the sanitized one remains the correctness gate).
build/qwen4-g1/forward_real: tests/test_qwen4_forward.c $(QW4_G1_CPU) src/gpu_weights.m tests/metal_device.m $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

# G2: ordinary graph on the GPU over the G0 session.
QW4_G2_GPU := src/qwen4_gpu.m src/qwen4_packed_gpu.m src/qwen4_recurrent_gpu.m src/qwen4_moe.m \
 src/qwen4_attention_gpu.m src/qwen4_ple_gpu.m src/qwen4_forward_gpu.m src/qwen4_profile.m src/gpu_weights.m src/qwen4_expert_pool.c

build/qwen4-g2/test_parts: tests/test_qwen4_gpu_parts.m $(QW4_G1_CPU) $(QW4_G2_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen4-g2-parts: build/qwen4-g2/test_parts build/qwen4-g1/graph.gguf $(METALLIB)
	$(QW4_LOCK) $(QW4_ENV) MTL_DEBUG_LAYER=1 ./$< build/qwen4-g1/graph.gguf "$(abspath $(METALLIB))"

.PHONY: test-qwen4-g2-parts

build/qwen4-g2/test_forward: tests/test_qwen4_forward_gpu.m $(QW4_G1_CPU) $(QW4_G2_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-g2/forward_real: tests/test_qwen4_forward_gpu.m $(QW4_G1_CPU) $(QW4_G2_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

test-qwen4-g2: build/qwen4-g2/test_forward build/qwen4-g1/graph.gguf $(METALLIB)
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward.py --native $< --gguf build/qwen4-g1/graph.gguf -v

.PHONY: test-qwen4-g2

# State/fault and >32-token MoE microbatch gate (same sanitized harness).
test-qwen4-g2-state: build/qwen4-g2/test_forward build/qwen4-g1/graph.gguf $(METALLIB)
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward_gpu.py --native $< --gguf build/qwen4-g1/graph.gguf -v
.PHONY: test-qwen4-g2-state

build/qwen4-g2/cli.gguf: tools/synth/make_qwen4_cli.py tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py tests/fixtures/qwen4/text/manifest.json
	$(QW4_ENV) $(PYTHON) -B $< --reference-root "$(QW4_REFERENCE_ROOT)"

test-qwen4-cli: pf build/qwen4-g1/test_forward build/qwen4-g2/cli.gguf
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_cli.py -v
.PHONY: test-qwen4-cli

# Upstream defaults that Flash-Next overrides (no PLE, SiLU GDN gate, QSA off the
# every-4th pattern), on the same tiny graph: CPU and GPU graph against the oracle,
# the GPU parts and state gates, and the CLI against the CPU graph.
build/qwen4-g1/variant.gguf: tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py
	$(QW4_ENV) $(PYTHON) -B $< --variant --output $@

build/qwen4-g2/cli-variant.gguf: tools/synth/make_qwen4_cli.py tools/synth/make_qwen4_graph.py tools/oracle/qwen4_config.py tests/fixtures/qwen4/text/manifest.json
	$(QW4_ENV) $(PYTHON) -B $< --reference-root "$(QW4_REFERENCE_ROOT)" --variant --output $@

test-qwen4-variants: pf build/qwen4-g1/test_forward build/qwen4-g2/test_forward build/qwen4-g2/test_parts build/qwen4-g1/variant.gguf build/qwen4-g2/cli-variant.gguf $(METALLIB)
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g1/test_forward --gguf build/qwen4-g1/variant.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/variant.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward_gpu.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/variant.gguf -v
	$(QW4_LOCK) $(QW4_ENV) MTL_DEBUG_LAYER=1 ./build/qwen4-g2/test_parts build/qwen4-g1/variant.gguf "$(abspath $(METALLIB))"
	$(QW4_LOCK) env QW4_CLI_GGUF="$(abspath build/qwen4-g2/cli-variant.gguf)" $(QW4_TEST) tests/test_qwen4_cli.py -v
.PHONY: test-qwen4-variants

test-qwen4-repl: pf build/qwen4-g2/cli.gguf
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_repl.py -v
.PHONY: test-qwen4-repl

# Explicit, heavyweight real-checkpoint gate. Never part of the tiny suite.
# G3_OUTPUT must name a fresh directory; the script retains failed evidence.
G3_OUTPUT ?= build/qwen4-g3
qwen4-g3: pf build/qwen4-g2/forward_real build/qwen4-g2/test_parts build/qwen4-g1/graph.gguf build/process_memory
	$(QW4_LOCK) $(QW4_ENV) $(PYTHON) -B tools/qwen4_g3.py --checkpoint "$(QW4_CHECKPOINT)" --dwarfstar "$(QW4_DWARFSTAR)" --output "$(G3_OUTPUT)"
.PHONY: qwen4-g3

# Persistent engine tests take the existing cross-worktree lock once.
test-qwen4-session: pf build/qwen4-g2/cli.gguf
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_session.py -v

ui-qwen4: pf
	$(PYTHON) tools/ui/qwen4/server.py --checkpoint "$(QW4_CHECKPOINT)"

# The server's resident worker acquires the lock; do not wrap this target again.
test-qwen4-server: pf build/qwen4-g2/cli.gguf build/qwen4-mtp/cli.gguf
	$(QW4_ENV) $(PYTHON) -B tests/test_qwen4_server.py -v

build/qwen4-attention-bench: tools/probe/qwen4_attention_bench.m src/qwen4_attention_internal.h
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) -fobjc-arc $< $(FRAMEWORKS) -lm -o $@

.PHONY: test-qwen4-session test-qwen4-server ui-qwen4

test-qwen4-native-tools: build/process_memory
	$(QW4_ENV) $(PYTHON) -B tests/test_qwen4_native_suite.py -v
	$(QW4_ENV) $(PYTHON) -B tests/test_qwen4_native_bounded.py -v
	$(QW4_ENV) $(PYTHON) -B tests/test_run_observed.py -v
.PHONY: test-qwen4-native-tools

build/qwen4-mtp/test_mtp: tests/test_qwen4_mtp.m $(QW4_G1_CPU) $(QW4_G2_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) $(QW4_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-mtp/mtp_real: tests/test_qwen4_mtp.m $(QW4_G1_CPU) $(QW4_G2_GPU) $(QW4_HEADERS) src/qwen4_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(QW4_FLAGS) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

build/qwen4-mtp/cli.gguf: tools/synth/make_qwen4_cli.py tools/synth/make_qwen4_graph.py
	$(QW4_ENV) $(PYTHON) -B $< --reference-root "$(QW4_REFERENCE_ROOT)" --mtp --output $@

test-qwen4-mtp: build/qwen4-mtp/test_mtp build/qwen4-mtp/cli.gguf $(METALLIB)
	@mkdir -p build/qwen4-mtp/tiny build/qwen4-mtp/tiny33
	$(QW4_LOCK) $(QW4_ENV) MTL_DEBUG_LAYER=1 ./$< build/qwen4-mtp/cli.gguf build/qwen4-mtp/tiny 3 1 12 45 19 71 83 120 64 20 49
	# R3.1: chunk 33 runs the GEMMs and 1-token appends the per-row kernels; the replay is bitwise under gemm
	$(QW4_LOCK) $(QW4_ENV) MTL_DEBUG_LAYER=1 QW4_REF_KERNELS=gemm ./$< build/qwen4-mtp/cli.gguf build/qwen4-mtp/tiny33 33 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
.PHONY: test-qwen4-mtp

test-qwen4-speculative: pf build/qwen4-mtp/cli.gguf $(METALLIB)
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_speculative.py -v
.PHONY: test-qwen4-speculative
