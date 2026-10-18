# R0/R1 measurement and decode-speed targets. Included by tools/qwen38.mk.

# Metal capability probe (R0.1): plain ObjC like build/memory_budget, no engine code.
build/qwen38-capabilities: tools/probe/qwen38_capabilities.m
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fobjc-arc -o $@ $< $(FRAMEWORKS)

# The recipe owns the GPU lock; the test only runs the probe (no nested wrapper).
test-qwen38-r0-tools: build/qwen38-capabilities
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_capabilities.py --native $< -v

.PHONY: test-qwen38-r0-tools

# R0.2 per-op GPU profile on the tiny CLI graph; one lock in the recipe only.
test-qwen38-profile: picoforge build/qwen38-g2/cli.gguf
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_profile.py -v

.PHONY: test-qwen38-profile

# R0.3 profile summary on fixture CSVs: pure Python, no GPU, so no lock.
test-qwen38-profile-summary:
	$(Q38_TEST) tests/test_qwen38_profile_summary.py -v

.PHONY: test-qwen38-profile-summary

# R0.4 paired harness logic with fake tokenizers: pure Python, no GPU, so no lock.
test-qwen38-compare:
	$(Q38_TEST) tests/test_qwen38_compare.py -v

.PHONY: test-qwen38-compare

# G3 launch fencing (R1.0) with a fake Launcher: pure Python, no GPU, so no lock.
test-qwen38-g3-tools:
	$(Q38_TEST) tests/test_qwen38_g3_tools.py -v

.PHONY: test-qwen38-g3-tools

# R1.2 fast matvec kernels, same flags as build/qwen38-b/packed.air; linked into $(METALLIB).
build/qwen38-r1/matvec.air: src/qwen38_matvec.metal src/qwen38_quant_decode.h
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

# R1.1 fast-kernel harness: sanitized like build/qwen38-b/test_packed_gpu.
build/qwen38-r1/test_fast_kernels: tests/test_qwen38_fast_kernels.m src/qwen38_packed_gpu.m src/qwen38_packed.c $(Q38_STORAGE) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

# R1.1 fast kernels vs oracle vs FP64: real row slices (never the checkpoint)
# and seeded shapes. One lock for the recipe; the test adds no wrapper.
Q38_R1_SLICES := $(foreach kind,0 1 8 10 12 16 30 39,--slices build/qwen38-b/slices-$(kind))
test-qwen38-fast-kernels: build/qwen38-r1/test_fast_kernels $(METALLIB)
	@mkdir -p build/qwen38-r1
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" $(Q38_TEST) tests/test_qwen38_fast_kernels.py --native $< $(Q38_R1_SLICES) --report "$$(mktemp -d build/qwen38-r1/fast-kernels.XXXXXX)/report.json" -v

.PHONY: test-qwen38-fast-kernels

# R1.2 isolated-kernel timing on synthetic Q8_0 (never the checkpoint; the
# LM-head shape is 644 MiB), without the validation layer, K = 2560. Specs are
# mode:n_out:rows:reps; reps 1 = one command buffer per encode (the protocol),
# reps 20 = 20 encodes back to back per command buffer, as in the decode graph
# (a lone short dispatch runs at an idle GPU clock). One JSON line per run plus
# `pmset -g therm` before and after, into a new build/qwen38-r1 file.
Q38_R1_BENCH := fast:248320:1:1 fast:248320:1:20 oracle:248320:1:1 \
	$(foreach m,1 2 4 8 32,$(foreach r,1 20,fast:10240:$(m):$(r) oracle:10240:$(m):$(r))) \
	$(foreach m,128 1024,fast:10240:$(m):1 oracle:10240:$(m):1)
bench-qwen38-matvec: build/qwen38-r1/test_fast_kernels $(METALLIB)
	@out="$$(mktemp build/qwen38-r1/bench.XXXXXX)"; \
	$(Q38_LOCK) env Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" sh -c 'pmset -g therm; \
	  for spec in $(Q38_R1_BENCH); do set -- $$(echo $$spec | tr : " "); \
	    ./$< bench $$1 $$2 2560 $$3 $$4 || exit 1; done; pmset -g therm' > "$$out"; \
	status=$$?; cat "$$out"; echo "bench: $$out"; exit $$status

.PHONY: bench-qwen38-matvec
