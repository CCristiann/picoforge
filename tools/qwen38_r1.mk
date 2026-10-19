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
# LM-head shape is 644 MiB), without the validation layer. Specs are
# mode:n_out:K:rows:reps; reps 1 = one command buffer per encode (the
# protocol), reps 20 = 20 encodes back to back per command buffer, as in the
# decode graph (a lone short dispatch runs at an idle GPU clock).
# Ruling R23: the raw evidence goes to a NEW BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/matvec-q8_0,
# never overwritten: runs.jsonl, the conditions before and after (therm, batt,
# swap, NSProcessInfo.thermalState from the capabilities probe) and a manifest.
# The README.md next to them is written by hand from runs.jsonl.
Q38_R1_BENCH := fast:248320:2560:1:1 fast:248320:2560:1:20 oracle:248320:2560:1:1 \
	$(foreach m,1 2 4 8 32,$(foreach r,1 20,fast:10240:2560:$(m):$(r) oracle:10240:2560:$(m):$(r))) \
	$(foreach m,128 1024,fast:10240:2560:$(m):1 oracle:10240:2560:$(m):1) \
	$(foreach s,2560:640 2560:6144 640:2560,$(foreach m,1 4 32,fast:$(s):$(m):20 oracle:$(s):$(m):20) \
	  fast:$(s):1024:1 oracle:$(s):1024:1)
Q38_R1_CONDITIONS := date -u +%Y-%m-%dT%H:%M:%SZ; pmset -g therm; pmset -g batt; sysctl vm.swapusage; ./build/qwen38-capabilities
bench-qwen38-matvec: build/qwen38-r1/test_fast_kernels build/qwen38-capabilities $(METALLIB)
	@case "$(BENCH_OUT)" in bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/matvec-q8_0) ;; \
	  *) echo "set BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/matvec-q8_0 (a new directory)" >&2; exit 2;; esac
	@mkdir -p "$(dir $(BENCH_OUT))" && mkdir "$(BENCH_OUT)" # an existing directory fails here: never overwritten
	@{ git rev-parse HEAD; git status --porcelain; shasum -a 256 $< $(METALLIB); } > "$(BENCH_OUT)/manifest.txt"
	$(Q38_LOCK) env Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" sh -c '{ $(Q38_R1_CONDITIONS); } > "$(BENCH_OUT)/before.txt" && \
	  for spec in $(Q38_R1_BENCH); do set -- $$(echo $$spec | tr : " "); \
	    ./$< bench $$1 $$2 $$3 $$4 $$5 || exit 1; done > "$(BENCH_OUT)/runs.jsonl" && \
	  { $(Q38_R1_CONDITIONS); } > "$(BENCH_OUT)/after.txt"'
	@cat "$(BENCH_OUT)/runs.jsonl"; echo "bench: $(BENCH_OUT); add README.md from runs.jsonl"

.PHONY: bench-qwen38-matvec
