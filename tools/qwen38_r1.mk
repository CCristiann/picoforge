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
build/qwen38-r1/matvec.air: src/qwen38_matvec.metal src/qwen38_quant_decode.h src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

# R1.5b fused MoE kernels, same flags and the same row dots as matvec.air.
build/qwen38-r1/moe_fused.air: src/qwen38_moe_fused.metal src/qwen38_quant_decode.h src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c $< -o $@

# R1.1 fast-kernel harness: sanitized like build/qwen38-b/test_packed_gpu; C's GPU ops
# (R1.5b) run the unfused gate/up sequence the fused kernel is checked against.
build/qwen38-r1/test_fast_kernels: tests/test_qwen38_fast_kernels.m src/qwen38_packed_gpu.m src/qwen38_recurrent_gpu.m $(Q38_RECURRENT) $(Q38_STORAGE) $(Q38_HEADERS) src/qwen38_iq2_grid.inc
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.c %.m,$^) $(FRAMEWORKS) -lm -o $@

# R1.1 fast kernels vs oracle vs FP64: real row slices (never the checkpoint)
# and seeded shapes. One lock for the recipe; the test adds no wrapper.
Q38_R1_SLICES := $(foreach kind,0 1 8 10 12 16 30 39,--slices build/qwen38-b/slices-$(kind))
test-qwen38-fast-kernels: build/qwen38-r1/test_fast_kernels $(METALLIB)
	@mkdir -p build/qwen38-r1
	$(Q38_LOCK) env MTL_DEBUG_LAYER=1 Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" $(Q38_TEST) tests/test_qwen38_fast_kernels.py --native $< $(Q38_R1_SLICES) --report "$$(mktemp -d build/qwen38-r1/fast-kernels.XXXXXX)/report.json" -v

.PHONY: test-qwen38-fast-kernels

# R1.2/R1.3 isolated-kernel timing on synthetic data (never the checkpoint;
# the LM-head shape is 644 MiB), without the validation layer. Specs are
# mode:n_out:K:rows:reps[:type] (type 8 Q8_0 by default, 1 F16, 0 F32);
# reps 1 = one command buffer per encode (the protocol), reps 20 = 20 encodes
# back to back per command buffer, as in the decode graph (a lone short
# dispatch runs at an idle GPU clock).
# Ruling R23: the raw evidence goes to a NEW directory, never overwritten:
# BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/matvec-q8_0 (Q38_R1_BENCH) or
# .../matvec-f16-f32 (Q38_R1_BENCH_DENSE). It holds runs.jsonl, the conditions
# before and after (therm, batt, swap, NSProcessInfo.thermalState from the
# capabilities probe) and a manifest.
# The README.md next to them is written by hand from runs.jsonl.
Q38_R1_BENCH := fast:248320:2560:1:1 fast:248320:2560:1:20 oracle:248320:2560:1:1 \
	$(foreach m,1 2 4 8 32,$(foreach r,1 20,fast:10240:2560:$(m):$(r) oracle:10240:2560:$(m):$(r))) \
	$(foreach m,128 1024,fast:10240:2560:$(m):1 oracle:10240:2560:$(m):1) \
	$(foreach s,2560:640 2560:6144 640:2560,$(foreach m,1 4 32,fast:$(s):$(m):20 oracle:$(s):$(m):20) \
	  fast:$(s):1024:1 oracle:$(s):1024:1)
# F16/F32 (R1.3): 500 MiB tensors for the honest bandwidth, the real HC down
# (320x10240), inject (4x10240), up (10240x320), router (512x2560) and
# ssm_alpha (48x2560) shapes; reps 200 steadies the GPU clock of dispatches
# under ~30 us. Then fast vs oracle by rows of x (exit criterion 4).
# q38_dense(n_out,K,rows,reps list,type): fast and oracle specs.
q38_dense = $(foreach r,$(4),fast:$(1):$(2):$(3):$(r):$(5) oracle:$(1):$(2):$(3):$(r):$(5))
q38_hc_router = $(call q38_dense,320,10240,$(1),$(2),1) $(call q38_dense,512,2560,$(1),$(2),0) \
	$(call q38_dense,10240,320,$(1),$(2),1) $(call q38_dense,4,10240,$(1),$(2),1)
Q38_R1_BENCH_DENSE := $(call q38_dense,25600,10240,1,1,1) fast:25600:10240:1:20:1 \
	$(call q38_dense,51200,2560,1,1,0) fast:51200:2560:1:20:0 \
	$(call q38_dense,320,10240,1,1 20 200,1) $(call q38_dense,4,10240,1,1 20 200,1) \
	$(call q38_dense,10240,320,1,1 20 200,1) $(call q38_dense,512,2560,1,1 20 200,0) \
	$(call q38_dense,48,2560,1,1 20 200,0) \
	$(foreach m,2 4 8 32,$(call q38_hc_router,$(m),20)) $(foreach m,128 1024,$(call q38_hc_router,$(m),1))
# R1.4 multi-GEMV (BENCH_OUT=.../matvec-multi): the QSA (5 x Q8_0), GDN
# (QKV, Z Q8_0 + alpha, beta F32) and shared-expert (router, slogit F32 + gate,
# up Q8_0) groups at 1 and 4 rows, one multi call vs the fast single calls,
# each followed by one barrier, 20 per command buffer as in the graph; the
# order alternates between the two row counts.
# Specs are mode@rows@reps@type:physical:logical:n@...
q38_k = $(foreach p,$(1),$(2):2560:2560:$(p))
Q38_R1_MULTI_GROUPS := $(subst $(eval) ,@,$(call q38_k,12288 512 512 512 128,8)) \
	$(subst $(eval) ,@,$(call q38_k,10240 6144,8) $(call q38_k,48 48,0)) \
	$(subst $(eval) ,@,$(call q38_k,512 1,0) $(call q38_k,640 640,8))
Q38_R1_BENCH_MULTI := $(foreach g,$(Q38_R1_MULTI_GROUPS),multi@1@20@$(g) single@1@20@$(g) single@4@20@$(g) multi@4@20@$(g))
# R1.5 indexed experts (BENCH_OUT=.../matvec-indexed): fast vs oracle on
# 512-expert synthetic tensors, rows = tokens x 10 slots with distinct experts
# per token. Gate shape IQ2_XXS 2560->640 (x divisor 10) and down Q2_K
# 640/768->2560 (divisor 1) at 1, 4, 32, 128 tokens; the mixed layer's Q4_K
# gate and MXFP4 down at 1 and 32 tokens. 20 encodes per buffer up to 4
# tokens, as in the graph. Specs are ibench:mode:type:n:physical:logical:experts:rows:divisor:reps.
# q38_ib(type:n:physical:logical,rows,divisor,reps): fast and oracle specs.
q38_ib = $(foreach m,fast oracle,ibench:$(m):$(1):512:$(2):$(3):$(4))
q38_ib_tokens = $(call q38_ib,$(1),10,$(2),20) $(call q38_ib,$(1),40,$(2),20) \
	$(call q38_ib,$(1),320,$(2),1) $(call q38_ib,$(1),1280,$(2),1)
Q38_R1_BENCH_INDEXED := $(call q38_ib_tokens,16:640:2560:2560,10) $(call q38_ib_tokens,10:2560:768:640,1) \
	$(call q38_ib,12:640:2560:2560,10,10,20) $(call q38_ib,12:640:2560:2560,320,10,1) \
	$(call q38_ib,39:2560:640:640,10,1,20) $(call q38_ib,39:2560:640:640,320,1,1)
# R1.5b fused gate+up+SwiGLU (BENCH_OUT=.../moe-gate-up): the fused kernel vs
# the unfused fast sequence (indexed gate, indexed up, ACT SILU, MUL: barriers
# included) on the IQ2_XXS layer and the Q4_K layer-48 shape (640 x 2560, 512
# experts), 1, 4, 32, 128 tokens x 10 slots. Specs as ibench, mode fused|sequence.
q38_gb = $(foreach m,fused sequence,gbench:$(m):$(1):640:2560:2560:512:$(2):10:$(3))
Q38_R1_BENCH_GATE_UP := $(foreach t,16 12,$(call q38_gb,$(t),10,20) $(call q38_gb,$(t),40,20) \
	$(call q38_gb,$(t),320,1) $(call q38_gb,$(t),1280,1))
Q38_R1_CONDITIONS := date -u +%Y-%m-%dT%H:%M:%SZ; pmset -g therm; pmset -g batt; sysctl vm.swapusage; ./build/qwen38-capabilities
bench-qwen38-matvec: build/qwen38-r1/test_fast_kernels build/qwen38-capabilities $(METALLIB)
	@case "$(BENCH_OUT)" in bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/matvec-q8_0) ;; \
	  bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/matvec-f16-f32) ;; \
	  bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/matvec-multi) ;; \
	  bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/matvec-indexed) ;; \
	  bench/qwen38_r1_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/moe-gate-up) ;; \
	  *) echo "set BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/matvec-q8_0, .../matvec-f16-f32, .../matvec-multi, .../matvec-indexed or .../moe-gate-up (a new directory)" >&2; exit 2;; esac
	@mkdir -p "$(dir $(BENCH_OUT))" && mkdir "$(BENCH_OUT)" # an existing directory fails here: never overwritten
	@{ git rev-parse HEAD; git status --porcelain; shasum -a 256 $< $(METALLIB); } > "$(BENCH_OUT)/manifest.txt"
	$(Q38_LOCK) env Q38_PACKED_LIBRARY="$(abspath $(METALLIB))" sh -c '{ $(Q38_R1_CONDITIONS); } > "$(BENCH_OUT)/before.txt" && \
	  for spec in $(if $(filter %/matvec-f16-f32,$(BENCH_OUT)),$(Q38_R1_BENCH_DENSE),$(if $(filter %/matvec-multi,$(BENCH_OUT)),$(Q38_R1_BENCH_MULTI),$(if $(filter %/matvec-indexed,$(BENCH_OUT)),$(Q38_R1_BENCH_INDEXED),$(if $(filter %/moe-gate-up,$(BENCH_OUT)),$(Q38_R1_BENCH_GATE_UP),$(Q38_R1_BENCH))))); do \
	    case $$spec in *@*) ./$< mbench $$(echo $$spec | tr @ " ") || exit 1;; \
	    ibench:*|gbench:*) ./$< $$(echo $$spec | tr : " ") || exit 1;; \
	    *) set -- $$(echo $$spec | tr : " "); ./$< bench $$1 $$2 $$3 $$4 $$5 $$6 || exit 1;; esac; done > "$(BENCH_OUT)/runs.jsonl" && \
	  { $(Q38_R1_CONDITIONS); } > "$(BENCH_OUT)/after.txt"'
	@cat "$(BENCH_OUT)/runs.jsonl"; echo "bench: $(BENCH_OUT); add README.md from runs.jsonl"

.PHONY: bench-qwen38-matvec
