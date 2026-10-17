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
