# R0/R1 measurement and decode-speed targets. Included by tools/qwen38.mk.

# Metal capability probe (R0.1): plain ObjC like build/memory_budget, no engine code.
build/qwen38-capabilities: tools/probe/qwen38_capabilities.m
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fobjc-arc -o $@ $< $(FRAMEWORKS)

# The recipe owns the GPU lock; the test only runs the probe (no nested wrapper).
test-qwen38-r0-tools: build/qwen38-capabilities
	$(Q38_LOCK) $(Q38_TEST) tests/test_qwen38_capabilities.py --native $< -v

.PHONY: test-qwen38-r0-tools
