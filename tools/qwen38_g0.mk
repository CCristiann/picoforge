# G0, common GPU lifecycle. Coordinator-owned.
# The ledger test is CPU-only (no Metal device), so it takes no GPU lock.
build/qwen38-g/test_gpu_lifecycle: tests/test_qwen38_gpu_lifecycle.m src/qwen38_gpu.m $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.m,$^) $(FRAMEWORKS) -o $@

test-qwen38-g0-ledger: build/qwen38-g/test_gpu_lifecycle
	$(Q38_ENV) ./$<

.PHONY: test-qwen38-g0-ledger
