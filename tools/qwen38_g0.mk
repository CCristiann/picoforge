# G0, common GPU lifecycle. Coordinator-owned.
# The binary opens a Metal device, so it runs under the common GPU lock.
build/qwen38-g/test_gpu_lifecycle: tests/test_qwen38_gpu_lifecycle.m src/qwen38_gpu.m $(Q38_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(Q38_FLAGS) $(Q38_SAN) -fobjc-arc $(filter %.m,$^) $(FRAMEWORKS) -o $@

test-qwen38-g0: build/qwen38-g/test_gpu_lifecycle
	$(Q38_LOCK) $(Q38_ENV) MTL_DEBUG_LAYER=1 ./$<

.PHONY: test-qwen38-g0
