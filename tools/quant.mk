# Quantization pipeline (tools/quant/): the qw4q encoder library and its tests.
# CPU only, no model files. Same float flags as the DwarfStar goldens: -O3, no
# fast-math, so the compat encoders reproduce their rounding bit for bit.

QW4Q_CFLAGS := -O3 -std=c11 -Wall -Wextra -Werror -Isrc

# The DwarfStar translation unit keeps its own flags (no -Werror: not our code).
build/quant-ds4_quants.o: tools/quant/third_party/ds4_quants.c tools/quant/third_party/quants.h
	@mkdir -p build
	$(CC) -O3 -std=c11 -fPIC -c -o $@ $<

build/libqw4q.dylib: build/quant-ds4_quants.o tools/quant/qw4q.c tools/quant/third_party/ds4_quants.c tools/quant/third_party/quants.h tools/quant/qw4q.h src/qwen4_iq2_grid.inc
	@mkdir -p build
	$(CC) $(QW4Q_CFLAGS) -dynamiclib -fPIC -o $@ tools/quant/qw4q.c build/quant-ds4_quants.o -lm -pthread

test-quant: build/libqw4q.dylib build/qwen4-g1/graph.gguf build/qwen4-g1/graph-q4.gguf
	@$(PYTHON) tests/test_qw4q.py
	@$(PYTHON) tests/test_quant_io.py
	@$(PYTHON) tests/test_quantize_layer.py
	@$(PYTHON) tests/test_layer_lab.py
	@$(PYTHON) tests/test_mixed_fixture.py
	@$(PYTHON) tests/test_build_mixed.py

# Engine side (GPU, synthetic checkpoint): QW4_ACT_LOG capture. Part of `make test-qwen4`.
test-quant-engine: pf build/qwen4-g2/cli.gguf build/qwen4-g2/cli-mixed.gguf build/qwen4-g1/test_forward build/qwen4-g2/test_forward build/qwen4-g1/graph-mixed.gguf build/qwen4-g1/graph-built-mixed.gguf $(METALLIB)
	@$(PYTHON) tests/test_act_log.py
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g1/test_forward --gguf build/qwen4-g1/graph-mixed.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/graph-mixed.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 PICOFORGE_GROUPED_MIN_PAIRS=1 $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/graph-mixed.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 QW4_REF_KERNELS=exact $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/graph-mixed.gguf -v
	$(QW4_LOCK) $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g1/test_forward --gguf build/qwen4-g1/graph-built-mixed.gguf -v
	$(QW4_LOCK) env MTL_DEBUG_LAYER=1 $(QW4_TEST) tests/test_qwen4_forward.py --native build/qwen4-g2/test_forward --gguf build/qwen4-g1/graph-built-mixed.gguf -v
	$(QW4_LOCK) env QW4_CLI_GGUF=$(abspath build/qwen4-g2/cli-mixed.gguf) $(QW4_TEST) tests/test_qwen4_cli.py -v

# Picoforge mixed experts (notes M4): the synthetic graph with one Q4 expert per trunk layer.
build/qwen4-g1/graph-mixed.gguf: tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py
	$(PYTHON) tools/synth/make_qwen4_graph.py --output $@ --mixed-hot 1

# M4c: the published Q4 layout (byte source of the hot experts) and a file assembled by build_mixed.py.
build/qwen4-g1/graph-q4.gguf: tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py
	$(PYTHON) tools/synth/make_qwen4_graph.py --output $@ --q4-experts

build/qwen4-g1/graph-built-mixed.gguf: tests/test_build_mixed.py tools/quant/build_mixed.py build/qwen4-g1/graph.gguf build/qwen4-g1/graph-q4.gguf
	$(PYTHON) tests/test_build_mixed.py --fixture $@

build/qwen4-g2/cli-mixed.gguf: tools/synth/make_qwen4_cli.py tools/synth/make_qwen4_graph.py tools/synth/make_qwen4_small.py tools/oracle/qwen4_config.py tests/fixtures/qwen4/text/manifest.json
	$(QW4_ENV) $(PYTHON) -B $< --reference-root "$(QW4_REFERENCE_ROOT)" --output $@ --mixed-hot 1

.PHONY: test-quant test-quant-engine
