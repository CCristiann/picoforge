# Qwen3.8: native GGUF storage increment

Implemented on 2026-10-12 after the first successful local Q2 generation and
verified against the complete [phase-0 checkpoint](qwen38-q2-baseline.md).
This is C11/Metal code inside picoforge, independent of DwarfStar's runtime.
It establishes storage and GPU addressing, **not native Qwen3.8 inference**.

## Interfaces and limits

`src/gguf_reader.c` implements bounded little-endian reads, overflow checks,
strings and metadata values. `src/gguf.c` inventories the selected `qwen4exp`
GGUF v3 architecture. It supports the eight formats actually present: F32,
F16, BF16, Q8_0, IQ2_XXS, Q2_K, Q4_K and MXFP4. Format IDs remain distinct
from the existing safetensors/row-quantization types; no weights are expanded
or relabeled as a different numeric representation.

The CPU header mapping is capped at 128 MiB. Parsing touches metadata and
tensor descriptors, not tensor payload. There are limits of 100,000 tensor
and metadata records, 16 MiB per string, two million array elements, and four
tensor dimensions. Keys and tensor names use the ASCII identifier subset of
the selected architecture; tokenizer value strings remain arbitrary bytes.
Duplicate names/keys, nested arrays, invalid types, ranks, strides, alignment,
overlaps, arithmetic overflow and truncated files are rejected before use.
Inspection validates layout and explicitly reports `payload_verified=false`:
the separately recorded SHA-256 authenticates the complete artifact.

`Gguf` owns the file descriptor, bounded CPU mapping, metadata and tensor
descriptors. Metadata value ranges borrow the header and remain valid until
`gguf_close`. Tensor shape is physical GGUF order: dimension zero is the
contiguous row. Q2_K down projections retain **768 physical columns**;
the future graph must read the **640 logical inputs** from model metadata.

`src/gguf_plan.c` creates `GgufBinding` records with a copied tensor descriptor,
buffer ID, inner offset and explicit plan owner. Plans must not be copied by
value after construction because bindings identify that owner. All spans are
checked against file size, aligned to host pages and bounded by a supplied
buffer cap. Splits occur at tensor boundaries so no tensor crosses buffers;
adjacent views can overlap by at most one page. The BF16 n-gram table has
page-aligned boundaries and is excluded even if resident tensors follow it.

`src/gpu_weights.m` wraps each planned file span in a separate shared,
read-only mmap and no-copy `MTLBuffer`. It verifies the file size again,
uses the actual device buffer limit, and rejects resident view totals above
the recommended working set **before runtime/context buffers are added**.
That recommendation is a separate constraint from each buffer's maximum.
There is no complete payload copy, format conversion, n-gram mapping or
explicit whole-model prefetch in this increment.

Each Metal buffer owns its mmap via its deallocator. `GpuWeightStore` owns
the plan and strong buffer references; callers borrow bindings and buffers.
The existing engine uses ordinary `MTLCommandBuffer` objects that retain
encoded resources. Those references allow pending GPU work to finish after
the C store is closed, then release the mapping. Callers using unretained
command buffers or Metal 4 command buffers must keep their own strong
ownership through completion; early store closure is not supported there.
Checkpoint contents must remain immutable while mappings are live.

## Verification

```sh
make
make test-gguf test-gpu-weights
./picoforge --inspect-gguf \
  models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
MTL_DEBUG_LAYER=1 ./build/test_gpu_weights \
  models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
make test-all
```

The default tests use tiny fixtures, not the large model. They cover every
header truncation in the fixture, all eight packed layouts, malformed
metadata, overflows, overlapping tensors, forced small view caps, a disk-only
table between resident tensors, foreign owners, a changed file size, and
mapping release. Binary parsing and planning run under ASan/UBSan. The Metal
test enables API validation and closes both the GGUF reader and C store
**before submitting** the command; reads must still succeed, and the live
mapping counter must return to its initial value after resource release.

The actual Q2 artifact also passes:

- All **1,256 tensor descriptors** agree with the independent Python inventory.
- **1,255 resident tensors** fit in **two views**; the 95.37 GiB n-gram table is excluded.
- **3,765 GPU byte probes** match `pread` at the first, middle and last byte of
  every resident tensor, including after closing the C owners.
- View spans total **44,795,625,472 bytes**. The probe reports approximately
  41.72 GiB of Metal allocations and returns its mapping counter to zero.

These are storage checks, not dequantization or logits tests. The byte probe
touches only three positions per tensor; its duration and sparse memory
samples are **not** a full-residency or inference benchmark. Compare
neither its allocation total nor its duration directly with DwarfStar's
complete graph. Evidence is under
[`bench/qwen38_q2_20261012_storage/`](../bench/qwen38_q2_20261012_storage/).

## Remaining work

The existing Qwen3 safetensors, CPU/GPU, sharding checks, quantization and
speculative paths are retained. Qwen3.8 configuration semantics, independent
dequantization references, hybrid operators, bounded n-gram I/O and the full
generation graph are still required. Start those with small numerical fixtures;
do not infer model support from a successful container inventory.

Format references: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
[pinned DwarfStar Qwen documentation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/QWEN38_FLASH_NEXT.md).
Metal ownership follows Apple's
[no-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
and the SDK 26.5 `MTLCommandBuffer.retainedReferences` declaration, checked on
the actual M5 Pro with Metal API validation.
