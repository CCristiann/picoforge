"""CPU-only Qwen3.8 references; no checkpoint or inference engine is loaded.

The numerical definitions are selected, unchanged, from pinned Transformers
source (Qwen Team / Hugging Face, Apache-2.0). Only optional kernel-dispatch
decorators are removed. We neither import that module nor execute its imports.
See docs/specs/qwen38/C.md for layouts and the separate BF16-quality question.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_REV = "812a950d7037ba93bb624b154030e9068afdce4d"
UPSTREAM_SHA256 = "f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa"
UPSTREAM_URL = (f"https://raw.githubusercontent.com/huggingface/transformers/{UPSTREAM_REV}/"
                "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py")
UPSTREAM_DEFAULT = Path("build/qwen38-c/upstream/modeling_qwen4_exp.py")
SOURCE_LIMIT = 512 * 1024
EXPORT_LIMIT = 32 * 1024 * 1024
FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests/fixtures/qwen38/c"
NODES = (
    "l2norm", "causal_conv1d_update", "causal_conv1d_fn",
    "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
    "Qwen4ExpTextRMSNorm", "Qwen4ExpTextRMSNormGated", "Qwen4ExpTextGatedResidual",
)


def verify_source(raw):
    if len(raw) > SOURCE_LIMIT or hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Qwen4Exp upstream SHA-256 mismatch")
    return raw


def source_bytes(path):
    """Verify the complete source before parsing or executing selected nodes."""
    with Path(path).open("rb") as handle:
        return verify_source(handle.read(SOURCE_LIMIT + 1))


def fetch_upstream(path):
    """Explicit opt-in, bounded download of one pinned source file, never kernels."""
    from urllib.request import urlopen

    path = Path(path)
    if path.exists():
        source_bytes(path)
        return
    with urlopen(UPSTREAM_URL, timeout=30) as response:
        raw = verify_source(response.read(SOURCE_LIMIT + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def load_upstream(path=UPSTREAM_DEFAULT):
    raw = source_bytes(path)
    tree = ast.parse(raw, filename=str(path))
    selected, removed = [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NODES:
            removed[node.name] = [ast.unparse(d) for d in node.decorator_list]
            node.decorator_list = []
            selected.append(node)
    if {node.name for node in selected} != set(NODES) or len(selected) != len(NODES):
        raise ValueError("upstream reference definitions missing or duplicated")
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[]))
    namespace = {
        "__name__": "qwen38_pinned_reference", "torch": torch, "nn": nn, "F": F,
        "ACT2FN": {"sigmoid": torch.sigmoid, "silu": F.silu},
        # The oracle is eager CPU execution, never an export/dynamo context.
        "is_torchdynamo_exporting": lambda: False,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in NODES},
                           removed_decorators=removed)


def require_cpu(*tensors):
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("Qwen3.8 oracle accepts CPU tensors only")


def require_fp32(*tensors):
    require_cpu(*tensors)
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("this upstream adapter requires finite FP32 tensors")


def head_permutation(key_heads, value_heads):
    """Physical GGUF head h contains HF head p[h] (spec C §3.2)."""
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
        raise ValueError("value_heads must be a positive multiple of key_heads")
    h = torch.arange(value_heads, device="cpu")
    return (h % key_heads) * (value_heads // key_heads) + h // key_heads


def gate_reference(a, b, dt_bias, ssm_a):
    """FP32 gates from projected inputs, all in physical GGUF head order.

    Upstream GatedDeltaNet.forward uses -exp(A_log)*softplus(a+dt_bias).
    The converter has ALREADY stored -exp(A_log) in ssm_a: never exp it again.
    This adapter is checked against NumPy logaddexp, not treated as a second
    independent implementation of the upstream forward function.
    """
    require_cpu(a, b, dt_bias, ssm_a)
    if a.ndim != 2 or b.shape != a.shape or dt_bias.shape != a.shape[-1:] or ssm_a.shape != dt_bias.shape:
        raise ValueError("GDN projected gates shape mismatch")
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (a, b, dt_bias, ssm_a)):
        raise ValueError("GDN projected gates require finite FP32 tensors")
    if (ssm_a > 0).any():
        raise ValueError("ssm_a must contain already-nonpositive decay coefficients")
    return ssm_a * F.softplus(a + dt_bias), torch.sigmoid(b)


def conv_reference(x, taps, history, dilation=1, apply_silu=True):
    """Torch depthwise correlation, [T,C], taps [C,K], oldest-first history.

    A complete sequence and a continuation use exactly the same halo contract.
    Returned history owns a copy of the raw input, not the activated output.
    """
    require_cpu(x, taps, history)
    if x.ndim != 2 or taps.ndim != 2 or history.ndim != 2:
        raise ValueError("conv expects rank-2 tensors")
    channels, kernel = taps.shape
    if channels < 1 or kernel < 1 or dilation < 1 or x.shape[1] != channels:
        raise ValueError("invalid conv dimensions")
    history_len = (kernel - 1) * dilation
    if history.shape != (history_len, channels):
        raise ValueError("conv history shape mismatch")
    if any(t.dtype != x.dtype for t in (taps, history)):
        raise ValueError("conv dtypes must match")
    raw = torch.cat((history, x), dim=0)
    next_history = raw[-history_len:].clone() if history_len else raw[:0].clone()
    if not len(x):
        return x.clone(), next_history
    y = F.conv1d(raw.T.unsqueeze(0), taps.unsqueeze(1),
                 groups=channels, dilation=dilation).squeeze(0).T
    return (F.silu(y) if apply_silu else y).contiguous(), next_history


def gdn_reference(ref, q, k, v, log_decay, beta, state, *, algorithm="chunk",
                  chunk_size=64, normalize_qk=True):
    """Run upstream FP32 GDN with GGUF heads and state [Hv,Dv,Dk].

    q/k are post-conv and NOT scaled by 1/sqrt(Dk); upstream applies that
    scale once. normalize_qk=False is only for explicitly prepared fixtures.
    All returned tensors are canonical GGUF order, without the batch axis.
    """
    tensors = (q, k, v, log_decay, beta, state)
    require_cpu(*tensors)
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("upstream GDN reference is FP32")
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError("q/k/v shape mismatch")
    tokens, hk, dk = q.shape
    tv, hv, dv = v.shape
    p = head_permutation(hk, hv)
    if tv != tokens or min(dk, dv) < 1 or state.shape != (hv, dv, dk):
        raise ValueError("GDN value/state shape mismatch")
    if log_decay.shape != (tokens, hv) or beta.shape != (tokens, hv):
        raise ValueError("GDN gates shape mismatch")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("GDN reference requires finite tensors")
    if (log_decay > 0).any() or (beta < 0).any() or (beta > 1).any():
        raise ValueError("GDN gate domain violation")
    if algorithm not in ("chunk", "recurrent") or chunk_size < 1:
        raise ValueError("invalid GDN algorithm/chunk size")
    if tokens == 0:
        return v.clone(), state.clone()
    inverse = torch.argsort(p)
    kwargs = dict(initial_state=state[inverse].transpose(-1, -2).unsqueeze(0),
                  output_final_state=True, use_qk_l2norm_in_kernel=normalize_qk)
    if algorithm == "chunk":
        fn = ref.torch_chunk_gated_delta_rule
        kwargs["chunk_size"] = chunk_size
    else:
        fn = ref.torch_recurrent_gated_delta_rule
    with torch.inference_mode():
        out, final = fn(q.repeat_interleave(hv // hk, dim=1).unsqueeze(0),
                        k.repeat_interleave(hv // hk, dim=1).unsqueeze(0),
                        v[:, inverse].unsqueeze(0), log_decay[:, inverse].unsqueeze(0),
                        beta[:, inverse].unsqueeze(0), **kwargs)
    return out[0, :, p].contiguous(), final[0, p].transpose(-1, -2).contiguous()


def grouped_rms_reference(ref, x, gain, eps=1e-6):
    """Use the upstream zero-centered norm with an effective GGUF gain."""
    require_fp32(x, gain)
    if x.ndim != 3 or gain.shape not in (x.shape[1:], x.shape[-1:]):
        raise ValueError("grouped RMS shape mismatch")
    groups, width = x.shape[1:]
    with torch.device("cpu"), torch.inference_mode():
        norm = ref.Qwen4ExpTextRMSNorm(groups * width, group_size=width, eps=eps).float()
        norm.weight.copy_(zero_centered_weight(gain.expand(groups, width).flatten()))
        return norm(x.flatten(1)).reshape_as(x)


def zero_centered_weight(gain):
    """Do not silently change an effective GGUF gain while adapting it to HF.

    Tiny/arbitrary FP32 gains may not be in the image of FP32 (1+w).
    Such gains need original HF weights or an explicitly different oracle.
    """
    weight = gain - 1
    if not torch.equal(1 + weight, gain):
        raise ValueError("effective gain cannot be reconstructed exactly as upstream 1+w")
    return weight


def gdn_norm_reference(ref, out, z, gain, eps=1e-6):
    """The GDN gated norm uses direct gain and sigmoid, NOT zero-centered gain."""
    require_fp32(out, z, gain)
    if out.ndim != 3 or out.shape != z.shape or gain.shape != out.shape[-1:]:
        raise ValueError("GDN norm shape mismatch")
    with torch.device("cpu"), torch.inference_mode():
        norm = ref.Qwen4ExpTextRMSNormGated(out.shape[-1], eps, activation="sigmoid").float()
        norm.weight.copy_(gain)
        return norm(out, z)


def hc_reference(ref, residual, gain, down, up, inject=None, block=None, eps=1e-6):
    """Upstream GatedResidual; optional combine is explicit for fixture output."""
    require_fp32(residual, gain, down, up)
    if inject is not None:
        require_fp32(inject)
    if block is not None:
        require_fp32(block)
    if residual.ndim != 3 or down.ndim != 2:
        raise ValueError("HC rank mismatch")
    tokens, streams, width = residual.shape
    rank, flat_width = down.shape
    if flat_width != streams * width or up.shape != (flat_width, rank):
        raise ValueError("HC projection shape mismatch")
    if gain.shape != (streams, width):
        raise ValueError("HC gain shape mismatch")
    if inject is not None and inject.shape != (streams, flat_width):
        raise ValueError("HC injection shape mismatch")
    if block is not None and (inject is None or block.shape != (tokens, width)):
        raise ValueError("HC block requires matching injection")
    config = SimpleNamespace(hc_count=streams, hidden_size=width, hc_lowrank=rank,
                             rms_norm_eps=eps)
    with torch.device("cpu"), torch.inference_mode():
        hc = ref.Qwen4ExpTextGatedResidual(config, use_combine=inject is not None).float()
        hc.hc_norm.weight.copy_(zero_centered_weight(gain.flatten()))
        hc.input_mix_weight_down.weight.copy_(down)
        hc.input_mix_weight_up.weight.copy_(up)
        if inject is not None:
            hc.block_inject_weight.weight.copy_(inject)
        flat = residual.flatten(1)
        result = hc(flat)
        normed = hc.hc_norm(flat)
        low = hc.input_mix_weight_down(normed)
        gate = hc.input_mix_weight_up(F.silu(low / streams))
        outputs = dict(normed=normed.reshape_as(residual), low=low,
                       gate_logits=gate.reshape_as(residual))
        if inject is None:
            outputs["mixed"] = result
        else:
            mixed, _, injection = result
            outputs.update(mixed=mixed, injection=injection,
                           injection_logits=hc.block_inject_weight(normed))
            if block is not None:
                outputs["residual"] = residual + injection.unsqueeze(-1) * block.unsqueeze(1)
    return outputs


def affine_reference(q, k, v, log_decay, beta, state, normalize_qk=True, scale_q=True):
    """Small FP64 matrix-transition oracle, independent of the scalar delta loop.

    This intentionally materializes I-beta*k*k^T. Never use it for inference
    or real-sized heads. Input/output state uses GGUF [Hv,Dv,Dk] order.
    For the native scan's already-prepared Q/K, disable BOTH normalization
    and query scaling, so the comparison consumes the exact same FP32 values.
    """
    require_cpu(q, k, v, log_decay, beta, state)
    q, k, v, log_decay, beta, state = [t.detach().double().numpy().copy()
                                     for t in (q, k, v, log_decay, beta, state)]
    tokens, hk, dk = q.shape
    hv, dv, _ = state.shape
    if dk > 16 or dv > 16 or tokens > 129 or hv > 16:
        raise ValueError("affine oracle is limited to small synthetic cases")
    if normalize_qk:
        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    if scale_q:
        q /= np.sqrt(dk)
    output = np.empty((tokens, hv, dv), dtype=np.float64)
    for t in range(tokens):
        for h in range(hv):
            key = k[t, h % hk]
            transition = np.exp(log_decay[t, h]) * (np.eye(dk) - beta[t, h] * np.outer(key, key))
            write = beta[t, h] * np.outer(v[t, h], key)
            state[h] = state[h] @ transition + write
            output[t, h] = state[h] @ q[t, h % hk]
    return torch.from_numpy(output), torch.from_numpy(state)


def error_metrics(got, expected, local_ndim=1):
    """Errors scaled per output row/head (1 axis), or per state head (2 axes)."""
    require_cpu(got, expected)
    if got.shape != expected.shape or not 1 <= local_ndim <= expected.ndim:
        raise ValueError("invalid comparison shapes")
    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite reference/comparison result")
    if not got.numel():
        return dict(max_abs=0., scaled_inf=0., scaled_rms=0., worst_index=[])
    axes = tuple(range(-local_ndim, 0))
    diff = (got.double() - expected.double()).abs()
    scale = expected.double().abs().amax(axes, keepdim=True).clamp_min(1)
    scaled = diff / scale
    index = np.unravel_index(scaled.argmax().item(), got.shape)
    return dict(max_abs=diff.max().item(), scaled_inf=scaled.max().item(),
                scaled_rms=scaled.square().mean(axes).sqrt().max().item(),
                worst_index=[int(i) for i in index])


def write_bundle(directory, tensors, metadata):
    """Export an experimental test dataset, not a native engine ABI.

    tensors maps names to (role, axis_names, CPU tensor). Arrays own no state
    in the engine; every exported state is a separate snapshot. Never overwrite
    an existing directory. Validate and bound the payload before creating it.
    """
    payload, records = bytearray(), {}
    roles = {"input", "intermediate", "output", "state_initial", "state_final"}
    for name, (role, axes, tensor) in tensors.items():
        require_cpu(tensor)
        if tensor.dtype not in (torch.float32, torch.float64) or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid floating fixture: {name}")
        if role not in roles or len(axes) != tensor.ndim:
            raise ValueError(f"invalid role/axes: {name}")
        dtype = "<f4" if tensor.dtype == torch.float32 else "<f8"
        array = np.ascontiguousarray(tensor.detach().numpy(), dtype=dtype)
        offset = (len(payload) + 63) // 64 * 64
        if offset + array.nbytes > EXPORT_LIMIT:
            raise ValueError("fixture payload exceeds 32 MiB budget")
        raw = array.tobytes(order="C")
        payload.extend(b"\0" * (offset - len(payload)))
        payload.extend(raw)
        records[name] = dict(role=role, axes=list(axes), shape=list(array.shape), dtype=dtype,
                             strides_bytes=list(array.strides), offset_bytes=offset, nbytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest())
    document = dict(schema="qwen38-c-cpu-bundle-v1", metadata=metadata, tensors=records,
                    data=dict(path="tensors.bin", nbytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "tensors.bin").open("xb") as handle:
        handle.write(payload)
    with (directory / "manifest.json").open("x") as handle:
        handle.write(encoded)
    return document


def gdn_fixture(ref, recipe, gen, eps, tolerance):
    """Small synthetic scan fixture; no projections/dequantization are simulated."""
    t, hk, hv, dk, dv = (recipe[n] for n in ("tokens", "key_heads", "value_heads", "key_dim", "value_dim"))
    if not (0 < t <= 129 and 0 < hk <= hv <= 16 and hv % hk == 0 and 0 < dk <= 16 and 0 < dv <= 16):
        raise ValueError("fixture exceeds small affine-oracle dimensions")
    rand = lambda *shape: torch.randn(*shape, generator=gen, device="cpu", dtype=torch.float32)
    q, k, v, z = rand(t, hk, dk), rand(t, hk, dk), rand(t, hv, dv), rand(t, hv, dv)
    a, b, dt, ssm_a = rand(t, hv), rand(t, hv), rand(hv) / 4, -rand(hv).abs() - .01
    gain, initial = rand(dv), rand(hv, dv, dk)
    decay, beta = gate_reference(a, b, dt, ssm_a)
    inputs = (q, k, v, decay, beta, initial)
    chunk = gdn_reference(ref, *inputs, chunk_size=recipe["chunk_size"])
    recurrent = gdn_reference(ref, *inputs, algorithm="recurrent")
    affine = affine_reference(*inputs)
    tensors = {}

    def add(name, role, axes, value):
        tensors["gdn." + name] = (role, axes.split(), value)

    for name, value in (("q", q), ("k", k)):
        add(name, "input", "T Hk Dk", value)
        add(name + "_l2", "intermediate", "T Hk Dk", ref.l2norm(value))
    for name, value in (("v", v), ("z", z)):
        add(name, "input", "T Hv Dv", value)
    for name, value in (("a", a), ("b", b)):
        add(name, "input", "T Hv", value)
    for name, value in (("dt_bias", dt), ("ssm_a", ssm_a)):
        add(name, "input", "Hv", value)
    add("norm_gain", "input", "Dv", gain)
    add("log_decay", "intermediate", "T Hv", decay)
    add("beta", "intermediate", "T Hv", beta)
    add("state_initial", "state_initial", "Hv Dv Dk", initial)
    for name, result in (("chunk", chunk), ("recurrent", recurrent), ("affine64", affine)):
        add("output_" + name, "output", "T Hv Dv", result[0])
        add("state_" + name, "state_final", "Hv Dv Dk", result[1])
    add("norm_sigmoid", "output", "T Hv Dv", gdn_norm_reference(ref, chunk[0], z, gain, eps))
    comparisons = {}
    for name, result in (("chunk", chunk), ("recurrent", recurrent)):
        for part, got, expected, axes in (("output", result[0], affine[0], 1), ("state", result[1], affine[1], 2)):
            metrics = error_metrics(got, expected, axes)
            if metrics["scaled_inf"] > tolerance:
                raise ValueError(f"{name}/{part} exceeds the proposed tolerance: {metrics}")
            comparisons[name + "_vs_affine64_" + part] = dict(limit=tolerance, **metrics)
    return tensors, comparisons


def export_synthetic(source, directory):
    """Create small CPU fixtures from the checked-in recipe and pinned source."""
    ref = load_upstream(source)
    recipe_raw = (FIXTURE_ROOT / "manifest.json").read_bytes()
    recipe = json.loads(recipe_raw)
    if recipe["upstream"]["sha256"] != UPSTREAM_SHA256 or recipe["upstream"]["revision"] != UPSTREAM_REV:
        raise ValueError("fixture recipe and adapter source pins disagree")
    gen = torch.Generator(device="cpu").manual_seed(recipe["seed"])
    tensors, comparisons = gdn_fixture(ref, recipe["gdn"], gen, recipe["rms_epsilon"],
                                       recipe["tolerances_proposed"]["gdn"])
    rand = lambda *shape: torch.randn(*shape, generator=gen, device="cpu", dtype=torch.float32)

    def add(prefix, name, role, axes, value):
        tensors[prefix + "." + name] = (role, axes.split(), value)

    for conv in recipe["conv"]:
        t, c, kernel, d = (conv[n] for n in ("tokens", "channels", "kernel", "dilation"))
        x, taps, history = rand(t, c), rand(c, kernel), rand((kernel - 1) * d, c)
        raw, last = conv_reference(x, taps, history, d, False)
        for name, role, axes, value in (
            ("x", "input", "T C", x), ("taps", "input", "C K", taps),
            ("history_initial", "state_initial", "history_time C", history),
            ("raw", "output", "T C", raw), ("silu", "output", "T C", F.silu(raw)),
            ("history_final", "state_final", "history_time C", last),
        ):
            add(conv["name"], name, role, axes, value)
    t, h, width, rank = (recipe["hc"][n] for n in ("tokens", "streams", "width", "rank"))
    x, down, up = rand(t, h, width), rand(rank, h * width) / 5, rand(h * width, rank) / 5
    inject, block = rand(h, h * width) / 5, rand(t, width)
    # Dyadic effective gain survives reconstruction of upstream's (1+w) exactly.
    gain = torch.randint(-4, 9, (h, width), generator=gen, device="cpu").float() / 4
    for name, axes, value in (("residual", "T H E", x), ("gain", "H E", gain),
                              ("down", "rank HE", down), ("up", "HE rank", up),
                              ("inject", "H HE", inject), ("block", "T E", block)):
        add("hc", name, "input", axes, value)
    outputs = hc_reference(ref, x, gain, down, up, inject, block, recipe["rms_epsilon"])
    axes = dict(normed="T H E", low="T rank", gate_logits="T H E", mixed="T E",
                injection_logits="T H", injection="T H", residual="T H E")
    for name, value in outputs.items():
        role = "output" if name in ("mixed", "residual") else "intermediate"
        add("hc", "result_" + name, role, axes[name], value)
    add("norm", "x", "input", "T H E", x)
    add("norm", "shared_gain", "input", "E", gain[0])
    add("norm", "output", "output", "T H E", grouped_rms_reference(ref, x, gain[0], recipe["rms_epsilon"]))
    metadata = dict(recipe=recipe, comparisons=comparisons, device="cpu", accumulator="FP32 upstream; FP64 affine",
                    torch_version=torch.__version__, numpy_version=np.__version__, threads=torch.get_num_threads(),
                    source_url=UPSTREAM_URL, removed_decorators=ref.removed_decorators,
                    recipe_sha256=hashlib.sha256(recipe_raw).hexdigest(),
                    analytic_sha256=hashlib.sha256((FIXTURE_ROOT / "analytic.json").read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return write_bundle(directory, tensors, metadata)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=UPSTREAM_DEFAULT)
    parser.add_argument("--fetch-upstream", action="store_true", help="explicitly download pinned Python source only")
    parser.add_argument("--synthetic", action="store_true", help="export input/output/state tensors")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("build/qwen38-c/fixtures"))
    args = parser.parse_args()
    if not args.fetch_upstream and not args.synthetic:
        parser.error("select --fetch-upstream and/or --synthetic")
    torch.set_num_threads(1)
    try:
        if args.fetch_upstream:
            fetch_upstream(args.source)
            print(f"Verified upstream {UPSTREAM_REV}: {args.source}")
        if args.synthetic:
            document = export_synthetic(args.source, args.out)
            print(f"Exported {len(document['tensors'])} CPU tensors, {document['data']['nbytes']} bytes: {args.out}")
            print(f"SHA-256 tensors.bin: {document['data']['sha256']}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"qwen38 oracle: {error}\n")


if __name__ == "__main__":
    main()
