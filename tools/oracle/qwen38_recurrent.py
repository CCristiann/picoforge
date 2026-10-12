"""CPU-only Qwen3.8 references; no checkpoint or inference engine is loaded.

The numerical definitions are selected, unchanged, from pinned Transformers
source (Qwen Team / Hugging Face, Apache-2.0). Only optional kernel-dispatch
decorators are removed. We neither import that module nor execute its imports.
See docs/specs/qwen38/C.md for layouts and the separate BF16-quality question.
"""

import ast
import hashlib
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
NODES = (
    "l2norm", "causal_conv1d_update", "causal_conv1d_fn",
    "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
    "Qwen4ExpTextRMSNorm", "Qwen4ExpTextRMSNormGated", "Qwen4ExpTextGatedResidual",
)


def source_bytes(path):
    """Verify the complete source before parsing or executing selected nodes."""
    with Path(path).open("rb") as handle:
        raw = handle.read(SOURCE_LIMIT + 1)
    if len(raw) > SOURCE_LIMIT or hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Qwen4Exp upstream SHA-256 mismatch")
    return raw


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


def head_permutation(key_heads, value_heads):
    """Physical GGUF head h contains HF head p[h] (spec C §3.2)."""
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
        raise ValueError("value_heads must be a positive multiple of key_heads")
    h = torch.arange(value_heads, device="cpu")
    return (h % key_heads) * (value_heads // key_heads) + h // key_heads


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
    require_cpu(x, gain)
    if x.ndim != 3 or gain.shape not in (x.shape[1:], x.shape[-1:]):
        raise ValueError("grouped RMS shape mismatch")
    groups, width = x.shape[1:]
    with torch.device("cpu"), torch.inference_mode():
        norm = ref.Qwen4ExpTextRMSNorm(groups * width, group_size=width, eps=eps)
        norm.weight.copy_(gain.expand(groups, width).flatten() - 1)
        return norm(x.flatten(1)).reshape_as(x)


def gdn_norm_reference(ref, out, z, gain, eps=1e-6):
    """The GDN gated norm uses direct gain and sigmoid, NOT zero-centered gain."""
    require_cpu(out, z, gain)
    if out.ndim != 3 or out.shape != z.shape or gain.shape != out.shape[-1:]:
        raise ValueError("GDN norm shape mismatch")
    with torch.device("cpu"), torch.inference_mode():
        norm = ref.Qwen4ExpTextRMSNormGated(out.shape[-1], eps, activation="sigmoid")
        norm.weight.copy_(gain)
        return norm(out, z)


def hc_reference(ref, residual, gain, down, up, inject=None, block=None, eps=1e-6):
    """Upstream GatedResidual; optional combine is explicit for fixture output."""
    require_cpu(residual, gain, down, up)
    if inject is not None:
        require_cpu(inject)
    if block is not None:
        require_cpu(block)
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
        hc = ref.Qwen4ExpTextGatedResidual(config, use_combine=inject is not None)
        hc.hc_norm.weight.copy_(gain.flatten() - 1)
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


def affine_reference(q, k, v, log_decay, beta, state, normalize_qk=True):
    """Small FP64 matrix-transition oracle, independent of the scalar delta loop.

    This intentionally materializes I-beta*k*k^T. Never use it for inference
    or real-sized heads. Input/output state uses GGUF [Hv,Dv,Dk] order.
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
