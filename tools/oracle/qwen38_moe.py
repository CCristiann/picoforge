"""CPU-only Qwen3.8 MoE fixtures; no engine API or quantization code.

The independent oracle executes hash-pinned Transformers class bodies.
Only the optional expert-dispatch decorator is removed (eager CPU path).
Upstream: Hugging Face / Qwen, Apache-2.0; LICENSE is cached with the source.
"""
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REVISION = "812a950d7037ba93bb624b154030e9068afdce4d"
SOURCE_DIR = "src/transformers/models/qwen4_exp/"
HASHES = {
    "modeling_qwen4_exp.py": "f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa",
    "configuration_qwen4_exp.py": "ac0950ab34516d26fe4f6b28cffe0649d763449effe44e42b5913463806f7e0b",
    "LICENSE": "77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049",
}
CACHE = Path(__file__).resolve().parents[2] / "build/qwen38-f/upstream"
CLASSES = tuple("Qwen4ExpText" + n for n in
                ("MLP", "Experts", "TopKRouter", "SparseMoeBlock"))


def checked_source(cache, name):
    data = (Path(cache) / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError(f"upstream hash mismatch: {name}")
    return data


def prepare_upstream(cache=CACHE):
    """Explicit network step; normal oracle/test execution is offline."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        path = cache / name
        if path.exists():
            checked_source(cache, name)
            continue
        relative = name if name == "LICENSE" else SOURCE_DIR + name
        url = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(1 << 20)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"download hash mismatch: {name}")
        with path.open("xb") as output:
            output.write(data)


def upstream_class(cache=CACHE):
    sources = {name: checked_source(cache, name) for name in HASHES}
    config_tree = ast.parse(sources["configuration_qwen4_exp.py"])
    config = next(n for n in config_tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpTextConfig")
    default = next(n.value for n in config.body if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == "norm_topk_prob")
    if ast.literal_eval(default) is not True:
        raise ValueError("unexpected upstream norm_topk_prob default")
    tree = ast.parse(sources["modeling_qwen4_exp.py"])
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASSES]
    if tuple(n.name for n in nodes) != CLASSES:
        raise ValueError("unexpected upstream class selection")
    for node in nodes:
        expected = ["use_experts_implementation"] if node.name.endswith("Experts") else []
        if [ast.unparse(d) for d in node.decorator_list] != expected:
            raise ValueError("unexpected upstream decorator")
        node.decorator_list = []
    namespace = {"torch": torch, "nn": nn, "F": F, "ACT2FN": {"silu": F.silu}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_qwen4_moe", "exec"), namespace)
    return namespace[CLASSES[-1]]


@dataclass(frozen=True)
class FixtureConfig:
    hidden_size: int = 8
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 5
    shared_expert_intermediate_size: int = 3
    norm_topk_prob: bool = True
    hidden_act: str = "silu"

    def __post_init__(self):
        for name in ("hidden_size", "num_experts", "num_experts_per_tok",
                     "moe_intermediate_size", "shared_expert_intermediate_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("top-k exceeds expert count")
        if type(self.norm_topk_prob) is not bool or self.hidden_act != "silu":
            raise ValueError("expected boolean norm_topk_prob and silu")


def weight_shapes(config):
    h, e, i, s = (config.hidden_size, config.num_experts,
                  config.moe_intermediate_size, config.shared_expert_intermediate_size)
    return {"gate.weight": (e, h), "experts.gate_up_proj": (e, 2*i, h),
              "experts.down_proj": (e, h, i), "shared_expert.gate_proj.weight": (s, h),
              "shared_expert.up_proj.weight": (s, h), "shared_expert.down_proj.weight": (h, s),
              "shared_expert_gate.weight": (1, h)}


def synthetic(config=FixtureConfig(), tokens=7, seed=3811):
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    weights = {name: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
               for name, shape in weight_shapes(config).items()}
    return rng.standard_normal((tokens, config.hidden_size)).astype(np.float32), weights


def validate_inputs(config, x, weights):
    shapes = weight_shapes(config)
    if set(weights) != set(shapes):
        raise ValueError("unexpected or missing weight names")
    if x.ndim != 2 or x.shape[1] != config.hidden_size:
        raise ValueError("expected X[T,H]")
    for name, array in {"input": x, **weights}.items():
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{name}: expected finite FP32")
        if name != "input" and array.shape != shapes[name]:
            raise ValueError(f"{name}: incorrect shape")


def route_logits(logits, k, normalize=True):
    """Native proposed tie policy: equal FP32 probabilities -> smaller ID."""
    if (logits.ndim != 2 or logits.dtype != np.float32 or
            not np.isfinite(logits).all() or type(k) is not int or
            not 1 <= k <= logits.shape[1] or type(normalize) is not bool):
        raise ValueError("invalid FP32 logits / top-k / normalization")
    # A negative overflow in the shifted logits is harmless: exp(-inf)=0.
    with np.errstate(over="ignore", under="ignore"):
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    ids = np.argsort(-probabilities, axis=-1, kind="stable")[:, :k]
    scores = np.take_along_axis(probabilities, ids, axis=-1)
    if normalize:
        scores /= scores.sum(axis=-1, keepdims=True)
    return probabilities, ids.astype(np.uint32), scores


def sigmoid(x):
    result = np.empty_like(x)
    positive = x >= 0
    with np.errstate(under="ignore"):
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1 + exp_x)
    return result


def numpy_forward(config, x, weights):
    """Transparent FP32 reference, validated against the upstream oracle."""
    validate_inputs(config, x, weights)
    t, h = x.shape
    k, i = config.num_experts_per_tok, config.moe_intermediate_size
    trace = {}
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        trace["logits"] = x @ weights["gate.weight"].T
        if not np.isfinite(trace["logits"]).all():
            raise FloatingPointError("nonfinite router projection")
        p, ids, scores = route_logits(trace["logits"], k, config.norm_topk_prob)
        trace.update(probabilities=p, ids=ids, weights=scores)
        for name in ("gate", "up", "mid"):
            trace[name] = np.empty((t, k, i), dtype=np.float32)
        trace["down"] = np.empty((t, k, h), dtype=np.float32)
        routed = np.zeros((t, h), dtype=np.float32)
        for token in range(t):
            for slot, expert in enumerate(ids[token]):
                g, u = np.split(weights["experts.gate_up_proj"][expert] @ x[token], 2)
                mid = (g * sigmoid(g)) * u
                down = weights["experts.down_proj"][expert] @ mid
                for name, value in (("gate", g), ("up", u), ("mid", mid), ("down", down)):
                    trace[name][token, slot] = value
                routed[token] += scores[token, slot] * down
        gs = x @ weights["shared_expert.gate_proj.weight"].T
        us = x @ weights["shared_expert.up_proj.weight"].T
        ms = gs * sigmoid(gs) * us
        shared = ms @ weights["shared_expert.down_proj.weight"].T
        logit = x @ weights["shared_expert_gate.weight"].T
        trace.update(routed=routed, shared=shared, shared_logit=logit,
                     shared_gate=gs, shared_up=us, shared_mid=ms,
                     y=routed + sigmoid(logit) * shared)
    if any(not np.isfinite(a).all() for a in trace.values()):
        raise FloatingPointError("nonfinite MoE intermediate")
    return trace


def upstream_forward(config, x, weights, cache=CACHE):
    """No full model import, device discovery, checkpoint or GPU allocation."""
    validate_inputs(config, x, weights)
    with torch.device("cpu"), torch.no_grad():
        block = upstream_class(cache)(config).float().eval()
        block.load_state_dict({name: torch.from_numpy(np.array(w, copy=True))
                               for name, w in weights.items()}, strict=True)
        xt = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
        trace = {}
        handles = [module.register_forward_hook(
            lambda _m, _args, out, key=key: trace.__setitem__(key, out))
            for key, module in (("route", block.gate), ("routed", block.experts),
                                ("shared", block.shared_expert),
                                ("shared_logit", block.shared_expert_gate))]
        y = block(xt[None])[0]
        for handle in handles:
            handle.remove()
        logits, scores, ids = trace.pop("route")
        trace.update(y=y, logits=logits, weights=scores, ids=ids)
        result = {name: value.detach().numpy().copy() for name, value in trace.items()}
        if any(not np.isfinite(a).all() for a in result.values()):
            raise FloatingPointError("nonfinite upstream MoE output")
        return result
