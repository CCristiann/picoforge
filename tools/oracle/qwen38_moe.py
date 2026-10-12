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


def synthetic(config=FixtureConfig(), tokens=7, seed=3811):
    h, e, i, s = (config.hidden_size, config.num_experts,
                  config.moe_intermediate_size, config.shared_expert_intermediate_size)
    shapes = {"gate.weight": (e, h), "experts.gate_up_proj": (e, 2*i, h),
              "experts.down_proj": (e, h, i), "shared_expert.gate_proj.weight": (s, h),
              "shared_expert.up_proj.weight": (s, h), "shared_expert.down_proj.weight": (h, s),
              "shared_expert_gate.weight": (1, h)}
    rng = np.random.default_rng(seed)
    weights = {name: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
               for name, shape in shapes.items()}
    return rng.standard_normal((tokens, h)).astype(np.float32), weights


def upstream_forward(config, x, weights, cache=CACHE):
    """No full model import, device discovery, checkpoint or GPU allocation."""
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
        return {name: value.detach().numpy().copy() for name, value in trace.items()}
