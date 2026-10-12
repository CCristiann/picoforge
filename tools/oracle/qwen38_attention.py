"""CPU-only QSA reference, independent of picoforge's future C/Metal code.

Official Apache-2.0 Transformers definitions are loaded unchanged from a
hash-checked local source file. No network access occurs on import/load.
Use --fetch-source explicitly to acquire that file into this worktree.
This tests checkpoint arithmetic, not Q2-versus-original-BF16 quality.

From the worktree root, with the existing tools venv's Python:
  python tools/oracle/qwen38_attention.py --fetch-source   # first use only
  python tests/test_qwen38_attention.py -v
  python tools/oracle/qwen38_attention.py --case boundary --output build/qsa-new
"""
from __future__ import annotations

import ast
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import struct
import subprocess
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
import torch
from torch import nn

REVISION = "db2304e8694db3ecd6f39d353beba361688afe97"
SOURCE_HASH = "f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa"
SOURCE_URL = (f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/"
              "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py")
SOURCE_LIMIT = 256 * 1024
INPUT_ELEMENTS_LIMIT = 32_000_000  # 128 MB F32, before diagnostic copies
DEFINITIONS = frozenset({
    "Qwen4ExpTextRMSNorm", "Qwen4ExpTextRotaryEmbedding", "rotate_half",
    "apply_rotary_pos_emb", "Qwen4ExpTextQSAIndexer", "repeat_kv",
    "eager_attention_forward",
})


def verified_source(raw: bytes) -> bytes:
    if len(raw) > SOURCE_LIMIT or hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError("HF reference source does not match the pinned SHA-256")
    return raw


def fetch_source(path: Path) -> None:
    """Explicit, bounded download; never replace existing local evidence."""
    if path.exists():
        raise FileExistsError(path)
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        raw = verified_source(response.read(SOURCE_LIMIT + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def load_reference(path: Path) -> SimpleNamespace:
    with path.open("rb") as stream:
        raw = verified_source(stream.read(SOURCE_LIMIT + 1))
    tree = ast.parse(raw)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in DEFINITIONS]
    if {node.name for node in nodes} != DEFINITIONS:
        raise ValueError("Incomplete HF definition allowlist")
    # The dynamic-context decorator is unnecessary for rope_type='default'.
    # Replacing that wrapper is the sole semantic adapter; method bodies stay
    # unchanged. Config below rejects dynamic/YaRN and non-textual positions.
    namespace = {"torch": torch, "nn": nn, "math": math,
                 "dynamic_rope_update": lambda function: function}
    future = ast.ImportFrom(module="__future__", level=0,
                            names=[ast.alias(name="annotations")])
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in DEFINITIONS})


@dataclass(frozen=True)
class Config:
    width: int = 16
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    index_heads: int = 2
    index_dim: int = 8
    rotary_dim: int = 4
    ratio: int = 4
    budget: int = 8
    eps: float = 1e-6
    theta: float = 10000.0

    def __post_init__(self):
        dims = list(asdict(self).values())[:-2]
        if any(type(value) is not int or value <= 0 for value in dims):
            raise ValueError("Dimensions must be positive integers")
        if self.heads % self.kv_heads or self.budget % self.ratio:
            raise ValueError("Non-integral GQA group or block budget")
        if self.rotary_dim % 2 or self.rotary_dim > min(self.head_dim, self.index_dim):
            raise ValueError("Invalid partial rotary dimension")
        if not all(math.isfinite(x) and x > 0 for x in (self.eps, self.theta)):
            raise ValueError("Invalid norm epsilon or rotary base")

    def hf(self):
        # Text repeats its position over all three coordinates; sections do
        # not alter the result. Keep their total equal to the rotary pair count.
        pairs = self.rotary_dim // 2
        return SimpleNamespace(
            hidden_size=self.width, head_dim=self.head_dim,
            num_attention_heads=self.heads, indexer_n_heads=self.index_heads,
            indexer_kv_heads=1, indexer_head_dim=self.index_dim,
            indexer_budget=self.budget, indexer_compress_ratio=self.ratio,
            rms_norm_eps=self.eps, max_position_embeddings=262144,
            rope_parameters={"rope_type": "default", "rope_theta": self.theta,
                             "partial_rotary_factor": self.rotary_dim / self.head_dim,
                             "mrope_section": [(pairs + 2) // 3, (pairs + 1) // 3, pairs // 3]},
        )


@dataclass
class Inputs:
    """Full contiguous text prefix, already projected; effective F32 gamma."""
    qg: np.ndarray
    k: np.ndarray
    v: np.ndarray
    iq: np.ndarray
    ik: np.ndarray
    gamma_q: np.ndarray
    gamma_k: np.ndarray
    gamma_iq: np.ndarray
    gamma_ik: np.ndarray

    def arrays(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def validate(self, cfg: Config) -> int:
        n = self.qg.shape[0] if self.qg.ndim else 0
        shapes = {
            "qg": (n, cfg.heads, 2, cfg.head_dim),
            "k": (n, cfg.kv_heads, cfg.head_dim),
            "v": (n, cfg.kv_heads, cfg.head_dim),
            "iq": (n, cfg.index_heads, cfg.index_dim), "ik": (n, cfg.index_dim),
            "gamma_q": (cfg.head_dim,), "gamma_k": (cfg.head_dim,),
            "gamma_iq": (cfg.index_dim,), "gamma_ik": (cfg.index_dim,),
        }
        if not 0 < n <= 32768:
            raise ValueError("Oracle prefix must contain 1..32768 tokens")
        if sum(x.size for x in self.arrays().values()) > INPUT_ELEMENTS_LIMIT:
            raise ValueError("Oracle input budget exceeded")
        for name, value in self.arrays().items():
            if value.dtype != np.float32 or value.shape != shapes[name]:
                raise ValueError(f"{name}: expected F32 {shapes[name]}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite input")
        return n


def synthetic(cfg: Config, tokens: int, seed: int = 3811) -> Inputs:
    """Small CPU fixtures; RNG is local and never modifies torch's global seed."""
    if type(tokens) is not int or not 0 < tokens <= 32768:
        raise ValueError("Invalid fixture length")
    elements = (tokens * (2 * cfg.heads * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim
                         + (cfg.index_heads + 1) * cfg.index_dim)
                + 2 * cfg.head_dim + 2 * cfg.index_dim)
    if elements > INPUT_ELEMENTS_LIMIT:
        raise ValueError("Oracle input budget exceeded")
    rng = np.random.default_rng(seed)
    def values(shape):
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    def gamma(n):
        return (rng.integers(4, 13, size=n) / 8).astype(np.float32)
    return Inputs(
        values((tokens, cfg.heads, 2, cfg.head_dim)),
        values((tokens, cfg.kv_heads, cfg.head_dim)),
        values((tokens, cfg.kv_heads, cfg.head_dim)),
        values((tokens, cfg.index_heads, cfg.index_dim)),
        values((tokens, cfg.index_dim)), gamma(cfg.head_dim), gamma(cfg.head_dim),
        gamma(cfg.index_dim), gamma(cfg.index_dim),
    )


def _set_gamma(norm, array):
    gamma = torch.from_numpy(np.array(array, copy=True))
    offset = gamma - 1.0
    # Never hide an additional rounding boundary in the HF offset adapter.
    if not torch.equal(offset + 1.0, gamma):
        raise ValueError("Effective gamma cannot round-trip through HF offset")
    with torch.no_grad():
        norm.weight.copy_(offset)


class _ProjectedQK(nn.Module):
    """Inject specified projected inputs, without replacing indexer math."""
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, hidden_states):
        if hidden_states.shape[:2] != self.values.shape[:2]:
            raise ValueError("Projected input and query lengths disagree")
        return self.values


class _PrefixKeys:
    """Read-only prefix input to HF's cache hook, not a native cache oracle."""
    def __init__(self, prefix):
        self.prefix = prefix

    def update_indexer(self, current, layer_idx):
        if layer_idx != 0:
            raise ValueError("The oracle evaluates one isolated layer")
        return torch.cat((self.prefix, current), dim=1)


@torch.no_grad()
def evaluate(hf, cfg: Config, inputs: Inputs, start: int = 0) -> dict[str, np.ndarray]:
    """Recompute one prefix; return suffix queries at absolute start..N-1.

    The selected mask comes from official HF, not our own top-k. Eager FP32
    and CPU SDPA FP64 consume the same prepared Q/K/V and mask, separating
    softmax reduction error from norm/RoPE error and discrete selection.
    """
    n = inputs.validate(cfg)
    if type(start) is not int or not 0 <= start < n:
        raise ValueError("Invalid query start")
    if (n - start) * n * cfg.heads > 8_000_000:
        raise ValueError("Oracle matrix budget exceeded; evaluate fewer suffix queries")
    if cfg.width * (cfg.index_heads + 1) * cfg.index_dim > 16_000_000:
        raise ValueError("Oracle projection allocation budget exceeded")
    # Retain no native state. Repeated calls always recompute the prefix.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        return _evaluate_cpu(hf, cfg, inputs, start, n)


def _evaluate_cpu(hf, cfg, inputs, start, n):
    data = {name: torch.from_numpy(np.array(value, copy=True))
            for name, value in inputs.arrays().items()}
    positions = torch.arange(n, dtype=torch.long)[None, None, :]
    hidden = torch.zeros(1, n - start, cfg.width)
    rope = hf.Qwen4ExpTextRotaryEmbedding(cfg.hf())
    cos, sin = rope(hidden, positions)
    indexer = hf.Qwen4ExpTextQSAIndexer(cfg.hf(), layer_idx=0).eval()
    current_qk = torch.cat((data["iq"][start:].flatten(1), data["ik"][start:]), -1)
    indexer.index_qk_proj = _ProjectedQK(current_qk[None])
    _set_gamma(indexer.q_layernorm, inputs.gamma_iq)
    _set_gamma(indexer.k_layernorm, inputs.gamma_ik)
    causal = (torch.arange(n)[None, :] <= torch.arange(start, n)[:, None])[None, None]
    # Observe the official pooling/norm output of the final query. Keeping
    # only the final hook value bounds diagnostic storage for long prefixes.
    observed = {}
    def pooled_hook(_module, arguments, output):
        observed["pooled"] = arguments[0]
        observed["normed"] = output
    hook = indexer.k_layernorm.register_forward_hook(pooled_hook)
    try:
        mask = indexer(hidden, (cos, sin), causal,
                       _PrefixKeys(data["ik"][:start][None]) if start else None)
    finally:
        hook.remove()
    def norm(name, gamma, dim):
        op = hf.Qwen4ExpTextRMSNorm(dim, eps=cfg.eps)
        _set_gamma(op, gamma)
        return op(name)
    q = norm(data["qg"][start:, :, 0][None], inputs.gamma_q, cfg.head_dim)
    k = norm(data["k"][None], inputs.gamma_k, cfg.head_dim)
    q = hf.apply_rotary_pos_emb(q, cos=cos[:, start:], sin=sin[:, start:], unsqueeze_dim=2)
    k = hf.apply_rotary_pos_emb(k, cos=cos, sin=sin, unsqueeze_dim=2)
    iq = indexer.q_layernorm(data["iq"][start:][None])
    iq = hf.apply_rotary_pos_emb(iq, cos=cos[:, start:], sin=sin[:, start:], unsqueeze_dim=2)
    v = data["v"][None]
    additive = torch.where(mask, torch.tensor(0.0), torch.tensor(-torch.inf))
    module = SimpleNamespace(num_key_value_groups=cfg.heads // cfg.kv_heads, training=False)
    eager, _ = hf.eager_attention_forward(
        module, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), additive,
        scaling=cfg.head_dim ** -0.5, dropout=0.0)
    # Explicit GQA expansion is limited to the oracle, never an engine layout.
    group = cfg.heads // cfg.kv_heads
    sdpa = nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).double(),
        k.transpose(1, 2).double().repeat_interleave(group, dim=1),
        v.transpose(1, 2).double().repeat_interleave(group, dim=1),
        attn_mask=mask, dropout_p=0.0, is_causal=False,
        scale=cfg.head_dim ** -0.5).transpose(1, 2)
    gate = data["qg"][start:, :, 1][None]
    factor = torch.sigmoid(gate)
    result = {"q": q[0], "k": k[0], "v": v[0], "iq": iq[0],
              "gate": gate[0], "mask": mask[0, 0],
              "attention_eager": eager[0], "attention_fp64": sdpa[0],
              "gated_eager": (eager * factor)[0],
              "gated_fp64": (sdpa * factor.double())[0]}
    if observed:
        blocks = observed["normed"].shape[0]
        indices = torch.arange(blocks) * cfg.ratio
        result["pooled_last"] = observed["pooled"]
        result["block_keys_last"] = hf.apply_rotary_pos_emb(
            observed["normed"].unsqueeze(1), cos=cos[0, indices],
            sin=sin[0, indices]).squeeze(1)
    else:
        result["pooled_last"] = torch.empty(0, cfg.index_dim)
        result["block_keys_last"] = torch.empty(0, cfg.index_dim)
    result = {name: value.numpy().copy() for name, value in result.items()}
    result["indexer_scores_fp64_unscaled"] = indexer_scores64(
        result["iq"], result["block_keys_last"], cfg.ratio, start)
    return result


def indexer_scores64(iq, block_keys, ratio, start):
    """Independent diagnostic dots on official prepared F32 operands.

    HF divides the summed ReLUs by sqrt(Di). This unscaled diagnostic follows
    the DS/spec convention and uses FP64 BLAS, not the HF FP32 score path.
    It is not evidence that the two top-k sets agree for near-ties.
    """
    queries = iq.astype(np.float64)
    keys = block_keys.astype(np.float64)
    scores = np.zeros((len(iq), len(block_keys)), dtype=np.float64)
    for head in range(iq.shape[1]):
        scores += np.maximum(queries[:, head] @ keys.T, 0)
    visible = (np.arange(start, start + len(iq)) + 1) // ratio
    scores[np.arange(len(block_keys))[None, :] >= visible[:, None]] = -np.inf
    return scores


def online_attention(cfg, prepared, tile_size=3, splits=2, precision="float32"):
    """CPU diagnostic of the proposed tiled reduction, NOT a second oracle.

    Inputs are F32 Q[T,H,D], K/V[N,J,D] and bool mask[T,N] returned by
    evaluate(). GQA maps h to h//(H/J). Partition the *token domain*, skip
    empty tiles/splits, merge unnormalized (maximum, denominator, numerator),
    and normalize once. Compare against independent official eager/SDPA.
    No gate, persistent cache, native API or GPU behavior is implemented.
    """
    if any(type(x) is not int or x <= 0 for x in (tile_size, splits)):
        raise ValueError("Tile size and split count must be positive integers")
    if precision not in ("float32", "float64"):
        raise ValueError("Unsupported accumulator precision")
    q, k, v, mask = (prepared[name] for name in ("q", "k", "v", "mask"))
    if (q.ndim != 3 or k.ndim != 3 or q.shape[1:] != (cfg.heads, cfg.head_dim)
            or k.shape[1:] != (cfg.kv_heads, cfg.head_dim) or v.shape != k.shape
            or mask.shape != (len(q), len(k)) or mask.dtype != np.bool_
            or not len(q) or not len(k) or not mask.any(axis=1).all()):
        raise ValueError("Invalid prepared shapes or empty selected row")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Prepared operands must be finite F32")
    if splits > len(k):
        raise ValueError("Split count exceeds token count")
    dtype = np.dtype(precision).type
    q, k, v = (x.astype(dtype) for x in (q, k, v))
    output = np.zeros(q.shape, dtype=dtype)
    maximum = np.full(q.shape[:2], -np.inf, dtype=dtype)
    denominator = np.zeros(q.shape[:2], dtype=dtype)
    scale = dtype(cfg.head_dim ** -0.5)

    def merge(left, right):
        # The identity is (-inf, 0, 0); never evaluate -inf - (-inf).
        if right[1] == 0:
            return left
        if left[1] == 0:
            return right
        m = np.maximum(left[0], right[0])
        a, b = np.exp(left[0] - m), np.exp(right[0] - m)
        return m, a * left[1] + b * right[1], a * left[2] + b * right[2]

    for row in range(len(q)):
        for head in range(cfg.heads):
            kv_head = head // (cfg.heads // cfg.kv_heads)
            identity = (dtype(-np.inf), dtype(0), np.zeros(cfg.head_dim, dtype=dtype))
            total = identity
            for split in range(splits):
                lo, hi = len(k) * split // splits, len(k) * (split + 1) // splits
                partial = identity
                for begin in range(lo, hi, tile_size):
                    ids = np.flatnonzero(mask[row, begin:min(begin + tile_size, hi)]) + begin
                    if not len(ids):
                        continue
                    scores = (k[ids, kv_head] @ q[row, head]) * scale
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite online logits")
                    m = scores.max()
                    weights = np.exp(scores - m)
                    partial = merge(partial, (m, weights.sum(), weights @ v[ids, kv_head]))
                total = merge(total, partial)
            maximum[row, head], denominator[row, head] = total[:2]
            output[row, head] = total[2] / total[1]
    return {"attention": output, "maximum": maximum, "denominator": denominator,
            "logsumexp": maximum + np.log(denominator)}


def native_public(native, cfg, x, ends):
    config = struct.pack("=9I2f", cfg.heads, cfg.kv_heads, cfg.head_dim,
                         cfg.index_heads, 1, cfg.index_dim, cfg.ratio, cfg.budget,
                         cfg.rotary_dim, cfg.eps, cfg.theta)
    n = len(x.qg)
    payload = config + struct.pack("=2I", n, len(ends)) + np.array(ends, np.uint32).tobytes()
    payload += b"".join(a.tobytes() for a in x.arrays().values())
    run = subprocess.run([str(native), "public"], input=payload, capture_output=True, check=True)
    if run.stderr: raise RuntimeError(run.stderr.decode())
    code, = struct.unpack("=i", run.stdout[:4])
    values = np.frombuffer(run.stdout[4:], np.float32)
    offset = 0
    result = {}
    for key, shape in (("out", (n, cfg.heads, cfg.head_dim)),
                       ("k", (n, cfg.kv_heads, cfg.head_dim)), ("ik", (n, cfg.index_dim)),
                       ("blocks", (n // cfg.ratio, cfg.index_dim)),
                       ("trace", (n, 2 * cfg.heads * cfg.head_dim + cfg.index_heads * cfg.index_dim)),
                       ("scores", (n // cfg.ratio,))):
        size = int(np.prod(shape))
        result[key] = values[offset:offset+size].reshape(shape)
        offset += size
    selected = min(n // cfg.ratio,cfg.budget // cfg.ratio)*cfg.ratio+n%cfg.ratio
    result["ids"] = values[offset:offset+selected].view(np.uint32)
    offset += selected
    if offset != len(values): raise ValueError("Native fixture output length mismatch")
    return code, result

def fixture_case(name: str):
    if name == "tiny":
        return Config(), 17, 0
    if name == "boundary":
        return Config(budget=2048), 2053, 2046
    if name == "real-shape":
        return Config(width=2560, heads=24, kv_heads=2, head_dim=256,
                      index_heads=4, index_dim=128, rotary_dim=64,
                      budget=2048, theta=10_000_000.0), 9, 0
    raise ValueError(f"Unknown fixture case: {name}")


def write_fixture(hf, source: Path, output: Path, case: str, seed: int = 3811):
    """New output directory only, with inputs/intermediates and provenance."""
    if output.exists():
        raise FileExistsError(output)
    cfg, tokens, start = fixture_case(case)
    inputs = synthetic(cfg, tokens, seed)
    result = evaluate(hf, cfg, inputs, start)
    np.testing.assert_allclose(result["attention_eager"], result["attention_fp64"],
                               atol=3e-5, rtol=2e-4)
    online = online_attention(cfg, result)
    np.testing.assert_allclose(online["attention"], result["attention_fp64"],
                               atol=3e-5, rtol=2e-4)
    arrays = {f"input_{k}": v for k, v in inputs.arrays().items()}
    arrays.update({f"reference_{k}": v for k, v in result.items()})
    arrays.update({f"candidate_online_{k}": v for k, v in online.items()})
    record = {
        "schema_version": 1, "case": case, "seed": seed,
        "config": asdict(cfg), "prefix_tokens": tokens, "query_start": start,
        "source_url": SOURCE_URL, "source_sha256": SOURCE_HASH,
        "source_path": str(source.resolve()), "source_revision": REVISION,
        "source_license": "Apache-2.0 (original header retained in source cache)",
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter": "AST allowlist, unchanged bodies; default-only RoPE wrapper; projected QK; prefix keys",
        "dtype": "F32 preparation and eager; FP64 SDPA on identical prepared operands",
        "tie_policy": "Official torch.topk; tied boundary order unspecified, no native tie parity claimed",
        "score_diagnostic": "Unscaled FP64 dots on official prepared F32 IQ/block keys; not an HF score dump",
        "online_diagnostic": {"role": "Candidate reduction checked against independent SDPA, not an oracle",
                              "tile_size": 3, "splits": 2, "precision": "float32",
                              "gate_applied": False},
        "scope": "Synthetic operator reference; no C/Metal or checkpoint/BF16 quality claim",
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers")},
        "cpu_threads": torch.get_num_threads(),
        "max_abs_attention_eager_vs_fp64": float(np.max(np.abs(
            result["attention_eager"].astype(np.float64) - result["attention_fp64"]))),
        "max_abs_attention_online_vs_fp64": float(np.max(np.abs(
            online["attention"].astype(np.float64) - result["attention_fp64"]))),
        "arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                           "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                   for name, value in arrays.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "fixture.npz"
    np.savez(artifact, **arrays)
    record["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_real_slice(hf, native, reference, checkpoint, output):
    """A catalog -> B projections -> public D, bounded to one QSA layer.

    Synthetic activations on actual Q2 weights; NOT a BF16-quality test.
    Invoke under the repository GPU/workload lock, with fresh output.
    """
    import qwen38_packed as packed
    root=Path(__file__).resolve().parents[2]
    proof=json.loads((root/"bench/qwen38_q2_20261011_phase0/checkpoint.verification.json").read_text())
    before=checkpoint.stat()
    if not proof["matched"] or (before.st_size,before.st_ino,before.st_mtime_ns)!=(
            proof["size_bytes"],proof["file_inode"],proof["file_mtime_ns"]):
        raise ValueError("Checkpoint identity differs from recorded full verification")
    output.mkdir(parents=True,exist_ok=False)
    catalog=json.loads(subprocess.check_output([str(native),"catalog",str(checkpoint),"3"]))
    cfg=Config(**catalog["config"]); tokens=5
    x=np.random.default_rng(3811).uniform(-.05,.05,(tokens,cfg.width)).astype(np.float32)
    arrays={"x":x}; records=[]; read_bytes=0
    fd=os.open(checkpoint,os.O_RDONLY)
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def read_tensor(t,index,row,rows):
        nonlocal read_bytes
        row_bytes=t["bytes"]//t["rows"]; count=rows*row_bytes
        offset=t["offset"]+row*row_bytes
        if offset<0 or offset+count>before.st_size or read_bytes+count>64*1024**2:
            raise ValueError("Real slice exceeds extent/read budget")
        raw=packed.pread_exact(fd,offset,count); read_bytes+=count
        name=f"slice-{index}-row-{row}.bin"; (output/name).write_bytes(raw)
        records.append(dict(file=name,tensor=t["name"],offset=offset,bytes=count,
                            sha256=hashlib.sha256(raw).hexdigest()))
        return raw
    metrics=[]
    def project(t,index,activation):
        parts=[]; refs=[]; max_ratio=0.; max_error=0.
        for row in range(0,t["rows"],128):
            rows=min(128,t["rows"]-row); raw=read_tensor(t,index,row,rows)
            weights=packed.decode(reference,raw,t["type"],t["k"],rows)
            ref,absolute=packed.matmul_reference(weights,activation,t["k"])
            payload=struct.pack("=4I",len(activation),rows,t["k"],t["type"])+raw+activation.tobytes()
            run=subprocess.run([str(native),"linear"],input=payload,capture_output=True,check=True)
            if run.stderr or struct.unpack("=i",run.stdout[:4])[0]: raise ValueError("B projection failed")
            y=np.frombuffer(run.stdout[4:],np.float32).reshape(len(activation),rows)
            u=2**-24; g=(2*t["k"]+8)*u/(1-(2*t["k"]+8)*u)
            bound=g*absolute+2*np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64)
            error=np.abs(y-ref)
            if not np.all(error<=bound): raise AssertionError("B projection bound exceeded")
            max_ratio=max(max_ratio,float(np.max(error/np.maximum(bound,np.finfo(float).tiny))))
            max_error=max(max_error,float(error.max())); parts.append(y.copy()); refs.append(ref)
        metrics.append(dict(tensor=t["name"],max_abs_error=max_error,max_error_over_bound=max_ratio))
        return np.concatenate(parts,axis=1),np.concatenate(refs,axis=1)
    try:
        projections=[]
        for i,t in enumerate(catalog["tensors"][:5]):
            y,r=project(t,i,x); projections.append(y); arrays[f"projection_{i}"]=y; arrays[f"projection_ref64_{i}"]=r
        gains=[]
        for i,t in enumerate(catalog["tensors"][6:],6):
            raw=read_tensor(t,i,0,1)
            gains.append(packed.decode(reference,raw,t["type"],t["k"],1).ravel().copy())
        inputs=Inputs(projections[0].reshape(tokens,cfg.heads,2,cfg.head_dim),
            projections[1].reshape(tokens,cfg.kv_heads,cfg.head_dim),
            projections[2].reshape(tokens,cfg.kv_heads,cfg.head_dim),
            projections[3].reshape(tokens,cfg.index_heads,cfg.index_dim),projections[4],*gains)
        ref=evaluate(hf,cfg,inputs); runs=[]
        for ends in ([5],[1,2,3,4,5],[1,3,5]):
            code,got=native_public(native,cfg,inputs,ends)
            if code: raise AssertionError(f"Public attention failed: {code}")
            np.testing.assert_allclose(got["out"],ref["gated_fp64"],atol=3e-5,rtol=2e-4)
            np.testing.assert_allclose(got["k"],ref["k"],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(got["blocks"],ref["block_keys_last"],atol=2e-5,rtol=2e-5)
            runs.append(got["out"])
        np.testing.assert_array_equal(runs[0],runs[1]); np.testing.assert_array_equal(runs[0],runs[2])
        arrays["attention_native"]=runs[0]; arrays["attention_ref64"]=ref["gated_fp64"]
        arrays["out_projection"],arrays["out_projection_ref64"]=project(catalog["tensors"][5],5,runs[0].reshape(tokens,-1))
        # Small independent decoder/offset probes at two later trunk layers.
        for layer in (27,47):
            other=json.loads(subprocess.check_output([str(native),"catalog",str(checkpoint),str(layer)]))
            for i,t in enumerate(other["tensors"]):
                raw=read_tensor(t,f"l{layer}-{i}",0,min(2,t["rows"]))
                packed.decode(reference,raw,t["type"],t["k"],min(2,t["rows"]))
        if identity(os.fstat(fd))!=identity(before) or identity(checkpoint.stat())!=identity(before):
            raise ValueError("Checkpoint changed while reading")
    finally:
        os.close(fd)
    arrays.update({f"input_{k}":v for k,v in inputs.arrays().items()})
    np.savez(output/"fixture.npz",**arrays)
    manifest=dict(schema_version=1,checkpoint_sha256=proof["sha256"],checkpoint_revision=proof["revision"],
        source_sha256=SOURCE_HASH,base="6a6b886e985185d7c9104eda888e6a406553aedf",catalog=catalog,
        reads=records,read_bytes=read_bytes,read_cap_bytes=64*1024**2,temporary_cap_bytes=256*1024**2,
        projection_metrics=metrics,max_abs_attention_error=float(np.max(np.abs(runs[0]-ref["gated_fp64"]))),
        fixture_sha256=hashlib.sha256((output/"fixture.npz").read_bytes()).hexdigest(),
        scope="Synthetic X, actual Q2 layer 3 weights; A catalog, B products, C prep, D attention; no BF16 loss or real activation capture")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--source", type=Path,
                        default=root / "build/qwen38-attention/modeling_qwen4_exp.py")
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=["tiny", "boundary", "real-shape"], default="tiny")
    parser.add_argument("--seed", type=int, default=3811)
    parser.add_argument("--slice-checkpoint",type=Path)
    parser.add_argument("--native",type=Path)
    parser.add_argument("--packed-reference",type=Path)
    args = parser.parse_args()
    if not args.fetch_source and args.output is None:
        parser.error("Specify --fetch-source or --output (tests never fetch implicitly)")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; use a new directory")
    torch.set_num_threads(1)
    if args.fetch_source:
        fetch_source(args.source)
        print(f"Verified source: {args.source}")
    if args.output is not None:
        hf = load_reference(args.source)
        if args.slice_checkpoint:
            if not args.native or not args.packed_reference: parser.error("Slice needs --native and --packed-reference")
            record=verify_real_slice(hf,args.native.resolve(),args.packed_reference.resolve(),args.slice_checkpoint,args.output)
            print(json.dumps({"read_bytes":record["read_bytes"],"max_abs_attention_error":record["max_abs_attention_error"]}))
            return
        record = write_fixture(hf, args.source, args.output, args.case, args.seed)
        print(json.dumps({"output": str(args.output), "case": args.case,
                          "max_abs_error": record["max_abs_attention_eager_vs_fp64"]}))


if __name__ == "__main__":
    main()
