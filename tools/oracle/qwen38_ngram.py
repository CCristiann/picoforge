"""CPU-only ID oracle: run the pinned Transformers methods, not a translation.

The source is cached only under build/ and verified before AST extraction.
No model construction, weight download, GPU, or Transformers upgrade occurs.
See docs/specs/qwen38/E.md for the derivation and checkpoint provenance.
"""
import ast
import hashlib
from pathlib import Path
import urllib.request

import torch
from torch import nn

REVISION = "db2304e8694db3ecd6f39d353beba361688afe97"
SOURCE_SHA256 = "f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa"
CLASS_SHA256 = "874ee437373ae5ed3419da959f5163f1a5669549c66eccbdd7833e6869f87362"
URL = (f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/"
       "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "build/qwen38-e/modeling_qwen4_exp.py"


def verified_source(path=SOURCE, *, fetch=False):
    """Network access is opt-in; never execute an unverified cache entry."""
    path = Path(path)
    if fetch and not path.exists():
        with urllib.request.urlopen(URL, timeout=30) as response:
            raw = response.read((1 << 20) + 1)
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("upstream source checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream source checksum mismatch")
    return raw.decode("utf-8")


def validate(config):
    """Restrict arithmetic to the signed-int64 domain verified in the spec."""
    n, g, v, eos = (config[k] for k in ("ngram_size", "heads_per_ngram", "vocab", "eos"))
    if any(type(x) is not int for x in (n, g, v, eos)):
        raise ValueError("integer geometry required")
    if not (2 <= n <= 16 and 1 <= g <= 64 and 1 <= v <= 2**31 and 0 <= eos < v):
        raise ValueError("unsupported geometry")
    mult, sizes, offsets = (config[k] for k in ("multipliers", "sizes", "offsets"))
    if len(mult) != n or len(sizes) != (n-1)*g or len(offsets) != len(sizes):
        raise ValueError("hash metadata length mismatch")
    if any(type(m) is not int or not 0 < m <= 2**63-1 or m*(v-1) > 2**63-1 for m in mult):
        raise ValueError("hash product outside signed-int64 domain")
    end = 0
    for offset, size in zip(offsets, sizes, strict=True):
        if type(offset) is not int or type(size) is not int or offset != end or size <= 0:
            raise ValueError("invalid head interval")
        end += size
    if type(config["rows"]) is not int or not end <= config["rows"] <= 2**32-1:
        raise ValueError("head interval outside table")


class Capture(nn.Module):
    """Replace only the enormous embedding table; expose its exact indices."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, device="cpu"), requires_grad=False)

    def forward(self, ids):
        return ids.unsqueeze(-1)


class IdOracle:
    def __init__(self, config, source=SOURCE):
        validate(config)
        raw = verified_source(source)
        node = next(n for n in ast.parse(raw).body
                    if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpTextNGramEmbedding")
        if hashlib.sha256(ast.get_source_segment(raw, node).encode()).hexdigest() != CLASS_SHA256:
            raise ValueError("upstream class checksum mismatch")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        scope = {"torch": torch, "nn": nn}
        exec(compile(module, str(source), "exec"), scope)
        cls = scope[node.name]
        obj = cls.__new__(cls)  # do NOT allocate the original embedding table
        nn.Module.__init__(obj)
        obj.layer_idx = 1
        obj.ngram_size = config["ngram_size"]
        obj.context_len = obj.ngram_size - 1
        obj.heads_per_ngram = config["heads_per_ngram"]
        obj.eos_token_id = config["eos"]
        for name, key in (("layer_multipliers", "multipliers"),
                          ("ngram_heads_vocab_sizes", "sizes"), ("ngram_heads_offsets", "offsets")):
            setattr(obj, name, torch.tensor(config[key], dtype=torch.int64, device="cpu"))
        obj.ngram_embedding = Capture()
        self.model, self.vocab = obj, config["vocab"]

    def ids(self, tokens, previous=()):
        """Previous tokens are chronological; retain only N-1 for an append.

        Use upstream full-sequence evaluation, without a custom Cache shim.
        No state is committed here; a caller may discard the returned IDs.
        """
        tokens, previous = list(tokens), list(previous)
        if len(tokens) > 32768 or len(previous) > self.model.context_len:
            raise ValueError("oracle input exceeds bounded history/token capacity")
        if any(type(t) is not int or not 0 <= t < self.vocab for t in previous + tokens):
            raise ValueError("invalid token")
        if not tokens:
            return []
        with torch.no_grad():
            result = self.model(torch.tensor([previous + tokens], dtype=torch.int64, device="cpu"), None)
        return result[0, -len(tokens):].tolist()


def fixture():
    import json
    config = dict(ngram_size=3, heads_per_ngram=2, vocab=100, eos=99,
                  multipliers=[3, 5, 7], sizes=[11, 13, 17, 19],
                  offsets=[0, 11, 24, 41], rows=64)
    inventory = ROOT / "bench/qwen38_q2_20261011_phase0/inventory.json"
    metadata = json.loads(inventory.read_text())["metadata"]
    real = {k: metadata["qwen4exp.ple." + key] for k, key in
            (("ngram_size", "ngram_size"), ("heads_per_ngram", "heads_per_ngram"),
             ("multipliers", "layer_multipliers"), ("sizes", "head_vocab_sizes"),
             ("offsets", "head_offsets"), ("rows", "row_count"), ("eos", "eos_token_id"))}
    real["vocab"] = 248320  # pinned config; only fixture generation, not engine geometry
    cases = []
    for name, cfg, tokens in (("synthetic", config, [1, 2, 99, 3, 4, 99, 99, 5]),
                              ("checkpoint_metadata", real, [1, 2, 248044, 3, 4])):
        cases.append(dict(name=name, config=cfg, tokens=tokens, ids=IdOracle(cfg).ids(tokens)))
    return dict(schema_version=1, upstream_revision=REVISION, source_sha256=SOURCE_SHA256,
                class_sha256=CLASS_SHA256, torch_version=torch.__version__,
                inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
                cases=cases, bf16_bits=[0, 32768, 1, 32769, 127, 128, 16256, 49024, 32639],
                nonfinite_bf16_bits=[32640, 65408, 32704, 32641])


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-source", action="store_true", help="fetch pinned source into build only")
    parser.add_argument("--output", type=Path, help="write reproducible golden JSON, refusing overwrite")
    args = parser.parse_args()
    verified_source(fetch=args.fetch_source)
    if args.output:
        result = fixture()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    else:
        print("PASS: verified upstream source; no weights loaded")
