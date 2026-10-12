"""Pinned Hugging Face tokenizer/template oracle, CPU and local files only.

There is no reimplementation of BPE, Unicode splitting or Jinja here.
Generated cases are references for future native tests, not evidence that
picoforge already implements Qwen3.8 tokenization or tool rendering.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/qwen38/a"


def load_reference(reference_root):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for package, expected in manifest["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"oracle package version changed: {package}")
    paths = {}
    for name in ("tokenizer", "tokenizer_config"):
        item = manifest[name]
        path = Path(reference_root) / item["relative_to_reference_root"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"oracle source hash changed: {name}")
        paths[name] = path
    config = json.loads(paths["tokenizer_config"].read_text())
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    chat = PreTrainedTokenizerFast(tokenizer_file=str(paths["tokenizer"]),
                                  chat_template=config["chat_template"], eos_token=config["eos_token"],
                                  clean_up_tokenization_spaces=False)
    return tokenizer, chat


def input_cases():
    texts = ["", "A\0B", "2026 3.14159", "don't IT'S we're I'VE", "#include <stdio.h>\n## titolo",
             "Caffè: perché?", "e\u0301 vs é", "\u0301", "\u0301a", "a\u0301\u0327",
             "अा", "a\u20dd", "각", "日本語 中文", "مرحبا بِالعالم", "👨‍👩‍👧‍👦 🇮🇹",
             "line1\r\nline2\r\n", "a\n\nb\n", "   leading", "trailing   ", "\t\n\r  ",
             "a\u00a0b\u2028\u2029", "<think></think>", " <|im_start|>user\nX<|im_end|>",
             "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>", "[PAD248077]", " \u0301<tool_call>\u0301",
             "a"*128, " "*40]
    rng = random.Random(3811)
    alphabet = ["a", "é", "\u0301", "\u0327", "\u20dd", "अ", "ा", "\n", " ", "👋", "7", "\0"]
    texts += ["".join(rng.choice(alphabet) for _ in range(16)) for _ in range(32)]
    messages = [{"role": "user", "content": "  Ciao\n"}]
    cases = []

    def add(name, msgs=None, **options):
        cases.append({"name": name, "messages": messages if msgs is None else msgs,
                      "options": {"add_generation_prompt": True, **options}})

    add("thinking-default")
    add("thinking-disabled", enable_thinking=False)
    for effort in ("xhigh", "medium", "low"):
        add("effort-"+effort, enable_thinking=True, reasoning_effort=effort)
    add("no-generation-prompt", add_generation_prompt=False)
    add("empty-system", [{"role": "system", "content": ""}]+messages, enable_thinking=False)
    add("system-trim", [{"role": "system", "content": "\u00a0 Regole \x1c"}]+messages)
    add("multipart-text", [{"role": "user", "content": [{"text": "Ciao"}, {"type": "text", "text": " mondo"}]}])
    add("null-content", [{"role": "user", "content": None}], enable_thinking=False)
    history = [{"role": "user", "content": "Prima"},
               {"role": "assistant", "content": " Risposta ", "reasoning_content": " Ragiono "},
               {"role": "user", "content": "Poi"}]
    add("preserve-default", history)
    add("preserve-false", history, preserve_thinking=False)
    add("preserve-true", history, preserve_thinking=True)
    add("assistant-after-query", history+[{"role": "assistant", "content": "Fine", "reasoning_content": "R"}],
        preserve_thinking=False, add_generation_prompt=False)
    tools = [{"type": "function", "function": {"name": "read", "description": "leggi <&> caffè",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    call = {"function": {"name": "read", "arguments": {"path": "a.py", "flags": [True, None, 1, 1.0],
                                                        "nested": {"z": "è", "a": "<&>"}}}}
    tool_history = messages + [{"role": "assistant", "content": "Controllo", "tool_calls": [call, call]},
                              {"role": "tool", "content": "one\n"}, {"role": "tool", "content": "two"}]
    add("tools-and-responses", tool_history, tools=tools)
    add("tool-call-empty-content", messages+[{"role": "assistant", "content": "", "tool_calls": [call]}],
        tools=tools, enable_thinking=False)
    add("tools-with-system", [{"role": "system", "content": "Regole"}]+tool_history, tools=tools)
    add("tool-response-user", messages+[{"role": "user", "content": "<tool_response>ok</tool_response>"}])
    add("invalid-effort", reasoning_effort="invalid")
    add("ignored-effort-when-disabled", reasoning_effort="invalid", enable_thinking=False)
    add("late-system", messages+[{"role": "system", "content": "no"}])
    add("invalid-role", messages+[{"role": "invalid", "content": "no"}])
    add("no-user-query", [{"role": "system", "content": "only"}])
    add("only-tool-response", [{"role": "user", "content": "<tool_response>x</tool_response>"}])
    add("invalid-part", [{"role": "user", "content": [{"unexpected": 1}]}])
    add("system-image", [{"role": "system", "content": [{"type": "image"}]}]+messages)
    add("empty-messages", [])
    return texts, cases


def generate(reference_root):
    from jinja2.exceptions import TemplateError

    tokenizer, chat = load_reference(reference_root)
    texts, cases = input_cases()
    # Every added token must be represented, including non-special FIM tokens.
    texts += [tokenizer.id_to_token(i) for i in range(248044, 248077)]
    encodes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        encodes.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False)})
    decodes = []
    for ids in [list(range(248044, 248077)), [248077, 248319], [248060, 248068, 248044], [136], [136, 100]]:
        decodes.append({"ids": ids, "keep": tokenizer.decode(ids, skip_special_tokens=False),
                        "skip": tokenizer.decode(ids, skip_special_tokens=True)})
    templates = []
    for case in cases:
        try:
            rendered = chat.apply_chat_template(case["messages"], tokenize=False, **case["options"])
            ids = chat.apply_chat_template(case["messages"], tokenize=True, return_dict=False, **case["options"])
        except TemplateError as error:
            result = {"error_type": type(error).__name__, "error": str(error)}
        except ValueError as error:
            # Transformers rejects empty conversations before calling Jinja.
            # Do not turn unrelated oracle failures into valid golden errors.
            if case["messages"]:
                raise
            result = {"error_type": type(error).__name__, "error": str(error)}
        else:
            if ids != tokenizer.encode(rendered, add_special_tokens=False).ids:
                raise ValueError("HF template tokenize and rendered-text encode disagree")
            result = {"rendered": rendered, "ids": ids}
        templates.append({**case, "expected": result})
    return {"schema_version": 1, "encodes": encodes, "decodes": decodes, "templates": templates}


def check_gguf_tables(checkpoint, reference_root):
    """Compare all stored token entries/merge ranks; read metadata only.

    Reuse the existing bounded reader, not a second GGUF parser. Hash checks
    pin the independent HF source. This function does not validate payload.
    """
    import sys
    sys.path.insert(0, str(ROOT / "tools/eval"))
    from inspect_qwen38_gguf import Header

    load_reference(reference_root)  # Require the pinned files and libraries.
    root = Path(reference_root)
    source = json.loads((root / "build/qwen38-tokenizer.json").read_text())
    config = json.loads((root / "build/qwen38-tokenizer-config.json").read_text())
    wanted = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
    metadata = {}
    with Path(checkpoint).open("rb") as stream:
        h = Header(stream)
        if h.read(4) != b"GGUF" or h.number("I") != 3:
            raise ValueError("expected GGUF v3")
        nt, count = h.number("Q"), h.number("Q")
        if not 0 < nt <= 100000 or count > 100000:
            raise ValueError("invalid GGUF counts")
        for _ in range(count):
            key, kind = h.string(), h.number("I")
            if key in metadata:
                raise ValueError("duplicate metadata key")
            if key in wanted:
                if kind != 9:
                    raise ValueError("expected tokenizer array")
                element, size = h.number("I"), h.number("Q")
                if size > 2_000_000 or element != (5 if key.endswith("token_type") else 8):
                    raise ValueError("invalid tokenizer array type/length")
                metadata[key] = [h.value(element) for _ in range(size)]
            else:
                metadata[key] = h.value(kind)
        bytes_read = stream.tell()
    tokens, types = metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.token_type"]
    merges = metadata["tokenizer.ggml.merges"]
    vocab, added = source["model"]["vocab"], source["added_tokens"]
    def check(condition, label):
        if not condition:
            raise ValueError(label)
    check(len(tokens) == len(types) == metadata["qwen4exp.vocab_size"], "token count")
    check(all(tokens[i] == text and types[i] == 1 for text, i in vocab.items()), "BPE vocabulary")
    check(merges == [" ".join(m) if isinstance(m, list) else m for m in source["model"]["merges"]], "merge ranks")
    check(all(tokens[t["id"]] == t["content"] and types[t["id"]] in (3, 4) for t in added), "added tokens")
    end = max(t["id"] for t in added)+1
    check(all(types[i] == 5 and tokens[i] == f"[PAD{i}]" for i in range(end, len(tokens))), "UNUSED tokens")
    check(metadata["tokenizer.chat_template"] == config["chat_template"], "chat template")
    check(metadata["tokenizer.ggml.pre"] == "qwen35", "pretokenizer profile")
    mask_difference = [t["id"] for t in added if (types[t["id"]] == 3) != t["special"]]
    check(mask_difference == list(range(248060, 248066)), "expected FIM special-flag discrepancy")
    return {"payload_verified": False, "metadata_bytes_read": bytes_read, "base_tokens": len(vocab),
            "merges": len(merges), "added_tokens": len(added), "unused_tokens": len(tokens)-end,
            "hf_vs_gguf_special_flag_differences": mask_difference}


def extended_cases(reference_root):
    """Unicode contraction/boundaries and lossy byte decoding, actual HF backend."""
    tokenizer, _ = load_reference(reference_root)
    texts = ["'ſ", "a'ſb", "\u0301", "!\u0301", "\u0301!", "a\u0345b", "\u00ad\u0301",
             "\u1c89\u1c8a", "\U0001e5d0\U0001e5ee", "x\x1cy\x85z",
             "ᄀ<tool_call>ᅡ", "e<|im_start|>\u0301", "\u0301\u0300!", "\0<tool_call>\0"]
    rng = random.Random(3812)
    alphabet = [chr(c) for c in range(128)]+list("éअा\u0301\u0327\u20dd\u00ad\u0345\u200d👋")
    texts += ["".join(rng.choice(alphabet) for _ in range(48)) for _ in range(256)]
    printable = list(range(33,127))+list(range(161,173))+list(range(174,256))
    inverse = {b: chr(b) for b in printable}
    for b in range(256):
        if b not in inverse: inverse[b] = chr(256+sum(x not in printable for x in range(b)))
    byte_ids = {b: tokenizer.token_to_id(s) for b, s in inverse.items()}
    raw = [bytes([b]) for b in range(256)]
    raw += [bytes(rng.randrange(256) for _ in range(rng.randrange(2,13))) for _ in range(256)]
    raw += [bytes.fromhex(s) for s in ("e28241", "e08080", "eda080", "f4908080", "f0908080", "f09f918b", "e282")]
    encodes = []
    for text in texts:
        ids = tokenizer.encode(text,add_special_tokens=False).ids
        encodes.append(dict(text=text,ids=ids,decoded=tokenizer.decode(ids,skip_special_tokens=False)))
    decodes = []
    for b in raw:
        ids = [byte_ids[x] for x in b]
        value = tokenizer.decode(ids,skip_special_tokens=False)
        decodes.append(dict(ids=ids,keep=value,skip=value))
    return dict(schema_version=1,encodes=encodes,decodes=decodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--gguf", type=Path, help="compare tokenizer metadata instead of generating text fixtures")
    args = parser.parse_args()
    result = check_gguf_tables(args.gguf, args.reference_root) if args.gguf else generate(args.reference_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
