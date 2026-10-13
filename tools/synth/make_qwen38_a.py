"""Small GGUF schema fixture from the independent pinned HF-config oracle.

Writes sparse, zero payloads solely to exercise descriptors, never inference.
All generated files belong in the current worktree's build/qwen38-a directory.
"""
import json
import math
from pathlib import Path
import struct
import sys
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_config as ref


def small_config():
    hf = json.loads((ROOT / "tests/fixtures/qwen38/a/config.json").read_text())
    c = hf["text_config"]
    c.update(hidden_size=256, vocab_size=512, hc_count=2, hc_lowrank=32,
             num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
             full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=32, indexer_n_heads=2, indexer_head_dim=32, indexer_budget=8,
             linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
             linear_num_value_heads=4, num_experts=4, num_experts_per_tok=2,
             moe_intermediate_size=64, shared_expert_intermediate_size=32,
             heads_per_ngram=2, ple_embed_dim=256, eos_token_id=511)
    c["rope_parameters"]["partial_rotary_factor"] = 0.5
    return hf


def string(value):
    b = value.encode("utf-8")
    return struct.pack("<Q", len(b))+b


def write_fixture(path, hf):
    meta = ref.reference_metadata(hf)
    meta.update({"general.architecture": "qwen4exp", "general.alignment": 16384,
                 ref.PREFIX+"ple.row_count": 128,
                 ref.PREFIX+"ple.layer_multipliers": [17, 31, 43],
                 ref.PREFIX+"ple.head_offsets": [0, 7, 14, 21],
                 ref.PREFIX+"ple.head_vocab_sizes": [7]*4,
                 "ds4.qwen4.down.logical_input": 64, "ds4.qwen4.down.physical_input": 256})
    tensors = ref.reference_tensors(hf, 128)
    header = bytearray(b"GGUF"+struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        if isinstance(value, str): kind, payload = 8, string(value)
        elif isinstance(value, list):
            kind, payload = 9, struct.pack("<IQ", 10, len(value))+struct.pack("<"+"Q"*len(value), *value)
        elif key.endswith(("freq_base", "epsilon")): kind, payload = 6, struct.pack("<f", value)
        else: kind, payload = 4, struct.pack("<I", value)
        header += string(key)+struct.pack("<I", kind)+payload
    offset = 0
    expected = []
    for name, (shape, fmt, domain) in tensors.items():
        type_id, block, size = ref.FORMATS[fmt]
        offset = (offset+16383)//16384*16384
        count = math.prod(shape)//block*size
        header += string(name)+struct.pack("<I", len(shape))+struct.pack("<"+"Q"*len(shape), *shape)
        header += struct.pack("<IQ", type_id, offset)
        expected.append(dict(name=name, shape=shape, type_id=type_id, format=fmt,
                             size_bytes=count, offset=offset, domain=domain))
        offset += count
    start = (len(header)+16383)//16384*16384
    for t in expected: t["file_offset"] = start+t.pop("offset")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.truncate(start+offset)
    path.with_suffix(".json").write_text(json.dumps(dict(metadata=meta, tensors=expected), indent=2)+"\n")
    return expected


def text_cases():
    """Embed the frozen independent oracle as C data, not C algorithms."""
    cases = json.loads((ROOT / "tests/fixtures/qwen38/a/text.json").read_text())
    extra_path = ROOT / "tests/fixtures/qwen38/a/text_extended.json"
    if extra_path.exists():
        extra = json.loads(extra_path.read_text())
        for group in ("encodes", "decodes"): cases[group] += extra[group]
    lines = []

    def blob(name, text):
        b = text.encode()
        lines.append(f"static const uint8_t {name}[] = {{"+",".join(map(str,b or b'\0'))+"};")
        return f"{{{name},{len(b)}}}"

    enc = []
    for i, case in enumerate(cases["encodes"]):
        name = f"enc{i}"
        text = blob(name,case["text"])
        decoded = blob(name+"_decoded",case["decoded"])
        lines.append(f"static const int32_t {name}_ids[] = {{"+",".join(map(str,case["ids"] or [0]))+"};")
        enc.append(f"{{{text},{name}_ids,{len(case['ids'])},{decoded}}}")
    lines.append("static const EncodeCase encodes[] = {"+",\n".join(enc)+"};")
    dec = []
    for i, case in enumerate(cases["decodes"]):
        name = f"dec{i}"
        keep, skip = blob(name+"_keep",case["keep"]), blob(name+"_skip",case["skip"])
        lines.append(f"static const int32_t {name}_ids[] = {{"+",".join(map(str,case["ids"]))+"};")
        dec.append(f"{{{name}_ids,{len(case['ids'])},{keep},{skip}}}")
    lines.append("static const DecodeCase decodes[] = {"+",\n".join(dec)+"};")
    (ROOT / "build/qwen38-a/text_cases.inc").write_text("\n".join(lines)+"\n")


def chat_cases():
    cases = json.loads((ROOT / "tests/fixtures/qwen38/a/text.json").read_text())["templates"]
    extra = ROOT / "tests/fixtures/qwen38/a/chat_extended.json"
    if extra.exists(): cases += json.loads(extra.read_text())
    lines, records = [], []

    def data(text):
        raw = text.encode()
        return "{(const uint8_t[]) {"+",".join(map(str,raw or b'\0'))+"},"+str(len(raw))+"}"

    def node(value):
        if value is None: return "{.kind=Q38_JSON_NULL}"
        if type(value) is bool: return "{.kind=Q38_JSON_BOOL,.as.boolean="+str(value).lower()+"}"
        if type(value) is int:
            num = "(-INT64_C(9223372036854775807)-1)" if value == -2**63 else f"INT64_C({value})"
            return "{.kind=Q38_JSON_INT,.as.integer="+num+"}"
        if type(value) is float: return "{.kind=Q38_JSON_FLOAT,.as.number="+repr(value)+"}"
        if type(value) is str: return "{.kind=Q38_JSON_STRING,.as.string="+data(value)+"}"
        if type(value) is list:
            array = "(const Qwen38Json[]){"+",".join(node(v) for v in value)+"}" if value else "NULL"
            return "{.kind=Q38_JSON_ARRAY,.as.array={"+array+","+str(len(value))+"}}"
        members = ["{"+data(k)+",&(const Qwen38Json)"+node(v)+"}" for k, v in value.items()]
        obj = "(const Qwen38JsonMember[]){"+",".join(members)+"}" if value else "NULL"
        return "{.kind=Q38_JSON_OBJECT,.as.object={"+obj+","+str(len(value))+"}}"

    for i, case in enumerate(cases):
        opt, result = case["options"], case["expected"]
        lines.append(f"static const Qwen38Json chat{i} = "+node(case["messages"])+";")
        tools = "NULL"
        if "tools" in opt:
            lines.append(f"static const Qwen38Json tools{i} = "+node(opt["tools"])+";")
            tools = f"&tools{i}"
        opts = ["Q38_OMITTED" if k not in opt else "Q38_TRUE" if opt[k] else "Q38_FALSE"
                for k in ("enable_thinking", "preserve_thinking")]
        opts += [{None:"Q38_EFFORT_OMITTED", "low":"Q38_EFFORT_LOW", "medium":"Q38_EFFORT_MEDIUM",
                  "xhigh":"Q38_EFFORT_XHIGH"}.get(opt.get("reasoning_effort"), "99"),
                 str(opt.get("add_generation_prompt",False)).lower()]
        ids = result.get("ids",[])
        lines.append(f"static const int32_t chat_ids{i}[] = {{"+",".join(map(str,ids or [0]))+"};")
        records.append(f"{{&chat{i},{tools},{{"+",".join(opts)+"},"+data(result.get("rendered",""))+","+
                       str("error_type" in result).lower()+f",chat_ids{i},{len(ids)}}}")
    lines.append("static const ChatCase chats[] = {"+",\n".join(records)+"};")
    (ROOT / "build/qwen38-a/chat_cases.inc").write_text("\n".join(lines)+"\n")


def verify(args):
    """Build the A harnesses used by the coordinator Makefile targets."""
    build = ROOT / "build/qwen38-a"
    flags = ["clang","-std=c11","-O1","-g","-Wall","-Wextra","-Wpedantic","-Wshadow",
             "-Wconversion","-Werror","-ffp-contract=off","-fsanitize=address,undefined","-fno-sanitize-recover=all","-Isrc"]
    common = ["src/qwen38_config.c","src/gguf.c","src/gguf_reader.c","src/util.c"]
    groups = {
        "config": ["tests/test_qwen38_config.c"]+common,
        "bindings": ["tests/test_qwen38_bindings.c","src/qwen38_weights.c","src/gguf_plan.c"]+common,
        "text": ["tests/test_qwen38_text.c","src/qwen38_tokenizer.c","src/qwen38_unicode.c","src/qwen38_chat.c",
                 "src/tokenizer.c","src/unicode_tables.c","src/normalize.c","src/json.c"]+common}
    for name, sources in groups.items():
        subprocess.run(flags+["-Ibuild/qwen38-a"]+sources+["-lm","-o",str(build/f"test_{name}")],cwd=ROOT,check=True)
        paths = [] if name == "text" else [build/f"{variant}.gguf" for variant in ("small","no-mtp","last-gdn")]
        if args.checkpoint: paths.append(args.checkpoint)
        for path in paths:
            command = [str(build/f"test_{name}"),str(path)]
            if name == "config" and path == args.checkpoint: command.append(str(build/"native-config.json"))
            subprocess.run(command,cwd=ROOT,check=True)
    if args.reference_root:
        env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1",
               "QWEN38_A_REFERENCE_ROOT":str(args.reference_root),"QWEN38_A_TEXT_EXE":str(build/"test_text")}
        if args.checkpoint:
            env["QWEN38_A_NATIVE_CONFIG"] = str(build/"native-config.json")
            env["QWEN38_A_GGUF"] = str(args.checkpoint)
            subprocess.run(flags+["tests/gguf_inspect.c"]+common[1:]+["-lm","-o",str(build/"inspect")],cwd=ROOT,check=True)
            path = build/"native-inventory.json"
            with path.open("w") as f: subprocess.run([str(build/"inspect"),str(args.checkpoint)],stdout=f,check=True)
            env["QWEN38_A_NATIVE_INVENTORY"] = str(path)
        subprocess.run([sys.executable,"-B","tools/qwen38_test.py","tests/test_qwen38_a.py"],cwd=ROOT,env=env,check=True)
    if args.gpu:
        sources = groups["bindings"]+["src/gpu_weights.m","src/metal.m"]
        exe = build/"test_storage"
        subprocess.run(flags+["-DQ38_A_GPU","-fobjc-arc"]+sources+
                       ["-framework","Metal","-framework","Foundation","-lm","-o",str(exe)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,"-B","tools/qwen38_gpu_lock.py","--timeout","60","--",str(exe),str(build/"small.gguf")],
                       cwd=ROOT,env={**os.environ,"MTL_DEBUG_LAYER":"1"},check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--checkpoint",type=Path,help="read only GGUF header/catalog, never tensor payload")
    parser.add_argument("--reference-root",type=Path,help="existing pinned tokenizer assets/venv")
    parser.add_argument("--gpu",action="store_true",help="small fixture only, common lock, needs local metallib")
    args = parser.parse_args()
    destination = ROOT / "build/qwen38-a/small.gguf"
    write_fixture(destination, small_config())
    no_mtp = small_config()
    no_mtp["text_config"].update(mtp_num_hidden_layers=0,mtp={"layer_types":[],"num_hidden_layers":0})
    write_fixture(destination.with_name("no-mtp.gguf"),no_mtp)
    no_mtp["text_config"].update(num_hidden_layers=1,layer_types=["linear_attention"],ple_layer_ids=[1])
    write_fixture(destination.with_name("last-gdn.gguf"),no_mtp)
    text_cases()
    chat_cases()
    print(destination)
    if args.verify: verify(args)
