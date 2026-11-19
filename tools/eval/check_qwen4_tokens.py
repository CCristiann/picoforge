"""Independent HF tokenizer/template reference for the frozen local prompts."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def reference(suite_dir, tokenizer_path, config_path):
    config = json.loads(config_path.read_text())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path),
                                        chat_template=config["chat_template"],
                                        eos_token=config["eos_token"])
    suite = json.loads((suite_dir / "suite.json").read_text())
    result = dict(revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
                  tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                  tokenizer_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  cases={})
    for case in suite["cases"]:
        text = (suite_dir / case["prompt_file"]).read_text()
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             return_dict=False, enable_thinking=case["mode"] == "think")
        if len(ids) + case["max_new"] >= suite["context_tokens"]:
            raise ValueError(f"context budget exceeded: {case['id']}")
        result["cases"][case["id"]] = ids
    return result


def compare(expected, pilot):
    result = {}
    for case, ids in expected["cases"].items():
        actual = [int(n) for n in (pilot / case / "answer.0.prompt.ids").read_text().split()]
        result[case] = dict(matched=actual == ids, actual_tokens=len(actual), expected_tokens=len(ids),
                            first_difference=next((i for i, (a, b) in enumerate(zip(actual, ids)) if a != b), None))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    expected = reference(args.suite, args.tokenizer, args.config)
    if args.pilot:
        results = compare(expected, args.pilot)
        print(json.dumps(results, indent=2))
        raise SystemExit(0 if all(r["matched"] for r in results.values()) else 1)
    print(json.dumps(expected, indent=2))
