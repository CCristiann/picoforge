"""Tiny graph with a complete byte vocabulary and the pinned chat template.

The 256 byte tokens have their byte value as ID, with no merges. Added tokens
keep their upstream spelling/order but are remapped to 256..288. This tests
the real text entrypoint without embedding the full model vocabulary.
"""
import argparse
import hashlib
import json
from pathlib import Path

from make_qwen38_graph import ROOT, graph_config, write_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'build/qwen38-g2/cli.gguf')
    parser.add_argument('--mtp', action='store_true')
    args = parser.parse_args()
    item = json.loads((ROOT/'tests/fixtures/qwen38/a/manifest.json').read_text())['tokenizer_config']
    raw = (args.reference_root/item['relative_to_reference_root']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('pinned tokenizer config changed')
    original = json.loads(raw)
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = iter(range(256, 512))
    tokens = [chr(b if b in visible else next(extra)) for b in range(256)]
    for i in range(248044, 248077):
        tokens.append(original['added_tokens_decoder'][str(i)]['content'])
    tokens.extend(f'[PAD{i}]' for i in range(289, 512))
    kinds = [1]*256 + [3 if i < 14 or i >= 26 or 16 <= i <= 21 else 4 for i in range(33)] + [5]*(512-289)
    meta = {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.chat_template': original['chat_template'],
            'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.token_type': kinds,
            'tokenizer.ggml.merges': [], 'tokenizer.ggml.eos_token_id': 258,
            'tokenizer.ggml.padding_token_id': 256, 'tokenizer.ggml.bos_token_id': 256,
            'tokenizer.ggml.add_bos_token': False}
    hf = graph_config(); hf['text_config']['eos_token_id'] = 256
    write_graph(args.output, hf, 3811, meta, enable_mtp=args.mtp)
    print(f'{args.output}: graph with byte tokenizer, {args.output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
