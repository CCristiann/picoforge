"""Paired picoforge/DwarfStar teacher-forced decode harness (R0.4).

Both engines get the same token IDs: a prompt prefilled up to a frontier, then
128 forced IDs appended one at a time (picoforge --teacher-forced, ds4-bench
--teacher-forced-decode). Forcing removes any dependence on what each engine
would choose, so tok/s compares the same work. `prepare` fixes the IDs and
requires both tokenizers to agree on them. Every process that maps the
checkpoint starts through Launcher (release wait, strict memory guard).
Real runs need the common lock:
  tools/qwen38_gpu_lock.py -- PYTHON tools/eval/qwen38_compare.py prepare OUT
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from qwen38_native_suite import ROOT, ENGINE, CHECKPOINT, dump, sha, require_lock
from run_observed import run as observe, sample
from summarize_qwen38_native import distribution

DWARFSTAR = ROOT.parent/'ds4-qwen38-baseline'
CORPUS = DWARFSTAR/'speed-bench/promessi_sposi.txt'
FROZEN = ROOT/'bench/qwen38_mtp_20261013/paired/coding'
# The frozen coding case ran --complete on this text (its runner-source.py);
# the short frontier must tokenize back to the recorded prompt IDs.
FROZEN_TEXT = 'def add(a, b):\n    return'
ENGINES = ('picoforge', 'dwarfstar')
FORCED, LONG_PROMPT, CONTEXT, CHUNK = 128, 8063, 8192, 1024
CUT = 1 << 16  # corpus bytes kept, cut at a newline: ~16K IDs, enough for 8063 + 128
STOP_TEXTS = ('<|endoftext|>', '<|im_end|>')  # picoforge pad/eos, DwarfStar endoftext/eos
# ds4-bench reads --prompt-file as plain BPE, but picoforge and `ds4 --dump-tokens
# --raw` split special tokens: a text holding a special spelling is refused.
SPECIALS = ('<|', '<｜', '｜DSML｜', '[gMASK]', '<sop>', 'think>', '<tool_', '</tool_', '<arg_', '</arg_')
# Ruling R9: stop a child at pressure level >= 2 (warn) or > 256 MiB of swap growth on
# two consecutive samples 0.25 s apart. Healthy runs of both engines stay at level 1.
GUARD, INTERVAL = dict(pressure=2, swap_bytes=256 << 20, samples=2), .25
TOKENIZE_TIMEOUT, GREEDY_TIMEOUT = 60, 1800


def schedule(counts, block):
    """Blocks of at most `block` samples; engines alternate while both have samples left.

    counts maps each engine to (warm-ups, measured). Within an engine the warm-ups
    (True) come first. Alternation spreads slow drift (heat, background load)
    over both engines instead of charging it to whichever ran second.
    """
    if set(counts) != set(ENGINES) or block < 1 or any(w < 0 or m < 1 for w, m in counts.values()):
        raise ValueError('need both engines, block >= 1, warm-ups >= 0 and measured >= 1')
    queues = {e: [True]*counts[e][0] + [False]*counts[e][1] for e in ENGINES}
    plan = []
    while any(queues.values()):
        for engine in ENGINES:
            if queues[engine]:
                plan.append((engine, queues[engine][:block]))
                del queues[engine][:block]
    return plan


def stats(values):
    """Median and nearest-sample p10/p90 (a real sample, never interpolated), as native-next."""
    return distribution(values)


def summary(samples):
    """Per-engine tok/s over measured samples only, and the ratio of the medians."""
    result = {e: stats([s['tok_s'] for s in samples if s['engine'] == e and not s['warmup']]) for e in ENGINES}
    p, d = result['picoforge'], result['dwarfstar']
    result['ratio_picoforge_over_dwarfstar'] = p['median']/d['median'] if p and d else None
    result['warmups_excluded'] = sum(bool(s['warmup']) for s in samples)
    return result


def agree(path, engines):
    """Both tokenizers must return identical IDs for the file; returns them."""
    a, b = (engines.tokenize(e, path) for e in ENGINES)
    if a != b:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        raise ValueError(f'tokenizers disagree on {path.name} at ID {i} ({len(a)} vs {len(b)} IDs)')
    return a


def frontier(out, name, text, prompt_tokens, engines, stops, expect=None):
    """Write one frontier's text, require both tokenizers to agree, split prompt and forced IDs."""
    if any(s in text for s in SPECIALS):
        raise ValueError(f'{name}: text holds a special-token spelling')
    path = out/f'{name}.txt'
    path.write_bytes(text.encode())  # the exact bytes both tokenizers and ds4-bench read
    ids = agree(path, engines)
    if expect is not None and ids != expect:
        raise ValueError(f'{name}: tokenizers do not reproduce the prompt + forced IDs')
    if len(ids) < prompt_tokens + FORCED:
        raise ValueError(f'{name}: {len(ids)} IDs, need {prompt_tokens + FORCED}')
    forced = ids[prompt_tokens:prompt_tokens + FORCED]
    if set(forced) & set(stops):
        raise ValueError(f'{name}: a stop ID is among the forced IDs (ds4-bench would stop)')
    return dict(text_file=path.name, text_sha256=sha(path), text_ids=len(ids),
                prompt_ids=ids[:prompt_tokens], forced_ids=forced, context=CONTEXT)


def prepare(out, engines, corpus, case):
    """Fix the IDs of both frontiers in a new suite directory OUT.

    8K: the first 8063 + 128 IDs of the corpus. Short: the frozen 9-ID coding
    prompt and 128 IDs of picoforge's own greedy continuation (ruling R7: the
    frozen record stops at 32). Its text, prompt + emitted, is what ds4-bench reads.
    """
    out.mkdir(parents=True, exist_ok=False)
    (out/'stop-tokens.txt').write_text(''.join(STOP_TEXTS))
    stops = agree(out/'stop-tokens.txt', engines)
    if len(stops) != len(STOP_TEXTS):
        raise ValueError('a stop spelling is not a single token')
    raw = corpus.read_bytes()[:CUT]
    long = frontier(out, '8k', raw[:raw.rindex(b'\n') + 1].decode(), LONG_PROMPT, engines, stops)
    ids, emitted = engines.greedy(case['prompt_ids'], FORCED, out/'short-greedy')
    if len(ids) != FORCED:
        raise ValueError(f'greedy continuation stopped early: {len(ids)} of {FORCED} IDs')
    if set(ids) & set(stops):
        raise ValueError('a stop ID is among the greedy continuation')
    short = frontier(out, 'short', case['prompt_text'] + emitted, len(case['prompt_ids']), engines, stops,
                     case['prompt_ids'] + ids)
    recorded = case['recorded_ids']
    short['frozen_prefix_match'] = ids[:len(recorded)] == recorded
    if not short['frozen_prefix_match']:
        print(f'warning: the frozen {len(recorded)} IDs are not a prefix of the new continuation', file=sys.stderr)
    dump(out/'suite.json', dict(version=1, checkpoint=str(CHECKPOINT.relative_to(ROOT)), corpus=corpus.name,
         corpus_sha256=sha(corpus), corpus_cut_bytes=CUT, frozen_case=case.get('source'),
         frozen_recorded_ids=recorded, stop_texts=STOP_TEXTS, stop_ids=stops, frontiers={'short': short, '8k': long}))


def bounded():
    """Release wait and pinned DwarfStar revision, from the bounded gate (imports numpy: lazy)."""
    from qwen38_native_bounded import settle_after_metal, REVISION
    return settle_after_metal, REVISION


def conditions():
    """Power source, thermal warnings and swap (powermetrics needs sudo: ruling R6)."""
    def pmset(what):
        return subprocess.run(['pmset', '-g', what], capture_output=True, text=True).stdout
    batt, memory = pmset('batt'), sample(ROOT/'build/process_memory')
    return dict(ac_power="'AC Power'" in batt, batt=batt, therm=pmset('therm'),
                swap_used_bytes=memory['system_swap_used_bytes'] if memory else None)


class Launcher:
    """The only way a process that maps the checkpoint starts (ruling R9).

    A Metal process's wired pages (43-46 GiB here) are still being released for
    ~1.5 s after it exits; mapping the checkpoint again inside that window froze
    the Mac twice (WindowServer watchdog panic). So every launch waits for the
    release before AND after its process, even a failed one, and runs it
    observed under GUARD. Power and swap are recorded around it.
    """
    def __init__(self, settle=None, observe=observe, conditions=conditions):
        self.settle, self.observe, self.conditions = settle, observe, conditions

    def __call__(self, directory, cwd, command, timeout):
        settle = self.settle or bounded()[0]
        directory.mkdir(parents=True)

        def wait(name):
            (directory/name).mkdir()
            settle(directory/name)
        wait('release-before')
        before = self.conditions()
        try:
            code = self.observe(directory/'process', cwd, command, timeout=timeout, interval=INTERVAL, guard=GUARD)
        finally:
            wait('release-after')
        dump(directory/'conditions.json', dict(before=before, after=self.conditions()))
        return code


class Engines:
    """The real tokenizers and greedy run, all through the Launcher; evidence stays in OUT."""
    def __init__(self, out, launch=None):
        self.out, self.launch = out, launch or Launcher()

    def tokenize(self, engine, path):
        directory = self.out/f'tokens-{path.stem}-{engine}'
        if engine == 'picoforge':
            command, cwd = [str(ENGINE), '--qwen38', str(CHECKPOINT), '--complete-file', str(path),
                            '--context', '32768', '--max-new', '0', '--encode-only'], ROOT
        else:
            command, cwd = [str(DWARFSTAR/'ds4'), '-m', str(CHECKPOINT), '--dump-tokens', '--raw',
                            '--prompt-file', str(path)], DWARFSTAR
        if self.launch(directory, cwd, command, TOKENIZE_TIMEOUT):
            raise RuntimeError(f'{engine} tokenizer failed on {path.name}; evidence retained')
        first = json.loads((directory/'process/stdout.log').read_text().split('\n', 1)[0])
        return first['prompt_ids'] if engine == 'picoforge' else first  # ds4 prints the ID list first

    def greedy(self, prompt_ids, count, directory):
        """picoforge's own greedy continuation; stdout is the emitted text plus one newline."""
        ids = directory/'ids.json'
        command = [str(ENGINE), '--qwen38', str(CHECKPOINT), '--tokens', ','.join(map(str, prompt_ids)),
                   '--max-new', str(count), '--context', str(CONTEXT), '--chunk', str(CHUNK),
                   '--runs-jsonl', str(directory/'runs.jsonl'), '--token-ids', str(ids)]
        if self.launch(directory, ROOT, command, GREEDY_TIMEOUT):
            raise RuntimeError('greedy continuation run failed; evidence retained')
        got = json.loads(ids.read_text())
        text = (directory/'process/stdout.log').read_bytes()
        if got['prompt_ids'] != prompt_ids or not text.endswith(b'\n'):
            raise RuntimeError('greedy run: prompt IDs differ or stdout lacks its final newline')
        return got['generated_ids'], text[:-1].decode()


def frozen_case(directory=FROZEN):
    """The 9-ID coding prompt and its recorded 32-ID greedy continuation."""
    command = json.loads((directory/'ordinary/process.json').read_text())['command']
    rows = [json.loads(line) for line in (directory/'ordinary.jsonl').read_text().splitlines()]
    if any(r['generated_ids'] != rows[0]['generated_ids'] for r in rows):
        raise ValueError('frozen repetitions differ')
    return dict(prompt_ids=[int(t) for t in command[command.index('--tokens') + 1].split(',')],
                prompt_text=FROZEN_TEXT, recorded_ids=rows[0]['generated_ids'],
                source=str(directory.relative_to(ROOT)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare', help='fix the IDs of both frontiers').add_argument('output', type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        p.error(f'output exists, never overwritten: {out}')
    if 'PICOFORGE_GPU_LOCK_FD' not in os.environ:
        p.error('run under tools/qwen38_gpu_lock.py')
    require_lock()
    prepare(out, Engines(out), CORPUS, frozen_case())


if __name__ == '__main__':
    main()
