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
import csv
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
# Ruling R13: also stop it when two sample attempts are > 5 s apart (the longest gap in
# 46 healthy short runs was 0.35 s): the Mac slept, and a sample that spans a sleep is
# not a measurement. The failed 8K baseline ran through a DarkWake with ~899 s gaps.
GUARD, INTERVAL = dict(pressure=2, swap_bytes=256 << 20, samples=2, gap_s=5), .25
TOKENIZE_TIMEOUT, GREEDY_TIMEOUT, RUN_TIMEOUT, SSD_TIMEOUT = 60, 1800, 3600, 900
PROTOCOL = ('One process per sample for both engines (ds4-bench measures one frontier per process): '
            'prefill to the frontier, then 128 forced one-token appends. tok/s = forced IDs / decode '
            'seconds (picoforge decode_s, ds4-bench gen_tps). Warm-ups discarded. OS file cache warm, '
            'not purged: no cold-start claim.')
COLUMNS = ('index', 'engine', 'warmup', 'tok_s', 'decode_tokens', 'ac_power', 'swap_delta_bytes',
           'peak_footprint_bytes', 'process_s')


def schedule(counts, block):
    """Blocks of at most `block` samples; engines alternate while both have samples left.

    counts maps each engine to (warm-ups, measured). Within an engine the warm-ups
    (True) come first. Alternation spreads slow drift (heat, background load)
    over both engines instead of charging it to whichever ran second. An engine
    given (0, 0) is not run (picoforge-only runs); a requested one needs measured >= 1.
    """
    requested = [c for c in counts.values() if any(c)]
    if set(counts) != set(ENGINES) or block < 1 or not requested or any(w < 0 or m < 1 for w, m in requested):
        raise ValueError('need both engines (an unused one as (0, 0)), block >= 1, and for each used one '
                         'warm-ups >= 0 and measured >= 1')
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


def graphics(systemstate):
    """True iff `pmset -g systemstate` lists Graphics among the current capabilities.

    Awake: `CPU Graphics Audio Network`. A DarkWake (the ~20 s maintenance slices a closed
    Mac wakes for) has `CPU Network`: no GPU work runs there.
    """
    line = next((l for l in systemstate.splitlines() if l.lstrip().startswith('Current System Capabilities are:')), '')
    return 'Graphics' in line.split()


def lid_open(ioreg_text):
    """True iff `ioreg -r -k AppleClamshellState -d 1` reports the lid open (a closed lid prints Yes)."""
    return '"AppleClamshellState" = No' in ioreg_text


def awake():
    """Is the Mac fully awake, lid open and on AC? A DarkWake cannot be told from AC power alone."""
    state = subprocess.run(['pmset', '-g', 'systemstate'], capture_output=True, text=True).stdout
    clamshell = subprocess.run(['ioreg', '-r', '-k', 'AppleClamshellState', '-d', '1'], capture_output=True, text=True).stdout
    batt = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout
    return dict(graphics=graphics(state), lid_open=lid_open(clamshell), ac_power="'AC Power'" in batt,
                systemstate=state.strip().split('\n')[0].strip())


def require_awake(state):
    """Refuse to start anything unless `awake()` says fully awake, lid open, on AC (ruling R13)."""
    failed = [name for name in ('graphics', 'lid_open', 'ac_power') if not state[name]]
    if failed:
        raise RuntimeError(f"{', '.join(failed)} is false: the Mac must be fully awake (Graphics), "
                           f"lid open, on AC (ruling R13); {state.get('systemstate', '')!r}")


def conditions():
    """Wake state, power source, thermal warnings and swap (powermetrics needs sudo: ruling R6)."""
    def pmset(what):
        return subprocess.run(['pmset', '-g', what], capture_output=True, text=True).stdout
    memory = sample(ROOT/'build/process_memory')
    return dict(batt=pmset('batt'), therm=pmset('therm'),
                swap_used_bytes=memory['system_swap_used_bytes'] if memory else None, **awake())


class Launcher:
    """The only way a process that maps the checkpoint starts (ruling R9).

    A Metal process's wired pages (43-46 GiB here) are still being released for
    ~1.5 s after it exits; mapping the checkpoint again inside that window froze
    the Mac twice (WindowServer watchdog panic). So every launch waits for the
    release before AND after its process, even a failed one, and runs it
    observed under GUARD. Power, wake state and swap are recorded around it. Before
    anything is created it also refuses unless the Mac is fully awake, lid open, on
    AC (ruling R13): a DarkWake runs for ~20 s per 15 min and ruined a whole baseline.
    """
    def __init__(self, settle=None, observe=observe, conditions=conditions, awake=awake):
        self.settle, self.observe, self.conditions, self.awake = settle, observe, conditions, awake

    def __call__(self, directory, cwd, command, timeout, env=None):
        require_awake(self.awake())
        settle = self.settle or bounded()[0]
        directory.mkdir(parents=True)

        def wait(name):
            (directory/name).mkdir()
            settle(directory/name)
        wait('release-before')
        before = self.conditions()
        try:
            code = self.observe(directory/'process', cwd, command, timeout=timeout, interval=INTERVAL, guard=GUARD,
                                env_overrides=env)
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


def pinned(revision):
    """DwarfStar must be the pinned revision with a clean tree (read and execute only)."""
    def git(*a):
        return subprocess.check_output(['git', '-C', str(DWARFSTAR), *a], text=True)
    head = git('rev-parse', 'HEAD').strip()
    if head != revision or git('status', '--porcelain'):
        raise RuntimeError(f'{DWARFSTAR.name} must be {revision[:7]} with a clean tree')
    return head


def thermal():
    """NSProcessInfo.thermalState from the R0.1 probe; taken at run start and end only."""
    probe = ROOT/'build/qwen38-capabilities'
    done = subprocess.run([str(probe)], capture_output=True, text=True) if probe.exists() else None
    return json.loads(done.stdout)['thermal_state'] if done and done.returncode == 0 else None


def provenance(revision):
    def git(*a):
        return subprocess.check_output(['git', *a], cwd=ROOT, text=True)
    binaries = {'picoforge': ENGINE, 'picoforge.metallib': ROOT/'picoforge.metallib',
                'ds4-bench': DWARFSTAR/'ds4-bench', 'ds4': DWARFSTAR/'ds4', 'tool': Path(__file__)}
    return dict(revision=git('rev-parse', 'HEAD').strip(), diff=git('diff', '--', 'src', 'tools', 'tests'),
                dwarfstar_revision=pinned(revision), checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                checkpoint_bytes=CHECKPOINT.stat().st_size, checkpoint_sha256_recomputed=False,
                sha256={name: sha(path) for name, path in binaries.items()})


def command(engine, f, text, d):
    """One teacher-forced sample. Never --logits: picoforge writes its rows inside decode_s."""
    n = str(len(f['prompt_ids']))
    if engine == 'picoforge':
        return [str(ENGINE), '--qwen38', str(CHECKPOINT), '--tokens', ','.join(map(str, f['prompt_ids'] + f['forced_ids'])),
                '--teacher-forced', str(FORCED), '--context', str(CONTEXT), '--chunk', str(CHUNK),
                '--runs-jsonl', str(d/'runs.jsonl'), '--token-ids', str(d/'ids.json')], ROOT
    return [str(DWARFSTAR/'ds4-bench'), '-m', str(CHECKPOINT), '--prefill-chunk', str(CHUNK), '--prompt-file', str(text),
            '--ctx-start', n, '--ctx-max', n, '--ctx-alloc', str(CONTEXT), '--gen-tokens', str(FORCED),
            '--teacher-forced-decode', '--csv', str(d/'bench.csv')], DWARFSTAR


def result(engine, f, d):
    """tok/s of one sample, after checking it appended all of this frontier's forced IDs."""
    if engine == 'picoforge':
        rows = [json.loads(line) for line in (d/'runs.jsonl').read_text().splitlines()]
        ids, r = json.loads((d/'ids.json').read_text()), rows[0] if len(rows) == 1 else {}
        if (r.get('mode') != 'teacher-forced' or r['decode_tokens'] != FORCED or r['stop_reason'] != 'length'
                or not r['decode_s'] > 0 or (ids['prompt_ids'], ids['forced_ids']) != (f['prompt_ids'], f['forced_ids'])):
            raise ValueError(f'{d.name}: not a complete teacher-forced sample of this frontier')
        return dict(tok_s=r['decode_tokens']/r['decode_s'], decode_tokens=r['decode_tokens'],
                    decode_s=r['decode_s'], prefill_s=r['prefill_s'])
    with (d/'bench.csv').open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1 or int(rows[0]['ctx_tokens']) != len(f['prompt_ids']) or int(rows[0]['gen_tokens']) != FORCED:
        raise ValueError(f'{d.name}: not a complete teacher-forced sample of this frontier')
    r = rows[0]  # ds4-bench stops early on EOS: prepare refused stop IDs among the forced ones
    return dict(tok_s=float(r['gen_tps']), decode_tokens=int(r['gen_tokens']),
                gen_first_ms=float(r['gen_first_ms']), prefill_tps=float(r['prefill_tps']))


def stop_note(directory):
    """What the observer recorded when it stopped a sample, for the error message."""
    path = directory/'process/process.json'
    reason = json.loads(path.read_text()).get('stop_reason') if path.exists() else None
    if reason == 'sleep gap':
        return ', sleep gap: the system slept, this sample is invalid and the run stopped'
    return f', {reason}' if reason else ''


def run(suite_dir, out, name, counts, block, timeout, launch=None, awake=awake):
    """Alternate engine blocks on one frontier: one observed process per sample."""
    plan, launch = schedule(counts, block), launch or Launcher()
    suite = json.loads((suite_dir/'suite.json').read_text())
    f = suite['frontiers'][name]
    text = suite_dir/f['text_file']
    if sha(text) != f['text_sha256']:
        raise ValueError('frontier text changed since prepare')
    require_awake(awake())  # no evidence directory on a sleeping, closed or battery-powered Mac
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(complete=False, frontier=name, engines=[e for e in ENGINES if any(counts[e])], counts=counts,
                    block=block, plan=plan, protocol=PROTOCOL,
                    suite_sha256=sha(suite_dir/'suite.json'), guard=GUARD, **provenance(bounded()[1]),
                    before=dict(conditions(), thermal_state=thermal()))
    dump(out/'manifest.json', manifest)
    samples = []
    for engine, flags in plan:
        for warmup in flags:
            d = out/(f'{len(samples):03d}-{engine}' + ('-warmup' if warmup else ''))
            cmd, cwd = command(engine, f, text, d)
            code = launch(d, cwd, cmd, timeout)
            if code:
                raise RuntimeError(f'{d.name} failed (code {code}{stop_note(d)}); raw evidence retained')
            info, around = (json.loads((d/x).read_text()) for x in ('process/process.json', 'conditions.json'))
            a, b = info['system_before'], info['system_after']
            samples.append(dict(index=len(samples), engine=engine, warmup=warmup, **result(engine, f, d),
                                ac_power=around['before']['ac_power'] and around['after']['ac_power'],
                                swap_delta_bytes=b['system_swap_used_bytes'] - a['system_swap_used_bytes'] if a and b else None,
                                peak_footprint_bytes=info['lifetime_peak_footprint_bytes'],
                                process_s=info['process_s'], command=cmd))
            with (out/'samples.jsonl').open('a') as fh:
                fh.write(json.dumps(samples[-1]) + '\n')
    with (out/'samples.csv').open('x', newline='') as fh:
        writer = csv.DictWriter(fh, COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    manifest.update(complete=True, after=dict(conditions(), thermal_state=thermal()))
    dump(out/'manifest.json', manifest)
    summarize(out)


def summarize(out):
    """Median, nearest-sample p10/p90 and the ratio; summary.json is derived and rewritable."""
    samples = [json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines()]
    result = dict(summary(samples), complete=json.loads((out/'manifest.json').read_text())['complete'],
                  on_battery=any(not s['ac_power'] for s in samples),
                  max_swap_delta_bytes=max(s['swap_delta_bytes'] or 0 for s in samples))
    dump(out/'summary.json', result)
    for e in ENGINES:
        r = result[e]
        print(f"{e}: median {r['median']:.3f} tok/s, p10 {r['p10']:.3f}, p90 {r['p90']:.3f}, n {r['n']}"
              if r else f'{e}: no measured sample')
    print(f"picoforge/DwarfStar {result['ratio_picoforge_over_dwarfstar']}, on battery {result['on_battery']}, "
          f"complete {result['complete']}")


def ssd_check(out, timeout, launch=None):
    """One attempt of `ds4 --ssd-streaming` on the Qwen3.8 file; its exit status is the finding."""
    launch, head = launch or Launcher(), pinned(bounded()[1])
    out.mkdir(parents=True, exist_ok=False)
    cmd = [str(DWARFSTAR/'ds4'), '-m', str(CHECKPOINT), '--ssd-streaming', '-c', str(CONTEXT),
           '-n', '8', '--temp', '0', '-p', FROZEN_TEXT]
    launch(out/'ssd', DWARFSTAR, cmd, timeout)
    info = json.loads((out/'ssd/process/process.json').read_text())
    dump(out/'ssd-check.json', dict(command=cmd, dwarfstar_revision=head, ds4_sha256=sha(DWARFSTAR/'ds4'),
         returncode=info['returncode'], stop_reason=info['stop_reason'],
         accepted=info['returncode'] == 0 and info['stop_reason'] is None))
    print(f"ds4 --ssd-streaming exit status {info['returncode']}" +
          (f" ({info['stop_reason']})" if info['stop_reason'] else ''))


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
    r = sub.add_parser('run', help='alternate engine blocks on one frontier')
    r.add_argument('suite', type=Path)
    r.add_argument('output', type=Path)
    r.add_argument('--frontier', choices=('short', '8k'), required=True)
    for name in ('warmup', 'measure', 'block'):
        r.add_argument('--' + name, type=int, required=True)
    r.add_argument('--picoforge-warmup', type=int, help='override for picoforge (the slow 8K baseline)')
    r.add_argument('--picoforge-measure', type=int)
    r.add_argument('--engines', choices=('both', 'picoforge'), default='both',
                   help='picoforge: no DwarfStar samples (the R1 ladder keeps the R0 DwarfStar baseline)')
    r.add_argument('--timeout', type=float, default=RUN_TIMEOUT, help='seconds per process')
    sub.add_parser('summarize', help='recompute the summary of a run').add_argument('output', type=Path)
    s = sub.add_parser('ssd-check', help='one ds4 --ssd-streaming attempt')
    s.add_argument('output', type=Path)
    s.add_argument('--timeout', type=float, default=SSD_TIMEOUT)
    a = p.parse_args()
    out = a.output.resolve()
    if a.action == 'summarize':
        return summarize(out)
    if out.exists():
        p.error(f'output exists, never overwritten: {out}')
    if a.action == 'run':
        counts = {'picoforge': (a.warmup if a.picoforge_warmup is None else a.picoforge_warmup,
                                a.measure if a.picoforge_measure is None else a.picoforge_measure),
                  'dwarfstar': (a.warmup, a.measure) if a.engines == 'both' else (0, 0)}
        try:
            schedule(counts, a.block)
        except ValueError as e:
            p.error(str(e))
    if 'PICOFORGE_GPU_LOCK_FD' not in os.environ:
        p.error('run under tools/qwen38_gpu_lock.py')
    require_lock()
    if a.action == 'prepare':
        prepare(out, Engines(out), CORPUS, frozen_case())
    elif a.action == 'run':
        run(a.suite.resolve(), out, a.frontier, counts, a.block, a.timeout)
    else:
        ssd_check(out, a.timeout)

if __name__ == '__main__':
    main()
