"""Reproduce the historical raw48 chooser with random three-row 48-token previews."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

PROJECT = Path(os.environ.get('SGMEA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
REFERENCE = PROJECT / 'output/zh_paged_evidence30_20261002'
FULL = PROJECT / 'output/zh_active_full30_20261002'
ROOT = PROJECT / 'output/zh_random48_direct30_20261002'
MODEL = os.environ.get('SGMEA_VL_MODEL', 'models/Qwen3-VL-8B-Instruct')
SEED = 20261002
CONDITIONS = ('historical_raw48', 'random3_raw48')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def compact(tokenizer, text, limit=48):
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True)


def random_rows(record, uri, modality, tokenizer):
    raw = list(record.get(modality, []))
    seed_text = json.dumps([SEED, uri, modality], ensure_ascii=False)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    positions = sorted(random.Random(seed).sample(range(len(raw)), min(3, len(raw))))
    selected = [raw[i] for i in positions]
    text = compact(tokenizer, '; '.join(selected), 48)
    return text, {'seed_sha256': hashlib.sha256(seed_text.encode()).hexdigest(),
                  'total': len(raw), 'positions': positions,
                  'selected_raw': selected, 'preview_text': text,
                  'preview_tokens': len(tokenizer.encode(text, add_special_tokens=False))}


def replace_preview(prompt, handle, field, text):
    pattern = rf'(^\[{re.escape(handle)}\.{field}\.preview\] ).*$'
    updated, count = re.subn(pattern, lambda match: match.group(1) + text, prompt, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'Expected one {handle}.{field}.preview line, got {count}')
    return updated


def prepare():
    assert not ROOT.exists(), 'Use a fresh output directory'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tasks = read(REFERENCE / 'tasks.json')
    ROOT.mkdir(); audits = {}
    for task in tasks:
        k = task['key']; old = read(REFERENCE / f'inputs/{k}.json'); row = read(FULL / f'inputs/{k}.json')
        by_uri = {row['source']: row['record'], **{x['uri']: x['record'] for x in row['candidates']}}
        random_prompt = old['prompt']; audit = {}
        for h, uri in old['handle_map'].items():
            if h == 'SRC':
                record = by_uri[old['source']] if old['source'] in by_uri else by_uri[uri]
            else:
                record = by_uri[uri]
            audit[h] = {}
            for modality in ('attribute', 'relation'):
                text, detail = random_rows(record, uri, modality, tokenizer)
                random_prompt = replace_preview(random_prompt, h, modality, text)
                audit[h][modality] = detail
        assert random_prompt != old['prompt']
        dump(ROOT / f'inputs/{k}.json', {'index': old['index'], 'key': k, 'source': old['source'],
            'handle_map': old['handle_map'], 'labels': old['labels'], 'historical_choice': old['historical_choice'],
            'historical_uri': old['historical_uri'], 'historical_prompt_sha256': old['historical_prompt_sha256'],
            'images': old['images'], 'prompts': {'historical_raw48': old['prompt'], 'random3_raw48': random_prompt}})
        audits[k] = audit
    dump(ROOT / 'tasks.json', tasks); dump(ROOT / 'sampling_audit.json', audits)
    snapshot = ROOT / 'code_snapshot'; snapshot.mkdir()
    shutil.copytree(REFERENCE / 'code_snapshot/unified_agent', snapshot / 'unified_agent', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    dump(ROOT / 'protocol.json', {'n': len(tasks), 'model': MODEL, 'seed': SEED, 'conditions': CONDITIONS,
        'historical_replay': 'Exact frozen prompt, candidate order, images and labels from paged raw48 pilot. Choice and prompt hash must reproduce historical output.',
        'random': 'For each URI and modality independently sample up to three distinct original rows uniformly without replacement using SHA256(seed, URI, modality), preserve selected row order, join with semicolon, then tokenizer-truncate the joined text to 48 tokens. Sampling never uses names, values, ranks, gold or predictions.',
        'difference': 'Only attribute.preview and relation.preview lines are changed in random3_raw48. Images, names, retrieval priors, record order, instructions, one-token label argmax and model are unchanged.',
        'not_active_agent': 'This is a direct chooser reproduction, not planner routing or modality-agent acquisition. It isolates the preview sampling question.',
        'labels_in_inference': False, 'training': False, 'workers': 8, 'created_at': time.time()})
    dump(ROOT / 'input_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*')
                                      if p.is_file() and 'code_snapshot' not in p.parts})
    dump(ROOT / 'code_sha256.json', {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')})
    print(json.dumps({'prepared': str(ROOT), 'n': len(tasks), 'conditions': CONDITIONS}), flush=True)


def verify():
    for rel, digest in read(ROOT / 'input_sha256.json').items(): assert sha(ROOT / rel) == digest, rel
    for rel, digest in read(ROOT / 'code_sha256.json').items(): assert sha(ROOT / 'code_snapshot' / rel) == digest, rel


def worker(wid):
    verify(); sys.path.insert(0, str(ROOT / 'code_snapshot'))
    import torch
    from unified_agent.model_backends import ComparisonVLM
    model = ComparisonVLM('qwen3', MODEL, PROJECT / 'output/unified_agent_dependencies_py39')
    tasks = read(ROOT / 'tasks.json')[wid::8]; done = 0
    for task in tasks:
        item = read(ROOT / f'inputs/{task["key"]}.json')
        for condition in CONDITIONS:
            dump(ROOT / f'progress/{wid}.json', {'worker': wid, 'index': task['index'], 'condition': condition,
                                                 'completed': done, 'time': time.time()})
            prompt = item['prompts'][condition]
            try:
                started = time.perf_counter()
                scores, meta = model.choose(prompt, item['images'], item['labels'])
                choice = max(scores, key=scores.get)
                if condition == 'historical_raw48':
                    assert choice == item['historical_choice'], 'Historical choice drift'
                    assert meta['prompt_sha256'] == item['historical_prompt_sha256'], 'Historical prompt hash drift'
                dump(ROOT / f'runs/{item["key"]}/{condition}.json', {'index': task['index'], 'key': item['key'],
                    'choice': choice, 'uri': item['handle_map'][choice], 'scores': scores, 'model': meta,
                    'seconds': time.perf_counter() - started, 'prompt_sha256': meta['prompt_sha256'],
                    'labels_loaded': False})
            except Exception as exc:
                dump(ROOT / f'failures/{item["key"]}/{condition}.json', {'error': repr(exc), 'time': time.time()})
        done += 1
    dump(ROOT / f'workers_done/{wid}.json', {'worker': wid, 'completed': done, 'time': time.time()})
    dump(ROOT / f'progress/{wid}.json', {'worker': wid, 'phase': 'complete', 'completed': done, 'time': time.time()})


def evaluate():
    labels = read(PROJECT / 'output/dbp_qwen_embedding_agent1000_20260929/agent_inputs/evaluation/zh_en.json')
    tasks = read(ROOT / 'tasks.json'); report = {'n': len(tasks), 'conditions': {}, 'paired': {}, 'failures': []}
    for f in (ROOT / 'failures').glob('*/*.json'): report['failures'].append(str(f.relative_to(ROOT)))
    cases = {}
    for condition in CONDITIONS:
        rows = []
        for task in tasks:
            f = ROOT / f'runs/{task["key"]}/{condition}.json'
            if not f.exists(): continue
            d = read(f); gold = labels[read(ROOT / f'inputs/{task["key"]}.json')['source']]
            baseline = read(ROOT / f'inputs/{task["key"]}.json')['historical_uri']
            rows.append({'index': task['index'], 'correct': d['uri'] == gold, 'gold': gold,
                'uri': d['uri'], 'baseline_correct': baseline == gold,
                'fixed': d['uri'] == gold and baseline != gold, 'broken': d['uri'] != gold and baseline == gold,
                'seconds': d['seconds'], 'input_tokens': d['model']['input_tokens']})
        fixed = [x['index'] for x in rows if x['fixed']]; broken = [x['index'] for x in rows if x['broken']]
        cases[condition] = {x['index']: x for x in rows}
        report['conditions'][condition] = {'completed': len(rows), 'correct': sum(x['correct'] for x in rows),
            'correct_over_30': sum(x['correct'] for x in rows) / len(tasks), 'fixed': fixed, 'broken': broken,
            'net': len(fixed) - len(broken), 'mean_seconds': sum(x['seconds'] for x in rows) / len(rows) if rows else None,
            'max_input_tokens': max((x['input_tokens'] for x in rows), default=0), 'cases': rows}
    left, right = cases.get('historical_raw48', {}), cases.get('random3_raw48', {}); common = left.keys() & right.keys()
    fixed = sorted(i for i in common if not left[i]['correct'] and right[i]['correct']); broken = sorted(i for i in common if left[i]['correct'] and not right[i]['correct'])
    report['paired']['historical_raw48 -> random3_raw48'] = {'n': len(common), 'fixed': fixed, 'broken': broken, 'net': len(fixed) - len(broken)}
    report['complete'] = all(v['completed'] == len(tasks) for v in report['conditions'].values()) and not report['failures']
    dump(ROOT / 'report.json', report)
    lines = ['# Random three rows, each modality truncated to 48 tokens', '',
             '| Condition | Completed | Correct | Fixed | Broken | Net | Mean sec | Max input tokens |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c, v in report['conditions'].items():
        seconds = f'{v["mean_seconds"]:.3f}' if v['mean_seconds'] is not None else '-'
        lines.append(f'| {c} | {v["completed"]} | {v["correct"]}/30 | {len(v["fixed"])} | {len(v["broken"])} | {v["net"]} | {seconds} | {v["max_input_tokens"]} |')
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'cases'} for k, v in report['conditions'].items()}, ensure_ascii=False), flush=True)
    return report


def launch():
    verify(); assert not (ROOT / 'launch.json').exists(); jobs = []
    (ROOT / 'logs').mkdir()
    for wid in range(8):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(wid), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
        with (ROOT / f'logs/{wid}.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'code_snapshot' / Path(__file__).name),
                'worker', '--worker-id', str(wid)], stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, cwd=PROJECT, start_new_session=True)
        jobs.append({'worker': wid, 'gpu': wid, 'pid': proc.pid})
    dump(ROOT / 'launch.json', {'jobs': jobs, 'time': time.time()})
    print(json.dumps(jobs), flush=True)


def monitor():
    jobs = read(ROOT / 'launch.json')['jobs']
    while True:
        report = evaluate(); alive = []
        for job in jobs:
            try: command = Path(f'/proc/{job["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError: command = ''
            if str(ROOT) in command and 'worker' in command: alive.append(job['worker'])
        dump(ROOT / 'status.json', {'alive': alive, 'time': time.time(), 'conditions': {
            c: {k: v for k, v in x.items() if k != 'cases'} for c, x in report['conditions'].items()}, 'failures': report['failures']})
        if not alive:
            dump(ROOT / 'final_status.json', {'time': time.time(), 'complete': report['complete'], 'failures': report['failures']}); return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'worker', 'launch', 'evaluate', 'monitor']); parser.add_argument('--worker-id', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'worker': worker(args.worker_id)
    elif args.mode == 'prepare': prepare()
    elif args.mode == 'launch': launch()
    elif args.mode == 'evaluate': evaluate()
    else: monitor()
