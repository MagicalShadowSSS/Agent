"""Original literal matching without a semantic datatype classifier.

Numeric and explicit calendar readings come only from literal syntax.
Four-digit-era numeric literals may also have a calendar reading. Their
affinity is the larger available precision-aware scalar/calendar score,
times frozen property cosine squared. No alignment files are opened.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from numeric_set_candidate_control import key_text, parsed_value_v2
from unified_raw_qwen_candidates import DATASETS, QwenEncoder, dump, read_jsonl


def literal_readings(raw):
    try:
        number = float(raw)
        scalar = math.isfinite(number)
    except ValueError:
        number, scalar = 0., False
    if not scalar:
        number = 0.
    original = parsed_value_v2(raw, False)
    calendar = parsed_value_v2(raw, True) if scalar and 500 <= number <= 3000 else original
    calendar_has = calendar is not None and calendar[2] == 1
    if not scalar and not calendar_has:
        return None
    digits = len(raw.rsplit('.', 1)[1]) if '.' in raw and 'e' not in raw.lower() else 0
    precision = 10. ** (-min(digits, 12))
    lo, hi = calendar[:2] if calendar_has else (0., 1.)
    return number, precision, float(scalar), lo, hi, float(calendar_has)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out)
    model = None
    for ds in a.datasets:
        if (out / 'done' / f'{ds}.json').exists():
            print(ds, 'already complete; preserving', flush=True)
            continue
        started = time.monotonic()
        raw_path = Path(a.raw_root) / 'inputs' / f'{ds}.jsonl'
        rows = read_jsonl(raw_path)
        ns = sum(r['role'] == 'query' for r in rows)
        nt = len(rows) - ns
        predicates = sorted({fact.split(': ', 1)[0] for row in rows for fact in row['attribute']
                             if ': ' in fact and literal_readings(fact.split(': ', 1)[1]) is not None})
        pid = {key: i for i, key in enumerate(predicates)}
        texts = [key_text(key) for key in predicates]
        if model is None:
            model = QwenEncoder()
        embeddings = model.encode(texts, ['target'] * len(texts))
        predicate_kernel = torch.tensor(np.maximum(embeddings @ embeddings.T, 0.) ** 2, device='cuda')
        dump(out / 'schemas' / f'{ds}.json', {'property_labels': predicates,
             'property_texts': texts, 'datatype_classifier': None, 'labels_read': False})
        facts = [[], []]
        for i, row in enumerate(rows):
            side, entity = (0, i) if i < ns else (1, i - ns)
            for fact in sorted(set(row['attribute'])):
                if ': ' not in fact:
                    continue
                key, value = fact.split(': ', 1)
                readings = literal_readings(value)
                if readings is not None:
                    facts[side].append((entity, pid[key], *readings))
        if not all(facts):
            raise ValueError((ds, 'no original literals on a side'))
        source, target = [np.asarray(f, np.float32) for f in facts]
        counts_s = np.bincount(source[:, 0].astype(int), minlength=ns)
        counts_t = np.bincount(target[:, 0].astype(int), minlength=nt)
        source_entities = torch.tensor(source[:, 0], device='cuda', dtype=torch.long)
        target_entities = torch.tensor(target[:, 0], device='cuda', dtype=torch.long)
        sp = torch.tensor(source[:, 1], device='cuda', dtype=torch.long)
        tp = torch.tensor(target[:, 1], device='cuda', dtype=torch.long)
        sv = torch.tensor(source[:, 2:], device='cuda')
        tv = torch.tensor(target[:, 2:], device='cuda')
        folder = out / 'kernels'
        folder.mkdir(parents=True, exist_ok=True)
        kernels = {key: np.lib.format.open_memmap(folder / f'{ds}_{key}.npy', mode='w+',
                   dtype=np.float32, shape=(ns, nt)) for key in ('forward', 'symmetric')}
        for begin in range(0, ns, 512):
            end = min(begin + 512, ns)
            batch = end - begin
            chosen = np.flatnonzero((source[:, 0] >= begin) & (source[:, 0] < end))
            source_best = torch.zeros((batch, nt), device='cuda')
            target_best = torch.zeros((batch, len(target)), device='cuda')
            for offset in range(0, len(chosen), 128):
                selected = torch.tensor(chosen[offset:offset + 128], device='cuda', dtype=torch.long)
                s = sv[selected]
                scale = torch.maximum(s[:, 0, None].abs(), tv[None, :, 0].abs()).clamp_min(1.)
                bandwidth = torch.maximum(s[:, 1, None], tv[None, :, 1])
                bandwidth = torch.maximum(bandwidth, .001 * scale)
                value = torch.exp(-.5 * ((s[:, 0, None] - tv[None, :, 0]) / bandwidth).square())
                value *= s[:, 2, None] * tv[None, :, 2]
                widths_s = (s[:, 4] - s[:, 3]).clamp_min(1e-8)
                widths_t = (tv[:, 4] - tv[:, 3]).clamp_min(1e-8)
                overlap = (torch.minimum(s[:, 4, None], tv[None, :, 4])
                           - torch.maximum(s[:, 3, None], tv[None, :, 3])).clamp_min(0.)
                overlap /= torch.sqrt(widths_s[:, None] * widths_t[None, :])
                overlap *= s[:, 5, None] * tv[None, :, 5]
                value = torch.maximum(value, overlap)
                kernel = predicate_kernel[sp[selected, None], tp[None, :]] * value
                best = torch.zeros((len(s), nt), device='cuda')
                best.scatter_reduce_(1, target_entities[None, :].expand(len(s), -1), kernel,
                                     reduce='amax', include_self=True)
                source_best.index_add_(0, source_entities[selected] - begin, best)
                target_best.scatter_reduce_(0, (source_entities[selected] - begin)[:, None].expand(-1, len(target)),
                                            kernel, reduce='amax', include_self=True)
                del s, selected, scale, bandwidth, value, widths_s, widths_t, overlap, kernel, best
            forward = source_best / torch.tensor(np.maximum(counts_s[begin:end], 1), device='cuda')[:, None]
            backward = torch.zeros((batch, nt), device='cuda')
            backward.scatter_add_(1, target_entities[None, :].expand(batch, -1), target_best)
            backward /= torch.tensor(np.maximum(counts_t, 1), device='cuda')[None, :]
            kernels['forward'][begin:end] = forward.cpu().numpy()
            kernels['symmetric'][begin:end] = torch.sqrt(forward * backward).cpu().numpy()
            del source_best, target_best, forward, backward
            dump(out / 'status' / f'{ds}.json', dict(done=end, total=ns, labels_read=False,
                 seconds=time.monotonic() - started))
            print(ds, 'grammar literals', end, '/', ns, flush=True)
        for matrix in kernels.values():
            matrix.flush()
            if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1.0001).any():
                raise ValueError((ds, 'invalid affinity kernel'))
        np.savez(folder / f'{ds}_availability.npz', source_has=counts_s > 0, target_has=counts_t > 0)
        dump(out / 'protocol' / f'{ds}.json', {
            'raw_input_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'input_node_order_sha256': hashlib.sha256(json.dumps([r['uri'] for r in rows]).encode()).hexdigest(),
            'property_encoder': 'frozen Qwen3-Embedding-8B, unprefixed',
            'property_texts_sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
            'datatype_classifier': None, 'fixed_datatype_prototypes': None,
            'literal_kernel': 'max available Gaussian and month-interval overlap; raw syntax only',
            'numeric_calendar_range': [500, 3000], 'relative_bandwidth': .001,
            'training': False, 'EA_checkpoint': None, 'labels_read': False,
            'external_lookup': False, 'source_count': ns, 'target_count': nt,
            'source_facts': len(source), 'target_facts': len(target),
            'seconds': time.monotonic() - started,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        dump(out / 'done' / f'{ds}.json', dict(complete=True, labels_read=False,
             stage='kernel construction only; no candidate accuracy'))


if __name__ == '__main__':
    main()
