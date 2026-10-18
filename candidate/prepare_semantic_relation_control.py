"""Restore predicate semantics and optionally render opaque neighbors' facts.

One same-graph serializer is used for all five datasets. No entity alignment
file is read; query IDs and catalogs are inherited without opening labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from unified_raw_qwen_candidates import (DATASETS, dataset_dir, dump, label, parse_fact,
    read_jsonl, read_map, sha, translated_names, write_jsonl)


def readable(value):
    value = label(value)
    value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
    return value.replace('.', ' ').replace('_', ' ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--neighbor-facts', action='store_true')
    parser.add_argument('--max-edges', type=int, default=12)
    args = parser.parse_args()
    root, out, audit = Path(args.root).resolve(), Path(args.out).resolve(), Path(args.audit).resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory')
    summary = {}
    for dataset in DATASETS:
        base = dataset_dir(dataset)
        maps = [read_map(base / f'ent_ids_{side}') for side in (1, 2)]
        predicates = json.loads((audit / f'{dataset}_id2predicate.json').read_text())
        names = translated_names(dataset)
        attributes = defaultdict(list)
        ids = {**maps[0], **maps[1]}
        by_uri = {uri: eid for eid, uri in ids.items()}
        for filename in ('att_triples1', 'att_triples2'):
            with (base / filename).open(encoding='utf-8') as stream:
                for line in stream:
                    fact = parse_fact(line)
                    if fact:
                        attributes[fact[0]].append(f'{readable(fact[1])}: {fact[2]}')
        for uri in attributes:
            attributes[uri] = sorted(set(attributes[uri]))[:3]
        def neighbor_description(uri):
            name = names.get(by_uri.get(uri, ''), '') or label(uri)
            if name:
                return name
            if args.neighbor_facts and attributes.get(uri):
                return '[' + '; '.join(attributes[uri]) + ']'
            return '[unidentified entity]'
        edges = defaultdict(set)
        for side, entities in enumerate(maps, 1):
            with (base / f'triples_{side}').open(encoding='utf-8') as stream:
                for line in stream:
                    values = line.split()
                    if len(values) != 3:
                        continue
                    h, r, t = values
                    h, t = entities.get(h, h), entities.get(t, t)
                    predicate = readable(predicates[r]) if r in predicates else 'unknown predicate'
                    edges[h].add(f'out {predicate}: {neighbor_description(t)}')
                    edges[t].add(f'in {predicate}: {neighbor_description(h)}')
        rows = read_jsonl(root / 'inputs' / f'{dataset}.jsonl')
        # Corpus document frequency, never correctness or cross-graph labels.
        df = Counter(word for uri in edges for word in set(re.findall(r'\w+', ' '.join(edges[uri]).lower())))
        def information(record):
            words = set(re.findall(r'\w+', record.lower()))
            return sum(math.log((1+len(edges))/(1+df[word])) for word in words) / max(1, len(words))
        counts = []
        for row in rows:
            original = sorted(edges.get(row['uri'], set()), key=lambda e: (-information(e), e))
            row['relation'] = original[:args.max_edges]
            row['relation_text'] = '\n'.join(row['relation'])
            counts.append({'uri': row['uri'], 'original_edges': len(original), 'visible_edges': len(row['relation'])})
        write_jsonl(out / 'inputs' / f'{dataset}.jsonl', rows)
        write_jsonl(out / 'coverage' / f'{dataset}.jsonl', counts)
        dest = out / 'labels' / f'{dataset}.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(root / 'labels' / f'{dataset}.jsonl')
        for field in ('name', 'attribute'):
            destination = out / 'embeddings' / dataset / field
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(root / 'embeddings' / dataset / field)
        summary[dataset] = {'records': len(rows), 'input_sha256': sha(out / 'inputs' / f'{dataset}.jsonl'),
                            'query_ids_and_target_catalog_unchanged': True}
        print(dataset, len(rows), flush=True)
    dump(out / 'protocol.json', {'parent': str(root), 'datasets': list(DATASETS),
         'relation_audit': str(audit), 'changed_views': ['relation'],
         'serializer': 'out/in original predicate and readable neighbor, unknown IDs never decoded',
         'opaque_neighbor': 'up to 3 original attributes' if args.neighbor_facts else 'unidentified entity',
         'max_edges': args.max_edges, 'selection': 'mean corpus IDF of complete records',
         'labels_read': False, 'entity_names_repaired': False, 'training': False,
         'task_checkpoint': None, 'summary': summary})


if __name__ == '__main__':
    main()
