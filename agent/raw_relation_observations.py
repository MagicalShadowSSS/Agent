"""Render original KG edges without a neighbor-name lookup or generated facts.

Readable resource titles are a deterministic URI rendering, not a translation.
Every emitted record retains a source file/line/URI provenance in the audit.
No alignment links or evaluation labels are read.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote


def title(uri):
    return unquote(uri.rsplit('/', 1)[-1]).replace('_', ' ')


class RawRelationObservations:
    def __init__(self, project, root, expected_sha256):
        self.graph = {}
        tasks = json.loads((root / 'tasks.json').read_text())
        nodes = {}
        for task in tasks:
            row = json.loads((root / f'inputs/{task["cohort"]}/{task["key"]}.json').read_text())
            nodes.setdefault(task['cohort'][:5], set()).update(
                [row['source']] + [candidate['uri'] for candidate in row['candidates']])
        for ds, wanted in nodes.items():
            graph = {uri: [] for uri in wanted}
            for language in (ds[:2], 'en'):
                relative = f'dataset/DBP15K/{ds}/{language}_rel_triples'
                path = project / relative
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256[relative], relative
                with path.open() as records:
                    for line_number, line in enumerate(records, 1):
                        fields = line.rstrip('\n').split('\t')
                        assert len(fields) == 3, (relative, line_number)
                        assert all(value.startswith('<') and value.endswith('>') for value in fields)
                        subject, predicate, obj = [value[1:-1] for value in fields]
                        for owner, direction, neighbor in ((subject, 'out', obj), (obj, 'in', subject)):
                            if owner in graph:
                                graph[owner].append({'direction': direction,
                                    'predicate_uri': predicate, 'neighbor_uri': neighbor,
                                    'source_file': relative, 'source_line': line_number})
            self.graph[ds] = graph

    def install(self, evidence, dataset, index_class):
        audit = {}
        for handle, owner in evidence.handles.items():
            rows = []
            for i, original in enumerate(self.graph[dataset][owner]):
                predicate, neighbor = title(original['predicate_uri']), title(original['neighbor_uri'])
                identifier = f'{handle}.relation.raw.{i}'
                row = {'id': identifier, 'direction': original['direction'],
                       'predicate': predicate, 'neighbor': neighbor,
                       'raw': f'{original["direction"]} {predicate}: {neighbor}'}
                rows.append(row)
                audit[identifier] = {'owner_uri': owner, **original,
                                    'rendered_predicate': predicate, 'rendered_neighbor': neighbor}
                evidence.originals[identifier] = row
            evidence.indices[handle, 'relation'] = index_class(rows)
        return audit
