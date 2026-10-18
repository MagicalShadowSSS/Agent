"""Predicate correspondence inferred only inside the node-consistency loop.

This removes the separate lexical predicate matcher and its encoder call.
Raw predicate IDs are used only as relation-group keys; their values are
never compared across graphs. No supplied entity/relation correspondences.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from degree_corrected_candidates import DegreeCorrectedOperator
from unified_raw_qwen_candidates import dataset_dir, read_map


class TopologyPredicateOperator(DegreeCorrectedOperator):
    def __init__(self, dataset, graph_root, audit, out, checkpoints, topk=1, freeze=False):
        self.dataset = dataset; self.out = Path(out); self.topk = topk
        self.directional = True; self.infer_schema = True; self.round = 0
        self.functionality = False; self.propagation = 'soft'
        self.audit_rounds = set(checkpoints); self.freeze = freeze
        root = dataset_dir(dataset)
        maps = [read_map(root / f'ent_ids_{side}') for side in (1, 2)]
        uris = [list(m.values()) for m in maps]
        indexes = [{uri: i for i, uri in enumerate(side)} for side in uris]
        self.predicates = json.loads((Path(audit) / f'{dataset}_id2predicate.json').read_text())
        self.edges = [defaultdict(list), defaultdict(list)]
        for side in (0, 1):
            with (root / f'triples_{side + 1}').open() as stream:
                for line in stream:
                    h, r, t = line.split()
                    if r not in self.predicates: continue
                    hu, tu = maps[side].get(h, h), maps[side].get(t, t)
                    if hu in indexes[side] and tu in indexes[side]:
                        self.edges[side][r].append((indexes[side][hu], indexes[side][tu]))
        self.ids = [sorted(side) for side in self.edges]
        self.directed = [{}, {}]
        for side in (0, 1):
            for r, edges in self.edges[side].items():
                pairs = np.asarray(edges)
                matrix = sp.csr_matrix((np.ones(len(pairs), np.float32),
                    (pairs[:, 0], pairs[:, 1])), shape=(len(uris[side]),) * 2)
                matrix.sum_duplicates(); matrix.data[:] = 1
                self.directed[side][r] = matrix
        self.shape = tuple(map(len, uris))
        manifest = json.loads((Path(graph_root) / 'manifests' / f'{dataset}.json').read_text())
        assert self.shape == (manifest['source_count'], manifest['target_count'])
        self.target_edge_types = defaultdict(list)
        for j, r in enumerate(self.ids[1]):
            for h, t in self.edges[1][r]: self.target_edge_types[h, t].append(j)
        self.semantic_matches = []; self.groups = []
        self.ordinary = []
        for side in (1, 2):
            graph = sp.load_npz(Path(graph_root) / 'graphs' / f'{dataset}_side{side}.npz').tocsr()
            graph.setdiag(0); graph.eliminate_zeros(); graph.data[:] = 1
            self.ordinary.append(graph)

    def update_schema(self, coupling):
        if self.freeze and self.round: return
        super().update_schema(coupling)
