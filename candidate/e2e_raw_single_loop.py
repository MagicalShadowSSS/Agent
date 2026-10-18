"""Single entry point: raw records -> frozen views -> correspondence updates.

No previously generated input/embedding/ranking asset is required.  The raw
release's independent per-entity visual vectors are dataset inputs.  Only
the separate evaluate command opens alignment labels.  Intermediate files
created by this invocation are checkpointed for an exact resume.
"""
from __future__ import annotations
import os

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

import numpy as np
import scipy.sparse as sp

from unified_raw_qwen_candidates import (DATASETS, MMKG, dataset_dir, dump,
    label, parse_fact, read_jsonl, read_map, visual_indexes, write_jsonl)
from e2e_raw_fourview import original_relation_paths

MODEL = os.environ.get('SGMEA_EMBEDDING_MODEL', 'models/Qwen3-Embedding-8B')
VIEWS = ('name', 'attribute', 'relation', 'image')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_raw(root, dataset, recover_predicates=True):
    """Recover same-graph predicate IDs and build every source before labels."""
    started = time.time()
    maps = [read_map(dataset_dir(dataset) / f'ent_ids_{i}') for i in (1, 2)]
    predicate_votes = defaultdict(Counter)
    input_hashes = {}
    for side, mapping in enumerate(maps, 1):
        path = dataset_dir(dataset) / f'triples_{side}'
        input_hashes[str(path)] = sha(path)
        if not recover_predicates:
            continue
        endpoints = defaultdict(set)
        wanted = set(mapping.values())
        for path in original_relation_paths(dataset, side):
            if not path.exists():
                raise FileNotFoundError(path)
            input_hashes[str(path)] = sha(path)
            with path.open(encoding='utf-8', errors='ignore') as stream:
                for line in stream:
                    bits = line.split()
                    if len(bits) < 3:
                        continue
                    h, p, t = [x.strip('<>') for x in bits[:3]]
                    if h in wanted and t in wanted:
                        endpoints[h, t].add(p)
        path = dataset_dir(dataset) / f'triples_{side}'
        input_hashes[str(path)] = sha(path)
        with path.open() as stream:
            for line in stream:
                h, p, t = line.split()
                h, t = mapping.get(h, h), mapping.get(t, t)
                options = endpoints.get((h, t), ())
                if len(options) == 1:
                    predicate_votes[p][next(iter(options))] += 1
    id2predicate = {p: sorted(v, key=lambda x: (-v[x], x))[0]
                    for p, v in predicate_votes.items()}
    dump(root / 'predicate_audit' / f'{dataset}_id2predicate.json', id2predicate)
    dump(root / 'predicate_audit' / f'{dataset}_votes.json',
         {p: dict(sorted(v.items())) for p, v in predicate_votes.items()})
    rec = {u: {'uri': u, 'name': label(u), 'attribute': [], 'relation': []}
           for mapping in maps for u in mapping.values()}
    for filename in ('att_triples1', 'att_triples2'):
        with (dataset_dir(dataset) / filename).open(encoding='utf-8', errors='ignore') as stream:
            for line in stream:
                fact = parse_fact(line)
                if fact and fact[0] in rec:
                    rec[fact[0]]['attribute'].append(f'{label(fact[1]) or fact[1]}: {fact[2]}')
    source, target = [], []
    for mapping, role, dest in ((maps[0], 'query', source), (maps[1], 'target', target)):
        for u in mapping.values():
            row = rec[u]
            row['attribute'] = sorted(set(row['attribute']))
            row['role'] = role
            row['attribute_text'] = '\n'.join(row['attribute'])
            row['relation_text'] = ''
            dest.append(row)
    write_jsonl(root / 'raw' / 'inputs' / f'{dataset}.jsonl', source + target)
    for filename in ('ent_ids_1', 'ent_ids_2', 'att_triples1', 'att_triples2'):
        p = dataset_dir(dataset) / filename
        input_hashes[str(p)] = sha(p)
    image = (f'{dataset}_GA_id_img_feature_dict.pkl' if dataset in DATASETS[:3]
             else f'{dataset}_id_img_feature_dict.pkl')
    input_hashes[str(MMKG / 'pkls' / image)] = sha(MMKG / 'pkls' / image)
    dump(root / 'receipts' / f'{dataset}_raw.json', {
        'source_count': len(source), 'target_count': len(target),
        'raw_input_sha256': input_hashes, 'labels_read': False,
        'elapsed_seconds': time.time() - started})
    return source, target


def build_features(root, dataset, device, attribute_mode='all', calendar_kernel='distance',
                   graph_relations_only=False, literal_kernel=False, attribute_literal_only=False,
                   literal_values_only=False):
    import torch
    from sentence_transformers import SentenceTransformer
    from full_graph_frozen_control import prepare, encode
    graph = root / 'generated'
    for folder in ('graphs', 'embeddings'):
        (graph / folder).mkdir(parents=True, exist_ok=True)
    if graph_relations_only:
        raw = read_jsonl(root / 'raw' / 'inputs' / f'{dataset}.jsonl')
        ns = sum(row['role'] == 'query' for row in raw)
        records = [{'uri': row['uri'], 'side': int(i >= ns), 'name': row['name'],
                    'attribute_text': row['attribute_text'], 'relation_text': ''}
                   for i, row in enumerate(raw)]
        write_jsonl(graph / 'inputs' / f'{dataset}.jsonl', records)
        dump(graph / 'manifests' / f'{dataset}.json', {
             'source_count': ns, 'target_count': len(raw) - ns,
             'labels_read': False, 'relations': 'all original binary undirected edges'})
        for side, subset in ((1, raw[:ns]), (2, raw[ns:])):
            mapping = read_map(dataset_dir(dataset) / f'ent_ids_{side}')
            index = {row['uri']: i for i, row in enumerate(subset)}
            left, right = [], []
            with (dataset_dir(dataset) / f'triples_{side}').open() as stream:
                for line in stream:
                    h, _, t = line.split()
                    h, t = mapping.get(h, h), mapping.get(t, t)
                    if h in index and t in index and h != t:
                        left.extend((index[h], index[t]))
                        right.extend((index[t], index[h]))
            adjacency = sp.csr_matrix((np.ones(len(left), np.float32), (left, right)),
                                      shape=(len(subset), len(subset)))
            adjacency.data[:] = 1
            sp.save_npz(graph / 'graphs' / f'{dataset}_side{side}.npz', adjacency)
    else:
        records = prepare(root / 'raw', root / 'predicate_audit', graph, dataset, 'raw')
    if attribute_mode == 'typed':
        from classify_numeric_schemas import numeric_literal
        for row in records:
            row['attribute_text'] = '\n'.join(a for a in row['attribute_text'].split('\n')
                if ': ' in a and not numeric_literal(a.split(': ', 1)[1]))
        write_jsonl(graph / 'inputs' / f'{dataset}.jsonl', records)
    model = SentenceTransformer(MODEL, device=device, local_files_only=True).eval()
    model.requires_grad_(False)
    encoding_records = ([dict(row, attribute_text='') for row in records]
                        if attribute_literal_only else records)
    encode(model, encoding_records, graph, dataset)
    del model
    torch.cuda.empty_cache()
    # This writes a new kernel from this invocation's original numeric facts.
    # The kernel-only mode never reads labels or preexisting candidate assets.
    if literal_kernel:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('build_raw_literal_kernel.py')),
            '--raw-root', str(root / 'raw'), '--out', str(root / 'literal'),
            '--datasets', dataset] + (['--values-only'] if literal_values_only else []), check=True)
    elif calendar_kernel != 'none':
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('numeric_set_candidate_control.py')),
            '--root', str(root / 'raw'), '--out', str(root / 'numeric'),
            '--datasets', dataset, '--calendar-version', '2', '--calendar-kernel', calendar_kernel,
            '--kernels-only'], check=True)


def rankings(root, dataset, mode, matrix, sources, targets):
    """Full-catalog Top20 with a deterministic original-target-order tie break."""
    import torch
    rows = []
    # sort full rows rather than treating argpartition's unordered subset as a rank.
    for begin in range(0, len(sources), 128):
        values = matrix[begin:begin + 128].detach().cpu().numpy()
        for offset, values_row in enumerate(values):
            idx = np.argsort(-values_row, kind='stable')[:20]
            rows.append({'source': sources[begin + offset],
                         'rank': [targets[j] for j in idx],
                         'scores': [float(values_row[j]) for j in idx]})
    write_jsonl(root / 'rankings' / mode / f'{dataset}.jsonl', rows)


def soft_correspondence(score, temperature=.05, mode='balanced'):
    """Three alternative correspondence rules; no additional candidate stage."""
    import torch
    if mode == 'balanced':
        from structure_consistency import transport
        return transport(score, temperature, 'balanced')
    if mode == 'semi_equal_fixed10':
        kernel = torch.exp((score - score.max(1).values[:, None]) / temperature)
        v = torch.ones(score.shape[1], dtype=score.dtype, device=score.device)
        target_mass = score.shape[0] / score.shape[1]
        for _ in range(10):
            u = (kernel @ v).clamp_min(1e-30).reciprocal()
            v = (target_mass / (kernel.T @ u).clamp_min(1e-30)).sqrt()
        u = (kernel @ v).clamp_min(1e-30).reciprocal()
        return kernel * u[:, None] * v[None, :]
    logits = score / temperature
    row_log = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    if mode == 'row_softmax':
        return row_log.exp()
    if mode == 'reciprocal':
        col_log = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        return (.5 * (row_log + col_log)).exp()
    raise ValueError(('unknown correspondence rule', mode))


def rank20_calibration(score):
    """Map positive within-view affinities to a Top20-scale rank confidence.

    Competition ranks count strictly greater scores, so exact ties receive
    the same confidence and zero/missing affinities remain zero. Batching
    limits temporary storage; this is a replacement score calibration.
    """
    import torch
    for begin in range(0, score.shape[0], 128):
        rows = score[begin:begin + 128]
        values, columns = rows.sort(dim=1, descending=True)
        position = torch.arange(1, score.shape[1] + 1, device=score.device)[None, :]
        distinct = torch.ones_like(values, dtype=torch.bool)
        distinct[:, 1:] = values[:, 1:] != values[:, :-1]
        rank = torch.where(distinct, position, 0).cummax(dim=1).values
        confidence = torch.exp(-(rank.to(score.dtype) - 1.) / 20.)
        confidence.mul_(values > 0)
        rows.scatter_(1, columns, confidence)
    return score


def correct_cosine_hubs(score, source_has, target_has, k=20):
    """Subtract bidirectional local cosine means over available entities.

    The same candidate width sets the local mean size. Missing rows and
    columns do not enter either mean. This operates before clipping and
    existing row normalization, without changing the literal kernel.
    """
    import torch
    ns, nt = int(source_has.sum()), int(target_has.sum())
    if not ns or not nt:
        return score
    row_mean = torch.zeros(score.shape[0], dtype=score.dtype, device=score.device)
    column_mean = torch.zeros(score.shape[1], dtype=score.dtype, device=score.device)
    for begin in range(0, score.shape[0], 256):
        rows = score[begin:begin + 256, target_has]
        row_mean[begin:begin + 256] = rows.topk(min(k, nt), dim=1).values.mean(1)
    for begin in range(0, score.shape[1], 256):
        columns = score[source_has, begin:begin + 256]
        column_mean[begin:begin + 256] = columns.topk(min(k, ns), dim=0).values.mean(0)
    score.mul_(2.).sub_(row_mean[:, None]).sub_(column_mean[None, :])
    return score


def marginal_pearson_support(observed, shape):
    """Positive Pearson excess against actual observed mass marginals."""
    row_mass = np.bincount(observed.row, weights=observed.data, minlength=shape[0])
    column_mass = np.bincount(observed.col, weights=observed.data, minlength=shape[1])
    total = max(float(row_mass.sum()), 1e-12)
    expected = row_mass[observed.row] * column_mass[observed.col] / total
    return np.maximum(observed.data - expected, 0) / np.sqrt(np.maximum(expected, 1e-12))


def generate_rankings(root, dataset, update, steps, feature_root=None, readout='coupling', scope='all',
                      fusion='arithmetic', center_embeddings=False, support_scale=1.0,
                      support_norm='rowmax', support_k=20, include_numeric=True,
                      graph_relations_only=False, one_shot=False, attribute_kernel_root=None,
                      attribute_literal_only=False, attribute_set_kernel='forward',
                      correspondence='balanced', support_statistic='pearson', temperature_rule='fixed',
                      support_matrix='topk', view_calibration='rowmax', attribute_semantic=True,
                      cosine_rule='raw', relation_view_root=None):
    import torch
    from sentence_transformers import SentenceTransformer
    started = time.time()
    feature_root = root if feature_root is None else Path(feature_root)
    graph = feature_root / 'generated'
    records = read_jsonl(graph / 'inputs' / f'{dataset}.jsonl')
    ns = json.loads((graph / 'manifests' / f'{dataset}.json').read_text())['source_count']
    sources = [r['uri'] for r in records[:ns]]
    targets = [r['uri'] for r in records[ns:]]
    sv, tv = visual_indexes(dataset)
    evidence, has = {}, {}

    def calibrate(score, sh, th):
        score.clamp_min_(0)
        score.mul_(sh[:, None]); score.mul_(th[None, :])
        score.div_(score.max(1).values[:, None].clamp_min(1e-12))
        return score

    views = ('name', 'attribute', 'image') if graph_relations_only else VIEWS
    for view in views:
        if view == 'attribute' and not attribute_semantic:
            if not include_numeric:
                raise ValueError('Numeric-only attributes require the original value kernel')
            p = Path(attribute_kernel_root) if attribute_kernel_root else feature_root / 'numeric' / 'kernels'
            availability = np.load(p / f'{dataset}_availability.npz')
            sh = torch.tensor(availability['source_has'], device='cuda')
            th = torch.tensor(availability['target_has'], device='cuda')
            score = calibrate(torch.tensor(np.load(p / f'{dataset}_{attribute_set_kernel}.npy'), device='cuda'), sh, th)
            if view_calibration == 'rank20':
                rank20_calibration(score)
            elif view_calibration != 'rowmax':
                raise ValueError(('Unknown view calibration', view_calibration))
            evidence[view], has[view] = score, (sh, th)
            print(dataset, 'view prepared attribute original values only', flush=True)
            continue
        if view == 'attribute' and attribute_literal_only:
            p = Path(attribute_kernel_root)
            availability = np.load(p / f'{dataset}_availability.npz')
            sh = torch.tensor(availability['source_has'], device='cuda')
            th = torch.tensor(availability['target_has'], device='cuda')
            score = calibrate(torch.tensor(np.load(p / f'{dataset}_forward.npy'), device='cuda'), sh, th)
            evidence[view], has[view] = score, (sh, th)
            print(dataset, 'view prepared attribute literal only', flush=True)
            continue
        if view == 'image':
            dim = len(next(iter(sv.values())))
            x = np.stack([(sv if i < ns else tv).get(row['uri'], np.zeros(dim, np.float32))
                          for i, row in enumerate(records)])
        else:
            embedding_root = (Path(relation_view_root) / 'generated'
                              if view == 'relation' and relation_view_root else graph)
            x = np.load(embedding_root / 'embeddings' / f'{dataset}_{view}.npy')
        available = np.linalg.norm(x, axis=1) > 1e-8
        if center_embeddings:
            x = x.copy()
            if available.any():
                x[available] -= x[available].mean(axis=0)
                x[available] /= np.maximum(np.linalg.norm(x[available], axis=1, keepdims=True), 1e-12)
        sh = torch.tensor(available[:ns], device='cuda')
        th = torch.tensor(available[ns:], device='cuda')
        left, right = torch.tensor(x[:ns], device='cuda'), torch.tensor(x[ns:], device='cuda')
        score = left @ right.T
        if cosine_rule == 'local20':
            correct_cosine_hubs(score, sh, th, 20)
        elif cosine_rule != 'raw':
            raise ValueError(('Unknown cosine rule', cosine_rule))
        score = calibrate(score, sh, th)
        del left, right, x
        if view == 'attribute' and include_numeric:
            p = Path(attribute_kernel_root) if attribute_kernel_root else feature_root / 'numeric' / 'kernels'
            availability = np.load(p / f'{dataset}_availability.npz')
            n_sh = torch.tensor(availability['source_has'], device='cuda')
            n_th = torch.tensor(availability['target_has'], device='cuda')
            numeric = calibrate(torch.tensor(np.load(p / f'{dataset}_{attribute_set_kernel}.npy'), device='cuda'), n_sh, n_th)
            score.add_(numeric).div_((sh.float() + n_sh.float())[:, None].clamp_min(1))
            sh, th = sh | n_sh, th | n_th
            del numeric
        if view_calibration == 'rank20':
            rank20_calibration(score)
        elif view_calibration != 'rowmax':
            raise ValueError(('Unknown view calibration', view_calibration))
        evidence[view], has[view] = score, (sh, th)
        print(dataset, 'view prepared', view, flush=True)
    if scope == 'ordinary':
        class PlainOperator:
            def __init__(self):
                self.ordinary = []
                for side in (1, 2):
                    g = sp.load_npz(graph / 'graphs' / f'{dataset}_side{side}.npz').tocsr()
                    g.setdiag(0); g.eliminate_zeros(); g.data[:] = 1
                    self.ordinary.append(g)
                self.shape = tuple(g.shape[0] for g in self.ordinary)
                self.degrees = [np.asarray(g.sum(1)).reshape(-1) for g in self.ordinary]
                if support_statistic == 'normalized_adjacency':
                    normalized = []
                    for g, degree in zip(self.ordinary, self.degrees):
                        scale = 1. / np.sqrt(np.maximum(degree, 1.))
                        normalized.append((sp.diags(scale) @ g @ sp.diags(scale)).tocsr())
                    self.propagation_graphs = normalized
                elif support_statistic in ('adamic_pearson', 'marginal_log_pearson'):
                    self.propagation_graphs = [
                        (g @ sp.diags(1. / np.sqrt(np.log1p(np.maximum(degree, 1.))))).tocsr()
                        for g, degree in zip(self.ordinary, self.degrees)]
                else:
                    self.propagation_graphs = self.ordinary
                self.support_degrees = [np.asarray(g.sum(1)).reshape(-1) for g in self.propagation_graphs]
                if support_matrix == 'full':
                    self.cuda_graphs = [torch.sparse_csr_tensor(
                        torch.tensor(g.indptr, dtype=torch.int64, device='cuda'),
                        torch.tensor(g.indices, dtype=torch.int64, device='cuda'),
                        torch.tensor(g.data, dtype=torch.float32, device='cuda'),
                        size=g.shape) for g in self.propagation_graphs]
                    da, db = (self.support_degrees if support_statistic in ('adamic_pearson', 'marginal_log_pearson')
                              else self.degrees)
                    self.cuda_degrees = [torch.tensor(d, dtype=torch.float32, device='cuda')
                                         for d in (da, db)]
            def propagate(self, coupling, k=20):
                if support_matrix == 'full':
                    # No intermediate candidate truncation: all row-normalized
                    # soft mass contributes to the same neighborhood kernel.
                    probabilities = coupling / coupling.sum(1, keepdim=True).clamp_min(1e-12)
                    a, b = self.cuda_graphs
                    left = torch.sparse.mm(a, probabilities)
                    del probabilities
                    observed = torch.sparse.mm(b, left.T.contiguous()).T.contiguous()
                    del left
                    da, db = self.cuda_degrees
                    if support_statistic == 'normalized_adjacency':
                        support = observed
                    elif support_statistic == 'degree_cosine':
                        support = observed / (da[:, None] * db[None, :]).clamp_min(1.).sqrt()
                    else:
                        if support_statistic in ('marginal_pearson', 'marginal_log_pearson'):
                            expected = (observed.sum(1, keepdim=True) * observed.sum(0, keepdim=True)
                                        / observed.sum().clamp_min(1e-12))
                        else:
                            expected = da[:, None] * db[None, :] / self.shape[1]
                        denominator = (expected.clamp_min(1e-12) if support_statistic == 'excess_ratio'
                                       else expected.clamp_min(1e-12).sqrt())
                        support = (observed - expected).clamp_min_(0).div_(denominator)
                    if support_norm == 'saturating':
                        support = support / (1. + support)
                    else:
                        support.div_(support.max(1, keepdim=True).values.clamp_min(1e-12))
                        if support_norm == 'sqrt':
                            support.sqrt_()
                    return support
                values, columns = coupling.topk(min(k, coupling.shape[1]), dim=1)
                values /= values.sum(1, keepdim=True).clamp_min(1e-12)
                values, columns = values.cpu().numpy(), columns.cpu().numpy()
                sparse = sp.csr_matrix((values.reshape(-1), columns.reshape(-1),
                    np.arange(0, values.size + 1, values.shape[1])), shape=self.shape)
                a, b = self.propagation_graphs
                observed = (a @ sparse @ b.T).tocsr()
                observed.sum_duplicates()
                observed = observed.tocoo()
                da, db = (self.support_degrees if support_statistic in ('adamic_pearson', 'marginal_log_pearson')
                          else self.degrees)
                support = np.zeros(self.shape, np.float32)
                if support_statistic == 'normalized_adjacency':
                    support[observed.row, observed.col] = observed.data
                elif support_statistic == 'degree_cosine':
                    denominator = np.sqrt(np.maximum(da[observed.row] * db[observed.col], 1.))
                    support[observed.row, observed.col] = observed.data / denominator
                elif support_statistic in ('marginal_pearson', 'marginal_log_pearson'):
                    support[observed.row, observed.col] = marginal_pearson_support(observed, self.shape)
                else:
                    expected = da[observed.row] * db[observed.col] / self.shape[1]
                    denominator = (np.maximum(expected, 1e-12) if support_statistic == 'excess_ratio'
                                   else np.sqrt(np.maximum(expected, 1e-12)))
                    support[observed.row, observed.col] = np.maximum(observed.data - expected, 0) / denominator
                if support_norm == 'saturating':
                    support = support / (1.0 + support)
                elif support_norm == 'sqrt':
                    support /= np.maximum(support.max(1, keepdims=True), 1e-12)
                    support = np.sqrt(support)
                else:
                    support /= np.maximum(support.max(1, keepdims=True), 1e-12)
                return torch.tensor(support, device=coupling.device)
        operator = PlainOperator()
    else:
        from degree_corrected_candidates import DegreeCorrectedOperator
        model = SentenceTransformer(MODEL, device='cuda', local_files_only=True).eval()
        model.requires_grad_(False)
        operator = DegreeCorrectedOperator(dataset, graph, feature_root / 'predicate_audit', None,
            root / 'operator' / dataset, model, directional=True, infer_schema=True,
            normalize_neighbors=False, topk=3, audit_rounds={1, 2},
            graph_root=graph, dataset=dataset)
        del model
    operator.statistic = 'pearson'
    operator.deviation = lambda observed, expected: np.maximum(observed - expected, 0) / np.sqrt(np.maximum(expected, 1e-12))
    operator.support_scope = scope

    def project(matrix):
        if temperature_rule == 'fixed':
            return soft_correspondence(matrix, .05, correspondence), .05
        if temperature_rule != 'top20_ratio' or correspondence != 'balanced':
            raise ValueError('Top20-derived temperature requires balanced correspondence')
        k = min(20, matrix.shape[1])
        if k < 2:
            raise ValueError('Top20-derived temperature requires at least two targets')
        leading = matrix.topk(k, dim=1).values
        temperature = ((leading[:, :1] - leading[:, -1:]) / np.log(k)).clamp_min(1e-6)
        # Before column competition, the first/20th kernel ratio is 20.
        # A floating-point floor keeps every target column positive.
        scaled = ((matrix - leading[:, :1]) / temperature).clamp_min(np.log(1e-30))
        return soft_correspondence(scaled, 1., 'balanced'), temperature

    for mode in ('all', 'no_image'):
        fields = list(views if mode == 'all' else tuple(v for v in views if v != 'image'))
        available_count = sum(has[v][0].float() for v in fields)
        count = available_count.clamp_min(1)
        if fusion == 'geometric':
            prior = sum(evidence[v].clamp_min(.1).log() * has[v][0][:, None] for v in fields)
            prior.div_(count[:, None]).exp_()
        elif fusion == 'harmonic':
            denominator = sum(evidence[v].clamp_min(1e-12).reciprocal() * has[v][0][:, None]
                              for v in fields)
            prior = count[:, None] / denominator.clamp_min(1e-12)
            prior.mul_((available_count > 0)[:, None])
        elif fusion == 'root_mean':
            prior = sum(evidence[v].clamp_min(0).sqrt() for v in fields) / count[:, None]
            prior.square_()
        else:
            prior = sum(evidence[v] for v in fields) / count[:, None]
        rankings(root, dataset, mode + '_initial', prior, sources, targets)
        direct_update = one_shot or update == 'neighbor_mean'
        coupling, temperature = (prior, .05) if direct_update else project(prior)
        update_steps = 1 if one_shot else steps
        for step in range(1, update_steps + 1):
            support = operator.propagate(coupling, support_k)
            if not direct_update:
                support.div_(count[:, None])
            support.mul_(support_scale)
            if one_shot:
                score = (prior * count[:, None] + support) / (count[:, None] + 1)
            elif update == 'neighbor_mean':
                score = .5 * (prior + support)
            elif update == 'proximal':
                score = temperature * coupling.clamp_min(1e-30).log() + support
            elif update == 'residual_quarter':
                score = .25 * prior + temperature * coupling.clamp_min(1e-30).log() + support
            elif update == 'residual':
                score = .5 * (prior + temperature * coupling.clamp_min(1e-30).log()) + support
            else:
                score = prior + support
            coupling, temperature = (score, .05) if direct_update else project(score)
            del support
            print(dataset, mode, 'update', step, flush=True)
            if step < update_steps:
                del score
        if readout == 'compatibility':
            final_score = score
        elif readout == 'prior_compatibility':
            final_score = 0.5 * (prior + score)
        else:
            final_score = coupling
        rankings(root, dataset, mode, final_score, sources, targets)
        del prior, coupling, score
    dump(root / 'done' / f'{dataset}.json', {'complete': True, 'rankings_persisted_before_labels': True,
        'sources': ns, 'targets': len(targets), 'update': update, 'steps': steps, 'readout': readout, 'scope': scope,
        'temperature_rule': temperature_rule,
        'support_matrix': support_matrix,
        'view_calibration': view_calibration,
        'attribute_semantic': attribute_semantic,
        'cosine_rule': cosine_rule,
        'seconds': time.time() - started, 'cuda_max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})


def evaluate(root, dataset, labels_root):
    labels = read_jsonl(labels_root / 'labels' / f'{dataset}.jsonl')
    result = {}
    positions = {}
    for mode in ('all_initial', 'all', 'no_image_initial', 'no_image'):
        rows = {r['source']: r['rank'] for r in read_jsonl(root / 'rankings' / mode / f'{dataset}.jsonl')}
        missing = [r['source'] for r in labels if r['source'] not in rows]
        if missing:
            raise ValueError((dataset, mode, 'missing labelled queries', len(missing)))
        pos = np.asarray([rows[r['source']].index(r['gold']) + 1 if r['gold'] in rows[r['source']] else 999
                          for r in labels])
        positions[mode] = pos
        result[mode] = {'queries': len(pos), 'H1': float(np.mean(pos == 1) * 100),
            'H10': float(np.mean(pos <= 10) * 100), 'H20': float(np.mean(pos <= 20) * 100),
            'gap_H10_H1': float(np.mean((pos > 1) & (pos <= 10)) * 100)}
    for k in (1, 10, 20):
        b, f = positions['all_initial'] <= k, positions['all'] <= k
        result[f'paired_H{k}'] = {'fixes': int((f & ~b).sum()), 'harms': int((b & ~f).sum()), 'net': int(f.sum() - b.sum())}
    dump(root / 'evaluation' / f'{dataset}.json', result)
    print(dataset, json.dumps(result), flush=True)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--stage', choices=('generate', 'verify', 'evaluate'), required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--update', choices=('proximal', 'residual', 'residual_quarter', 'anchored', 'neighbor_mean'), default='residual')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--readout', choices=('coupling', 'compatibility', 'prior_compatibility'), default='coupling')
    p.add_argument('--scope', choices=('all', 'ordinary'), default='all')
    p.add_argument('--feature-root', help='Verification only: reuse features from a completed raw-generation experiment')
    p.add_argument('--attribute-mode', choices=('all', 'typed'), default='all')
    p.add_argument('--calendar-kernel', choices=('distance', 'overlap', 'none'), default='distance')
    p.add_argument('--fusion', choices=('arithmetic', 'geometric', 'harmonic'), default='arithmetic')
    p.add_argument('--correspondence', choices=('balanced', 'row_softmax', 'reciprocal'), default='balanced')
    p.add_argument('--center-embeddings', action='store_true')
    p.add_argument('--support-scale', type=float, default=1.0)
    p.add_argument('--support-norm', choices=('rowmax', 'sqrt', 'saturating'), default='rowmax')
    p.add_argument('--support-k', type=int, default=20)
    p.add_argument('--disable-numeric-kernel', action='store_true',
                   help='Use the raw attribute text embedding without the separate value-matching kernel')
    p.add_argument('--graph-relations-only', action='store_true',
                   help='Represent relations only through the raw graph-neighbor support, not a text view')
    p.add_argument('--one-shot', action='store_true',
                   help='Use the initial top-k correspondence once and average graph support as an equal view')
    p.add_argument('--literal-attribute-kernel', action='store_true',
                   help='Replace numeric/date-specific matching with a uniform raw attribute character kernel')
    p.add_argument('--literal-kernel-root', help='Verify only: kernels from build_raw_literal_kernel.py')
    p.add_argument('--attribute-literal-only', action='store_true',
                   help='Represent attributes by the character kernel alone, without an attribute text encoder')
    p.add_argument('--literal-values-only', action='store_true',
                   help='Use only original literal values in the character attribute document')
    p.add_argument('--attribute-set-kernel', choices=('forward', 'symmetric'), default='forward',
                   help='Numeric/date set affinity: source best-match mean, or geometric mean of both directions')
    p.add_argument('--labels-root', default='output/unified_raw_qwen_1000_20261006_repaired_duplicates')
    p.add_argument('--support-statistic', choices=('pearson', 'degree_cosine', 'normalized_adjacency',
                   'adamic_pearson', 'marginal_pearson', 'marginal_log_pearson', 'excess_ratio'),
                   default='pearson')
    p.add_argument('--relation-view-root', help='Verification-only root containing freshly encoded relation vectors')
    a = p.parse_args(); root = Path(a.root)
    if a.one_shot and (a.scope != 'ordinary' or a.fusion != 'arithmetic' or a.steps != 1
                       or a.readout != 'compatibility'):
        p.error('--one-shot requires ordinary scope, arithmetic fusion, one step and compatibility readout')
    if a.disable_numeric_kernel and a.attribute_mode != 'all':
        p.error('Disabling the value kernel requires the complete raw attribute text')
    if a.update == 'neighbor_mean' and (a.scope != 'ordinary' or a.readout != 'compatibility'
                                       or a.one_shot):
        p.error('neighbor_mean requires ordinary scope, compatibility readout and no --one-shot')
    if a.literal_attribute_kernel and (a.attribute_mode != 'all' or a.calendar_kernel != 'none'
                                       or a.disable_numeric_kernel):
        p.error('Literal attributes require all raw text, calendar-kernel none and an enabled attribute kernel')
    if a.literal_kernel_root and (a.stage != 'verify' or not a.literal_attribute_kernel):
        p.error('literal-kernel-root is only allowed in a literal-attribute verification')
    if a.stage == 'verify' and a.literal_attribute_kernel and not a.literal_kernel_root:
        p.error('Literal verification requires --literal-kernel-root')
    if a.attribute_literal_only and not a.literal_attribute_kernel:
        p.error('attribute-literal-only requires --literal-attribute-kernel')
    if a.literal_values_only and not a.literal_attribute_kernel:
        p.error('literal-values-only requires --literal-attribute-kernel')
    if a.attribute_set_kernel != 'forward' and (a.disable_numeric_kernel or a.literal_attribute_kernel):
        p.error('Symmetric set affinity requires the numeric/date attribute kernel')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.stage == 'evaluate':
        for ds in a.datasets: evaluate(root, ds, Path(a.labels_root))
        return
    if a.stage == 'verify' and a.feature_root:
        donor = json.loads((Path(a.feature_root) / 'protocol.json').read_text())
        if donor.get('attribute_mode', a.attribute_mode) != a.attribute_mode:
            raise ValueError('Verification attribute mode differs from the donor representation')
        if not a.literal_attribute_kernel and not a.disable_numeric_kernel:
            if donor.get('calendar_kernel', a.calendar_kernel) != a.calendar_kernel:
                raise ValueError('Verification calendar kernel differs from the donor representation')
        if a.literal_attribute_kernel:
            for ds in a.datasets:
                kernel_protocol = json.loads((Path(a.literal_kernel_root).parent / 'done' / f'{ds}.json').read_text())
                if kernel_protocol.get('values_only', False) != a.literal_values_only:
                    raise ValueError('Literal kernel document differs from the declared representation')
    protocol = {'method': ('raw views with repeated equal neighbor support'
                          if a.update == 'neighbor_mean' else
                          'raw four-view evidence with one Pearson correspondence loop'),
        'top_k': 20, 'downstream_top_k': 10, 'steps': a.steps, 'update': a.update, 'readout': a.readout, 'scope': a.scope,
        'attribute_mode': a.attribute_mode, 'calendar_kernel': a.calendar_kernel,
        'fusion': a.fusion, 'geometric_floor': .1 if a.fusion == 'geometric' else None,
        'correspondence': a.correspondence,
        'center_embeddings': a.center_embeddings,
        'support_scale': a.support_scale,
        'support_norm': a.support_norm,
        'support_k': a.support_k,
        'numeric_attribute_kernel': not a.disable_numeric_kernel and not a.literal_attribute_kernel,
        'attribute_set_kernel': a.attribute_set_kernel,
        'literal_attribute_kernel': a.literal_attribute_kernel,
        'attribute_text_encoder': not a.attribute_literal_only,
        'literal_parameters': {'character_width': 3, 'min_df': 2} if a.literal_attribute_kernel else None,
        'literal_values_only': a.literal_values_only,
        'verification_literal_kernel_root': a.literal_kernel_root,
        'graph_relations_only': a.graph_relations_only,
        'one_shot': a.one_shot,
        'temperature': None if a.one_shot or a.update == 'neighbor_mean' else .05,
        'schema_top_k': None if a.scope == 'ordinary' else 3, 'encoder': MODEL,
        'training': False, 'EA_checkpoint': None, 'labels_read_by_generation': False,
        'image_gate': False, 'external_name_lookup': False, 'ILL_name_repair': False,
        'generation_requires_old_candidate_assets': False, 'verification_reuses_features': a.stage == 'verify',
        'verification_feature_root': a.feature_root if a.stage == 'verify' else None,
        'support_statistic': a.support_statistic,
        'relation_view_root': a.relation_view_root,
        'code_sha256': sha(__file__)}
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != protocol:
        raise ValueError('Refuse to resume with a different protocol')
    dump(root / 'protocol.json', protocol)
    snapshots = {}
    for name in ('e2e_raw_single_loop.py', 'e2e_raw_fourview.py',
                 'unified_raw_qwen_candidates.py', 'full_graph_frozen_control.py',
                 'numeric_set_candidate_control.py', 'classify_numeric_schemas.py',
                 'degree_corrected_candidates.py', 'structure_consistency.py',
                 'typed_relation_consistency.py', 'prepare_semantic_relation_control.py'):
        path = Path(__file__).with_name(name)
        destination = root / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
        snapshots[name] = sha(path)
    dump(root / 'dependency_sha256.json', snapshots)
    for ds in a.datasets:
        if (root / 'done' / f'{ds}.json').exists():
            print(ds, 'already complete, preserving results', flush=True); continue
        if a.stage == 'verify':
            if not a.feature_root:
                raise ValueError('verify requires --feature-root; generate never needs it')
            generate_rankings(root, ds, a.update, a.steps, a.feature_root, a.readout, a.scope,
                              a.fusion, a.center_embeddings, a.support_scale, a.support_norm,
                              a.support_k, include_numeric=not a.disable_numeric_kernel,
                              graph_relations_only=a.graph_relations_only, one_shot=a.one_shot,
                              attribute_kernel_root=a.literal_kernel_root,
                              attribute_literal_only=a.attribute_literal_only,
                              attribute_set_kernel=a.attribute_set_kernel,
                              correspondence=a.correspondence, support_statistic=a.support_statistic,
                              relation_view_root=a.relation_view_root)
        else:
            if a.feature_root:
                raise ValueError('generate refuses preexisting features; use verify for diagnostic reuse')
            prepare_raw(root, ds, recover_predicates=not a.graph_relations_only)
            build_features(root, ds, 'cuda:0', a.attribute_mode,
                           'none' if a.disable_numeric_kernel else a.calendar_kernel,
                           graph_relations_only=a.graph_relations_only,
                           literal_kernel=a.literal_attribute_kernel,
                           attribute_literal_only=a.attribute_literal_only,
                           literal_values_only=a.literal_values_only)
            generate_rankings(root, ds, a.update, a.steps, readout=a.readout, scope=a.scope,
                              fusion=a.fusion, center_embeddings=a.center_embeddings,
                              support_scale=a.support_scale, support_norm=a.support_norm,
                              support_k=a.support_k, include_numeric=not a.disable_numeric_kernel,
                              graph_relations_only=a.graph_relations_only, one_shot=a.one_shot,
                              attribute_kernel_root=root / 'literal' / 'kernels' if a.literal_attribute_kernel else None,
                              attribute_literal_only=a.attribute_literal_only,
                              attribute_set_kernel=a.attribute_set_kernel,
                              correspondence=a.correspondence)


if __name__ == '__main__': main()
