import os
"""One relation-consistent transport loop, without a second GW optimizer.

Frozen equal-view evidence initializes the coupling. Each iteration uses
the same directed typed-neighbor operator and balanced normalization.
The proximal variant carries the previous coupling as its KL reference;
the anchored control restarts from the frozen semantic evidence each time.
No dataset-specific parameter branches, learned weights, or labels enter it.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from collective_candidate_control import fuse_evidence
from structure_consistency import StructureConsistencyRefiner, transport
from typed_relation_consistency import TypedRelationConsistency
from unified_raw_qwen_candidates import DATASETS, VIEWS, dump, read_jsonl, visual_indexes
from write_unified_final import metric, write_rankings


def prepare_views(args, dataset):
    import torch
    root = Path(args.graph_root)
    manifest = json.loads((root / 'manifests' / f'{dataset}.json').read_text())
    rows = read_jsonl(root / 'inputs' / f'{dataset}.jsonl')
    ns = manifest['source_count']
    si = {row['uri']: i for i, row in enumerate(rows[:ns])}
    ti = {row['uri']: i for i, row in enumerate(rows[ns:])}
    qi = np.array([si[u] for u in manifest['queries']])
    target_indices = np.array([ti[u] for u in manifest['targets']])
    evidence, has = {}, {}
    sv, tv = visual_indexes(dataset)

    def calibrate(matrix, sh, th):
        matrix = matrix.clamp_min(0) * sh[:, None] * th[None, :]
        return matrix / matrix.max(1).values[:, None].clamp_min(1e-12)

    for field in VIEWS:
        if field == 'attribute' and args.fact_kernel_root:
            kernels = Path(args.fact_kernel_root) / 'kernels'
            availability = np.load(kernels / f'{dataset}_availability.npz')
            sh = torch.tensor(availability['source_has'], device='cuda')
            th = torch.tensor(availability['target_has'], device='cuda')
            matrix = torch.tensor(np.load(kernels / f'{dataset}_symmetric.npy'), device='cuda')
            evidence[field] = calibrate(matrix, sh, th)
            has[field] = (sh, th)
            del matrix
            continue
        if field == 'image':
            dim = len(next(iter(sv.values())))
            vectors = np.stack([(sv if i < ns else tv).get(row['uri'], np.zeros(dim, np.float32))
                                for i, row in enumerate(rows)])
        else:
            folder = Path(args.literal_root) if field == 'attribute' else root
            vectors = np.load(folder / 'embeddings' / f'{dataset}_{field}.npy')
        if getattr(args, 'embedding_postprocess', 'none') != 'none':
            from embedding_common_component import remove_common_component
            transformed, audit, artifacts = remove_common_component(
                torch.tensor(vectors, device='cuda'), args.embedding_postprocess)
            vectors = transformed.cpu().numpy()
            folder = Path(args.out) / 'embedding_postprocess'
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(artifacts, folder / f'{field}.pt')
            dump(folder / f'{field}.json', audit)
            del transformed
        available = np.linalg.norm(vectors, axis=1) > 1e-8
        if getattr(args, 'center_embeddings', False) and available.any():
            # One unlabeled pooled centering rule, applied to every vector view.
            # Missing rows remain exactly zero; this does not invent names.
            vectors = vectors.copy()
            vectors[available] -= vectors[available].mean(axis=0, keepdims=True)
            vectors[available] /= np.maximum(
                np.linalg.norm(vectors[available], axis=1, keepdims=True), 1e-12)
        sh = torch.tensor(available[:ns], device='cuda')
        th = torch.tensor(available[ns:], device='cuda')
        left = torch.tensor(vectors[:ns], device='cuda')
        right = torch.tensor(vectors[ns:], device='cuda')
        semantic = calibrate(left @ right.T, sh, th)
        del left, right, vectors
        if field == 'attribute':
            kernels = Path(args.kernel_root) / 'kernels'
            availability = np.load(kernels / f'{dataset}_availability.npz')
            n_sh = torch.tensor(availability['source_has'], device='cuda')
            n_th = torch.tensor(availability['target_has'], device='cuda')
            numeric = torch.tensor(np.load(kernels / f'{dataset}_{args.numeric_kernel}.npy'), device='cuda')
            structured = calibrate(numeric, n_sh, n_th)
            if getattr(args, 'attribute_missing_normalization', 'source') == 'paired':
                semantic.add_(structured)
                for start in range(0, len(semantic), 256):
                    end = min(start+256, len(semantic))
                    count = ((sh[start:end, None] & th[None]).float()
                             + (n_sh[start:end, None] & n_th[None]).float())
                    semantic[start:end] /= count.clamp_min(1)
            else:
                semantic = (semantic + structured) / (sh.float() + n_sh.float())[:, None].clamp_min(1)
            sh, th = sh | n_sh, th | n_th
            del structured, numeric
        evidence[field], has[field] = semantic, (sh, th)
    return evidence, has, qi, target_indices, manifest


def run(args, dataset):
    import torch
    from sentence_transformers import SentenceTransformer
    out = Path(args.out)
    started = time.perf_counter()
    evidence, has, qi, ti, manifest = prepare_views(args, dataset)
    dump(out / 'input_receipts' / f'{dataset}.json', {
        'queries': manifest['queries'], 'targets': manifest['targets'],
        'source_count': manifest['source_count'], 'target_count': manifest['target_count'],
        'training': False, 'labels_read': False,
    })
    model = SentenceTransformer(os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE'),
                                device='cuda', local_files_only=True).eval()
    model.requires_grad_(False)
    ranked = {}
    timings = {'prepare_seconds': time.perf_counter() - started, 'modes': {}}
    for mode in args.modes:
        mode_started = time.perf_counter()
        fields = list(VIEWS if mode == 'all' else VIEWS[:-1])
        counts = sum(has[f][0].float() for f in fields).clamp_min(1)
        prior = fuse_evidence(evidence, has, fields, 'log_mean', 'source')
        initial_rows = write_rankings(Path(args.parent), dataset, prior[qi][:, ti].cpu().numpy(),
                                      out / 'rankings' / f'{mode}_initial' / dataset)
        reference = Path('output/candidate_component_controls_20261007/gw_only/rankings') / f'hybrid_{mode}_step0_none' / dataset / 'rankings.jsonl'
        if reference.exists() and args.numeric_kernel == 'forward' and not args.fact_kernel_root:
            old = read_jsonl(reference)
            assert [r['source'] for r in old] == [r['source'] for r in initial_rows]
            changes = sum(a['rank'] != b['rank'] for a, b in zip(old, initial_rows))
            maximum_error = max(abs(x - y) for a, b in zip(old, initial_rows) for x, y in zip(a['scores'], b['scores']))
            dump(out / 'input_equivalence' / f'{dataset}_{mode}.json', {
                'ranking_changes': changes, 'max_score_error': maximum_error,
                'reference': str(reference), 'labels_read': False,
            })
            if changes or maximum_error > 1e-6:
                raise ValueError('Module controls require identical initial evidence')
        ranked[f'{mode}_initial'] = initial_rows
        temperature = args.temperature
        if args.temperature_scale == 'row_std':
            temperature = args.temperature * prior.std(1, unbiased=False).clamp_min(1e-6)[:, None]
            dump(out / 'scales' / f'{dataset}_{mode}.json', {
                'rule': 'fixed multiplier times source-row standard deviation of frozen four-view evidence',
                'multiplier': args.temperature, 'minimum': float(temperature.min()),
                'median': float(temperature.median()), 'maximum': float(temperature.max()),
                'labels_read': False,
            })
        refiner = None
        if args.update != 'typed_gw':
            base_weight = args.base_weight
            proximal_weight = args.proximal_weight
            support_weight = args.support_weight
            if args.joint_lambda is not None:
                base_weight = 1.0 - args.joint_lambda
                proximal_weight = 1.0
                support_weight = args.joint_lambda
            refiner = StructureConsistencyRefiner(
                dataset, args.graph_root, args.typed_audit, model,
                out / mode, infer_schema=not args.static_schema,
                static_schema=args.static_schema, use_context=args.context,
                update=args.update, audit_rounds=set(args.steps),
                base_weight=base_weight, proximal_weight=proximal_weight,
                support_weight=support_weight,
                support_calibration=args.support_calibration,
                temperature=temperature, transport_mode=args.transport_mode)
            coupling = None
        else:
            coupling = transport(prior, temperature, args.transport_mode)
            typed = TypedRelationConsistency(
                dataset, Path(args.graph_root), Path(args.typed_audit), None,
                out / mode, model, directional=True,
                infer_schema=not args.static_schema, audit_rounds=set(args.steps),
                normalize_neighbors=False)
        graph_fraction = None
        if args.update == 'typed_gw':
            import scipy.sparse as sp
            densities = []
            for side, count in ((1, manifest['source_count']), (2, manifest['target_count'])):
                graph = sp.load_npz(Path(args.graph_root) / 'graphs' / f'{dataset}_side{side}.npz').tocsr()
                graph.setdiag(0); graph.eliminate_zeros()
                densities.append(graph.nnz / count**2)
            graph_fraction = sum(densities) / 2
            coupling /= manifest['target_count']
        init_seconds = time.perf_counter() - mode_started
        loop_seconds = 0.0
        snapshots = {}
        module_results = None
        if refiner is not None:
            module_results = refiner.iter_refinements(
                prior, counts, steps=max(args.steps),
                checkpoints=range(1, max(args.steps) + 1))
        for step in range(1, max(args.steps) + 1):
            iteration_started = time.perf_counter()
            if refiner is not None:
                current_step, coupling, structural, elapsed = next(module_results)
                assert current_step == step
                loop_seconds = elapsed
            else:
                graph = typed.propagate(coupling, preserve_mass=True)
            if args.update == 'typed_gw':
                structural = graph_fraction * prior + 2 * (1 - graph_fraction) * graph
                kernel = coupling * torch.exp((structural - structural.max(1).values[:, None]) / .01)
                for _ in range(10):
                    kernel /= kernel.sum(1, keepdim=True).clamp_min(1e-30) * manifest['source_count']
                    kernel /= kernel.sum(0, keepdim=True).clamp_min(1e-30) * manifest['target_count']
                coupling = kernel
                del graph
            torch.cuda.synchronize()
            if refiner is None:
                loop_seconds += time.perf_counter() - iteration_started
            if step in args.steps:
                snapshots[str(step)] = {'loop_seconds': loop_seconds, 'init_seconds': init_seconds,
                                       'includes_schema_audit_io': True}
                # The coupling is the preregistered ranking; local scores are
                # diagnostic, with one common variant across all datasets.
                local = structural if args.update == 'typed_gw' else prior + structural
                for view in ('coupling', 'local', 'local_csls10'):
                    if view == 'coupling':
                        matrix = coupling
                    elif view == 'local':
                        matrix = local
                    else:
                        matrix = 2 * local - local.topk(10, dim=1).values.mean(1)[:, None] - local.topk(10, dim=0).values.mean(0)[None, :]
                    variant = f'{mode}_step{step}_{view}'
                    selected = matrix[qi][:, ti].cpu().numpy()
                    ranked[variant] = write_rankings(Path(args.parent), dataset, selected,
                                                     out / 'rankings' / variant / dataset)
                    del matrix, selected
                del local
                dump(out / 'status' / f'{dataset}.json', {
                    'mode': mode, 'step': step, 'seconds': time.perf_counter() - started,
                })
                print(dataset, mode, 'step', step, flush=True)
            del structural
        del module_results, refiner, prior, coupling
        if args.update == 'typed_gw':
            del typed
        timings['modes'][mode] = snapshots
        dump(out / 'timings' / f'{dataset}.json', timings)
        torch.cuda.empty_cache()
    # Gold is loaded only after every ranking is persisted.
    labels = {r['source']: r['gold'] for r in read_jsonl(Path(args.parent) / 'labels' / f'{dataset}.jsonl')}
    results = {variant: metric(rows, labels) for variant, rows in ranked.items()}
    dump(out / 'metrics' / f'{dataset}.json', results)
    dump(out / 'done' / f'{dataset}.json', {'complete': True, 'seconds': time.perf_counter() - started})
    print(dataset, results, flush=True)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--graph-root', required=True)
    parser.add_argument('--literal-root', required=True)
    parser.add_argument('--kernel-root', required=True)
    parser.add_argument('--fact-kernel-root')
    parser.add_argument('--typed-audit', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--steps', nargs='+', type=int, default=[8, 32])
    parser.add_argument('--temperature', type=float, default=.05)
    parser.add_argument('--temperature-scale', choices=['fixed', 'row_std'], default='fixed')
    parser.add_argument('--transport-mode', choices=['balanced', 'unbalanced'], default='balanced')
    parser.add_argument('--numeric-kernel', choices=['forward', 'symmetric'], default='forward')
    parser.add_argument('--static-schema', action='store_true')
    parser.add_argument('--context', action='store_true', help='Equal typed and untyped support in the same neighbor step')
    parser.add_argument('--update', choices=['proximal', 'anchored', 'joint', 'typed_gw'], default='proximal')
    parser.add_argument('--base-weight', type=float)
    parser.add_argument('--proximal-weight', type=float)
    parser.add_argument('--support-weight', type=float, default=1.0)
    parser.add_argument('--support-calibration', choices=['none', 'row_std'], default='none')
    parser.add_argument('--joint-lambda', type=float,
                        help='single convex-combination coefficient: (1-lambda) prior + lambda structural')
    parser.add_argument('--modes', nargs='+', choices=['all', 'no_image'], default=['all', 'no_image'])
    args = parser.parse_args()
    if min(args.steps) < 1 or args.temperature <= 0:
        parser.error('positive iteration checkpoints and temperature required')
    if any(value is not None and value < 0 for value in (args.base_weight, args.proximal_weight)) or args.support_weight < 0:
        parser.error('objective weights must be non-negative')
    if args.joint_lambda is not None and not 0 <= args.joint_lambda <= 1:
        parser.error('--joint-lambda must be in [0,1]')
    if args.temperature_scale == 'row_std' and args.transport_mode != 'balanced':
        parser.error('row standardization requires balanced normalization')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dump(Path(args.out) / 'protocol.json', dict(vars(args), **{
        'module': 'single structural-consistency refiner', 'separate_gw_stage': False,
        'typed_gw_objective': args.update == 'typed_gw',
        'structural_adaptive_branch': False, 'equal_modality_prior': True,
        'training': False, 'task_checkpoint': None, 'gold_loaded_after_rankings': True,
        'main_ranking': 'coupling', 'top_k': 20, 'image_gate': False,
    }))
    for ds in args.datasets:
        if not (Path(args.out) / 'done' / f'{ds}.json').exists():
            run(args, ds)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
