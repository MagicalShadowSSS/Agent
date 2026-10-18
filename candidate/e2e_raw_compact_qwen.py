import os
"""Raw -> frozen Qwen text/image affinities -> one ordinary graph loop.

Verification may reuse exactly audited text vectors; generate has no donor
argument and reconstructs its own raw graph, strings, vectors, and value
kernel. Evaluation is a separate command and never changes rankings.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from e2e_raw_single_loop import evaluate, generate_rankings, prepare_raw
from encode_graph_qwen_control import literal_texts
from classify_numeric_schemas import numeric_literal
from unified_raw_qwen_candidates import (DATASETS, MODEL, QwenEncoder,
    dataset_dir, dump, read_jsonl, read_map, write_jsonl)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_digest(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def prepare_features(root, ds, raw_donor=None, text_donor=None,
                     name_encoder='qwen', attribute_encoder='qwen', relation_text=False,
                     attribute_selection='idf12', schema_encoder='labse',
                     numeric_interpretation='semantic', text_prefix='query_instruction'):
    if raw_donor is None:
        prepare_raw(root, ds, recover_predicates=relation_text)
        raw = read_jsonl(root / 'raw' / 'inputs' / f'{ds}.jsonl')
    else:
        donor = json.loads((raw_donor / 'protocol.json').read_text())
        if donor.get('calendar_kernel') != 'overlap' or donor.get('labels_read_by_generation'):
            raise ValueError('Raw donor kernel or label protocol differs')
        if donor.get('schema_encoder','labse') != schema_encoder:
            raise ValueError('Raw donor property encoder differs')
        raw = read_jsonl(raw_donor / 'raw' / 'inputs' / f'{ds}.jsonl')
    ns = sum(r['role'] == 'query' for r in raw)
    texts = (literal_texts(raw, 12) if attribute_encoder == 'qwen' and attribute_selection == 'idf12' else
             ['\n'.join(a for a in r['attribute'] if ': ' in a
                        and not numeric_literal(a.split(': ', 1)[1])) for r in raw])
    rows = [dict(uri=r['uri'], side=int(i >= ns), name=r['name'],
                 attribute_text=t, relation_text='') for i, (r, t) in enumerate(zip(raw, texts))]
    graph = root / 'generated'
    if relation_text:
        if raw_donor is None:
            from full_graph_frozen_control import prepare
            for folder in ('graphs', 'embeddings'):
                (graph / folder).mkdir(parents=True, exist_ok=True)
            original = prepare(root / 'raw', root / 'predicate_audit', graph, ds, 'raw')
        else:
            original = read_jsonl(raw_donor / 'generated' / 'inputs' / f'{ds}.jsonl')
        if [r['uri'] for r in original] != [r['uri'] for r in rows]:
            raise ValueError('Relation text donor node order differs')
        for row, old_row in zip(rows, original):
            row['relation_text'] = old_row['relation_text']
    write_jsonl(graph / 'inputs' / f'{ds}.jsonl', rows)
    dump(graph / 'manifests' / f'{ds}.json', {'source_count': ns,
         'target_count': len(raw) - ns, 'labels_read': False,
         'relation': 'all raw binary undirected edges, no self loops'})
    (graph / 'graphs').mkdir(parents=True, exist_ok=True)
    (graph / 'embeddings').mkdir(parents=True, exist_ok=True)
    if text_donor is not None:
        old = read_jsonl(text_donor / 'inputs' / f'{ds}.jsonl')
        if [r['uri'] for r in old] != [r['uri'] for r in rows]:
            raise ValueError('Qwen verification node order differs')
        fingerprints = {}
        fields = [('name', [r['name'] for r in rows])]
        if attribute_encoder != 'none':
            fields.append(('attribute', texts))
        if relation_text:
            fields.append(('relation', [r['relation_text'] for r in rows]))
        for field, strings in fields:
            field_encoder = {'name': name_encoder, 'attribute': attribute_encoder, 'relation': 'labse'}[field]
            if field_encoder == 'qwen':
                donor_protocol = json.loads((text_donor / 'field_protocols' / f'{ds}_{field}.json').read_text())
                if donor_protocol['input_strings_sha256'] != text_digest(strings):
                    raise ValueError((ds, field, 'Qwen donor strings differ from raw reconstruction'))
                if donor_protocol.get('text_prefix', 'query_instruction') != text_prefix:
                    raise ValueError((ds, field, 'Qwen donor prefix policy differs'))
                source = text_donor / 'embeddings' / f'{ds}_{field}.npy'
            else:
                donor_rows = read_jsonl(raw_donor / 'generated' / 'inputs' / f'{ds}.jsonl')
                key = 'name' if field == 'name' else f'{field}_text'
                if [r['uri'] for r in donor_rows] != [r['uri'] for r in rows] or [r[key] for r in donor_rows] != strings:
                    raise ValueError((ds, field, 'LaBSE donor strings differ from raw reconstruction'))
                source = raw_donor / 'generated' / 'embeddings' / f'{ds}_{field}.npy'
            destination = graph / 'embeddings' / source.name
            if not destination.exists():
                destination.symlink_to(source.resolve())
            fingerprints[str(source)] = sha(source)
        for side in (1, 2):
            source = raw_donor / 'generated' / 'graphs' / f'{ds}_side{side}.npz'
            destination = graph / 'graphs' / source.name
            if not destination.exists():
                destination.symlink_to(source.resolve())
            fingerprints[str(source)] = sha(source)
        for which in ('forward', 'symmetric', 'availability'):
            suffix = 'npz' if which == 'availability' else 'npy'
            source = raw_donor / 'numeric' / 'kernels' / f'{ds}_{which}.{suffix}'
            fingerprints[str(source)] = sha(source)
        dump(root / 'receipts' / f'{ds}_verification.json', {
            'exact_raw_text_digest_matches': True, 'labels_read': False,
            'input_sha256': fingerprints, 'attribute_limit': 12 if attribute_encoder == 'qwen' and attribute_selection == 'idf12' else None,
            'schema_encoder':schema_encoder,
            'text_prefix':text_prefix,
            'name_sha256': text_digest([r['name'] for r in rows]),
            'attribute_sha256': text_digest(texts)})
        return
    for side, subset in ((1, raw[:ns]), (2, raw[ns:])):
        mapping = read_map(dataset_dir(ds) / f'ent_ids_{side}')
        index = {r['uri']: i for i, r in enumerate(subset)}
        left, right = [], []
        with (dataset_dir(ds) / f'triples_{side}').open() as stream:
            for line in stream:
                h, _, t = line.split()
                h, t = mapping.get(h, h), mapping.get(t, t)
                if h in index and t in index and h != t:
                    left.extend((index[h], index[t])); right.extend((index[t], index[h]))
        adjacency = sp.csr_matrix((np.ones(len(left), np.float32), (left, right)),
                                  shape=(len(subset), len(subset)))
        adjacency.data[:] = 1
        sp.save_npz(graph / 'graphs' / f'{ds}_side{side}.npz', adjacency)
    encoder = QwenEncoder() if 'qwen' in (name_encoder, attribute_encoder) else None
    roles = (['query'] * ns + ['target'] * (len(raw) - ns) if text_prefix == 'query_instruction'
             else ['query'] * len(raw) if text_prefix == 'shared_instruction'
             else ['target'] * len(raw))
    for field, strings in (('name', [r['name'] for r in rows]), ('attribute', texts)):
        if (name_encoder if field == 'name' else attribute_encoder) != 'qwen':
            continue
        path = graph / 'embeddings' / f'{ds}_{field}.npy'
        if path.exists():
            continue
        vectors = np.lib.format.open_memmap(path.with_suffix('.tmp'), mode='w+',
                    dtype=np.float32, shape=(len(rows), encoder.dim))
        for begin in range(0, len(rows), 256):
            end = min(begin + 256, len(rows))
            vectors[begin:end] = encoder.encode(strings[begin:end], roles[begin:end])
            dump(root / 'encoding_status' / f'{ds}.json', {
                'field': field, 'done': end, 'total': len(rows), 'labels_read': False})
            print(ds, field, end, '/', len(rows), flush=True)
        vectors.flush(); del vectors
        path.with_suffix('.tmp').replace(path)
        dump(root / 'receipts' / f'{ds}_{field}.json', {
            'input_strings_sha256': text_digest(strings), 'labels_read': False,
            'vectors_sha256': sha(path)})
        dump(graph / 'field_protocols' / f'{ds}_{field}.json', {
            'input_strings_sha256': text_digest(strings), 'labels_read': False,
            'encoder':'frozen Qwen3-Embedding-8B', 'field':field,
            'text_prefix':text_prefix,
            'attribute_selection':attribute_selection if field=='attribute' else None,
            'nodes':len(rows), 'vectors_sha256':sha(path)})
    del encoder
    import torch
    torch.cuda.empty_cache()
    if 'labse' in (name_encoder, attribute_encoder) or relation_text:
        from sentence_transformers import SentenceTransformer
        from full_graph_frozen_control import encode
        model = SentenceTransformer(os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE'),
                                    device='cuda:0', local_files_only=True).eval()
        model.requires_grad_(False)
        encode(model, rows, graph, ds)
        del model
        torch.cuda.empty_cache()
    if numeric_interpretation == 'grammar':
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('build_grammar_literal_kernel.py')),
            '--raw-root', str(root / 'raw'), '--out', str(root / 'numeric'), '--datasets', ds], check=True)
    else:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('numeric_set_candidate_control.py')),
            '--root', str(root / 'raw'), '--out', str(root / 'numeric'), '--datasets', ds,
            '--calendar-version', '2', '--calendar-kernel', 'overlap', '--kernels-only',
            '--schema-encoder',schema_encoder], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--stage', choices=('generate', 'verify', 'evaluate'), required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--feature-root')
    p.add_argument('--qwen-view-root')
    p.add_argument('--grammar-kernel-root', help='Verify only: raw grammar kernel folder')
    p.add_argument('--numeric-interpretation', choices=('semantic', 'grammar'), default='semantic')
    p.add_argument('--update', choices=('residual', 'proximal', 'anchored'), default='residual')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--support-k', type=int, default=10)
    p.add_argument('--attribute-set-kernel', choices=('forward', 'symmetric'), default='forward')
    p.add_argument('--fusion', choices=('arithmetic', 'geometric', 'harmonic', 'root_mean'), default='geometric')
    p.add_argument('--support-statistic', choices=('pearson', 'excess_ratio', 'degree_cosine', 'normalized_adjacency', 'adamic_pearson',
                                                  'marginal_pearson', 'marginal_log_pearson'), default='pearson')
    p.add_argument('--correspondence', choices=('balanced', 'row_softmax', 'reciprocal', 'semi_equal_fixed10'), default='balanced')
    p.add_argument('--name-encoder', choices=('qwen', 'labse'), default='qwen')
    p.add_argument('--attribute-encoder', choices=('qwen', 'labse', 'none'), default='qwen',
                   help='none deletes nonnumeric entity-text encoding and uses original numeric/date facts only')
    p.add_argument('--attribute-selection', choices=('idf12','complete'),default='idf12')
    p.add_argument('--schema-encoder',choices=('labse','qwen'),default='labse')
    p.add_argument('--text-prefix', choices=('query_instruction', 'none', 'shared_instruction'),
                   default='query_instruction')
    p.add_argument('--temperature-rule', choices=('fixed', 'top20_ratio'), default='fixed')
    p.add_argument('--support-matrix', choices=('topk', 'full'), default='topk',
                   help='Propagate row-normalized full soft mass instead of intermediate Top-k truncation')
    p.add_argument('--view-calibration', choices=('rowmax', 'rank20'), default='rowmax',
                   help='Replace normalized view affinities by exp(-(strict competition rank-1)/20)')
    p.add_argument('--support-norm', choices=('rowmax', 'sqrt', 'saturating'), default='rowmax',
                   help='Replace neighbor-score rowmax by square-root rowmax or R/(1+R)')
    p.add_argument('--cosine-rule', choices=('raw', 'local20'), default='raw',
                   help='Subtract source/target available-neighbor Top20 cosine means before view clipping')
    p.add_argument('--relation-text', action='store_true',
                   help='Include original relation-neighbor records in the same frozen multimodal matching step')
    p.add_argument('--labels-root', default='output/unified_raw_qwen_1000_20261006_repaired_duplicates')
    args = p.parse_args(); root = Path(args.root)
    if args.stage == 'evaluate':
        for ds in args.datasets:
            evaluate(root, ds, Path(args.labels_root))
        return
    if args.stage == 'verify' and not (args.feature_root and args.qwen_view_root):
        p.error('Verification requires both exact text and raw feature donors')
    if args.stage == 'generate' and (args.feature_root or args.qwen_view_root):
        p.error('Raw generation refuses donor assets')
    if args.grammar_kernel_root and (args.stage != 'verify' or args.numeric_interpretation != 'grammar'):
        p.error('Grammar kernel donor is only allowed for grammar verification')
    if args.numeric_interpretation == 'grammar':
        if args.schema_encoder != 'qwen':
            p.error('Grammar literal mode uses the same frozen Qwen property encoder')
        if args.stage == 'verify' and not args.grammar_kernel_root:
            p.error('Grammar verification requires --grammar-kernel-root')
    if args.temperature_rule != 'fixed' and args.correspondence != 'balanced':
        p.error('Top20-derived temperature requires balanced correspondence')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = {'method': 'raw frozen name/literal attributes/image plus one ordinary Pearson loop',
        'encoder': {k:v for k,v in {'qwen':str(MODEL),'labse':os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE')}.items()
                    if k in (args.name_encoder,args.attribute_encoder,args.schema_encoder)
                    or (k=='labse' and args.relation_text)},
        'property_encoder': 'frozen LaBSE' if args.schema_encoder=='labse' else 'frozen Qwen3-Embedding-8B',
        'schema_encoder':args.schema_encoder,
        'numeric_interpretation': args.numeric_interpretation,
        'datatype_classifier': 'fixed prototype cosine' if args.numeric_interpretation == 'semantic' else None,
        'verification_grammar_kernel_root': args.grammar_kernel_root,
        'name_encoder': args.name_encoder, 'attribute_encoder': args.attribute_encoder,
        'attribute_text_encoder':args.attribute_encoder != 'none',
        'text_prefix':args.text_prefix,
        'attribute_mode': 'typed', 'attribute_record_limit': 12 if args.attribute_encoder == 'qwen' and args.attribute_selection=='idf12' else None,
        'attribute_selection': ('mean unlabeled pooled word IDF, lexical tie break'
                                if args.attribute_encoder == 'qwen' and args.attribute_selection=='idf12' else 'complete sorted original nonnumeric records'),
        'attribute_evidence':('original numeric/date facts only; no nonnumeric entity-text evidence'
                              if args.attribute_encoder == 'none' else 'nonnumeric entity text and original numeric/date facts'),
        'calendar_kernel': 'overlap', 'calendar_version': 2, 'relative_numeric_bandwidth': .001,
        'fusion': args.fusion, 'geometric_floor': .1 if args.fusion == 'geometric' else None,
        'view_calibration':args.view_calibration,
        'cosine_rule':args.cosine_rule,
        'cosine_local_mean_k':20 if args.cosine_rule == 'local20' else None,
        'rank_confidence_scale':20 if args.view_calibration == 'rank20' else None,
        'correspondence': args.correspondence,
        'graph_relations_only': not args.relation_text, 'scope': 'ordinary', 'support_k': args.support_k,
        'support_matrix':args.support_matrix,
        'effective_support_k':args.support_k if args.support_matrix == 'topk' else None,
        'relation_text_encoder': 'frozen LaBSE' if args.relation_text else None,
        'relation_record_limit': 12 if args.relation_text else None,
        'relation_record_selection': 'pooled word IDF and lexical ties' if args.relation_text else None,
        'opaque_neighbor_description': 'first3 original attribute records' if args.relation_text else None,
        'predicate_recovery': 'unique original same-side endpoints vote for local predicate IDs' if args.relation_text else None,
        'support_norm': args.support_norm, 'support_statistic': args.support_statistic,
        'neighbor_expectation': ('observed row mass times column mass / total mass'
                                  if args.support_statistic in ('marginal_pearson','marginal_log_pearson')
                                  else 'propagation graph row degree product / target catalog size'),
        'support_scale': 1., 'update': args.update,
        'steps': args.steps, 'readout': 'compatibility', 'temperature': .05 if args.temperature_rule == 'fixed' else None,
        'temperature_rule':args.temperature_rule,
        'temperature_derivation': ('(current row maximum - current row 20th score)/log(20)'
                                   if args.temperature_rule == 'top20_ratio' else None),
        'numeric_temperature_floor':1e-6 if args.temperature_rule == 'top20_ratio' else None,
        'top_k': 20, 'downstream_top_k': 10, 'attribute_set_kernel': args.attribute_set_kernel,
        'training': False, 'EA_checkpoint': None, 'image_gate': False,
        'labels_read_by_generation': False, 'external_name_lookup': False, 'ILL_name_repair': False,
        'generation_requires_old_candidate_assets': False,
        'verification_reuses_features': args.stage == 'verify',
        'verification_feature_root': args.feature_root, 'verification_qwen_view_root': args.qwen_view_root,
        'code_sha256': sha(__file__), 'graph_loop_code_sha256': sha(Path(__file__).with_name('e2e_raw_single_loop.py'))}
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != protocol:
        raise ValueError('Refuse changed protocol in existing output')
    dump(root / 'protocol.json', protocol)
    for filename in ('e2e_raw_compact_qwen.py', 'e2e_raw_single_loop.py',
                     'unified_raw_qwen_candidates.py', 'encode_graph_qwen_control.py',
                     'e2e_raw_fourview.py', 'degree_corrected_candidates.py',
                     'full_graph_frozen_control.py',
                     'typed_relation_consistency.py', 'prepare_semantic_relation_control.py',
                     'numeric_set_candidate_control.py', 'classify_numeric_schemas.py',
                     'build_grammar_literal_kernel.py',
                     'structure_consistency.py'):
        source = Path(__file__).with_name(filename)
        destination = root / 'code' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    for ds in args.datasets:
        if (root / 'done' / f'{ds}.json').exists():
            print(ds, 'already complete; preserving', flush=True)
            continue
        raw_donor = Path(args.feature_root) if args.stage == 'verify' else None
        text_donor = Path(args.qwen_view_root) if args.stage == 'verify' else None
        prepare_features(root, ds, raw_donor, text_donor, args.name_encoder, args.attribute_encoder,
                         args.relation_text,args.attribute_selection,args.schema_encoder,args.numeric_interpretation,
                         args.text_prefix)
        kernel_root = raw_donor / 'numeric' / 'kernels' if raw_donor else None
        if args.grammar_kernel_root:
            kernel_root = Path(args.grammar_kernel_root)
            kp = json.loads((kernel_root.parent / 'protocol' / f'{ds}.json').read_text())
            raw_path = raw_donor / 'raw' / 'inputs' / f'{ds}.jsonl'
            if kp['labels_read'] or kp['datatype_classifier'] is not None or kp['raw_input_sha256'] != sha(raw_path):
                raise ValueError('Grammar kernel differs from the raw input or method')
            dump(root / 'receipts' / f'{ds}_grammar_kernel.json', {
                'labels_read': False, 'protocol': kp,
                'sha256': {f'{ds}_{which}.{suffix}': sha(kernel_root / f'{ds}_{which}.{suffix}')
                           for which, suffix in [('forward','npy'), ('symmetric','npy'), ('availability','npz')]}})
        generate_rankings(root, ds, args.update, args.steps, readout='compatibility',
            scope='ordinary', fusion=args.fusion, support_k=args.support_k,
            graph_relations_only=not args.relation_text, attribute_kernel_root=kernel_root,
            correspondence=args.correspondence, attribute_set_kernel=args.attribute_set_kernel,
            support_statistic=args.support_statistic, temperature_rule=args.temperature_rule,
            support_matrix=args.support_matrix, view_calibration=args.view_calibration,
            attribute_semantic=args.attribute_encoder != 'none', support_norm=args.support_norm,
            cosine_rule=args.cosine_rule)


if __name__ == '__main__':
    main()
