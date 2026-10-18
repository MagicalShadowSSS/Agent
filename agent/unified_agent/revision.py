"""Evidence provenance, competing identity verification, and conservative repair.

All policies are label-free. Evaluation-based release decisions live elsewhere.
"""
from copy import deepcopy
import re
from urllib.parse import unquote


REVISION_MODES = {
    'grounded_fixed_cycle', 'grounded_provenance', 'grounded_compare_identity',
    'grounded_provenance_compare', 'grounded_reflect', 'grounded_provenance_reflect',
    'grounded_provenance_compare_reflect', 'grounded_guarded_repair', 'grounded_guarded_provenance_reflect',
    'grounded_guarded_provenance_reflect_unicode',
}


def provenance_row(row):
    """Retain every original value, but distinguish record titles from claims."""
    result = deepcopy(row)
    normalize = lambda s: re.sub(r'\W+', ' ', s.casefold()).strip()
    for rec in [result['record']] + [c['record'] for c in result['candidates']]:
        title = unquote(rec['uri'].strip('<>').rsplit('/', 1)[-1]).replace('_', ' ')
        values = ['Record title from URI: ' + title]
        for name in rec.get('name', []):
            predicates = list(dict.fromkeys(
                p for a in rec.get('attribute', []) for p, sep, value in [a.partition(': ')]
                if sep and normalize(value) == normalize(name)))
            if normalize(name) == normalize(title):
                origin = 'URI label variant'
            elif predicates:
                origin = 'Attribute claim [' + ', '.join(predicates) + ']; ownership unverified'
            else:
                origin = 'Supplied name or translation; unverified'
            values.append(origin + ': ' + name)
        rec['name'] = values
    return result


def unique_choice(scores, fallback='U'):
    winners = [k for k, v in scores.items() if v == max(scores.values())]
    return winners[0] if len(winners) == 1 else fallback


def verify_competing(model, env, best, rival, strict=True):
    from .grounded import ask_json, focused_state, object_schema
    alternatives = sorted([best, rival])
    state, refs = focused_state(env, model, ['SRC'] + alternatives)
    prompt = (state + '\nCOMPARE COMPETING IDENTITIES. Identify the subject of SRC, not a related entity. '
        'The alternatives are ' + ' and '.join(alternatives) + '. Choose one only when the supplied evidence '
        'identifies that same real-world subject and distinguishes it from the other. '
        'Names, attributes, directed relations and original images are equally valid evidence sources. '
        'A record can have renamed historical aliases. A title or name inside an attribute can instead refer '
        'to an associated work, member or publisher. Preserve the entity type and granularity: '
        'an album differs from its tracks, a city from its encompassing territory, an office from its holder. '
        'Missing evidence is unknown; generic resemblance or a shared category is insufficient. '
        'Use visible image text/logos/non-face details; facial similarity cannot establish identity. '
        'U means neither identity is sufficiently established, including unresolved ambiguity. '
        'Return just one label: ' + ', '.join(alternatives + ['U']) + '.')
    scores, meta = model.choose(prompt, env.images(), alternatives + ['U'])
    choice = unique_choice(scores)
    trace = {'kind': 'verifier', 'action': {'choice': choice}, 'hypothesis': best, 'rival': rival,
             'accepted': False, 'comparison_scores': scores, 'comparison_model': meta,
             'strict_scope': True, 'reason': 'No uniquely supported competing identity'}
    if choice not in alternatives:
        return False, trace, 'uncertain'
    source_refs = sorted(r for r in refs if env.ledger[r]['entity'] == 'SRC')
    target_refs = sorted(r for r in refs if env.ledger[r]['entity'] == choice)
    if not source_refs or not target_refs:
        trace['reason'] = 'Missing visible evidence for source or selected candidate'
        return False, trace, 'uncertain'
    fields = {'source_ref': {'enum': source_refs}, 'target_ref': {'enum': target_refs},
              'source_scope': {'enum': ['self', 'related_entity', 'unclear']},
              'target_scope': {'enum': ['self', 'related_entity', 'unclear']},
              'correspondence': {'type': 'string', 'maxLength': 180},
              'comparison': {'enum': ['identifying_agreement', 'identity_conflict', 'shared_nondistinctive', 'insufficient']}}
    obj, citation_trace = ask_json(model, env, state +
        f'\nAUDIT PROPOSED IDENTITY SRC={choice}. This proposal can be wrong. '
        'Cite the strongest existing source and target evidence and describe its semantic correspondence. '
        'Determine whether each cited fact belongs to its record itself. '
        'Different predicates must refer to compatible facts; a station count is not a service type. '
        'A shared general category alone does not distinguish identity. '
        'Do not invent aliases or reinterpret related-entity names as subject names. '
        'Names, attributes, directed relations and original images have equal standing. Return the specified JSON.',
        object_schema(fields), 'verifier')
    trace.update(action=obj, raw=citation_trace['raw'], model=citation_trace['model'], error=citation_trace['error'])
    valid = obj.get('source_ref') in source_refs and obj.get('target_ref') in target_refs
    trace['evidence_refs_valid'] = valid
    scope = obj.get('source_scope') == obj.get('target_scope') == 'self'
    if citation_trace['error'] or not valid or not scope or obj.get('comparison') != 'identifying_agreement':
        trace['reason'] = 'Cited facts do not establish same-subject identifying agreement'
        return False, trace, 'uncertain'
    # Independently assess the actual cited data, without showing the generated explanation.
    a, b = env.ledger[obj['source_ref']], env.ledger[obj['target_ref']]
    check = (f'SOURCE record {env.handles["SRC"]}\n[{obj["source_ref"]}] {a["text"]}\n'
             f'TARGET record {env.handles[choice]}\n[{obj["target_ref"]}] {b["text"]}\n'
             'Do these supplied observations provide identifying agreement about the two record subjects? '
             'same requires compatible facts belonging to the subjects themselves, such as a distinctive '
             'name/alias correspondence, matching identifying properties, graph roles or a visual identifier. '
             'different means a real identity conflict. unknown means unrelated predicates, missing evidence, '
             'an associated entity mistaken for the subject, or only a generic shared property. '
             'Do not infer agreement just because the two URI titles seem related. '
             'Names, attributes, relations and images are peers. Face resemblance is not identifying evidence. '
             'Output only same, different, or unknown.')
    support, support_meta = model.choose(check, env.images(), ['same', 'different', 'unknown'])
    verdict = unique_choice(support, 'unknown')
    trace.update(support_scores=support, support_model=support_meta, support_verdict=verdict,
                 accepted=verdict == 'same', accepted_handle=choice,
                 reason=None if verdict == 'same' else 'Actual cited observations do not establish identity')
    return trace['accepted'], trace, verdict


def apply_repair_guard(incumbent, proposal):
    """No labels. Retain verified decisions and the incumbent top-ten set."""
    if incumbent['status'] == 'verified':
        return list(incumbent['rank']), 'retain_verified_incumbent'
    if proposal['status'] != 'verified':
        return list(incumbent['rank']), 'repair_unresolved'
    choice = proposal['rank'][0]
    if choice not in incumbent['rank'][:10]:
        return list(incumbent['rank']), 'retain_top10_membership'
    rank = [choice] + [u for u in incumbent['rank'] if u != choice]
    assert set(rank[:10]) == set(incumbent['rank'][:10])
    return rank, 'accepted_repair'


def merge_repair(incumbent, proposal=None, cached=True):
    """Replay the same policy on saved inference outputs; no labels accepted."""
    if incumbent['status'] == 'verified':
        result = deepcopy(incumbent)
        result.update(repair_gate='retain_verified_incumbent', incumbent_cached=cached,
                      new_model_calls=0 if cached else incumbent['calls'],
                      accounting='End-to-end cost includes incumbent, even when reused from cache')
        return result
    assert proposal is not None and proposal['source'] == incumbent['source']
    assert proposal['baseline_rank'] == incumbent['baseline_rank']
    result = deepcopy(proposal)
    rank, reason = apply_repair_guard(incumbent, result)
    result['proposal_rank'] = result['rank']
    result['rank'] = rank
    result['repair_gate'] = reason
    result['incumbent_cached'] = cached
    result['incumbent_status'] = incumbent['status']
    result['new_model_calls'] = result['calls'] + (0 if cached else incumbent['calls'])
    result['accounting'] = 'End-to-end cost includes incumbent, even when reused from cache; two stages have separate evidence contexts'
    if reason != 'accepted_repair':
        result['status'] = 'guard_retained'
    for key in ('calls', 'inspections', 'verification_calls', 'input_tokens', 'output_tokens', 'seconds'):
        result[key] += incumbent[key]
    result['trace'] = [dict(t, stage='incumbent') for t in incumbent['trace']] + [dict(t, stage='repair') for t in result['trace']]
    assert set(result['rank'][:10]) == set(incumbent['rank'][:10])
    return result


def solve_repair(row, index, model, asset_dir, budget, incumbent=None,
                 repair_mode='grounded_provenance_compare_reflect',unicode_json=False):
    from .grounded import solve_grounded
    cached = incumbent is not None
    if incumbent is None:
        incumbent = solve_grounded(row, index, model, asset_dir, 'grounded_route_pages', budget)
    assert incumbent['source'] == row['source']
    assert incumbent['baseline_rank'] == row['baseline']['rank']
    assert len(incumbent['rank']) == 20 and set(incumbent['rank']) == set(row['baseline']['rank'])
    if incumbent['status'] == 'verified':
        return merge_repair(incumbent, cached=cached)
    previous=getattr(model,'unicode_json',False)
    try:
        model.unicode_json=unicode_json
        result = solve_grounded(row, index, model, asset_dir, repair_mode, budget)
    finally:
        model.unicode_json=previous
    return merge_repair(incumbent, result, cached)
