"""Label-free modality routing; shares retrieval pages and verification with controls."""
import hashlib
import json
import random

from .environment import MODALITIES


ROUTING_MODES = {'grounded_route_pages', 'grounded_route_once',
                 'grounded_route_fields', 'grounded_random_once'}


def modality_options(env, focus, excluded=()):
    # Availability contains no hidden field values, labels, or future observations.
    result = {}
    for modality in MODALITIES:
        if modality in excluded:
            continue
        counts = {h: len(env.unread(h, modality)) for h in focus}
        handles = [h for h in focus if counts[h]]
        if handles:
            result[modality] = {'modality': modality, 'entities': handles,
                                'predicate': None, 'unread_counts': counts}
    return result


def random_order(source):
    order = list(MODALITIES)
    seed = int(hashlib.sha256(('route-control-v5:' + source).encode()).hexdigest(), 16)
    random.Random(seed).shuffle(order)
    return order


def select_modality(model, env, focus, best, rival, feedback, mode, used):
    from .grounded import ask_json, focused_state, object_schema
    excluded = used if mode in ('grounded_route_once', 'grounded_random_once') else ()
    options = modality_options(env, focus, excluded)
    if not options:
        return None, {'kind': 'planner', 'action': {'decision': 'abstain'},
                      'reason': 'No eligible unread modality'}, 'abstain'
    if mode == 'grounded_random_once':
        modality = next(m for m in random_order(env.source) if m in options)
        trace = {'kind': 'planner', 'action': {'modality': modality, 'decision': 'inspect'},
                 'policy': 'URI-seeded random permutation, no label or model choice'}
    else:
        state, _ = focused_state(env, model, focus)
        schema = object_schema({'modality': {'enum': list(options) + ['none']},
                                'distinction': {'enum': ['identity_name', 'entity_type_or_granularity',
                                    'time_or_place', 'relation_role', 'visual_identifier',
                                    'other_specific_fact', 'unresolvable']}})
        prompt = (state + '\nVERIFICATION FEEDBACK\n' + json.dumps(feedback, ensure_ascii=False) +
            f'\nNEXT MODALITY: distinguish SRC={best} from SRC={rival}. '
            'Select the evidence source whose unread observations could resolve the remaining identity question. '
            'name returns names and aliases; attribute returns literal properties; relation returns directed graph facts; '
            'image opens original pixels for visible text, logos and non-face details. '
            'All modalities are peers. The number of available fields is not a vote for a modality. '
            'Do not choose an irrelevant shared property just because many values are available. '
            'Evidence already read remains available for comparison: a query of only one record may complete a comparison '
            'with already observed evidence from another record. Missing evidence is unknown, not contradiction. '
            'Choose none when the remaining evidence cannot help resolve the identity. '
            'Choose a modality first, then the distinction category. Return the specified JSON only. '
            '\nAVAILABLE UNREAD MODALITIES\n' + json.dumps(options, ensure_ascii=False))
        obj, trace = ask_json(model, env, prompt, schema, 'planner')
        trace.update(policy='model selects among available modalities', available_modalities=options)
        if trace['error']:
            return None, trace, 'invalid_plan'
        modality = obj.get('modality')
        if modality == 'none':
            trace['action']['decision'] = 'abstain'
            return None, trace, 'abstain'
        if modality not in options:
            return None, trace, 'invalid_plan'
        trace['action']['decision'] = 'inspect'
    item = {k: v for k, v in options[modality].items() if k != 'unread_counts'}
    trace['resolved_original_field'] = item
    return item, trace, 'inspect'
