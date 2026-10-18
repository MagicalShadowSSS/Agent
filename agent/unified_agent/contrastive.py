"""A single contrastive action/observation loop over frozen candidate records.

Two hypotheses are an unordered pair of distinct handles. An evidence action
denotes concrete record handles and an original field, not a second free-text
question. One decision and its source/target citations replace stacked judges.
"""
from copy import deepcopy
import itertools
import json
import time

from .environment import parse_object
from .engine import render
from .grounded import GroundedEnv, catalog, object_schema
from .deliberative import focused_images


def record_state(env, model, focus=None):
    """Group every observation under its actual record owner."""
    _, visible = render(env, model.compact)
    handles = list(env.handles) if focus is None else list(focus)
    lines = []
    refs = set()
    for h in handles:
        lines.append('\nSOURCE RECORD' if h == 'SRC' else '\nCANDIDATE RECORD ' + h)
        ref = h + '.record'
        lines.append('[' + ref + '] ' + env.ledger[ref]['text']); refs.add(ref)
        if h in env.priors:
            lines.append('Retrieval prior (not a match label): ' + json.dumps(env.priors[h]))
        for r, item in env.ledger.items():
            if item['entity'] != h or r not in visible or r == ref:
                continue
            lines.append(f'[{r}] {item["text"]}'); refs.add(r)
        lines.append('END RECORD ' + h)
    lines.append('\nFields inside a record can mention associated entities; each field keeps its original '
                 'record owner. In relations, out means this record is the subject; in means this record '
                 'is the object. Pictures may illustrate a related object rather than the record subject.')
    return '\n'.join(lines), refs


def make_env(row, index, model, folder):
    env = GroundedEnv(row, index, model.compact, folder, novel_only=True)
    for h, uri in env.handles.items():
        env.ledger[h+'.record'] = {'entity': h, 'uri': uri, 'modality': 'name', 'preview': True,
            'location': 'record/uri', 'text': 'Record subject URI: ' + uri}
    return env


def ask(model, prompt, images, schema, kind):
    # Only executable enums and existing references are generated. There is no
    # unconstrained explanation field that can consume the whole action budget.
    raw, meta = model.generate(prompt, images, schema=schema, max_tokens=512)
    try:
        obj = parse_object(raw); error = None
    except (ValueError, TypeError) as exc:
        obj = {}; error = str(exc)
    return obj, {'kind': kind, 'action': obj, 'raw': raw, 'model': meta, 'error': error}


def pairs_for(env):
    return {a+'/'+b: (a, b) for a, b in itertools.combinations(env.priors, 2)}


def propose_pair(model, env, memory):
    state, _ = record_state(env, model)
    pairs = pairs_for(env)
    prompt = (state + '\nACTION / OBSERVATION HISTORY\n' + json.dumps(memory, ensure_ascii=True) +
        '\nFind the record describing the same encyclopedia subject as SRC in the other language. '
        'Select the two most plausible distinct candidates from ALL20 for a focused comparison. '
        'An entity mentioned in a record or depicted in an illustration is not necessarily that record\'s subject. '
        'Respect the complete subject, its role and granularity when considering names, properties, '
        'directed relations and images; all are valid evidence sources. Reconsider the pair after new '
        'observations, including previously unexamined candidates. Ranks are only retrieval priors. '
        'Return only one unordered pair as two handles separated by /, smaller letter first, '
        'in the required JSON object.')
    schema = object_schema({'pair': {'enum': list(pairs)}})
    obj, t = ask(model, prompt, env.images(), schema, 'hypothesis')
    if t['error'] or obj.get('pair') not in pairs:
        t['error'] = t['error'] or 'Unknown pair'
        return None, t
    return pairs[obj['pair']], t


def action_catalog(env, pair):
    available = catalog(env, ['SRC'] + list(pair))
    options = {}
    for key, item in available.items():
        field = item['predicate'] or item['modality']
        handles = item['entities']
        options[key] = {**item, 'operation': 'Read ' + field + ' of records ' + ', '.join(handles)}
    return options


def operation_table(options):
    # Render each executable operation once, without duplicating its contents
    # inside verbose JSON keys and a second textual description. No fields are
    # dropped, ranked by hand or truncated to fit the model context.
    lines = ['operation ID | modality | records to read | original field']
    for key, item in options.items():
        lines.append(' | '.join([key, item['modality'], ','.join(item['entities']),
                                 item['predicate'] or item['modality']]))
    return '\n'.join(lines)


def decision_schema(env, pair, refs, options):
    """Bind citation ownership to the actual selected candidate in the grammar."""
    source = sorted(r for r in refs if env.ledger[r]['entity'] == 'SRC')
    next_actions = ['finish'] + list(options)
    alternatives = []
    for h in pair:
        target = sorted(r for r in refs if env.ledger[r]['entity'] == h)
        if not source or not target:
            continue
        alternatives.append(object_schema({
            'choice': {'enum': [h]},
            'source_ref': {'enum': source}, 'target_ref': {'enum': target},
            'next_action': {'enum': next_actions}}))
    alternatives.append(object_schema({'choice': {'enum': ['U']},
                                       'next_action': {'enum': next_actions}}))
    return {'anyOf': alternatives}


def decide(model, env, pair, remaining, folder):
    focus = ['SRC'] + list(pair); state, refs = record_state(env, model, focus)
    options = action_catalog(env, pair) if remaining else {}
    prompt = (state + '\nCompare the two candidate records against the subject of SRC. '
        'Determine the identifying agreement or remaining difference from the observations. '
        'Keep the source and candidate subjects distinct from entities mentioned in their fields or '
        'pictured in their illustrations. Similar context alone does not establish the same identity. '
        'Choose the current best exact match from the two records, or U if evidence is insufficient. '
        'For a candidate choice cite one visible source observation and one observation of that candidate. '
        'All four modalities have equal evidential standing. '
        'Then choose finish if further observation is unnecessary, or one concrete read operation whose '
        'result could distinguish the candidates. The listed operation specifies exactly which records '
        'and field will be read. Its availability is not its value. Choose the operation itself; do not '
        'write an independent tool question. A candidate choice is provisional when a read is requested. '
        f'Remaining evidence operations: {remaining}. Return only the specified action JSON, no extra explanation.\n'
        'AVAILABLE OPERATIONS\n' + operation_table(options))
    obj, trace = ask(model, prompt, focused_images(env, focus, folder/'focus'),
                     decision_schema(env, pair, refs, options), 'decision')
    if trace['error']:
        return obj, trace, options
    if obj.get('choice') not in [*pair, 'U'] or obj.get('next_action') not in ['finish', *options]:
        trace['error'] = 'Invalid decision or action'
    elif obj['choice'] != 'U':
        for key, owner in [('source_ref', 'SRC'), ('target_ref', obj['choice'])]:
            ref = obj.get(key)
            if ref not in refs or env.ledger[ref]['entity'] != owner:
                trace['error'] = 'Citation does not belong to the selected record'
    return obj, trace, options


def execute_read(env, item):
    # One catalog selection is the complete executable question. No second
    # generated description can silently point to a different record/property.
    action = {'action': 'inspect', 'entities': item['entities'], 'modalities': [item['modality']],
              'question': item['operation'], 'expected': 'Original observations for this exact operation', 'offset': 0}
    if item['predicate'] is not None:
        action['predicate'] = item['predicate']
    return {'kind': 'tool', 'action': action, 'observation': env.inspect(action)}


def snapshot(env, model, start, trace, inspections, choice, status, mode):
    clock, calls, inp, out = start
    result = {'source': env.source, 'baseline_rank': list(env.baseline), 'rank': env.rank(choice),
        'status': status, 'trace': deepcopy(trace), 'handle_map': dict(env.handles),
        'evidence': deepcopy(env.ledger), 'inspections': inspections, 'verification_calls': 0,
        'calls': model.calls-calls, 'input_tokens': model.input_tokens-inp,
        'output_tokens': model.output_tokens-out, 'seconds': time.time()-clock, 'mode': mode,
        'semantic_scope_is_model_judgment': True}
    assert len(result['rank']) == 20 and set(result['rank']) == set(env.baseline)
    return result


def solve_contrastive(row, index, model, folder, budget=4):
    """Return the initial decision and the final decision of the same policy.

    The no-query arm freezes the first decision before any tool executes. Thus
    initial observations and reasoning are identical, and no labels pick arms.
    """
    env = make_env(row, index, model, folder)
    start = (time.time(), model.calls, model.input_tokens, model.output_tokens)
    trace = []; memory = []; initial = None; choice = 'U'; inspections = 0
    status = 'unresolved'; previous = getattr(model, 'unicode_json', False); model.unicode_json = True
    try:
        for turn in range(budget+1):
            pair, t = propose_pair(model, env, memory); trace.append(t)
            if pair is None:
                status = 'invalid_hypothesis'; choice = 'U'; break
            decision, t, options = decide(model, env, pair, budget-inspections, folder)
            trace.append(t)
            if t['error']:
                status = 'invalid_decision'; choice = 'U'; break
            choice = decision['choice']; status = 'selected' if choice != 'U' else 'unresolved'
            if initial is None:
                initial = snapshot(env, model, start, trace, 0, choice, status, 'initial_decision')
            if decision['next_action'] == 'finish':
                break
            item = options[decision['next_action']]
            tt = execute_read(env, item); trace.append(tt); inspections += 1
            if tt['observation'].get('error'):
                status = 'tool_error'; choice = 'U'; break
            memory.append({'compared_pair': list(pair), 'provisional_choice': choice,
                'operation': item['operation'], 'observed_refs': [x['ref'] for x in tt['observation']['evidence']]})
        final = snapshot(env, model, start, trace, inspections, choice, status, 'active_evidence')
        if initial is None:
            initial = deepcopy(final); initial['mode'] = 'initial_decision'
        return initial, final
    finally:
        model.unicode_json = previous
