"""Four acquisition policies with one frozen, all-candidate decision reader.

Predicted answers and planning notes never enter the reader's factual input.
The one-call no-acquisition baseline and every subsequent reader call have the
same prompt and all ten candidates; only actually acquired evidence changes.
"""
from copy import copy, deepcopy
import hashlib
import json
import time

from .ambiguity import make_env
from .active_top10 import options_for
from .environment import parse_object
from .grounded import object_schema
from .json_protocol import SHORT_TEXT
from .zh_active_evidence import semantic_operations, state, state_hash

METHODS = ('contrast', 'answerable', 'information', 'falsify')
READER = (
    '\nENTITY ALIGNMENT. Identify which of ALL TEN candidate records denotes '
    'the same entity as the SOURCE RECORD. Compare the available names, attributes, '
    'directed relations and actual attached images. Distinguish the record subject '
    'from associated people, organizations, places or works. Missing information '
    'does not imply disagreement. Retrieval ranks are fallible. Select the single '
    'best matching candidate and return only its handle A through J.')


def reader(model, env):
    text, refs = state(env, model)
    scores, meta = model.choose(text + READER, env.images(), list(env.priors))
    choice = max(scores, key=scores.get)
    return choice, {'kind': 'reader', 'choice': choice, 'scores': scores, 'model': meta,
        'visible_refs': sorted(refs), 'observation_state_sha256': state_hash(env, model),
        'error': None}


def initial(row, index, model, folder, source_last=False):
    env = make_env(row, index, model, folder)
    env.source_last = source_last
    choice, trace = reader(model, env)
    return {'source': row['source'], 'choice': choice, 'rank': env.rank(choice),
        'baseline_rank': env.baseline, 'initial_state_sha256': trace['observation_state_sha256'],
        'trace': [trace], 'handle_map': env.handles, 'inspections': 0,
        'candidate_count': 10, 'calls': 1, 'labels_loaded': False}


def structured(model, prompt, images, schema, kind, max_tokens=768):
    raw, meta = model.generate(prompt, images, max_tokens=max_tokens, schema=schema)
    obj = parse_object(raw)  # Fail visibly; no repaired JSON or invented prediction.
    return obj, {'kind': kind, 'action': obj, 'raw': raw, 'model': meta, 'error': None}


def executable_schema(tools):
    branches = [object_schema({'tool': {'enum': [key]},
        'entities': {'type': 'array', 'items': {'enum': item['entities']},
                     'minItems': 1, 'maxItems': min(3, len(item['entities']))}})
        for key, item in tools.items()]
    return object_schema({'operation': {'anyOf': branches},
        'question': SHORT_TEXT, 'expected_observation': SHORT_TEXT})


def plan(model, env, method, current, history, remaining):
    tools = semantic_operations(options_for(env, 'adaptive', set()))
    if not tools:
        return None, [{'kind': 'planner', 'action': {'action': 'stop'},
                       'reason': 'No unread operations', 'error': None}]
    text, _ = state(env, model)
    common = (text + '\nFALLIBLE CURRENT CANDIDATE (not evidence): ' + current +
        '\nPREVIOUS QUERIES AND ANSWER ASSESSMENTS (interpretations, not facts):\n' +
        json.dumps(history, ensure_ascii=False) +
        '\nAVAILABLE ORIGINAL OPERATIONS (addresses and owners only):\n' +
        json.dumps({k: v['entities'] for k, v in tools.items()}, ensure_ascii=False) +
        '\nIdentify a concrete unresolved identity distinction among the ten records. '
        'Choose a factual question together with the exact available operation and '
        'record owners that can answer it. Describe how possible answers would '
        'distinguish the identities; these predictions are never observations. '
        'An operation reads its literal predicate, or opens original pixels. '
        'All four modalities are peers and all ten candidates remain eligible. '
        'Stop when observed evidence resolves the competing interpretations or '
        'no remaining operation can address them. Do not query merely to use the '
        'budget. Keep each prose field to one short sentence. '
        f'Remaining operations: {remaining}. ')
    if method == 'falsify':
        common += ('Look for the strongest plausible alternative to the tentative '
            'identity. Select an observation that could reveal that the current '
            'candidate is a related entity or a different subject. Seek a test '
            'that can support either interpretation, not confirmation of a favored '
            'answer. Absence of a fact is not a refutation. ')
    elif method == 'answerable':
        common += ('Check that the question applies to the requested record subject '
            'and that the literal operation can supply an answer. Do not assume '
            'that a field describes its record owner when it describes a related '
            'object. Revise questions that previous observations did not answer. ')
    else:
        common += ('Prefer a question whose different plausible answers separate '
            'the remaining identity interpretations, rather than facts common to '
            'all of them. Reconsider the ambiguity after each new observation. ')
    stop = object_schema({'action': {'enum': ['stop']}, 'reason': SHORT_TEXT})
    query = executable_schema(tools)
    if method == 'information':
        inspect = object_schema({'action': {'enum': ['inspect']},
            'alternatives': {'type': 'array', 'items': query, 'minItems': 2, 'maxItems': 2}})
        common += ('Generate two different executable questions. Each expected_observation '
            'should describe how alternative answers distinguish identities. A later '
            'selection compares their answerability and discriminatory value; do not '
            'pretend to have read the unknown answers. ')
    else:
        inspect = object_schema({'action': {'enum': ['inspect']}, **query['properties']})
    schema = object_schema({'source_subject': SHORT_TEXT, 'distinction': SHORT_TEXT,
                            'next': {'anyOf': [inspect, stop]}})
    obj, trace = structured(model, common + 'Return source_subject, distinction, and next.',
                             env.images(), schema, 'planner')
    traces = [trace]; nxt = obj['next']
    if nxt['action'] == 'stop': return None, traces
    if method == 'information':
        alternatives = nxt['alternatives']
        selection = (text + '\nUNRESOLVED DISTINCTION (hypothesis): ' + obj['distinction'] +
            '\nTWO PROPOSED QUERIES (predicted outcomes are not facts):\n' +
            json.dumps(dict(zip(('A', 'B'), alternatives)), ensure_ascii=False) +
            '\nChoose which executable query is more likely to give a reliable answer '
            'that distinguishes the candidate identities. Prefer relevant new '
            'observations with different implications across candidates. Return A or B.')
        scores, meta = model.choose(selection, env.images(), ['A', 'B'])
        selected = max(scores, key=scores.get)
        traces.append({'kind': 'query_selection', 'choice': selected, 'scores': scores,
                       'model': meta, 'error': None})
        nxt = alternatives[0 if selected == 'A' else 1]
    op = nxt['operation']; item = tools[op['tool']]
    assert all(h in item['entities'] for h in op['entities'])
    action = {'action': 'inspect', 'entities': list(dict.fromkeys(op['entities'])),
              'modalities': [item['modality']], 'question': nxt['question'],
              'expected': nxt['expected_observation'], 'offset': 0}
    if item['predicate'] is not None: action['predicate'] = item['predicate']
    trace['resolved_operation'] = {'tool': op['tool'], **action}
    return action, traces


def assess(model, env, action, observation):
    """Answer the executed question without the planner's favored identity/outcome."""
    refs = [v['ref'] for v in observation['evidence']]
    schema = object_schema({'status': {'enum': ['answered', 'partial', 'unanswered', 'inapplicable']},
        'answer': SHORT_TEXT, 'usable_refs': {'type': 'array', 'items': {'enum': refs},
            'minItems': 0, 'maxItems': len(refs)}})
    text, _ = state(env, model, action['entities'])
    prompt = (text + '\nEXECUTED QUESTION:\n' + action['question'] +
        '\nRETURNED RAW OBSERVATIONS:\n' + json.dumps(observation['evidence'], ensure_ascii=False) +
        '\nAnswer this question using the returned observations about the requested '
        'record subjects. Check whether its premise applies and the predicate '
        'meaning matches. Cite returned values that actually answer it; unrelated '
        'values are not answers. Inspect the actual attached pixels for image '
        'questions. Missing or inapplicable evidence is not a negative identity '
        'judgment. Do not choose an alignment. Return status, a concise answer, '
        'and usable_refs. Return no usable_refs if unanswered or inapplicable.')
    result, trace = structured(model, prompt, env.images(), schema, 'answer_assessment')
    assert set(result['usable_refs']) <= set(refs)
    return result, trace


def reader_view(env, allowed):
    """Keep original values; never replace them with an LLM-written summary."""
    view = copy(env)
    view.ledger = {k: v for k, v in env.ledger.items() if v['preview'] or k in allowed}
    view.recent_images = [h for h in env.recent_images if h+'.image.full' in allowed]
    return view


def solve(row, index, model, folder, method, baseline, budget=4):
    assert method in METHODS
    env = make_env(row, index, model, folder)
    assert baseline['source'] == row['source']
    assert state_hash(env, model) == baseline['initial_state_sha256']
    current = baseline['choice']; initial_rank = list(baseline['rank'])
    before_calls = (model.calls, model.input_tokens, model.output_tokens); started = time.time()
    trace = []; history = []; allowed = set(); inspections = 0; status = 'stopped'
    reader_cache = {baseline['initial_state_sha256']: current}
    for step in range(budget):
        action, plans = plan(model, env, method, current, history, budget-step)
        trace.extend(plans)
        if action is None: break
        before_observation = state_hash(env, model)
        observation = env.inspect(action)
        assert not observation.get('error'), observation
        inspections += 1
        changed = before_observation != state_hash(env, model)
        trace.append({'kind': 'tool', 'action': action, 'observation': observation,
                      'observation_state_changed': changed})
        if not changed:
            history.append({'question': action['question'], 'operation': {
                k: v for k, v in action.items() if k not in ('question', 'expected')},
                'notice': 'No new visible observation; this does not answer the question.'})
            continue
        if method == 'answerable':
            answer, t = assess(model, env, action, observation); trace.append(t)
            if answer['status'] in ('answered', 'partial'):
                allowed.update(answer['usable_refs'])
            history.append({'question': action['question'], 'operation': {
                k: v for k, v in action.items() if k not in ('question', 'expected')},
                'answer_assessment': answer})
            view = reader_view(env, allowed)
        else:
            history.append({'question': action['question'], 'operation': {
                k: v for k, v in action.items() if k not in ('question', 'expected')},
                'observed_refs': [v['ref'] for v in observation['evidence']]})
            view = env
        digest = state_hash(view, model)
        if digest not in reader_cache:
            current, t = reader(model, view); trace.append(t); reader_cache[digest] = current
        else:
            current = reader_cache[digest]
            trace.append({'kind': 'reader_cache', 'choice': current,
                          'observation_state_sha256': digest, 'error': None})
        status = 'budget_reached' if inspections == budget else 'stopped'
    if inspections >= budget: status = 'budget_reached'
    result = {'source': row['source'], 'method': method, 'rank': env.rank(current),
        'initial_rank': initial_rank, 'baseline_rank': env.baseline, 'choice': current,
        'initial_choice': baseline['choice'], 'trace': trace, 'handle_map': env.handles,
        'evidence': deepcopy(env.ledger), 'inspections': inspections, 'status': status,
        'calls': model.calls-before_calls[0], 'input_tokens': model.input_tokens-before_calls[1],
        'output_tokens': model.output_tokens-before_calls[2], 'seconds': time.time()-started,
        'cached_initial_calls': 1, 'reader_calls': sum(t['kind']=='reader' for t in trace),
        'candidate_count': 10, 'labels_loaded': False}
    assert len(set(result['rank'])) == 10 and set(result['rank']) == set(initial_rank)
    if not inspections: assert result['rank'] == initial_rank
    return result
