"""Original ambiguity-driven acquisition and unified identity verification.

The complete Top10 remains eligible in every diagnosis and verification.
Questions/expected observations are model hypotheses, never tool evidence.
One main decision role and one verifier share a frozen VLM, not independent
weights. Fixed-order and joint modes are explicitly labelled controls.
"""
from copy import deepcopy
import json
import time

from .environment import parse_object
from .grounded import GroundedEnv, catalog, object_schema
from .contrastive import record_state, operation_table

MODES = ('adaptive', 'fixed_order', 'joint')
MODALITIES = ('name', 'attribute', 'relation', 'image')
# A shared serialization bound, not a decision threshold. It prevents free-text
# repetition from exhausting the JSON response before an executable action.
TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 240}


def top10(row):
    """Take the existing retrieval prefix, without selecting by answer labels."""
    rank = list(row['baseline']['rank'][:10])
    assert len(rank) == len(set(rank)) == 10
    by_uri = {c['uri']: c for c in row['candidates']}
    baseline = {'rank': rank}
    if 'scores' in row['baseline']:
        baseline['scores'] = list(row['baseline']['scores'][:10])
    # Deliberately copy only the inference contract, excluding evaluation data.
    return deepcopy({'source': row['source'], 'record': row['record'],
                     'candidates': [by_uri[u] for u in rank], 'baseline': baseline})


def make_env(row, index, model, folder):
    assert len(row['candidates']) == 10
    env = GroundedEnv(row, index, model.compact, folder, novel_only=True, candidate_count=10)
    for h, uri in env.handles.items():
        env.ledger[h+'.record'] = {'entity': h, 'uri': uri, 'modality': 'name', 'preview': True,
            'location': 'record/uri', 'text': 'Record subject URI: '+uri}
    return env


def ask(model, env, prompt, schema, kind):
    raw, meta = model.generate(prompt, env.images(), schema=schema, max_tokens=768)
    try:
        obj = parse_object(raw); error = None
    except (ValueError, TypeError) as exc:
        obj = {}; error = str(exc)
    return obj, {'kind': kind, 'action': obj, 'raw': raw, 'model': meta, 'error': error}


def diagnose(model, env, memory, remaining):
    state, _ = record_state(env, model)
    schema = object_schema({
        'source_subject': TEXT, 'ambiguity': TEXT,
        'hypothesis': {'enum': list(env.priors)},
        'question': TEXT, 'expected_observation': TEXT,
        'query_entities': {'type': 'array', 'items': {'enum': list(env.handles)},
                           'minItems': 1, 'maxItems': 3},
        'next': {'enum': ['inspect', 'verify', 'unresolved'] if remaining else ['verify', 'unresolved']}})
    prompt = (state+'\nUNREAD TOOL AVAILABILITY\n'+json.dumps(env.available_tools())+
        '\nPRIOR ACTIONS AND FALLIBLE VERIFIER FEEDBACK\n'+json.dumps(memory, ensure_ascii=True)+
        '\nMAIN AGENT: identify SRC among ALL TEN records. Reconsider every candidate, including those '
        'outside your previous query. First state what the SOURCE RECORD itself denotes, as a short '
        'subject description, not an associated object or a proposed candidate. Next identify the '
        'specific ambiguity between plausible identities. Only then propose a candidate hypothesis '
        'and a neutral factual question that can resolve the ambiguity, '
        'and what observation would favor one interpretation over another. Keep each text to one short '
        'sentence; if already resolved, say so and request verification. These are proposed interpretations '
        'and anticipated observations, NOT established facts. Do not simply request more facts or list '
        'a modality. Name, attribute, directed relation and original image have equal standing. '
        'Choose the records needed for this query; a tool reads up to three records per operation, '
        'but this NEVER eliminates any of the ten candidates from the decision. A prior negative or '
        'unknown judgment is fallible feedback, not permanent exclusion. Use the concrete unresolved '
        'question returned by the verifier to revise your search. Inspect if a useful new observation '
        'is needed, verify if existing evidence is enough, unresolved if no available observation can '
        'resolve the uncertainty. Do not select verify merely because a familiar name is present. '
        f'Remaining tool operations: {remaining}. Return the required JSON.')
    obj, trace = ask(model, env, prompt, schema, 'diagnosis')
    required = ('source_subject', 'ambiguity', 'question', 'expected_observation')
    if not trace['error']:
        if (obj.get('hypothesis') not in env.priors or
            any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in required) or
            not isinstance(obj.get('query_entities'), list) or not 1 <= len(obj['query_entities']) <= 3 or
            any(h not in env.handles for h in obj['query_entities']) or
            obj.get('next') not in (['inspect', 'verify', 'unresolved'] if remaining else ['verify', 'unresolved'])):
            trace['error'] = 'Invalid ambiguity diagnosis'
    return obj, trace


def operations(env, diagnosis, mode, used_modalities):
    focus = list(dict.fromkeys(diagnosis['query_entities']))
    options = catalog(env, focus)
    if mode == 'fixed_order':
        # A coverage control: each available modality at most once, in a fixed
        # order. Only this control overrides the main Agent's query modality.
        modality = next((m for m in MODALITIES if m not in used_modalities and
                         any(v['modality'] == m for v in options.values())), None)
        options = {k: v for k, v in options.items() if v['modality'] == modality}
    return {k: {**v, 'operation': 'Read '+(v['predicate'] or v['modality'])+
                ' of records '+', '.join(v['entities'])} for k, v in options.items()}


def acquire(model, env, diagnosis, mode, used_modalities):
    options = operations(env, diagnosis, mode, used_modalities)
    if not options:
        return None, [{'kind': 'planner', 'action': {'field_id': 'none'},
                       'reason': 'No unread operation in the requested tool scope', 'error': None}]
    state, _ = record_state(env, model, list(dict.fromkeys(diagnosis['query_entities'])))
    prompt = (state+'\nMAIN AGENT QUERY PLAN (anticipated observations are not facts)\n'+
        json.dumps(diagnosis, ensure_ascii=True)+'\nChoose the concrete existing operation that can answer '
        'the stated question and discriminate the identity hypotheses. The table binds each operation '
        'to exact records and original predicates. Check that the field meaning actually matches the '
        'question, rather than choosing an unrelated unread value. none means no useful operation. '
        'A name tool only reads names; an image tool opens original pixels. No modality has priority '
        'in the adaptive method. '+('This is the fixed-order control; only its next modality is offered. '
        if mode == 'fixed_order' else '')+'Return only the field ID.\n'+operation_table(options))
    obj, trace = ask(model, env, prompt, object_schema({'field_id': {'enum': list(options)+['none']}}), 'planner')
    if trace['error'] or obj.get('field_id') == 'none':
        return None, [trace]
    item = options.get(obj.get('field_id'))
    if item is None:
        trace['error'] = 'Unknown original evidence operation'; return None, [trace]
    action = {'action': 'inspect', 'entities': item['entities'], 'modalities': [item['modality']],
              'question': diagnosis['question'], 'expected': diagnosis['expected_observation'], 'offset': 0}
    if item['predicate'] is not None:
        action['predicate'] = item['predicate']
    trace['resolved_original_field'] = item
    observation = env.inspect(action)
    used_modalities.add(item['modality'])
    return observation, [trace, {'kind': 'tool', 'action': action, 'observation': observation}]


def verification_schema(env, refs):
    source = sorted(r for r in refs if env.ledger[r]['entity'] == 'SRC')
    choices = []
    for h in env.priors:
        target = sorted(r for r in refs if env.ledger[r]['entity'] == h)
        choices.append(object_schema({'choice': {'enum': [h]},
            'source_ref': {'enum': source}, 'target_ref': {'enum': target}}))
    choices.append(object_schema({'choice': {'enum': ['U']}, 'remaining_question': TEXT}))
    return object_schema({'source_subject': TEXT, 'evidence_assessment': TEXT, 'decision': {'anyOf': choices}})


def verify_identity(model, env):
    state, refs = record_state(env, model)
    prompt = (state+'\nUNIFIED IDENTITY VERIFIER. Reassess ALL TEN candidates from raw observations. '
        'Use only the original records and actual observations below their evidence IDs. '
        'First state what the SOURCE RECORD itself denotes, preserving its subject and granularity. '
        'Then give one concise evidence assessment: what do the strongest observations actually describe, '
        'which record subject or related object do they concern, and which competing identity do they '
        'distinguish? Then choose any candidate whose exact identity is supported, or U if still unresolved. '
        'A shared topic, associated name, creator, containing region or illustrative picture does not '
        'alone identify the record subject. Missing evidence is unknown, not contradiction; differences '
        'may reflect time or granularity, so interpret the original predicates. Treat all four modalities '
        'equally and inspect the actual attached images when relevant; text may revise a visual '
        'interpretation and an image may revise a textual interpretation. A visible URI is record '
        'provenance, not an alignment label. Cite observed SRC and chosen-target references that support '
        'your assessment. A reference being valid is not proof of semantic identity. If U, return one '
        'concrete remaining question for the main Agent to investigate. An unknown result must not '
        'become an assertion that the initial top candidate is correct. Return the specified JSON.')
    obj, trace = ask(model, env, prompt, verification_schema(env, refs), 'verifier')
    trace['compared_handles'] = list(env.priors)
    decision = obj.get('decision', {})
    choice = decision.get('choice', 'U')
    if not trace['error']:
        if not isinstance(obj.get('source_subject'), str) or not obj['source_subject'].strip():
            trace['error'] = 'Missing source subject assessment'
        elif not isinstance(obj.get('evidence_assessment'), str) or not obj['evidence_assessment'].strip():
            trace['error'] = 'Missing evidence assessment'
        elif choice == 'U':
            if not isinstance(decision.get('remaining_question'), str) or not decision['remaining_question'].strip():
                trace['error'] = 'Unresolved verification needs a remaining question'
        elif choice not in env.priors:
            trace['error'] = 'Candidate outside Top10'
        else:
            for key, owner in [('source_ref', 'SRC'), ('target_ref', choice)]:
                ref = decision.get(key)
                if ref not in refs or env.ledger[ref]['entity'] != owner:
                    trace['error'] = 'Citation not visible or belongs to a different record'
    accepted = not trace['error'] and choice in env.priors
    trace.update(accepted=accepted, accepted_handle=choice if accepted else 'U')
    return choice if accepted else 'U', trace


def solve_ambiguity(row, index, model, folder, mode='adaptive', budget=4):
    assert mode in MODES
    env = make_env(row, index, model, folder)
    clock = time.time(); start = (model.calls, model.input_tokens, model.output_tokens)
    trace = []; memory = []; used = set(); inspections = 0; choice = 'U'; status = 'unresolved'
    previous = getattr(model, 'unicode_json', False); model.unicode_json = True
    try:
        if mode == 'joint':
            choice, t = verify_identity(model, env)
            trace.append(t); status = 'invalid_verification' if t['error'] else ('verified' if choice != 'U' else 'unresolved')
        else:
            for turn in range(budget+1):
                remaining = min(budget-inspections, budget-turn)
                plan, t = diagnose(model, env, memory, remaining); trace.append(t)
                if t['error']:
                    status = 'invalid_diagnosis'; break
                observation = None
                acquisition = None
                if remaining and plan['next'] == 'inspect':
                    observation, ts = acquire(model, env, plan, mode, used); trace.extend(ts)
                    acquisition = {'selected_operation': ts[0].get('resolved_original_field'),
                                   'planner_action': ts[0].get('action'), 'notice': ts[0].get('reason')}
                    if any(t.get('error') for t in ts):
                        status = 'invalid_plan'; break
                    if observation is not None:
                        inspections += 1
                        if observation.get('error'):
                            status = 'tool_error'; break
                choice, t = verify_identity(model, env); trace.append(t)
                if t['error']:
                    status = 'invalid_verification'; choice = 'U'; break
                if choice != 'U':
                    status = 'verified'; break
                memory.append({'diagnosis': plan, 'observed_refs': [e['ref'] for e in (observation or {}).get('evidence', [])],
                               'acquisition': acquisition, 'verifier_feedback': t['action']})
                if plan['next'] == 'unresolved' or not remaining:
                    break
        result = {'source': env.source, 'baseline_rank': list(env.baseline), 'rank': env.rank(choice),
            'status': status, 'mode': mode, 'trace': trace, 'handle_map': dict(env.handles),
            'evidence': deepcopy(env.ledger), 'inspections': inspections,
            'verification_calls': sum(t['kind'] == 'verifier' for t in trace),
            'calls': model.calls-start[0], 'input_tokens': model.input_tokens-start[1],
            'output_tokens': model.output_tokens-start[2], 'seconds': time.time()-clock,
            'candidate_count': 10, 'semantic_scope_is_model_judgment': True}
        assert len(result['rank']) == len(set(result['rank'])) == 10
        assert set(result['rank']) == set(row['baseline']['rank'])
        return result
    finally:
        model.unicode_json = previous
