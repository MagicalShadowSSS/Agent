"""Question-conditioned acquisition over the unchanged Top10 candidate pool.

A verifier either identifies a record from observed evidence or returns an
unresolved question. The main role binds that question to an original field
and record handles. Verification runs once per observation state. All roles
share the same frozen model; there are no confidence thresholds or training.
"""
from copy import deepcopy
import hashlib
import json
import time

from .ambiguity import make_env, verify_identity as legacy_verify
from .contrastive import operation_table, record_state as raw_record_state
from .environment import parse_object
from .grounded import catalog, object_schema
from .json_protocol import SHORT_TEXT

MODES = ('adaptive', 'legacy_verifier', 'fixed_order', 'joint',
         'question_bound', 'evidence_first', 'evidence_joint',
         'legacy_bound', 'legacy_joint', 'legacy_fixed')
TEXT = SHORT_TEXT
MODALITIES = ('name', 'attribute', 'relation', 'image')
FIXED_MODES = ('fixed_order', 'legacy_fixed')


def record_state(env, model):
    """Keep image attachment metadata separate from factual text observations.

    The old ledger's placeholder text describes attachment availability, not
    image content. Repeating it inside each record let the VLM quote identical
    placeholder sentences as identifying evidence. Images still have visible
    citation IDs and exactly the same attached pixels.
    """
    state, refs = raw_record_state(env, model)
    image_refs = [r for r in refs if env.ledger[r]['modality']=='image']
    metadata_lines = {f'[{r}] {env.ledger[r]["text"]}' for r in image_refs}
    state = '\n'.join(line for line in state.splitlines() if line not in metadata_lines)
    if image_refs:
        state += ('\nIMAGE ATTACHMENT INDEX (addresses only, NOT observations or descriptions). '
                  'For these evidence IDs, observe the actual pixels. Do not quote an address or '
                  'attachment availability as identifying content.\n')
        state += '\n'.join(f'[{r}] -> '+('overview tile '+env.ledger[r]['entity']
                          if env.ledger[r]['preview'] else 'original attachment '+env.ledger[r]['entity'])
                           for r in sorted(image_refs))
    return state, refs


def ask(model, env, prompt, schema, kind):
    raw, meta = model.generate(prompt, env.images(), schema=schema, max_tokens=768)
    try:
        obj = parse_object(raw); error = None
    except (ValueError, TypeError) as exc:
        obj = {}; error = str(exc)
    return obj, {'kind': kind, 'action': obj, 'raw': raw, 'model': meta, 'error': error}


def evidence_refs(env, refs, owner):
    # A URI remains available as subject context, but is not a citation for an
    # attribute, relation or image claim. Actual name observations remain valid.
    return sorted(r for r in refs if env.ledger[r]['entity'] == owner
                  and env.ledger[r]['location'] != 'record/uri')


def verification_schema(env, refs, evidence_first=False):
    source = evidence_refs(env, refs, 'SRC')
    branches = []
    for h in env.priors:
        target = evidence_refs(env, refs, h)
        if source and target:
            props = {'choice': {'enum': [h]}}
            if not evidence_first:
                props['source_ref'] = {'enum': source}
            props['target_ref'] = {'enum': target}
            if evidence_first:
                props['target_observation'] = TEXT
            props['identifying_comparison'] = TEXT
            branches.append(object_schema(props))
    branches.append(object_schema({'choice': {'enum': ['U']},
                                   'ambiguity': TEXT, 'remaining_question': TEXT}))
    props = {'source_subject': TEXT}
    if evidence_first:
        props['source_evidence'] = object_schema({'ref': {'enum': source}, 'observation': TEXT})
    props['decision'] = {'anyOf': branches}
    return object_schema(props)


def verify(model, env, evidence_first=False):
    state, refs = record_state(env, model)
    prompt = (state+'\nIDENTITY VERIFICATION: identify the SOURCE RECORD among ALL TEN candidates. '
        'First describe the subject of SRC in a short phrase, preserving its role and granularity. '
        'Then decide whether observed evidence identifies exactly one candidate, or U if a material '
        'ambiguity remains. Consider every candidate; retrieval ranks are fallible. '
        'Do not equate a record with a person, organization, place or work mentioned in its fields or '
        'depicted in its illustration. Missing facts are unknown. Names, attributes, directed relations '
        'and the attached images have equal evidential standing. '
        'For a candidate decision cite an actual observation from SRC and from that candidate. '
        'Compare the specific identifying details in those two observations in ONE short sentence. '
        'Keep the same predicate meaning and the same record subject on both sides. '
        'Do not cite a name to claim an unseen date or an image detail; .record is provenance only. '
        'If the available observations do not distinguish the identities, return U, the concrete '
        'ambiguity, and a neutral factual question that could distinguish them. Do not invent a '
        'missing value, and do not assume a new query has already been answered. '
        'Use this exact JSON shape (replace placeholders with observed IDs):\n'
        '{"source_subject":"short subject phrase","decision":{"choice":"A through J",'
        '"source_ref":"observed SRC evidence ID","target_ref":"observed chosen record evidence ID",'
        '"identifying_comparison":"one sentence comparing the cited observations"}}\n'
        'OR {"source_subject":"short subject phrase","decision":{"choice":"U",'
        '"ambiguity":"one specific unresolved distinction","remaining_question":"one factual question"}}')
    if evidence_first:
        prompt = (state+'\nEVIDENCE-FIRST IDENTITY VERIFICATION. Find the exact SOURCE RECORD subject '
            'among ALL TEN candidates. Describe the source subject briefly, then select an actual '
            'source observation and state its content before choosing a candidate. '
            'For text, quote or closely paraphrase the cited value without adding facts. For an image, '
            'describe a concrete visible identifying detail, without using facial appearance. '
            'Then choose a candidate only if its own cited observation identifies the same subject. '
            'State the actual target observation and compare it with the source observation. '
            'The subjects must have the same role and granularity; attributes, neighbors and pictured '
            'objects may describe related entities. Do not transfer a value from one record or '
            'predicate to another. A common category, clothing, pose, scenery or generic appearance '
            'fits many different identities and cannot by itself identify a subject. A distinctive '
            'name, fact, relation or image detail can distinguish identity; no modality has priority. '
            'If the observations leave an unresolved distinction, choose U and ask a neutral factual '
            'question to resolve it. Missing evidence is unknown. All ten candidates remain eligible. '
            'Each text should be ONE short phrase or sentence. Return exactly:\n'
            '{"source_subject":"record subject","source_evidence":{"ref":"actual SRC evidence ID",'
            '"observation":"what the source observation actually contains"},"decision":{"choice":"A through J",'
            '"target_ref":"actual chosen-record evidence ID","target_observation":"what that observation actually contains",'
            '"identifying_comparison":"why those observed details identify the same record subject"}}\n'
            'OR the same source_subject and source_evidence with "decision":{"choice":"U",'
            '"ambiguity":"the unresolved identity distinction","remaining_question":"a factual question"}.')
    obj, trace = ask(model, env, prompt, verification_schema(env, refs, evidence_first), 'verifier')
    decision = obj.get('decision', {})
    choice = decision.get('choice', 'U')
    if not trace['error']:
        if not isinstance(obj.get('source_subject'), str) or not obj['source_subject'].strip():
            trace['error'] = 'Missing source subject'
        elif choice == 'U':
            if any(not isinstance(decision.get(k), str) or not decision[k].strip()
                   for k in ('ambiguity', 'remaining_question')):
                trace['error'] = 'Unresolved identity requires a concrete question'
        elif choice not in env.priors:
            trace['error'] = 'Choice outside Top10'
        elif not isinstance(decision.get('identifying_comparison'), str) or not decision['identifying_comparison'].strip():
            trace['error'] = 'Missing identifying comparison'
        else:
            for key, owner in (('source_ref', 'SRC'), ('target_ref', choice)):
                ref = (obj.get('source_evidence',{}).get('ref')
                       if evidence_first and owner=='SRC' else decision.get(key))
                if ref not in evidence_refs(env, refs, owner):
                    trace['error'] = 'Citation is invisible, wrong-owner or provenance-only'
        if evidence_first:
            source_evidence = obj.get('source_evidence',{})
            if (source_evidence.get('ref') not in evidence_refs(env, refs, 'SRC')
                or not isinstance(source_evidence.get('observation'),str)
                or not source_evidence['observation'].strip()):
                trace['error'] = 'Missing actual source observation'
            if choice!='U' and (not isinstance(decision.get('target_observation'),str)
                               or not decision['target_observation'].strip()):
                trace['error'] = 'Missing actual target observation'
    accepted = not trace['error'] and choice in env.priors
    trace.update(compared_handles=list(env.priors), accepted=accepted,
                 accepted_handle=choice if accepted else 'U')
    return choice if accepted else 'U', trace


def options_for(env, mode, used_modalities):
    options = catalog(env, list(env.handles))
    if mode in FIXED_MODES:
        modality = next((m for m in MODALITIES if m not in used_modalities
                         and any(v['modality'] == m for v in options.values())), None)
        options = {k: v for k, v in options.items() if v['modality'] == modality}
    return options


def query_schema(options, question_bound=False):
    if question_bound:
        # The selected field IS the action. There is no separate inspect/stop
        # token gate before considering the concrete operation alternatives.
        branches = [object_schema({'field_id': {'enum': [key]},
            'entities': {'type':'array','items':{'enum':item['entities']},
                         'minItems':1,'maxItems':min(3,len(item['entities']))},
            'expected_observation': TEXT}) for key,item in options.items()]
        branches.append(object_schema({'field_id': {'enum':['none']}, 'reason': TEXT}))
        return {'anyOf':branches}
    operations = [object_schema({'field_id': {'enum': [key]},
        'entities': {'type': 'array', 'items': {'enum': item['entities']},
                     'minItems': 1, 'maxItems': min(3, len(item['entities']))}})
        for key, item in options.items()]
    stop = object_schema({'action': {'enum': ['stop']}, 'reason': TEXT})
    if not operations:
        return stop
    inspect = object_schema({'action': {'enum': ['inspect']}, 'question': TEXT,
        'expected_observation': TEXT, 'operation': {'anyOf': operations}})
    return {'anyOf': [inspect, stop]}


def query(model, env, feedback, history, mode, used_modalities, remaining):
    options = options_for(env, mode, used_modalities)
    if not options:
        return None, {'kind': 'planner', 'action': {'action': 'stop', 'reason': 'No unread operations'},
                      'error': None}
    state, _ = record_state(env, model)
    prompt = (state+'\nFALLIBLE VERIFIER FEEDBACK (questions and interpretations, NOT new evidence)\n'+
        json.dumps(feedback, ensure_ascii=False)+'\nPREVIOUS EXECUTED QUERIES\n'+
        json.dumps(history, ensure_ascii=False)+'\nMAIN AGENT: the current observations left identity unresolved. '
        'Reconsider ALL TEN candidates and the feedback. Select one concrete available observation '
        'that can resolve a specific remaining ambiguity. State a neutral factual question, then what '
        'different observations would favor different interpretations. These expectations are hypotheses, '
        'not facts. Finally bind the question to an actual field_id and one to three eligible records '
        'from its table row. This executes the tool immediately. The list describes availability, '
        'not observed values. A name operation reads names; attributes and directed relations read '
        'the exact original predicate; image opens the selected original pixels. '
        'All four modalities have equal standing. No candidate is eliminated by selecting query records. '
        'Use only a field whose meaning answers your question. If no available observation can help, '
        'stop with a reason. You cannot verify the unchanged observations again. '
        f'Remaining evidence operations: {remaining}. '
        +('This is the FIXED-ORDER CONTROL: only its next available modality is offered. ' if mode in FIXED_MODES else '')+
        'Keep each text field to ONE short sentence. Return exactly one of these JSON forms:\n'
        '{"action":"inspect","question":"specific factual question",'
        '"expected_observation":"what would distinguish the competing interpretations",'
        '"operation":{"field_id":"ID from table","entities":["eligible handle"]}}\n'
        'OR {"action":"stop","reason":"why the remaining evidence cannot help"}\n'
        'AVAILABLE ORIGINAL OPERATIONS (each lists all eligible records; select at most three)\n'+
        operation_table(options))
    question_bound = mode in ('question_bound', 'evidence_first', 'fixed_order', 'legacy_bound')
    bound_question = feedback.get('decision',{}).get('remaining_question')
    if question_bound:
        assert isinstance(bound_question,str) and bound_question.strip()
        prompt = (state+'\nUNRESOLVED IDENTITY QUESTION\n'+bound_question+
            '\nFALLIBLE VERIFIER FEEDBACK (not new evidence)\n'+json.dumps(feedback,ensure_ascii=False)+
            '\nPREVIOUS EXECUTED QUERIES\n'+json.dumps(history,ensure_ascii=False)+
            '\nMAIN AGENT: choose a real field or original image operation that can answer the '
            'unresolved question above. This question is the current goal; do not substitute a '
            'different question. Consider all ten candidates and all four modalities equally. '
            'The table lists available fields, not their values. Choose field_id and one to three '
            'eligible entities first, then state in ONE sentence what possible observations from '
            'that exact operation would distinguish the identities. Do not assert an unseen answer. '
            'An attribute or relation reads the literal predicate named in its table row; it does '
            'not analyze pixels. An image operation opens original pixels. A name operation reads '
            'names. Select based on the meaning of the operation and question, not the ID number. '
            'The selected operation executes immediately; no unchanged-state verification is available. '
            'If no available operation can address the question, stop and explain why. '
            f'Remaining operations: {remaining}. '+
            ('This is the FIXED-ORDER CONTROL: only the next available modality is offered. ' if mode in FIXED_MODES else '')+
            'Select the concrete field ID itself, or none when no operation can help. Return exactly one form:\n'
            '{"field_id":"ID from table","entities":["eligible handle"],'
            '"expected_observation":"how an observation from that field can answer the question"}\n'
            'OR {"field_id":"none","reason":"why available observations cannot resolve this question"}\n'+
            operation_table(options))
    obj, trace = ask(model, env, prompt, query_schema(options, question_bound), 'planner')
    if question_bound and not trace['error']:
        if obj.get('field_id')=='none':
            obj['action'] = 'stop'
        else:
            obj['action'] = 'inspect'
            obj['operation'] = {'field_id':obj.get('field_id'), 'entities':obj.get('entities')}
            obj['question'] = bound_question
            trace['question_origin'] = 'verifier unresolved question, bound without rewriting'
    if trace['error']:
        return None, trace
    if obj.get('action') == 'stop':
        if not isinstance(obj.get('reason'), str) or not obj['reason'].strip():
            trace['error'] = 'Stop requires a reason'
        return None, trace
    op = obj.get('operation', {}); item = options.get(op.get('field_id'))
    entities = op.get('entities')
    if (obj.get('action') != 'inspect' or not item or not isinstance(entities, list)
        or not 1 <= len(entities) <= 3 or any(h not in item['entities'] for h in entities)
        or any(not isinstance(obj.get(k), str) or not obj[k].strip()
               for k in ('question', 'expected_observation'))):
        trace['error'] = 'Invalid executable query'; return None, trace
    action = {'action': 'inspect', 'entities': list(dict.fromkeys(entities)),
              'modalities': [item['modality']], 'question': obj['question'],
              'expected': obj['expected_observation'], 'offset': 0}
    if item['predicate'] is not None:
        action['predicate'] = item['predicate']
    trace['resolved_original_field'] = {**item, 'entities': action['entities']}
    return action, trace


def observation_state(env, model):
    state, _ = record_state(env, model)
    return hashlib.sha256((state+json.dumps(env.images(), ensure_ascii=False)).encode()).hexdigest()


def solve_active(row, index, model, folder, mode='adaptive', budget=4):
    assert mode in MODES
    env = make_env(row, index, model, folder)
    started = time.time(); begin = (model.calls, model.input_tokens, model.output_tokens)
    trace = []; history = []; used = set(); seen = set(); inspections = 0
    choice = 'U'; status = 'unresolved'
    old_unicode = getattr(model, 'unicode_json', False); model.unicode_json = True
    try:
        while True:
            state_id = observation_state(env, model)
            if state_id in seen:
                status = 'no_observation_progress'; break
            seen.add(state_id)
            choice, t = (legacy_verify(model, env) if mode in ('legacy_verifier','legacy_bound','legacy_joint','legacy_fixed')
                         else verify(model, env, evidence_first=mode in ('evidence_first','evidence_joint')))
            t['observation_state_sha256'] = state_id; trace.append(t)
            if t['error']:
                status = 'invalid_verification'; choice = 'U'; break
            if choice != 'U':
                status = 'verified'; break
            if mode in ('joint', 'evidence_joint', 'legacy_joint') or inspections >= budget:
                break
            action, p = query(model, env, t['action'], history, mode, used, budget-inspections)
            trace.append(p)
            if p['error']:
                status = 'invalid_plan'; break
            if action is None:
                break
            observation = env.inspect(action); inspections += 1
            trace.append({'kind': 'tool', 'action': action, 'observation': observation})
            if observation.get('error'):
                status = 'tool_error'; break
            used.update(action['modalities'])
            history.append({'action': action, 'observed_refs': [e['ref'] for e in observation['evidence']]})
            # Progress is defined by the actual visible input, including pixels.
            # Reopening earlier originals is legal if it changes that input.
        result = {'source': env.source, 'baseline_rank': list(env.baseline), 'rank': env.rank(choice),
            'status': status, 'mode': mode, 'trace': trace, 'handle_map': dict(env.handles),
            'evidence': deepcopy(env.ledger), 'inspections': inspections,
            'verification_calls': sum(t['kind']=='verifier' for t in trace),
            'calls': model.calls-begin[0], 'input_tokens': model.input_tokens-begin[1],
            'output_tokens': model.output_tokens-begin[2], 'seconds': time.time()-started,
            'candidate_count': 10, 'semantic_scope_is_model_judgment': True}
        assert len(result['rank']) == len(set(result['rank'])) == 10
        assert set(result['rank']) == set(row['baseline']['rank'])
        return result
    finally:
        model.unicode_json = old_unicode
