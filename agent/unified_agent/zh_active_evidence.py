"""Top10 ambiguity-driven evidence acquisition; frozen-model zh-en study.

The main policy runs before any stopping verifier. Concrete questions choose
original fields or pixels, observations remain separate from model hypotheses,
and every candidate stays eligible. Baselines deliberately isolate the gate.
"""
from copy import deepcopy
import hashlib
import json
import time

from .active_top10 import query, query_schema, options_for, evidence_refs
from .ambiguity import make_env
from .contrastive import record_state as raw_state, operation_table
from .deliberative import focused_images
from .environment import parse_object
from .grounded import object_schema, verify as old_verify

MODES = ('legacy_decomposed', 'semantic_joint', 'semantic_gate',
         'active_semantic', 'fixed_semantic', 'fact_active', 'fact_fixed', 'fact_joint', 'fact_gate',
         'direct_active', 'direct_fixed', 'direct_joint', 'direct_gate', 'direct_active_old_attribution',
         'single_active','single_gate','single_joint','single_fixed')
TEXT = {'type': 'string', 'minLength': 1}
RELATIONS = ('same', 'related', 'different', 'unknown')


def state(env, model, focus=None):
    if focus is None and getattr(env, 'source_last', False):
        # Keep all originals and candidates; place the query subject nearest
        # the decision/action instruction instead of before the candidate list.
        focus = list(env.priors) + ['SRC']
    text, refs = raw_state(env, model, focus)
    handle_order={h:i for i,h in enumerate(env.handles)}
    images = sorted((r for r in refs if env.ledger[r]['modality'] == 'image'),
                    key=lambda r:(handle_order[env.ledger[r]['entity']],r))
    metadata = {f'[{r}] {env.ledger[r]["text"]}' for r in images}
    text = '\n'.join(line for line in text.splitlines() if line not in metadata)
    if images:
        text += '\nIMAGE REFERENCES (addresses, not factual descriptions):\n'
        text += '\n'.join(f'[{r}] actual '+('thumbnail' if env.ledger[r]['preview'] else 'original')+
                          ' of '+env.ledger[r]['entity'] for r in images)
    return text, refs


def state_hash(env, model):
    text, _ = state(env, model)
    return hashlib.sha256((text+json.dumps(env.images())).encode()).hexdigest()


def ask(model, env, prompt, schema, kind, images=None):
    raw, meta = model.generate(prompt, env.images() if images is None else images,
                               schema=schema, max_tokens=768)
    try:
        action = parse_object(raw); error = None
    except (ValueError, TypeError) as exc:
        action = {}; error = str(exc)
    return action, {'kind': kind, 'action': action, 'raw': raw, 'model': meta, 'error': error}


def semantic_verify(model, env, folder, single_decision=False, explicit_subjects=False, unified_decision=False):
    """Independent raw-record comparisons, never given the planner's answer.

    Scores only implement argmax over semantic labels; not calibrated confidence.
    All ten records are tested. Relatedness is explicitly distinct from identity.
    """
    checks = []; verdicts = {}
    for handle in env.priors:
        text, _ = state(env, model, ['SRC', handle])
        prompt = (text+f'\nRECORD SUBJECT RELATION: SOURCE SRC and TARGET {handle}. '
            'Determine what relationship holds between the two RECORD SUBJECTS themselves. '
            'same: exactly the same entity, including cross-language names. '
            'related: distinct but associated subjects, such as a work and its creator, a club '
            'and its stadium, or an organization and its member. different: distinct subjects '
            'without an established identity. unknown: observations do not resolve identity. '
            'Interpret the original predicates and depicted objects: a record may mention or '
            'illustrate an associated subject. A shared topic, common appearance or category '
            'is not identifying evidence. Missing facts are unknown, not disagreement. Names, '
            'attributes, directed relations and actual images have equal standing. '
            'Read the source and target as separate records; never transfer facts between them. '
            'Answer only same, related, different, or unknown; do not generate a supporting story.')
        images = focused_images(env, ['SRC', handle], folder)
        scores, meta = model.choose(prompt, images, list(RELATIONS))
        highest = max(scores.values())
        winners = [label for label, score in scores.items() if score == highest]
        relation = winners[0] if len(winners) == 1 else 'unknown'
        verdicts[handle] = relation
        checks.append({'kind': 'candidate_check', 'handle': handle, 'relation': relation,
                       'scores': scores, 'model': meta, 'error': None})
    same = [h for h, relation in verdicts.items() if relation == 'same']
    summary = {'kind': 'verifier', 'action': {'relations': verdicts},
               'compared_handles': list(env.priors), 'accepted': False, 'error': None}
    if len(same) != 1:
        summary['action']['remaining_question'] = (
            'More than one record was judged identical; identify the unresolved distinction.' if same else
            'No record has established exact identity; identify a useful unresolved distinction.')
        summary['action']['decision'] = {'choice': 'U'}
        return 'U', checks+[summary]
    handle = same[0]
    if unified_decision:
        summary['accepted']=True
        summary['action']['decision']={'choice':handle}
        summary['decision_rule']='Unique same-subject decision across all ten raw-record comparisons; no second attribution veto.'
        return handle,checks+[summary]
    text, refs = state(env, model, ['SRC', handle])
    source_refs = evidence_refs(env, refs, 'SRC'); target_refs = evidence_refs(env, refs, handle)
    if not source_refs or not target_refs:
        summary['action']['decision'] = {'choice': 'U'}
        summary['action']['remaining_question'] = 'Identifying observations are missing for one record.'
        return 'U', checks+[summary]
    schema = object_schema({
        'source_ref': {'enum': source_refs}, 'target_ref': {'enum': target_refs},
        'source_scope': {'enum': ['self', 'related_entity', 'unclear']},
        'target_scope': {'enum': ['self', 'related_entity', 'unclear']},
        'comparison': {'enum': ['identifying_agreement', 'identity_conflict', 'shared_nondistinctive', 'insufficient']},
        'verdict': {'enum': ['same', 'different', 'uncertain']}})
    if single_decision:
        schema['properties'].pop('verdict');schema['required'].remove('verdict')
    prompt = (text+'\nEVIDENCE ATTRIBUTION CHECK. Independently assess exact identity of the '
        'two record subjects using observed evidence. Cite an actual observation from each. '
        'self means the identifying observation concerns that record subject; related_entity '
        'means it concerns an associated object; unclear means attribution is not established. '
        'identifying_agreement means both observations identify one subject; common category, '
        'topic, clothes, scenery or a shared associated object are shared_nondistinctive. '
        'Names, attributes, relations and pixels are peers. Inspect actual pixels for image '
        'references. URI provenance alone does not establish alignment. Missing or noisy '
        'values require interpretation, not automatic rejection. Return the specified JSON.')
    if explicit_subjects:
        prompt=(text+f'\nIDENTITY QUESTION: Does SOURCE {env.handles["SRC"]} denote the same '
            f'real-world entity as TARGET {env.handles[handle]}? SRC always denotes the source record. '
            'Select the most diagnostic observed item about EACH record subject. '
            'A strong identifying name, attribute, directed relation or actual image detail may suffice; '
            'no modality has priority. A translated name can identify the subject itself. '
            'self means the cited observation identifies or describes its own record subject; '
            'related_entity means it describes a distinct associated person, organization, place or work; '
            'unclear means ownership is not established. Read the predicate and role in context. '
            'identifying_agreement means the observations distinguish this identity, identity_conflict '
            'means incompatible identities, shared_nondistinctive means a common category or associated '
            'object, and insufficient means the observations cannot settle identity. Missing values '
            'are unknown; noisy values require interpretation. Inspect pixels when citing images. '
            'URI provenance is context, not a known alignment. Return the supplied JSON fields.')
    action, trace = ask(model, env, prompt, schema, 'attribution',
                        focused_images(env, ['SRC', handle], folder))
    valid = action.get('source_ref') in source_refs and action.get('target_ref') in target_refs
    if not trace['error'] and not valid:
        trace['error'] = 'Citation is invisible, wrong-owner or provenance-only'
    accepted = (not trace['error'] and valid and (single_decision or action.get('verdict') == 'same')
                and action.get('source_scope') == action.get('target_scope') == 'self'
                and action.get('comparison') == 'identifying_agreement')
    summary.update(accepted=bool(accepted), error=trace['error'])
    summary['action'].update(attribution=action, decision={'choice': handle if accepted else 'U'})
    if not accepted:
        summary['action']['remaining_question'] = 'The proposed identity lacks distinguishing evidence about both record subjects.'
    return handle if accepted else 'U', checks+[trace, summary]


def fact_plan(model, env, feedback, history, used, remaining, allow_verify, fixed=False):
    """Explicit ambiguity then scoped tool binding; no fabricated facts or pair pruning."""
    available = options_for(env, 'legacy_fixed' if fixed else 'adaptive', used) if remaining else {}
    query_branch = object_schema({'action': {'enum':['query']}, 'ambiguity':TEXT, 'question':TEXT,
        'expected_observation':TEXT,
        'entities':{'type':'array','items':{'enum':list(env.handles)},'minItems':1,'maxItems':3}})
    branches = [query_branch] if available else []
    if allow_verify:
        branches.append(object_schema({'action':{'enum':['verify']},'basis':TEXT}))
    branches.append(object_schema({'action':{'enum':['stop']},'reason':TEXT}))
    schema=object_schema({'source_subject':TEXT,'next':{'anyOf':branches}})
    text,_=state(env,model)
    available_modalities={m:sorted({h for v in available.values() if v['modality']==m for h in v['entities']})
                          for m in ('name','attribute','relation','image')}
    prompt=(text+'\nPRIOR FALLIBLE ASSESSMENTS\n'+json.dumps(feedback,ensure_ascii=True)+
        '\nEXECUTED QUERIES\n'+json.dumps(history,ensure_ascii=True)+
        '\nAMBIGUITY-DRIVEN MAIN AGENT. First state the SOURCE RECORD subject in a short phrase. '
        'Its picture and fields can concern associated objects; identify the record subject itself. '
        'Consider all ten candidates. If existing observations distinguish the identity, request '
        'verify with a specific observed basis. Otherwise identify the concrete competing '
        'interpretations, ask a neutral factual question, and select up to three records whose '
        'observations can resolve that distinction. Expected observations describe how DIFFERENT '
        'possible answers would change the identity decision, never assert the desired answer. '
        'Use meaningful subject names in the question. Tool selection follows in the next call. '
        'The records selected for querying do not remove any of the ten candidates. Name, '
        'attribute, directed relation and original image are equally legitimate evidence sources. '
        'An insufficient current preview is a reason to consider an available query, not by '
        'itself a reason to declare all unread records useless. Stop only if available observations '
        'cannot address the remaining distinction; explain that limitation. '
        +('Verification of unchanged observations is unavailable; choose query or stop. ' if not allow_verify else '')+
        ('Fixed-order control: currently only the listed modality can be read. ' if fixed else '')+
        'Keep every text field to one concise sentence. Return the schema, with next.action '
        'query and ambiguity/question/expected_observation/entities, or verify and basis, or stop and reason. '
        '\nAVAILABLE UNREAD MODALITIES AND RECORDS\n'+json.dumps(available_modalities))
    obj,trace=ask(model,env,prompt,schema,'diagnosis')
    if trace['error']:return 'error',None,[trace]
    decision=obj.get('next',{});move=decision.get('action')
    if not obj.get('source_subject'):
        trace['error']='Missing source subject';return 'error',None,[trace]
    if move in ('verify','stop'):
        if (move=='verify' and not allow_verify) or not decision.get('basis' if move=='verify' else 'reason'):
            trace['error']='Invalid diagnosis termination'
        return move,None,[trace]
    entities=decision.get('entities')
    if (move!='query' or not available or not isinstance(entities,list) or not 1<=len(entities)<=3
        or any(h not in env.handles for h in entities)
        or any(not isinstance(decision.get(k),str) or not decision[k].strip() for k in ('ambiguity','question','expected_observation'))):
        trace['error']='Invalid ambiguity question';return 'error',None,[trace]
    options={k:{**v,'entities':[h for h in v['entities'] if h in entities]} for k,v in available.items()}
    options={k:v for k,v in options.items() if v['entities']}
    fields=object_schema({'field_id':{'enum':list(options)+['none']}})
    focused,_=state(env,model,list(dict.fromkeys(['SRC']+entities)))
    prompt=(focused+'\nQUESTION TO EXECUTE (hypothesis, not observed fact)\n'+json.dumps(decision,ensure_ascii=True)+
        '\nSelect the ONE original operation that answers this exact question about these records. '
        'Use the literal predicate meaning and the table entities. Attribute names are literal '
        'fields, not instructions to analyze images. Image opens actual original pixels. '
        'Do not choose an unrelated property just because it is unread. none means no listed '
        'operation can answer this question. Return only field_id.\n'+operation_table(options))
    op,binding=ask(model,env,prompt,fields,'planner')
    if binding['error']:return 'error',None,[trace,binding]
    if op.get('field_id')=='none':return 'stop',None,[trace,binding]
    item=options.get(op.get('field_id'))
    if not item:
        binding['error']='Unknown scoped operation';return 'error',None,[trace,binding]
    action={'action':'inspect','entities':item['entities'],'modalities':[item['modality']],
        'question':decision['question'],'expected':decision['expected_observation'],'offset':0}
    if item['predicate'] is not None:action['predicate']=item['predicate']
    binding['resolved_original_field']=item
    return 'inspect',action,[trace,binding]


def legacy_verify(model, env):
    text, _ = state(env, model)
    handles = list(env.priors)
    schema = object_schema({'hypothesis': {'enum': handles}, 'rival': {'enum': handles}})
    action, proposal = ask(model, env, text+'\nOLD DECOMPOSED CONTROL: select the most plausible '
        'identity of SRC among ALL TEN records and a distinct competitive alternative. '
        'These are provisional hypotheses. Return hypothesis and rival.', schema, 'hypothesis')
    best, rival = action.get('hypothesis'), action.get('rival')
    if proposal['error'] or best not in handles or rival not in handles or best == rival:
        proposal['error'] = proposal['error'] or 'Invalid distinct hypothesis/rival'
        return 'U', [proposal]
    accepted, trace, _ = old_verify(model, env, best, rival, strict=True)
    trace['compared_handles'] = [best, rival]
    trace['action']['decision'] = {'choice': best if accepted else 'U'}
    return best if accepted else 'U', [proposal, trace]


def semantic_operations(options):
    """Literal tool names, preserving original predicate/direction and eligible owners."""
    result={}
    for item in options.values():
        label=item['modality']+': '+(item['predicate'] if item['predicate'] is not None else
                                    ('original_pixels' if item['modality']=='image' else 'labels_and_aliases'))
        assert label not in result
        result[label]=item
    return result


def direct_plan(model,env,feedback,history,used,remaining,allow_verify,fixed=False):
    """A semantic action choice, factual question, then literal tool binding."""
    available=options_for(env,'legacy_fixed' if fixed else 'adaptive',used) if remaining else {}
    operations=semantic_operations(available)
    catalog_text=json.dumps({k:v['entities'] for k,v in operations.items()},ensure_ascii=False)
    text,_=state(env,model)
    context=(text+'\nPRIOR FALLIBLE ASSESSMENTS\n'+json.dumps(feedback,ensure_ascii=False)+
        '\nPREVIOUS QUERIES\n'+json.dumps(history,ensure_ascii=False)+
        '\nAVAILABLE TOOLS (original field names and eligible records; values are not yet read)\n'+catalog_text)
    labels=(['inspect'] if operations else [])+(['verify'] if allow_verify else [])+['stop']
    prompt=(context+'\nChoose the next action for identifying SRC among all TEN records. '
        'inspect: an unanswered factual distinction can be addressed with an available tool. '
        'verify: currently observed evidence is ready for independent identity verification. '
        'stop: no remaining available observation can resolve the distinction. '
        'After a failed verification, ask what information is missing instead of repeating the same '
        'judgment. All ten remain eligible and every modality has equal standing. '
        'An attribute called image is a text field; image: original_pixels opens the actual image. '
        +('This control offers only its next available modality. ' if fixed else '')+
        'Return only one of: '+', '.join(labels)+'.')
    scores,meta=model.choose(prompt,env.images(),labels)
    best=max(scores.values());winners=[k for k,v in scores.items() if v==best]
    move=winners[0] if len(winners)==1 else 'stop'
    trace=[{'kind':'policy','action':{'action':move},'scores':scores,'model':meta,'error':None}]
    if move!='inspect':return move,None,trace
    question_schema=object_schema({'source_subject':TEXT,'ambiguity':TEXT,'question':TEXT,
        'expected_observation':TEXT,
        'entities':{'type':'array','items':{'enum':list(env.handles)},'minItems':1,'maxItems':3}})
    prompt=(context+'\nYour next action is to acquire evidence. State the source subject and ONE '
        'unresolved factual distinction between plausible identities, then ask ONE short neutral '
        'question whose answer can be read with an available tool. Ask for an observable fact, '
        'not for confirmation of a favored candidate or a general explanation of alignment. '
        'Describe how different possible answers distinguish the identities without asserting an '
        'unseen value. Select up to three records to inspect; this does not eliminate the other '
        'candidates. Use a short sentence per field. Return source_subject, ambiguity, question, '
        'expected_observation, entities in JSON.')
    question,t=ask(model,env,prompt,question_schema,'diagnosis');trace.append(t)
    entities=question.get('entities')
    if t['error']:return 'error',None,trace
    if (not isinstance(entities,list) or not 1<=len(entities)<=3 or any(h not in env.handles for h in entities)
        or any(not isinstance(question.get(k),str) or not question[k].strip()
               for k in ('source_subject','ambiguity','question','expected_observation'))):
        t['error']='Invalid factual question';return 'error',None,trace
    scoped={k:{**v,'entities':[h for h in v['entities'] if h in entities]} for k,v in operations.items()}
    scoped={k:v for k,v in scoped.items() if v['entities']}
    schema=object_schema({'tool':{'enum':list(scoped)+['none']}})
    prompt=('Bind this factual question to one original evidence tool. This call selects a tool; '
        'it does not answer the question or observe any new values. Choose by the literal modality '
        'and predicate meaning. Text fields such as attribute: image contain literal text, while '
        'image: original_pixels opens pixels. Do not substitute an unrelated field. none means no '
        'offered tool can answer this question.\nQUESTION\n'+json.dumps(question,ensure_ascii=False)+
        '\nTOOLS AND ELIGIBLE RECORDS\n'+json.dumps({k:v['entities'] for k,v in scoped.items()},ensure_ascii=False))
    obj,t=ask(model,env,prompt,schema,'planner',images=[]);trace.append(t)
    if t['error']:return 'error',None,trace
    if obj.get('tool')=='none':return 'stop',None,trace
    item=scoped.get(obj.get('tool'))
    if not item:
        t['error']='Unknown literal tool';return 'error',None,trace
    action={'action':'inspect','entities':item['entities'],'modalities':[item['modality']],
        'question':question['question'],'expected':question['expected_observation'],'offset':0}
    if item['predicate'] is not None:action['predicate']=item['predicate']
    t['resolved_original_field']=item
    return 'inspect',action,trace


def active_plan(model, env, feedback, history, used, remaining, allow_verify, fixed=False):
    options = options_for(env, 'legacy_fixed' if fixed else 'adaptive', used) if remaining else {}
    branches = query_schema(options)['anyOf'] if options else [object_schema({'action': {'enum': ['stop']}, 'reason': TEXT})]
    if allow_verify:
        branches = branches+[object_schema({'action': {'enum': ['verify']}, 'basis': TEXT})]
    text, _ = state(env, model)
    prompt = (text+'\nFALLIBLE PREVIOUS IDENTITY ASSESSMENTS (not facts)\n'+json.dumps(feedback, ensure_ascii=True)+
        '\nEXECUTED QUERIES (hypotheses remain distinct from returned observations)\n'+json.dumps(history, ensure_ascii=True)+
        '\nMAIN EVIDENCE AGENT. Identify SRC among all TEN candidate record subjects. '
        'Begin by considering whether observed facts distinguish the plausible identities. '
        'If a material ambiguity remains, formulate ONE neutral, answerable factual question '
        'and select an existing field or original image that would resolve it. State what '
        'different possible observations would distinguish the hypotheses, without inventing '
        'an unseen answer. Bind the question to field_id and eligible entities in the table. '
        'The exact operation runs immediately; the table exposes availability, not values. '
        'All ten candidates remain eligible after every query; do not permanently reject one '
        'based on a previous model judgment. Name, attribute, directed relation and original '
        'image are peers. Associated objects must not replace the record subject. '
        'Query meaning must match the chosen predicate and records. '
        +('If existing observations already distinguish a single identity, request verify and '
          'give a brief basis grounded in observed facts. A separate verifier will examine raw '
          'records, not your claimed answer. ' if allow_verify else
          'These observations were already checked; only a new evidence operation or stop is available. ')+
        'Choose stop if no available observation can resolve the ambiguity. No mandatory '
        'minimum number of queries. Use one short sentence per text field. '+
        ('This is the FIXED-ORDER CONTROL; the table offers only its next available modality. ' if fixed else '')+
        f'Remaining tool operations: {remaining}. Return one form:\n'+
        '{"action":"inspect","question":"concrete factual question","expected_observation":"how answers discriminate",'
        '"operation":{"field_id":"table ID","entities":["eligible handle"]}}\n'+
        ('OR {"action":"verify","basis":"observed distinguishing facts"}\n' if allow_verify else '')+
        'OR {"action":"stop","reason":"why no available evidence can help"}\n'+operation_table(options))
    action, trace = ask(model, env, prompt, {'anyOf': branches}, 'planner')
    if trace['error']:
        return 'error', None, trace
    move = action.get('action')
    if move in ('verify', 'stop'):
        if (move == 'verify' and not allow_verify) or not action.get('basis' if move == 'verify' else 'reason'):
            trace['error'] = 'Invalid stopping or verification action'
        return move, None, trace
    op = action.get('operation', {}); item = options.get(op.get('field_id')); entities = op.get('entities')
    if (move != 'inspect' or not item or not isinstance(entities, list) or not 1 <= len(entities) <= 3
        or any(h not in item['entities'] for h in entities)
        or any(not isinstance(action.get(k), str) or not action[k].strip() for k in ('question', 'expected_observation'))):
        trace['error'] = 'Invalid executable evidence question'
        return 'error', None, trace
    resolved = {'action': 'inspect', 'entities': list(dict.fromkeys(entities)), 'modalities': [item['modality']],
                'question': action['question'], 'expected': action['expected_observation'], 'offset': 0}
    if item['predicate'] is not None:
        resolved['predicate'] = item['predicate']
    trace['resolved_original_field'] = {**item, 'entities': resolved['entities']}
    return 'inspect', resolved, trace


def solve(row, index, model, folder, mode, budget=4):
    assert mode in MODES
    env = make_env(row, index, model, folder)
    started = time.time(); begin = (model.calls, model.input_tokens, model.output_tokens)
    previous = getattr(model, 'unicode_json', False); model.unicode_json = True
    trace = []; seen = set(); history = []; feedback = {}; used = set()
    inspections = 0; choice = 'U'; status = 'unresolved'; no_query_rank = None; must_verify = False
    proactive = mode in ('active_semantic', 'fixed_semantic', 'fact_active', 'fact_fixed',
                         'direct_active','direct_fixed','direct_active_old_attribution','single_active','single_fixed')
    direct=mode.startswith(('direct_','single_'))
    factored = mode.startswith('fact_') or direct
    try:
        while True:
            before = state_hash(env, model)
            action = None
            if proactive and not must_verify and inspections < budget:
                if factored:
                    plan=direct_plan if direct else fact_plan
                    move,action,plans=plan(model,env,feedback,history,used,budget-inspections,
                                               before not in seen,fixed=mode in ('fact_fixed','direct_fixed','single_fixed'))
                else:
                    move, action, t = active_plan(model, env, feedback, history, used,
                        budget-inspections, before not in seen, fixed=mode == 'fixed_semantic')
                    plans=[t]
                trace.extend(plans)
                if any(t['error'] for t in plans):
                    status = 'invalid_plan'; break
                if move == 'stop':
                    break
            else:
                move = 'verify'
            if move == 'verify':
                if before in seen:
                    status = 'no_observation_progress'; break
                seen.add(before)
                must_verify = False
                if mode=='legacy_decomposed':choice,records=legacy_verify(model,env)
                elif mode.startswith('single_'):choice,records=semantic_verify(model,env,folder/'focused',unified_decision=True)
                elif direct:choice,records=semantic_verify(model,env,folder/'focused',single_decision=True,
                                                          explicit_subjects=mode!='direct_active_old_attribution')
                elif factored:choice,records=semantic_verify(model,env,folder/'focused',single_decision=True)
                else:choice,records=semantic_verify(model,env,folder/'focused')
                for t in records:
                    if t['kind'] == 'verifier':
                        t['observation_state_sha256'] = before
                trace.extend(records)
                if not inspections:
                    no_query_rank = env.rank(choice)
                if any(t.get('error') for t in records):
                    status = 'invalid_verification'; choice = 'U'; break
                feedback = records[-1]['action']
                if choice != 'U':
                    status = 'verified'; break
                if mode in ('semantic_joint','fact_joint','direct_joint','single_joint') or inspections >= budget:
                    break
                if proactive:
                    continue
                if factored:
                    plan=direct_plan if direct else fact_plan
                    _,action,plans=plan(model,env,feedback,history,used,budget-inspections,False)
                else:
                    action,t=query(model,env,feedback,history,'legacy_verifier',used,budget-inspections);plans=[t]
                trace.extend(plans)
                if any(t['error'] for t in plans):
                    status = 'invalid_plan'; break
                if action is None:
                    break
            observation = env.inspect(action); inspections += 1
            trace.append({'kind': 'tool', 'action': action, 'observation': observation})
            if observation.get('error'):
                status = 'tool_error'; break
            used.update(action['modalities'])
            history.append({'question': action['question'], 'operation': {k:v for k,v in action.items() if k not in ('expected','question')},
                            'observed_refs': [e['ref'] for e in observation['evidence']]})
            if state_hash(env, model) == before:
                status = 'no_observation_progress'; break
            must_verify = True
        result = {'source': env.source, 'baseline_rank': list(env.baseline), 'rank': env.rank(choice),
            'status': status, 'mode': mode, 'trace': trace, 'handle_map': dict(env.handles),
            'evidence': deepcopy(env.ledger), 'inspections': inspections, 'no_query_rank': no_query_rank,
            'verification_calls': sum(t['kind'] == 'verifier' for t in trace),
            'calls': model.calls-begin[0], 'input_tokens': model.input_tokens-begin[1],
            'output_tokens': model.output_tokens-begin[2], 'seconds': time.time()-started,
            'candidate_count': 10, 'labels_loaded': False, 'semantic_scope_is_model_judgment': True}
        assert len(result['rank']) == len(set(result['rank'])) == 10
        assert set(result['rank']) == set(row['baseline']['rank'])
        return result
    finally:
        model.unicode_json = previous
