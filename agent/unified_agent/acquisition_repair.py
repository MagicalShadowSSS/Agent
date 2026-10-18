"""Persistent original observations and executable identity questions.

The reader is unchanged from acquisition146. Generated conflict descriptions
are fallible policy memory, never facts supplied to that reader. Stop remains
a model action with the same four-tool maximum; no learned stopping policy.
"""
from copy import copy, deepcopy
import json
import time

from .acquisition146 import reader, structured
from .active_top10 import options_for
from .ambiguity import make_env
from .grounded import object_schema
from .json_protocol import SHORT_TEXT
from .zh_active_evidence import semantic_operations, state, state_hash
from .type_guided import (POLICIES, TYPE_TEXT, REVIEW_GUIDANCE, PLAN_GUIDANCE,
                          PREFERENCE, candidate_type_schema)
from .review_improvements import (settings, options, type_update_schema, merge_types,
                                  COMPACT_GUIDANCE, FRESH_GUIDANCE, PENDING_GUIDANCE)


def ref_array(refs, minimum=0):
    refs = sorted(set(refs))
    # An empty enum is not a valid grammar; an empty result has no item choices.
    return {'type': 'array', 'items': {'enum': refs} if refs else {'type': 'string'},
            'minItems': minimum, 'maxItems': len(refs)}


class EvidenceMemory:
    """Maintain acquired and accepted originals separately, without image eviction."""
    def __init__(self, env):
        self.env = env
        self.accepted = set()

    def view(self, additional=()):
        allowed = self.accepted | set(additional)
        assert allowed <= set(self.env.ledger)
        view = copy(self.env)
        view.ledger = {r: item for r, item in self.env.ledger.items()
                       if item['preview'] or r in allowed}
        # Derive attachments from accepted originals, not the last tool call.
        view.recent_images = [h for h in self.env.handles if h+'.image.full' in allowed]
        return view

    def inspect(self, action):
        previous = set(self.env.recent_images)
        observation = self.env.inspect(action)
        if observation.get('error'):
            raise ValueError(observation)
        acquired = previous | set(self.env.recent_images)
        self.env.recent_images = [h for h in self.env.handles if h in acquired]
        return observation

    def accept(self, observation, refs):
        available = {item['ref'] for item in observation['evidence']}
        if not set(refs) <= available:
            raise ValueError('Cannot accept a reference absent from the actual receipt')
        self.accepted.update(refs)

    def pending_refs(self, exclude=()):
        return sorted(r for r, item in self.env.ledger.items()
                      if not item['preview'] and r not in self.accepted and r not in exclude)

    def reconsider(self, refs):
        if not set(refs) <= set(self.pending_refs()):
            raise ValueError('Reassessment requires an acquired, unaccepted original')
        self.accepted.update(refs)


def receipt_rows(env, observation):
    """Expose direction/owner literally; no entity or predicate is inferred."""
    result = []
    for item in observation['evidence']:
        value = {k: item[k] for k in ('ref', 'entity', 'modality', 'uri', 'text')}
        if item['modality'] == 'relation':
            field, _, neighbor = item['text'].partition(': ')
            direction, _, predicate = field.partition(' ')
            if direction in ('in', 'out'):
                value['relation_roles'] = {
                    'predicate': predicate,
                    'subject': item['entity'] if direction == 'out' else neighbor,
                    'object': neighbor if direction == 'out' else item['entity'],
                    'neighbor_text_is_raw_not_an_alignment': True}
        result.append(value)
    return result


def review(model, memory, previous=None, observation=None, request=None):
    """One verifier role checks record identity and returns the remaining issue.

    It does not receive the planner's current candidate, proposed answer or
    rationale. The old issue is explicitly a revisable hypothesis.
    """
    improvements = options(memory.env)
    refs = [v['ref'] for v in observation['evidence']] if observation else []
    pending = memory.pending_refs(refs) if improvements['revisit_pending'] else []
    view = memory.view(refs + pending)
    text, visible = state(view, model)
    source_refs = [r for r in visible if view.ledger[r]['entity'] == 'SRC'
                   and view.ledger[r]['location'] != 'record/uri']
    policy = getattr(memory.env, 'planning_policy', 'original')
    subject_fields = {'description': SHORT_TEXT, 'refs': ref_array(source_refs)}
    if policy != 'original':
        subject_fields = {'possible_type': TYPE_TEXT, **subject_fields}
    subject = object_schema(subject_fields)
    issue = object_schema({
        'status': {'enum': ['open', 'resolved', 'insufficient']},
        'candidates': ref_array(list(view.priors)),
        'distinction': SHORT_TEXT, 'needed_fact': SHORT_TEXT})
    properties = {'source_subject': subject}
    compact_types = policy != 'original' and previous is not None and improvements['compact_types']
    if policy != 'original':
        if compact_types:
            properties['type_update'] = type_update_schema(view.priors)
        else:
            properties['candidate_types'] = candidate_type_schema(view.priors)
    prompt = (text + '\nIDENTITY EVIDENCE REVIEW. Identify what SRC itself denotes, '
        'using source observations and keeping the record subject distinct from '
        'participants or objects mentioned in its fields and pictures. All ten '
        'candidates remain eligible. Describe the unresolved distinction between '
        'candidate identities and the missing observation needed to resolve it. '
        'If it is already resolved, say so; do not invent a new ambiguity. '
        'Citations establish provenance, not truth. Interpret incoming relations '
        'with their recorded subject/object direction. Use all modalities on equal terms. '
        'Return short descriptions; no final candidate selection is requested.\n')
    if policy != 'original':
        prompt += REVIEW_GUIDANCE
    if previous:
        prior = previous
        if improvements['fresh_receipt']:
            prior = {k: v for k, v in previous.items()
                     if k not in ('observation_assessment', 'pending_assessment')}
        prompt += '\nPREVIOUS FALLIBLE IDENTITY ISSUE (revise from raw observations):\n' + json.dumps(prior, ensure_ascii=False)
    if compact_types:
        prompt += COMPACT_GUIDANCE
    if observation is not None:
        rejected = {'status': {'enum': ['context_only', 'unusable']}}
        if policy == 'original' and not improvements['safe_rejection']:
            rejected['usable_refs'] = ref_array([])
        # LMFE 0.10.11 can admit a first array item despite maxItems=0.
        # New runs use the safe structure for every policy. The legacy branch
        # is available only for explicit historical reproduction.
        decisions = [object_schema(rejected)]
        if refs:
            decisions.insert(0, object_schema({'status': {'enum': ['identity_evidence']},
                                               'usable_refs': ref_array(refs, minimum=1)}))
        properties['observation_assessment'] = object_schema({
            'observed_fact': SHORT_TEXT, 'decision': {'anyOf': decisions}})
        if improvements['fresh_receipt']:
            assessment_schema = properties.pop('observation_assessment')
            properties = {'observation_assessment': assessment_schema, **properties}
            prompt += FRESH_GUIDANCE
        prompt += ('\nEXECUTED REQUEST (literal owners and field, not an assumed answer):\n' +
            json.dumps(request, ensure_ascii=False) + '\nACTUAL RETURNED ORIGINALS:\n' +
            json.dumps(receipt_rows(view, observation), ensure_ascii=False) +
            '\nFirst state what the returned originals actually say about their '
            'record owners. Then classify their use: identity_evidence means '
            'the cited fact helps establish or contradict an exact identity '
            'among the candidates; context_only means it describes context '
            'without distinguishing identities; unusable means no interpretable '
            'fact was obtained. A fact may be relevant even if the previous '
            'issue was thought resolved. Missing evidence is unknown, not a '
            'contradiction. Cite only returned originals; context_only/unusable '
            'have no usable_refs. Update the remaining identity distinction '
            'about SRC itself, without substituting a mentioned participant '
            'for the record owner.\n')
    if pending:
        properties['pending_assessment'] = {'anyOf': [
            object_schema({'status': {'enum': ['unchanged']}}),
            object_schema({'status': {'enum': ['accept']},
                           'usable_refs': ref_array(pending, minimum=1)})]}
        prompt += PENDING_GUIDANCE + '\nPREVIOUSLY ACQUIRED, NOT ACCEPTED ORIGINALS:\n' + json.dumps(
            receipt_rows(view, {'evidence': [{'ref': r, **view.ledger[r]} for r in pending]}),
            ensure_ascii=False)
    properties['issue'] = issue
    if policy == 'original':
        result, trace = structured(model, prompt, view.images(), object_schema(properties), 'identity_review')
    else:
        result, trace = structured(model, prompt, view.images(), object_schema(properties),
                                   'identity_review', max_tokens=1536)
        if compact_types:
            result = merge_types(result, previous, view.priors)
            trace['effective_candidate_types'] = deepcopy(result['candidate_types'])
        assert set(result['candidate_types']) == set(view.priors)
        trace['planning_policy'] = policy
    assert set(result['source_subject']['refs']) <= set(source_refs)
    assert set(result['issue']['candidates']) <= set(view.priors)
    accepted = []
    if observation is not None:
        assessment = result['observation_assessment']['decision']
        usable_refs = assessment.get('usable_refs', [])
        assert set(usable_refs) <= set(refs)
        if assessment['status'] == 'identity_evidence':
            assert usable_refs, 'Identity evidence requires actual references'
            accepted = usable_refs
        else:
            assert not usable_refs, 'Non-evidence cannot carry accepted references'
    accepted_current = list(accepted)
    accepted_pending = []
    if pending:
        assessment = result['pending_assessment']
        if assessment['status'] == 'accept':
            accepted_pending = assessment['usable_refs']
            assert accepted_pending and set(accepted_pending) <= set(pending)
        else:
            assert assessment['status'] == 'unchanged' and not assessment.get('usable_refs')
        accepted = list(dict.fromkeys(accepted + accepted_pending))
    trace.update(actual_returned_refs=refs, accepted_refs=accepted,
                 observation_state_sha256=state_hash(view, model))
    if improvements['revisit_pending']:
        trace.update(pending_refs=pending, accepted_current_refs=accepted_current,
                     accepted_pending_refs=accepted_pending)
    return result, accepted, trace


def bind_request(env, tools, operation, issue):
    """The read question is constructed from executable addresses, not free text.

    In particular, a generated question cannot ask to read B while executing
    against A. The semantic usefulness of that operation is still fallible.
    """
    item = tools[operation['tool']]
    owners = list(dict.fromkeys(operation['entities']))
    if not owners or len(owners) > 3 or not set(owners) <= set(item['entities']):
        raise ValueError('Unavailable operation owner')
    field = item['predicate'] or item['modality']
    request = {'identity_subject': 'SRC', 'record_owners': owners,
        'owner_uris': {h: env.handles[h] for h in owners},
        'modality': item['modality'], 'literal_field': field,
        'identity_distinction': issue['distinction'],
        'missing_observation': issue['needed_fact']}
    # Tool argument is only the literal requested field: generated hypotheses
    # cannot select values via fabricated answer words in lexical retrieval.
    question = 'Read ' + field + ' from record owners ' + ', '.join(owners) + '.'
    action = {'action': 'inspect', 'entities': owners, 'modalities': [item['modality']],
              'question': question, 'offset': 0}
    if item['predicate'] is not None:
        action['predicate'] = item['predicate']
    return action, request


def plan(model, memory, feedback, history, current, remaining):
    tools = semantic_operations(options_for(memory.env, 'adaptive', set()))
    if not tools:
        return None, None, {'kind': 'planner', 'reason': 'No unread operations', 'error': None}
    text, _ = state(memory.view(), model)
    branches = [object_schema({'tool': {'enum': [key]},
        'entities': {'type': 'array', 'items': {'enum': value['entities']},
                     'minItems': 1, 'maxItems': min(3, len(value['entities']))}})
        for key, value in tools.items()]
    schema = object_schema({'next': {'anyOf': [
        object_schema({'action': {'enum': ['stop']}, 'reason': SHORT_TEXT}),
        object_schema({'action': {'enum': ['inspect']},
                       'operation': {'anyOf': branches}, 'relevance': SHORT_TEXT})]}})
    prompt = (text + '\nFALLIBLE CURRENT CHOICE: ' + current +
        '\nPERSISTENT VERIFIER FEEDBACK (interpretations, not additional facts):\n' +
        json.dumps(feedback, ensure_ascii=False) + '\nPREVIOUS QUERIES AND UPDATES:\n' +
        json.dumps(history, ensure_ascii=False) + '\nUNREAD ORIGINAL OPERATIONS:\n' +
        json.dumps({key: value['entities'] for key, value in tools.items()}, ensure_ascii=False) +
        '\nChoose a literal field or original image and its record owners that '
        'can resolve the remaining identity distinction about SRC. Explain the '
        'connection between the field meaning and that distinction. The source '
        'record remains the object to identify; reading an associated object '
        'does not change the alignment target. Requests are executed exactly '
        'on the selected owners and field. All ten candidates remain eligible '
        'and no modality has priority. Stop when observed evidence resolves '
        'the competing interpretations or no remaining operation can address '
        'them. Do not query merely to use the budget. '
        f'Remaining operations: {remaining}.')
    policy = getattr(memory.env, 'planning_policy', 'original')
    if policy != 'original':
        prompt += PLAN_GUIDANCE
    if policy == 'type_prefer':
        prompt += PREFERENCE
    result, trace = structured(model, prompt, memory.view().images(), schema, 'planner')
    nxt = result['next']
    if nxt['action'] == 'stop':
        return None, None, trace
    action, request = bind_request(memory.env, tools, nxt['operation'], feedback['issue'])
    trace.update(resolved_operation=action, bound_request=request)
    return action, request, trace


def solve(row, index, model, folder, baseline, budget=4, source_last=False,
          planning_policy='original', improvements=None):
    if planning_policy not in POLICIES:
        raise ValueError('Unknown planning policy: '+planning_policy)
    env = make_env(row, index, model, folder)
    env.source_last = source_last
    env.planning_policy = planning_policy
    env.review_improvements = settings(improvements)
    memory = EvidenceMemory(env)
    assert state_hash(memory.view(), model) == baseline['initial_state_sha256']
    started = time.time()
    counts = model.calls, model.input_tokens, model.output_tokens
    current = baseline['choice']
    cache = {baseline['initial_state_sha256']: current}
    feedback, _, first = review(model, memory)
    trace, history, inspections = [first], [], 0
    for step in range(budget):
        action, request, t = plan(model, memory, feedback, history, current, budget-step)
        trace.append(t)
        if action is None:
            break
        before = state_hash(memory.view(), model)
        observation = memory.inspect(action)
        inspections += 1
        trace.append({'kind': 'tool', 'action': action, 'bound_request': request,
                      'observation': observation})
        updated, accepted, t = review(model, memory, feedback, observation, request)
        trace.append(t)
        returned_refs = {item['ref'] for item in observation['evidence']}
        memory.accept(observation, [r for r in accepted if r in returned_refs])
        reconsidered = [r for r in accepted if r not in returned_refs]
        if reconsidered:
            assert env.review_improvements['revisit_pending']
            assert set(reconsidered) <= set(t['pending_refs'])
            memory.reconsider(reconsidered)
        view = memory.view()
        digest = state_hash(view, model)
        if not accepted:
            assert digest == before, 'Rejected receipt changed accepted observation state'
        if digest not in cache:
            current, t = reader(model, view)
            trace.append(t)
            cache[digest] = current
        else:
            current = cache[digest]
            trace.append({'kind': 'reader_cache', 'choice': current,
                          'observation_state_sha256': digest, 'error': None})
        history.append({'request': request, 'previous_issue': feedback['issue'],
                        'assessment': updated['observation_assessment'],
                        'updated_issue': updated['issue'], 'accepted_refs': accepted})
        if 'pending_assessment' in updated:
            history[-1]['pending_assessment'] = updated['pending_assessment']
        feedback = updated
    result = {'source': row['source'], 'method': 'persistent_bound_feedback',
        'planning_policy': planning_policy,
        'review_improvements': env.review_improvements,
        'record_order': 'candidates_then_source' if source_last else 'source_then_candidates',
        'rank': env.rank(current), 'choice': current, 'initial_choice': baseline['choice'],
        'initial_rank': list(baseline['rank']), 'handle_map': env.handles,
        'inspections': inspections, 'trace': trace, 'issue_history': history,
        'accepted_refs': sorted(memory.accepted), 'evidence': deepcopy(env.ledger),
        'final_feedback': feedback, 'candidate_count': 10, 'labels_loaded': False,
        'status': 'budget_reached' if inspections == budget else 'stopped',
        'calls': model.calls-counts[0], 'input_tokens': model.input_tokens-counts[1],
        'output_tokens': model.output_tokens-counts[2], 'seconds': time.time()-started,
        'reader_calls': sum(t['kind'] == 'reader' for t in trace)}
    assert len(result['rank']) == 10 and set(result['rank']) == set(baseline['rank'])
    if not inspections:
        assert result['rank'] == baseline['rank']
    return result
