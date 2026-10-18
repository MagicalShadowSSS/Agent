"""Modality selection with fixed evidence checks; no model-generated questions."""
from copy import deepcopy

from .modality_agents import LABELS, ROLES, TYPES, RoleClient, digest, enum, obj, refs_schema
from .modality_agents_v2 import Evidence as AvailableEvidence, SHORT

STATUSES = ['support', 'conflict', 'mixed', 'uncertain']
FIXED_TASKS = {
    role: f'Using the supplied {kind}, names and tentative types, identify evidence supporting '
          'or conflicting with SRC and each candidate referring to the same entity. '
          'Summarize the evidence and report mixed or uncertain when appropriate.'
    for role, kind in [('attribute', 'original attributes'), ('relation', 'original relations'),
                       ('visual', 'actual images')]
}


class Evidence(AvailableEvidence):
    def retrieve(self, action):
        if set(action) != {'agent', 'candidates', 'types'}:
            raise ValueError('Fixed evidence checks accept only agent, candidates and types')
        self.validate_operation(action)
        owners, role = ['SRC']+action['candidates'], action['agent']
        task = {'instruction': FIXED_TASKS[role], 'source': {
            'handle': 'SRC', 'name': self.names['SRC'], 'type': action['types']['SRC']},
            'candidates': [{'handle': h, 'name': self.names[h], 'type': action['types'][h]}
                           for h in action['candidates']]}
        # Only observed names drive lexical retrieval, never instructions or generated facts.
        query = ' '.join(n for h in owners for n in self.names[h])
        records, audit, images = {}, {}, []
        for h in owners:
            if role == 'visual':
                records[h] = [{'id': h+'.image.full'}]
                images.append((h+'.image.full', self.images[h]))
            else:
                selected, audit[h] = self.indices[h, role].search(query, 10)
                records[h] = [self.public(item) for item in selected]
        return {'TASK': task, 'RECORDS': records}, images, audit


def planner_schema(refs, available):
    finish = obj({'action': enum(['finish']), 'choice': enum(LABELS), 'refs': refs_schema(refs)})
    branches = [finish]
    if available:
        operations = [obj({'agent': enum([m]), 'candidates': {
            'type': 'array', 'items': enum(candidates), 'minItems': 1,
            'maxItems': min(3, len(candidates))}}) for m, candidates in available.items()]
        branches.append(obj({'action': enum(['inspect']), 'operation': {'anyOf': operations}}))
    return obj({'types': obj({h: enum(TYPES) for h in ['SRC']+LABELS}),
                'assessment': SHORT, 'current_choice': enum(LABELS),
                'next': {'anyOf': branches} if available else finish})


def reader_schema(candidates, records):
    branches = []
    for h in candidates:
        refs = [r['id'] for owner in ['SRC', h] for r in records[owner]]
        branches.append(obj({'candidate': enum([h]),
            'summary': {'type': 'string', 'minLength': 1, 'maxLength': 400},
            'evidence_status': enum(STATUSES), 'refs': refs_schema(refs)}))
    return obj({'explanations': {'type': 'array', 'items': {'anyOf': branches},
                                'minItems': len(candidates), 'maxItems': len(candidates)}})


def validate_reader(result, candidates, records):
    actual = [item['candidate'] for item in result['explanations']]
    if len(actual) != len(set(actual)) or set(actual) != set(candidates):
        raise ValueError('Reader must explain each requested candidate once')
    for item in result['explanations']:
        allowed = {r['id'] for h in ['SRC', item['candidate']] for r in records[h]}
        if not set(item['refs']) <= allowed:
            raise ValueError('Reader cited evidence outside the compared pair')
        if item['evidence_status'] not in STATUSES or not item['summary'].strip():
            raise ValueError('Invalid evidence status or empty explanation')
        if item['evidence_status'] != 'uncertain' and not item['refs']:
            raise ValueError('A directional evidence assessment requires supplied evidence refs')


def solve(evidence, preview, images, client, max_inspections=2, on_event=None):
    if max_inspections < 0:
        raise ValueError('Negative inspection budget')
    results, trace, extra_images = [], [], []
    visible = {r['id']: deepcopy(r) for record in preview.values()
               for key in ('attributes', 'relations', 'images') for r in record[key]}
    read_ids, used_modalities = set(visible), set()
    inspections, current, initial_choice = 0, None, None

    def emit(event):
        trace.append(event)
        if on_event:
            on_event(event)

    while True:
        available = {m: candidates for m, candidates in evidence.available().items()
                     if m not in used_modalities} if inspections < max_inspections else {}
        payload = {'RECORDS': preview, 'CURRENT_JUDGMENT': current, 'AGENT_RESULTS': results,
                   'AVAILABLE_EVIDENCE': available,
                   'AVAILABLE_ACTIONS': {'inspect': bool(available), 'finish': True}}
        decision = client('planner', payload, images+extra_images, planner_schema(visible, available))
        if set(decision) != {'types', 'assessment', 'current_choice', 'next'}:
            raise ValueError('Planner must judge and route without generating a question')
        if decision['current_choice'] not in LABELS:
            raise ValueError('Invalid current choice')
        if set(decision['types']) != set(['SRC']+LABELS) or not set(decision['types'].values()) <= set(TYPES):
            raise ValueError('Invalid type hypotheses')
        current = {'choice': decision['current_choice'], 'assessment': decision['assessment']}
        if initial_choice is None:
            initial_choice = current['choice']
        action = decision['next']
        emit({'kind': 'planner', 'decision': decision, 'action': action,
              'current_judgment': deepcopy(current), 'available': available})
        if action['action'] == 'finish':
            if action['choice'] != current['choice'] or not set(action['refs']) <= set(visible):
                raise ValueError('Final answer disagrees with current judgment or cites unavailable evidence')
            target = evidence.handles[action['choice']]
            return {'choice': action['choice'], 'rank': [target]+[u for u in evidence.rank if u != target],
                    'handle_map': evidence.handles, 'inspections': inspections, 'trace': trace,
                    'initial_choice': initial_choice, 'current_judgment': current,
                    'evidence': visible, 'initial_preview_sha256': digest(preview),
                    'max_inspections': max_inspections, 'labels_loaded': False, 'protocol_version': 3,
                    'modality_reuse': False, 'inspection_task': 'fixed_evidence_check'}
        if action['action'] != 'inspect' or not available:
            raise ValueError('Inspection is disabled by budget or evidence availability')
        if set(action) != {'action', 'operation'} or set(action['operation']) != {'agent', 'candidates'}:
            raise ValueError('Inspection selects modality and candidates only')
        operation = {**action['operation'], 'types': decision['types']}
        if operation['agent'] not in available:
            raise ValueError('Modality is unavailable or has already been inspected for this SRC')
        payload, acquired_images, retrieval = evidence.retrieve(operation)
        inspections += 1
        used_modalities.add(operation['agent'])
        supplied = {r['id']: r for records in payload['RECORDS'].values() for r in records}
        emit({'kind': 'retrieval', 'agent': operation['agent'], 'inspection': inspections,
              'instruction': payload['TASK']['instruction'], 'records': payload['RECORDS'],
              'retrieval': retrieval, 'new_evidence_ids': sorted(set(supplied)-read_ids),
              'images': acquired_images})
        result = client(operation['agent'], payload, acquired_images,
                        reader_schema(operation['candidates'], payload['RECORDS']))
        validate_reader(result, operation['candidates'], payload['RECORDS'])
        cited = {r for item in result['explanations'] for r in item['refs']}
        visible.update({r: supplied[r] for r in cited})
        read_ids.update(supplied)
        for label, path in acquired_images:
            if label in cited and (label, path) not in extra_images:
                extra_images.append((label, path))
        receipt = {'agent': operation['agent'], 'candidates': operation['candidates'],
                   'instruction': payload['TASK']['instruction'], 'result': result,
                   'evidence': [supplied[r] for r in sorted(cited)]}
        results.append(receipt)
        emit({'kind': 'reader', **receipt})
