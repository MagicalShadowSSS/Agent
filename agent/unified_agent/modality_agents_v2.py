"""Evidence availability masks, explicit judgments, and scoped evidence explanations."""
from copy import deepcopy

from .modality_agents import (Evidence as BaseEvidence, LABELS, ROLES, TYPES,
                             RoleClient, digest, enum, obj, refs_schema)

SHORT = {'type': 'string', 'minLength': 1, 'maxLength': 240}


class Evidence(BaseEvidence):
    def available(self):
        """Only allow pairs with a real observation on both SRC and the candidate."""
        def has(h, modality):
            if modality == 'visual':
                from pathlib import Path
                return h in self.images and Path(self.images[h]).is_file()
            field = 'value' if modality == 'attribute' else 'neighbor'
            return any(r[field].strip() for r in self.indices[h, modality].rows)
        return {m: candidates for m in ROLES[1:] if has('SRC', m)
                if (candidates := [h for h in LABELS if has(h, m)])}

    def validate_operation(self, action):
        available = self.available()
        modality, candidates = action['agent'], action['candidates']
        if modality not in available or not 1 <= len(candidates) <= 3:
            raise ValueError('Unavailable modality or invalid comparison size')
        if len(set(candidates)) != len(candidates) or not set(candidates) <= set(available[modality]):
            raise ValueError('Candidate lacks the requested modality or is duplicated')

    def retrieve(self, action):
        self.validate_operation(action)
        return super().retrieve(action)


def planner_schema(refs, available):
    finish = obj({'action': enum(['finish']), 'choice': enum(LABELS), 'refs': refs_schema(refs)})
    branches = [finish]
    if available:
        operations = [obj({'agent': enum([m]), 'candidates': {
            'type': 'array', 'items': enum(candidates), 'minItems': 1,
            'maxItems': min(3, len(candidates))}}) for m, candidates in available.items()]
        inspect = obj({'action': enum(['inspect']), 'question': SHORT,
                       'operation': {'anyOf': operations}})
        branches.append(inspect)
    # The decoder enforces this order: interpretation before commitment and routing.
    return obj({'types': obj({h: enum(TYPES) for h in ['SRC']+LABELS}),
                'assessment': SHORT, 'current_choice': enum(LABELS),
                'unresolved': {'type': 'string', 'maxLength': 240},
                'next': {'anyOf': branches} if available else finish})


def reader_schema(candidates, records):
    branches = []
    for h in candidates:
        refs = [r['id'] for owner in ['SRC', h] for r in records[owner]]
        branches.append(obj({'candidate': enum([h]),
            'summary': {'type': 'string', 'minLength': 1, 'maxLength': 400},
            'refs': refs_schema(refs),
            'uncertainty': {'type': 'string', 'maxLength': 200}}))
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
        if not item['refs'] and not item['uncertainty'].strip():
            raise ValueError('An uncited explanation must state its evidence limitation')


def solve(evidence, preview, images, client, max_inspections=2, on_event=None):
    if max_inspections < 0:
        raise ValueError('Negative inspection budget')
    results, trace, extra_images = [], [], []
    visible = {r['id']: deepcopy(r) for record in preview.values()
               for key in ('attributes', 'relations', 'images') for r in record[key]}
    read_ids = set(visible)
    inspections, current, initial_choice = 0, None, None
    used_modalities = set()

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
        if decision['current_choice'] not in LABELS:
            raise ValueError('Invalid current choice')
        if set(decision['types']) != set(['SRC']+LABELS) or not set(decision['types'].values()) <= set(TYPES):
            raise ValueError('Invalid type hypotheses')
        current = {'choice': decision['current_choice'], 'assessment': decision['assessment'],
                   'unresolved': decision['unresolved']}
        if initial_choice is None:
            initial_choice = current['choice']
        action = decision['next']
        emit({'kind': 'planner', 'decision': decision, 'action': action,
              'current_judgment': deepcopy(current), 'available': available})
        if action['action'] == 'finish':
            if action['choice'] != current['choice'] or not set(action['refs']) <= set(visible):
                raise ValueError('Final answer disagrees with current judgment or cites unavailable evidence')
            choice = action['choice']
            target = evidence.handles[choice]
            return {'choice': choice, 'rank': [target]+[u for u in evidence.rank if u != target],
                    'handle_map': evidence.handles, 'inspections': inspections, 'trace': trace,
                    'initial_choice': initial_choice, 'current_judgment': current,
                    'evidence': visible, 'initial_preview_sha256': digest(preview),
                    'max_inspections': max_inspections, 'labels_loaded': False, 'protocol_version': 2,
                    'modality_reuse': False}
        if action['action'] != 'inspect' or not available:
            raise ValueError('Inspection is disabled by budget or evidence availability')
        if not decision['unresolved'].strip() or not action['question'].strip():
            raise ValueError('Inspection requires a concrete unresolved distinction and question')
        operation = {**action['operation'], 'question': action['question'], 'types': decision['types']}
        if operation['agent'] not in available:
            raise ValueError('Modality is unavailable or has already been inspected for this SRC')
        evidence.validate_operation(operation)
        payload, acquired_images, retrieval = evidence.retrieve(operation)
        inspections += 1
        used_modalities.add(operation['agent'])
        supplied = {r['id']: r for records in payload['RECORDS'].values() for r in records}
        new_ids = set(supplied)-read_ids
        emit({'kind': 'retrieval', 'agent': operation['agent'], 'inspection': inspections,
              'query': operation['question'], 'records': payload['RECORDS'], 'retrieval': retrieval,
              'new_evidence_ids': sorted(new_ids), 'images': acquired_images})
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
                   'question': operation['question'], 'result': result,
                   'evidence': [supplied[r] for r in sorted(cited)]}
        results.append(receipt)
        emit({'kind': 'reader', **receipt})
