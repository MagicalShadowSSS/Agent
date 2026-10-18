"""Fixed evidence checks followed by an explicit full-Top10 reselection."""
from copy import deepcopy

from .modality_agents import LABELS, ROLES, TYPES, RoleClient, digest, enum, obj, refs_schema
from .modality_agents_fixed import (Evidence, FIXED_TASKS, SHORT, reader_schema,
                                    validate_reader)


def planner_schema(refs, available):
    finish = obj({'action': enum(['finish']), 'refs': refs_schema(refs)})
    branches = [finish]
    if available:
        operations = [obj({'agent': enum([m]), 'candidates': {
            'type': 'array', 'items': enum(candidates), 'minItems': 1,
            'maxItems': min(3, len(candidates))}}) for m, candidates in available.items()]
        branches.append(obj({'action': enum(['inspect']), 'operation': {'anyOf': operations}}))
    return obj({'types': obj({h: enum(TYPES) for h in ['SRC']+LABELS}),
                'assessment': SHORT, 'choice': enum(LABELS),
                'next': {'anyOf': branches} if available else finish})


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
        if set(decision) != {'types', 'assessment', 'choice', 'next'}:
            raise ValueError('Planner must return one choice and one action')
        if decision['choice'] not in LABELS:
            raise ValueError('Invalid choice')
        if set(decision['types']) != set(['SRC']+LABELS) or not set(decision['types'].values()) <= set(TYPES):
            raise ValueError('Invalid type hypotheses')
        current = {'choice': decision['choice'], 'assessment': decision['assessment']}
        if initial_choice is None:
            initial_choice = current['choice']
        action = decision['next']
        emit({'kind': 'planner', 'decision': decision, 'action': action,
              'current_judgment': deepcopy(current), 'available': available,
              'full_top10_reselection': bool(results)})
        if action['action'] == 'finish':
            if set(action) != {'action', 'refs'} or not set(action['refs']) <= set(visible):
                raise ValueError('Finish cites unavailable evidence or repeats a choice field')
            target = evidence.handles[current['choice']]
            return {'choice': current['choice'],
                    'rank': [target]+[u for u in evidence.rank if u != target],
                    'handle_map': evidence.handles, 'inspections': inspections, 'trace': trace,
                    'initial_choice': initial_choice, 'current_judgment': current,
                    'evidence': visible, 'initial_preview_sha256': digest(preview),
                    'max_inspections': max_inspections, 'labels_loaded': False, 'protocol_version': 4,
                    'modality_reuse': False, 'inspection_task': 'fixed_evidence_check',
                    'full_top10_reselection': True, 'single_choice_field': True}
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
