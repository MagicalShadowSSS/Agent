"""Persistent candidate states, irreversible rejection and native VLM role calls."""
import json
import time
from copy import deepcopy
from pathlib import Path

from .modality_agents import LABELS, ROLES, TYPES, digest, enum, obj, refs_schema
from .modality_agents_fixed import Evidence, validate_reader


def reader_schema(candidates, records):
    branches = []
    for h in candidates:
        refs = [r['id'] for owner in ['SRC', h] for r in records[owner]]
        for statuses, minimum in [(['uncertain'], 0), (['support', 'conflict', 'mixed'], 1)]:
            cited = refs_schema(refs)
            cited['minItems'] = minimum
            branches.append(obj({'candidate': enum([h]),
                'summary': {'type': 'string', 'minLength': 1, 'maxLength': 400},
                'evidence_status': enum(statuses), 'refs': cited}))
    return obj({'explanations': {'type': 'array', 'items': {'anyOf': branches},
                                'minItems': len(candidates), 'maxItems': len(candidates)}})


def planner_schema(refs, available, states, rejectable=()):
    eligible = [h for h in LABELS if states[h] != 'rejected']
    if not eligible:
        raise ValueError('All candidates rejected; alignment unresolved')
    common = {'types': obj({h: enum(TYPES) for h in ['SRC']+LABELS}),
              'assessment': {'type': 'string', 'minLength': 1, 'maxLength': 600}}
    def table(anchor, state):
        return obj({h: enum(['rejected']) if states[h] == 'rejected' else
                    enum([state]) if h == anchor else enum(['undetermined', 'suspected']+(['rejected'] if h in rejectable else []))
                    for h in LABELS})
    terminal = [obj({'confirmed': enum([chosen]), 'states': obj({h:
                    enum(['rejected']) if states[h] == 'rejected' else enum(['undetermined', 'suspected']+(['rejected'] if h in rejectable else []))
                    for h in LABELS if h != chosen})}) for chosen in eligible]
    finish = obj({**common, 'action': enum(['finish']), 'decision': {'anyOf': terminal}, 'refs': refs_schema(refs)})
    if not available:
        return finish
    usable = sorted({h for hs in available.values() for h in hs if h in eligible})
    inspect = obj({**common, 'action': enum(['inspect']),
                   'states': {'anyOf': [table(h, 'suspected') for h in usable]+[table(h, 'confirmed') for h in eligible]}})
    return {'anyOf': [finish, inspect]}


def tool_schema(available):
    return {'anyOf': [obj({'agent': enum([m]), 'candidates': {'type': 'array',
            'items': enum(hs), 'minItems': 1, 'maxItems': min(3, len(hs))}})
            for m, hs in available.items()]}


def validate_tool(operation, available):
    targets = operation['candidates']
    if not 1 <= len(targets) <= 3 or len(set(targets)) != len(targets):
        raise ValueError('Inspect one to three distinct candidates')
    if operation['agent'] not in available or not set(targets) <= set(available[operation['agent']]):
        raise ValueError('Tool must inspect suspected candidates with available evidence')


def transition(states, decision, available, rejectable=()):
    """Validate a complete transaction before changing any persisted state."""
    if decision['action'] == 'finish':
        chosen = decision['decision']['confirmed']
        remaining = decision['decision']['states']
        if chosen not in LABELS or set(remaining) != set(LABELS)-{chosen}:
            raise ValueError('Final decision must state the other nine candidates exactly once')
        new = {**remaining, chosen: 'confirmed'}
    else:
        new = dict(decision['states'])
    if set(new) != set(LABELS) or not set(new.values()) <= {'undetermined', 'suspected', 'rejected', 'confirmed'}:
        raise ValueError('Every candidate must have one valid state')
    if any(old == 'rejected' and new[h] != 'rejected' for h, old in states.items()):
        raise ValueError('Cannot reopen a rejected candidate')
    if any(old != 'rejected' and new[h] == 'rejected' and h not in rejectable for h, old in states.items()):
        raise ValueError('Irreversible rejection requires an explicit inspected conflict citing both entities')
    confirmations = sum(v == 'confirmed' for v in new.values())
    if confirmations > 1:
        raise ValueError('At most one confirmed candidate')
    if decision['action'] == 'finish':
        if confirmations != 1:
            raise ValueError('Finish requires exactly one confirmed candidate')
    elif decision['action'] == 'inspect':
        if not any(new[h] == 'suspected' for hs in available.values() for h in hs):
            raise ValueError('Inspect requires a suspected candidate with available unused evidence')
    else:
        raise ValueError('Unknown action')
    return {h: new[h] for h in LABELS}


def solve(evidence, preview, images, client, max_inspections=2, on_event=None):
    if max_inspections < 0:
        raise ValueError('Negative inspection budget')
    states = {h: 'undetermined' for h in LABELS}
    results, trace, extra_images = [], [], []
    visible = {r['id']: deepcopy(r) for rec in preview.values()
               for k in ('attributes', 'relations', 'images') for r in rec[k]}
    used, read_ids, rejectable = set(), set(visible), set()
    inspections, initial_choice, planner_round = 0, None, 0
    errors = []

    def emit(event):
        trace.append(event)
        if on_event:
            on_event(event)

    while True:
        available = {m: [h for h in hs if states[h] != 'rejected']
                     for m, hs in evidence.available().items() if m not in used}
        available = {m: hs for m, hs in available.items() if hs} if inspections < max_inspections else {}
        payload = {'RECORDS': preview, 'CANDIDATE_STATES': states, 'AGENT_RESULTS': results,
                   'REJECTABLE_CANDIDATES': sorted(rejectable),
                   'AVAILABLE_EVIDENCE': available,
                   'AVAILABLE_ACTIONS': {'inspect': bool(available), 'finish_requires_unique_confirmation': True}}
        schema = planner_schema(visible, available, states, rejectable)
        for attempt in range(3):
            decision = client('planner', payload, images+extra_images, schema)
            try:
                new = transition(states, decision, available, rejectable)
                action = {'action': decision['action']}
                if action['action'] == 'finish' and not set(decision['refs']) <= set(visible):
                    raise ValueError('Unavailable final evidence reference')
                break
            except ValueError as exc:
                errors.append(str(exc))
                emit({'kind': 'protocol_error', 'decision': decision, 'error': str(exc), 'attempt': attempt})
                if attempt == 2:
                    raise
                payload = {**payload, 'INVALID_PREVIOUS_OUTPUT': decision, 'FORMAT_CORRECTION': str(exc)}
        before, states = states, new
        if planner_round == 0:
            initial_choice = next((h for h, s in states.items() if s == 'confirmed'), None)
        planner_round += 1
        emit({'kind': 'planner', 'decision': decision, 'action': action, 'states_before': before,
              'states_after': deepcopy(states), 'available': available})
        if action['action'] == 'finish':
            choice = next(h for h, s in states.items() if s == 'confirmed')
            target = evidence.handles[choice]
            return {'choice': choice, 'rank': [target]+[u for u in evidence.rank if u != target],
                    'handle_map': evidence.handles, 'inspections': inspections, 'trace': trace,
                    'initial_choice': initial_choice, 'states': states, 'protocol_errors': errors,
                    'evidence': visible, 'initial_preview_sha256': digest(preview),
                    'max_inspections': max_inspections, 'labels_loaded': False,
                    'protocol_version': 9, 'modality_reuse': False, 'irreversible_rejection': True,
                    'rejection_requires_pair_cited_conflict': True}
        usable = {m: [h for h in hs if states[h] == 'suspected'] for m, hs in available.items()}
        usable = {m: hs for m, hs in usable.items() if hs}
        selection_payload = {'PHASE': 'select_tool', 'CANDIDATE_STATES': states,
                             'CURRENT_ASSESSMENT': decision['assessment'], 'TYPES': decision['types'],
                             'AVAILABLE_EVIDENCE': usable, 'AGENT_RESULTS': results,
                             'RECORDS': {h: r for h, r in preview.items() if h == 'SRC' or states[h] == 'suspected'}}
        for attempt in range(3):
            selected = client('planner', selection_payload, [], tool_schema(usable))
            try:
                validate_tool(selected, usable)
                break
            except ValueError as exc:
                errors.append(str(exc))
                emit({'kind': 'protocol_error', 'decision': selected, 'error': str(exc), 'attempt': attempt})
                if attempt == 2:
                    raise
                selection_payload = {**selection_payload, 'FORMAT_CORRECTION': str(exc)}
        emit({'kind': 'tool_selection', 'operation': selected, 'available': usable})
        operation = {**selected, 'types': decision['types']}
        acquired, acquired_images, retrieval = evidence.retrieve(operation)
        inspections += 1
        used.add(operation['agent'])
        supplied = {r['id']: r for rows in acquired['RECORDS'].values() for r in rows}
        emit({'kind': 'retrieval', 'agent': operation['agent'], 'inspection': inspections,
              'records': acquired['RECORDS'], 'retrieval': retrieval,
              'new_evidence_ids': sorted(set(supplied)-read_ids), 'images': acquired_images})
        result = client(operation['agent'], acquired, acquired_images,
                        reader_schema(operation['candidates'], acquired['RECORDS']))
        validate_reader(result, operation['candidates'], acquired['RECORDS'])
        for item in result['explanations']:
            if item['evidence_status'] == 'conflict' and any(r.startswith('SRC.') for r in item['refs']) and any(
                    r.startswith(item['candidate']+'.') for r in item['refs']):
                rejectable.add(item['candidate'])
        cited = {r for item in result['explanations'] for r in item['refs']}
        visible.update({r: supplied[r] for r in cited})
        read_ids.update(supplied)
        for label, path in acquired_images:
            if label in cited and (label, path) not in extra_images:
                extra_images.append((label, path))
        receipt = {'agent': operation['agent'], 'candidates': operation['candidates'],
                   'result': result, 'evidence': [supplied[r] for r in sorted(cited)]}
        results.append(receipt)
        emit({'kind': 'reader', **receipt})


class RoleClient:
    """Log identical structured tasks, using each model's native image/chat input."""
    def __init__(self, model, prompts, folder, validate):
        from .decoder_optimization import optimize_union_alphabet
        optimize_union_alphabet()
        self.model, self.prompts, self.folder, self.validate = model, prompts, Path(folder), validate
        self.folder.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def __call__(self, role, payload, images, schema):
        import torch
        from PIL import Image
        from .engine import constrained_decoder
        stem = self.folder/f'{len(self.calls):02d}_{role}'
        system = self.prompts[role]
        prompt = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        stem.with_suffix('.system.txt').write_text(system)
        stem.with_suffix('.user.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        stem.with_suffix('.schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
        started = time.perf_counter()
        call = {'role': role, 'images': images, 'started_at': time.time(), 'backend': self.model.backend}
        self.calls.append(call)
        opened = []
        try:
            if self.model.backend == 'internvl3':
                batch, embeds, formatted, eos = self.model._internvl_input(prompt, images, system)
                generator = self.model.model.language_model
                inputs = {'input_ids': batch.input_ids, 'inputs_embeds': embeds,
                          'attention_mask': batch.attention_mask}
            else:
                content = []
                for label, path in images:
                    content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
                    with Image.open(path) as source:
                        opened.append(source.convert('RGB'))
                content.append({'type': 'text', 'text': prompt})
                messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
                formatted = self.model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch = self.model.processor(text=[formatted], images=opened or None,
                    return_tensors='pt', padding=True).to('cuda:0')
                inputs, generator = dict(batch), self.model.model
                eos = self.model.tokenizer.eos_token_id
            stem.with_suffix('.formatted.txt').write_text(formatted)
            n, max_tokens = batch.input_ids.shape[1], (512 if role == "planner" else 1024)
            if n+max_tokens > 262144:
                raise ValueError(f'Context overflow {n}+{max_tokens}; no truncation')
            decoder = constrained_decoder(self.model.format_tokens, schema, unicode_json=True)
            call['input_preparation_seconds'] = time.perf_counter()-started
            decoding = {'seconds': 0.0, 'calls': 0}
            def timed_decoder(batch_id, input_ids):
                tick = time.perf_counter()
                try:
                    return decoder(batch_id, input_ids)
                finally:
                    decoding['seconds'] += time.perf_counter()-tick
                    decoding['calls'] += 1
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = generator.generate(**inputs, logits_to_keep=1, max_new_tokens=max_tokens, do_sample=False,
                    use_cache=True, pad_token_id=eos, eos_token_id=eos,
                    prefix_allowed_tokens_fn=timed_decoder)
            call.update(generation_seconds=time.perf_counter()-generation_started,
                        constraint_callback_seconds=decoding['seconds'], constraint_callback_calls=decoding['calls'])
            if not torch.equal(output[0, :n], batch.input_ids[0]):
                raise ValueError('Unexpected generation prefix')
            tail = output[0, n:]
            raw = self.model.tokenizer.decode(tail, skip_special_tokens=True)
            stem.with_suffix('.output.txt').write_text(raw)
            call.update(input_tokens=n, output_tokens=len(tail), prompt_sha256=digest(formatted))
            result = json.loads(raw)
            self.validate(result, schema)
            call['valid_json_schema'] = True
            return result
        except Exception as exc:
            call['error'] = str(exc)
            raise
        finally:
            call['seconds'] = time.perf_counter()-started
            stem.with_suffix('.meta.json').write_text(json.dumps(call, ensure_ascii=False, indent=2))
            for image in opened:
                image.close()
