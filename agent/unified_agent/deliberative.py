"""Label-free hypothesis revision and question-directed multimodal acquisition.

The frozen discovery result is an input, never a truth label. Two independently
testable interventions repair unresolved searches and audit proposed changes.
Unknown audit outcomes preserve the incoming decision, not the retrieval rank.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time

from .environment import parse_object
from .engine import render
from .grounded import GroundedEnv, catalog, focused_state, object_schema, verify


def restore_environment(row, index, model, asset_dir, incumbent):
    assert incumbent['source'] == row['source']
    assert incumbent['baseline_rank'] == row['baseline']['rank']
    env = GroundedEnv(row, index, model.compact, asset_dir, novel_only=True)
    assert incumbent['handle_map'] == env.handles
    # Only observations enter the evidence ledger; prior model claims remain
    # explicitly fallible feedback. Original handle/ref namespaces are retained.
    for ref, item in incumbent.get('evidence', {}).items():
        if item['preview']:
            continue
        assert item['uri'] == env.handles[item['entity']]
        if item['modality'] == 'image':
            if item['entity'] not in env.assets:
                continue
            env.recent_images.append(item['entity'])
        env.ledger[ref] = deepcopy(item)
    env.recent_images = list(dict.fromkeys(env.recent_images))
    return env


def prior_feedback(incumbent):
    memory = []
    for t in incumbent.get('trace', []):
        if t['kind'] != 'verifier':
            continue
        entry = {k: t.get(k) for k in ('hypothesis', 'rival', 'accepted', 'reason')}
        entry['judgment'] = t.get('action', {})
        entry['rival_judgment'] = t.get('rival_verdict')
        if not memory or entry != memory[-1]:
            memory.append(entry)
    return memory


def ask(model, env, prompt, fields, kind, images=None, max_tokens=448):
    raw, meta = model.generate(prompt, env.images() if images is None else images,
                               schema=object_schema(fields), max_tokens=max_tokens)
    try:
        obj = parse_object(raw); error = None
    except (ValueError, TypeError) as exc:
        obj = {}; error = str(exc)
    return obj, {'kind': kind, 'action': obj, 'raw': raw, 'model': meta, 'error': error}


def propose(model, env, memory):
    state, _ = render(env, model.compact)
    subjects = '\n'.join(h + ': ' + u for h, u in env.handles.items())
    handles = [h for h in env.handles if h != 'SRC']
    fields = {
        'source_subject': {'type': 'string', 'maxLength': 160},
        'unresolved_difference': {'type': 'string', 'maxLength': 240},
        'hypothesis': {'enum': handles}, 'rival': {'enum': handles},
    }
    prompt = (state + '\nRECORD SUBJECT URIS (provenance, not known alignments)\n' + subjects +
        '\nPREVIOUS FALLIBLE VERIFICATION JUDGMENTS\n' + json.dumps(memory, ensure_ascii=True) +
        '\nREVISE THE IDENTITY SEARCH. First describe the subject denoted by SRC, including its '
        'entity granularity. Then identify the unresolved distinction using actual visible evidence. '
        'Reconsider ALL20 records, including candidates never inspected. A rejected proposal is not '
        'a fact about SRC. If a previous proposal was judged different, reconsider the alternatives '
        'instead of repeatedly gathering facts about that same associated entity. A prior verifier '
        'can be wrong: retain a hypothesis only when the observations address its rejection. '
        'Unknown evidence must not permanently eliminate a candidate. Distinguish the record subject '
        'from names of works, people or organizations mentioned in its fields. Select the most plausible '
        'exact match and a distinct, genuinely competitive alternative. Use all four modalities equally. '
        'Do not infer identity from a shared picture, topic, substring or common category alone. '
        'Descriptions are provisional reasoning, not new observed facts. Return the required JSON.')
    return ask(model, env, prompt, fields, 'hypothesis')


def acquire(model, env, focus, memory, images=None):
    options = catalog(env, focus)
    if not options:
        return None, [{'kind': 'planner', 'action': {'decision': 'abstain'},
                       'reason': 'No unread evidence'}]
    state, _ = focused_state(env, model, focus)
    fields = {'question': {'type': 'string', 'maxLength': 240},
              'field_id': {'enum': list(options) + ['none']}}
    prompt = (state + '\nFALLIBLE COMPARISON FEEDBACK\n' + json.dumps(memory[-2:], ensure_ascii=True) +
        '\nPLAN A DISCRIMINATING OBSERVATION. Write one concrete factual question whose answer could '
        'distinguish the competing subjects. Use meaningful names or property words, not just handle '
        'letters and a modality name. Then select the existing field or original image that can answer it. '
        'A field ID selects its exact predicate and only its available entities; name reads names and '
        'image opens original pixels. Relational direction and predicate roles matter. A question about '
        'authorship is different from asking for more generic related names. All modalities have equal '
        'standing; choose according to the unresolved distinction. Do not request irrelevant fields '
        'merely because they are unread. The catalog reveals availability only, not answers. '
        'Select none if no available evidence could help.\nCATALOG\n' + json.dumps(options, ensure_ascii=True))
    obj, t = ask(model, env, prompt, fields, 'planner', images=images, max_tokens=288)
    if t['error'] or obj.get('field_id') == 'none':
        return None, [t]
    item = options.get(obj.get('field_id'))
    if not item or not obj.get('question', '').strip():
        t['error'] = 'Invalid question or catalog field'; return None, [t]
    action = {'action': 'inspect', 'entities': item['entities'], 'modalities': [item['modality']],
              'question': obj['question'], 'expected': 'Resolve the stated identity distinction', 'offset': 0}
    if item['predicate'] is not None:
        action['predicate'] = item['predicate']
    t['resolved_original_field'] = item
    observation = env.inspect(action)
    return observation, [t, {'kind': 'tool', 'action': action, 'observation': observation}]


def focused_images(env, focus, folder):
    """Match the visual comparison scope to the text comparison scope."""
    from PIL import Image, ImageDraw
    present = {h: env.assets[h] for h in focus if h in env.assets}
    if not present:
        return []
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([focus, present], sort_keys=True).encode()).hexdigest()
    path = folder / (key + '.jpg')
    if not path.exists():
        sheet = Image.new('RGB', (160 * len(focus), 180), 'white'); draw = ImageDraw.Draw(sheet)
        for i, h in enumerate(focus):
            draw.text((160*i+4, 2), h, fill='black')
            if h in present:
                with Image.open(present[h]) as src:
                    im = src.convert('RGB'); im.thumbnail((152, 152)); sheet.paste(im, (160*i+4, 22)); im.close()
            else:
                draw.text((160*i+4, 50), 'image unavailable', fill='gray')
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.jpg', delete=False) as stream:
            temp = Path(stream.name)
        try:
            sheet.save(temp, format='JPEG', quality=90); temp.replace(path)
        finally:
            temp.unlink(missing_ok=True); sheet.close()
    images = [('FOCUSED RECORD THUMBNAILS: ' + ', '.join(focus), str(path))]
    images += [(f'ORIGINAL IMAGE {h}; evidence {h}.image.full', env.assets[h])
               for h in env.recent_images if h in focus and h in env.assets]
    return images


def grounded_choice(obj, refs, env, alternatives):
    choice = obj.get('choice')
    if choice not in alternatives:
        return 'U'
    for key, owner in [('source_ref', 'SRC'), ('target_ref', choice)]:
        ref = obj.get(key)
        if ref not in refs or env.ledger[ref]['entity'] != owner:
            return 'U'
    if obj.get('source_scope') != 'self' or obj.get('target_scope') != 'self':
        return 'U'
    return choice if obj.get('relationship') == 'same_entity' else 'U'


def compare_subjects(model, env, alternatives, folder, memory):
    focus = ['SRC'] + list(alternatives)
    state, refs = focused_state(env, model, focus)
    source_refs = sorted(r for r in refs if env.ledger[r]['entity'] == 'SRC')
    target_refs = sorted(r for r in refs if env.ledger[r]['entity'] in alternatives)
    if not source_refs or not target_refs:
        return 'U', {'kind': 'verifier', 'action': {'choice': 'U'}, 'error': None,
                     'accepted': False, 'reason': 'Missing visible source or target evidence'}
    fields = {'source_subject': {'type': 'string', 'maxLength': 160},
              'identity_comparison': {'type': 'string', 'maxLength': 360},
              'source_ref': {'enum': source_refs}, 'target_ref': {'enum': target_refs},
              'source_scope': {'enum': ['self', 'related_entity', 'unclear']},
              'target_scope': {'enum': ['self', 'related_entity', 'unclear']},
              'relationship': {'enum': ['same_entity', 'related_entity', 'different_entity', 'unknown']},
              'choice': {'enum': list(alternatives) + ['U']}}
    prompt = (state + '\nPREVIOUS FALLIBLE COMPARISONS\n' + json.dumps(memory[-2:], ensure_ascii=True) +
        '\nCOMPARE RECORD SUBJECTS. Describe what SRC itself denotes. Then compare EACH alternative '
        'subject to SRC using the observed evidence, preserving the roles and granularity of all records. '
        'Distinguish exact identity from authorship, membership, a channel versus its network, a model '
        'variant versus its family, a government versus its leader, and a work versus its performer. '
        'Names in a record can name associated entities. URI titles identify the record being discussed '
        'but are not known alignment labels. Explain the actual identity distinction, not just a shared '
        'word or generic category. Do not invent an alias or claim a query answered something it did not. '
        'Names, attributes, directed relations and original images have equal standing; use the strongest '
        'distinguishing observations available. Missing evidence is unknown. No facial identification. '
        'Cite actual source and selected-target evidence and classify its ownership. Choose a record '
        'only for same-subject identifying agreement that distinguishes the alternatives. U means further '
        'evidence is needed, not that either alternative is wrong. The selected target_ref must belong '
        'to the chosen record. Return the required JSON.')
    obj, t = ask(model, env, prompt, fields, 'verifier',
                 images=focused_images(env, focus, folder), max_tokens=512)
    choice = 'U' if t['error'] else grounded_choice(obj, refs, env, alternatives)
    t.update(accepted=choice != 'U', accepted_handle=choice, compared_handles=list(alternatives))
    return choice, t


def combine_result(incumbent, env, model, start, trace, inspections, checks, choice, stage, status):
    result = deepcopy(incumbent)
    result['rank'] = ([env.handles[choice]] + [u for u in incumbent['rank'] if u != env.handles[choice]]) if choice != 'U' else list(incumbent['rank'])
    result['status'] = status
    result['trace'] = deepcopy(incumbent['trace']) + [dict(t, stage=stage) for t in trace]
    result['evidence'] = deepcopy(env.ledger)
    result['handle_map'] = dict(env.handles)
    elapsed, calls, inp, out = start
    added = {'seconds': time.time()-elapsed, 'calls': model.calls-calls,
             'input_tokens': model.input_tokens-inp, 'output_tokens': model.output_tokens-out,
             'inspections': inspections, 'verification_calls': checks}
    for key, value in added.items():
        result[key] = incumbent.get(key, 0) + value
    result.setdefault('interventions', []).append({'stage': stage, 'status': status,
        'choice': choice, 'changed': result['rank'][0] != incumbent['rank'][0], **added})
    result['accounting'] = 'End-to-end costs include the frozen discovery and all applied intervention stages.'
    assert len(result['rank']) == 20 and set(result['rank']) == set(incumbent['rank'])
    return result


def replan_unresolved(row, index, model, asset_dir, incumbent, budget=4):
    if incumbent['status'] == 'verified':
        return deepcopy(incumbent)
    env = restore_environment(row, index, model, asset_dir, incumbent)
    memory = prior_feedback(incumbent); trace = []; inspections = checks = 0; choice = 'U'
    start = (time.time(), model.calls, model.input_tokens, model.output_tokens)
    previous = getattr(model, 'unicode_json', False); model.unicode_json = True
    status = 'replan_unresolved'
    try:
        for step in range(budget+1):
            obj, t = propose(model, env, memory); trace.append(t)
            best, rival = obj.get('hypothesis'), obj.get('rival')
            if t['error'] or best == rival or best not in env.priors or rival not in env.priors:
                status = 'invalid_replan'; break
            ok, vt, verdict = verify(model, env, best, rival, strict=True)
            trace.append(vt); checks += 1
            if ok:
                choice = best; status = 'replan_verified'; break
            memory.append({'hypothesis': best, 'rival': rival, 'judgment': vt['action'],
                           'reason': vt.get('reason'), 'rival_judgment': vt.get('rival_verdict')})
            if step == budget:
                break
            obs, ts = acquire(model, env, ['SRC', best, rival], memory); trace.extend(ts)
            if obs is None:
                if any(t.get('error') for t in ts): status = 'invalid_query'
                break
            inspections += 1
            if obs.get('error'):
                status = 'tool_error'; break
    finally:
        model.unicode_json = previous
    return combine_result(incumbent, env, model, start, trace, inspections, checks, choice, 'replan', status)


def audit_change(row, index, model, asset_dir, incumbent, budget=4):
    original = row['baseline']['rank'][0]
    if incumbent['rank'][0] == original:
        return deepcopy(incumbent)
    env = restore_environment(row, index, model, asset_dir, incumbent)
    # Handle order is independent of retrieval rank and the current proposal.
    alternatives = sorted([env.reverse[original], env.reverse[incumbent['rank'][0]]])
    focus = ['SRC'] + alternatives; trace = []; memory = []; inspections = checks = 0; choice = 'U'
    start = (time.time(), model.calls, model.input_tokens, model.output_tokens)
    previous = getattr(model, 'unicode_json', False); model.unicode_json = True
    status = 'compare_unresolved_keep_incoming'
    try:
        for step in range(budget+1):
            choice, t = compare_subjects(model, env, alternatives, Path(asset_dir)/'focused', memory)
            trace.append(t); checks += 1
            if choice != 'U':
                status = 'compare_verified'; break
            memory.append({'judgment': t['action'], 'accepted': False})
            if t.get('error'):
                status = 'invalid_comparison'; break
            if step == budget:
                break
            obs, ts = acquire(model, env, focus, memory,
                images=focused_images(env, focus, Path(asset_dir)/'focused')); trace.extend(ts)
            if obs is None:
                if any(t.get('error') for t in ts): status = 'invalid_query'
                break
            inspections += 1
            if obs.get('error'):
                status = 'tool_error'; break
    finally:
        model.unicode_json = previous
    return combine_result(incumbent, env, model, start, trace, inspections, checks, choice, 'compare', status)
