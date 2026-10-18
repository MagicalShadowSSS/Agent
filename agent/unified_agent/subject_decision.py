"""One identity decision with record-owned evidence, shared by both conditions.

This is a diagnostic reader, not a trained policy or a truth checker. Every
candidate is considered; citations can validate ownership but not entailment.
Generated interpretations never become original evidence in the environment.
"""
from .environment import parse_object
from .acquisition146 import READER
from .grounded import object_schema
from .json_protocol import SHORT_TEXT
from .zh_active_evidence import state, state_hash


def prior_free_reader(model, env):
    """Keep the old reader and all originals; withhold numeric retrieval priors."""
    text, visible = state(env, model)
    lines = text.splitlines()
    removed = [line for line in lines if line.startswith('Retrieval prior (not a match label): ')]
    assert len(removed) == len(env.priors)
    text = '\n'.join(line for line in lines if not line.startswith('Retrieval prior (not a match label): '))
    scores, meta = model.choose(text + READER, env.images(), list(env.priors))
    choice = max(scores, key=scores.get)
    return choice, {'kind': 'prior_free_reader', 'choice': choice, 'scores': scores,
        'model': meta, 'visible_refs': sorted(visible),
        'observation_state_sha256': state_hash(env, model),
        'withheld_metadata': 'numeric retrieval ranks only; all original observations and images unchanged',
        'error': None}


def text_only_diagnostic(model, env):
    """Pixel ablation for diagnosis only; never used as the proposed agent."""
    text, visible = state(env, model)
    image_refs = {r for r in visible if env.ledger[r]['modality']=='image'}
    lines = [line for line in text.splitlines() if line != 'IMAGE REFERENCES (addresses, not factual descriptions):'
             and not any(line.startswith('['+r+']') for r in image_refs)]
    scores, meta = model.choose('\n'.join(lines)+READER, [], list(env.priors))
    choice = max(scores, key=scores.get)
    return choice, {'kind':'text_only_diagnostic', 'choice':choice, 'scores':scores,
        'model':meta, 'visible_refs':sorted(visible-image_refs),
        'observation_state_sha256':state_hash(env,model),
        'withheld_metadata':'All pixels and image address lines; pixel ablation, NOT a proposed multimodal method.',
        'error':None}


def source_last_reader(model, env):
    """Present the query record last; retain every original, image and prior."""
    text, visible = state(env, model, list(env.priors)+['SRC'])
    scores, meta = model.choose(text+READER, env.images(), list(env.priors))
    choice = max(scores, key=scores.get)
    return choice, {'kind':'source_last_reader', 'choice':choice, 'scores':scores,
        'model':meta, 'visible_refs':sorted(visible),
        'observation_state_sha256':state_hash(env,model),
        'presentation':'candidate records A-J followed by SRC; identical observations and attachments',
        'error':None}


def decision(model, env, compare=False):
    text, visible = state(env, model)
    owned = {h: sorted(r for r in visible if env.ledger[r]['entity'] == h
                      and env.ledger[r]['location'] != 'record/uri')
             for h in env.handles}

    def references(h):
        refs = owned[h]
        return {'type': 'array', 'items': {'enum': refs} if refs else {'type': 'string'},
                'minItems': min(1, len(refs)), 'maxItems': min(2, len(refs))}

    checks = {}
    for h in env.priors:
        properties = {'refs':references(h),
            'subject':{'type':'string','minLength':1,'maxLength':96}}
        if compare:
            properties.update(source_refs=references('SRC'), support=SHORT_TEXT, conflict=SHORT_TEXT)
        properties['relation_to_source'] = {'enum':['same','related','different','unknown']}
        checks[h] = object_schema(properties)
    schema = object_schema({
        'source_subject': object_schema({'refs': references('SRC'), 'description': SHORT_TEXT}),
        'candidate_checks': object_schema(checks),
        'decision': object_schema({'choice': {'enum': list(env.priors)},
            'reason': SHORT_TEXT,
            'remaining_distinction': SHORT_TEXT, 'needed_observation': SHORT_TEXT})})
    prompt = (text + '\nIDENTITY DECISION WITH RECORD-OWNED OBSERVATIONS. '
        'Identify which of all ten candidate records denotes the same entity as SRC. '
        'First describe the subject of SRC using its own cited observations. '
        "For every candidate, cite that record's observations, describe its own "
        'subject, and distinguish exact identity from relatedness or unknown identity. '
        'A mentioned participant, associated work or pictured object need not be the '
        'record subject. Missing observations remain unknown. Names, attributes, '
        'directed relations and original images are all valid evidence sources. '
        'Citations establish ownership, not correctness. After these comparisons, '
        'select the best candidate among all ten and state any remaining identity '
        'distinction and observation needed; say none if resolved. '
        'Use the original observations rather than assuming retrieval ranks are correct. '
        'Keep each subject description brief. Return the requested JSON.')
    if compare:
        prompt += (' For each candidate compare the cited source and candidate observations: '
            'state the strongest observed support for exact identity and the strongest '
            'observed conflict. State unknown when observations provide neither; do not '
            'invent agreement or contradictions. A common category or related subject '
            'is weaker than a fact identifying the particular entity. Resolve these '
            'comparisons before making the final choice. All modalities can supply '
            'support or conflict; none is automatically decisive.')
    raw, meta = model.generate(prompt, env.images(), max_tokens=3584 if compare else 2048, schema=schema)
    result = parse_object(raw)
    assert set(result['candidate_checks']) == set(env.priors)
    assert set(result['source_subject']['refs']) <= set(owned['SRC'])
    for h, check in result['candidate_checks'].items():
        assert set(check['refs']) <= set(owned[h])
        if compare:
            assert set(check['source_refs']) <= set(owned['SRC'])
    choice = result['decision']['choice']
    assert choice in env.priors
    return choice, {'kind': 'subject_decision', 'choice': choice, 'action': result,
        'raw': raw, 'model': meta, 'visible_refs': sorted(visible),
        'observation_state_sha256': state_hash(env, model), 'error': None}


def comparison_decision(model, env):
    return decision(model, env, compare=True)
