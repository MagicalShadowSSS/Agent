"""Revisable semantic types for acquisition planning, never candidate filtering."""
from .grounded import object_schema

POLICIES = ('original', 'type_guided', 'type_prefer')
TYPE_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 64}

REVIEW_GUIDANCE = (
    '\nSUBJECT AND TYPE BEFORE QUERY PLANNING. First infer what SRC itself denotes '
    'and its possible semantic type; then infer the possible type of each candidate '
    'from that candidate record. Use concise natural-language types, or unknown '
    'when uncertain. A type is a revisable interpretation, not an observed fact. '
    'Mark whether each candidate type is compatible, incompatible or uncertain '
    'relative to SRC. Type compatibility alone does not establish exact identity. '
    'Use these subject hypotheses and the actual candidate differences to describe '
    'the remaining identity distinction and the observation that would resolve it. '
    'Revisit the hypotheses when new originals disagree with them. '
    'No candidate is removed and no modality has a fixed position in the plan. '
    'Keep each description to one concise sentence.\n')

PLAN_GUIDANCE = (
    '\nUse the tentative subject and type comparisons to reconsider the remaining '
    'identity issue against the original observations. Identify a record-level '
    'fact whose possible values would distinguish the competing candidate '
    'identities. Select an available operation that can actually reveal that '
    'fact, and connect its meaning to the distinction in relevance. Type '
    'hypotheses may be revised; they are neither identity decisions nor facts.\n')

PREFERENCE = (
    'Prioritize comparisons among candidates whose possible types are compatible '
    'with SRC; this is a soft attention preference, not an exclusion rule, and '
    'uncertain or conflicting type judgments may be checked and revised.\n')


def candidate_type_schema(handles):
    return object_schema({h: object_schema({
        'possible_type': TYPE_TEXT,
        'compatibility': {'enum': ['compatible', 'incompatible', 'uncertain']}})
        for h in handles})
