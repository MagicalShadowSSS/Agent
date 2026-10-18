"""Independent review experiments; no changes to reader facts or candidates."""
from copy import deepcopy

from .grounded import object_schema
from .type_guided import candidate_type_schema


DEFAULTS = {'compact_types': False, 'fresh_receipt': False,
            'revisit_pending': False, 'safe_rejection': True}
VARIANTS = {
    'control': ('type_prefer', {}),
    'compact_types': ('type_prefer', {'compact_types': True}),
    'fresh_receipt': ('type_prefer', {'fresh_receipt': True}),
    'revisit_pending': ('type_prefer', {'revisit_pending': True}),
    'combined': ('type_prefer', {'compact_types': True, 'fresh_receipt': True,
                                 'revisit_pending': True}),
    'original_protocol': ('original', {}),
}


def settings(overrides=None):
    overrides = overrides or {}
    if set(overrides) - set(DEFAULTS) or any(type(v) is not bool for v in overrides.values()):
        raise ValueError('Unknown review improvement or non-boolean setting')
    return {**DEFAULTS, **overrides}


def options(env):
    return settings(getattr(env, 'review_improvements', None))


def type_update_schema(handles):
    # No empty-array branch, and no permanently frozen candidate types.
    return {'anyOf': [
        object_schema({'status': {'enum': ['unchanged']}}),
        object_schema({'status': {'enum': ['revised']},
                       'candidate_types': candidate_type_schema(handles)})]}


def merge_types(result, previous, handles):
    result = deepcopy(result)
    change = result.pop('type_update')
    if change['status'] == 'unchanged':
        table = previous['candidate_types']
    elif change['status'] == 'revised':
        table = change['candidate_types']
    else:
        raise ValueError('Unknown type update')
    assert set(table) == set(handles)
    result['candidate_types'] = deepcopy(table)
    return result


COMPACT_GUIDANCE = (
    '\nThe previous candidate type table is available as revisable memory. '
    'Check it against the current original observations. In type_update return '
    'unchanged if all ten judgments remain appropriate; otherwise return revised '
    'with a complete replacement table. Any candidate may be revised, including '
    'one not queried. Reconsider compatibility if the source subject type changes. '
    'Do not repeat an unchanged table.\n')

FRESH_GUIDANCE = (
    '\nFor this update, assess the CURRENT RETURNED ORIGINALS first. '
    'observed_fact describes this receipt and its actual record owners, not a '
    'previous image or field. Text values and image references are supplied '
    'literally below; interpretations must remain tied to these references. '
    'Then reconsider the source subject, candidate types and remaining issue.\n')

PENDING_GUIDANCE = (
    '\nPREVIOUSLY ACQUIRED, NOT ACCEPTED ORIGINALS are available below for '
    'reassessment in light of the current identity issue. They are neither '
    'automatically evidence nor new tool results. pending_assessment may accept '
    'specific references that now establish or contradict exact identity, or '
    'leave them unaccepted. Only explicitly accepted originals reach the reader.\n')
