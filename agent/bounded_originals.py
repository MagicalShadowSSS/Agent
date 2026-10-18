"""Keep ordered original rows under a per-entity observation token budget."""
from copy import deepcopy
import json

def bound_rows(rows, tokenizer, budget, modality, owner):
    """Retain whole rows, then at most one clearly marked original-value prefix.

    No summaries, semantic filters or reordered records are introduced. Relation
    triples mirror the exact visible neighbor after clipping; IDs stay unchanged.
    """
    field = 'value' if modality == 'attribute' else 'neighbor'
    result=[]
    def encode_count(value):
        return len(tokenizer.encode(json.dumps(value,ensure_ascii=False,separators=(',',':')),add_special_tokens=False))
    def synchronize(row):
        if modality=='relation':
            row['triple']={'subject':owner if row['direction']=='out' else row[field],
                          'predicate':row['predicate'],
                          'object':row[field] if row['direction']=='out' else owner}
        return row
    for original in rows:
        row=synchronize(deepcopy(original))
        if encode_count(result+[row])<=budget:
            result.append(row)
            continue
        text=row[field]
        row['truncated']=True
        # Clip by characters so the visible value is an exact Unicode prefix.
        # Binary search need not find the maximal token packing; it always
        # validates the actual serialized observation before retaining it.
        lo,hi=0,len(text)
        best=None
        while lo<=hi:
            mid=(lo+hi)//2
            trial=deepcopy(row);trial[field]=text[:mid];synchronize(trial)
            if encode_count(result+[trial])<=budget:
                if mid>0:best=trial
                lo=mid+1
            else:hi=mid-1
        if best is not None:result.append(best)
        break
    tokens=encode_count(result)
    assert tokens<=budget
    originals={row['id']:row for row in rows}
    for row in result:
        original=originals[row['id']]
        assert row['predicate']==original['predicate']
        assert original[field].startswith(row[field])
        if not row.get('truncated'):assert row[field]==original[field]
        if modality=='relation':assert row['direction']==original['direction']
    return result,{'token_budget':budget,'visible_tokens':tokens,'input_rows':len(rows),
                   'visible_rows':len(result),'truncated_ids':[r['id'] for r in result if r.get('truncated')],
                   'row_order_preserved':True,'prefix_is_original_text':True}
