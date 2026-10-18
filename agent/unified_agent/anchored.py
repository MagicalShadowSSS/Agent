"""Compare every proposed change against the retrieval incumbent, without labels.

Frozen discovery traces are reusable observations, not proof of identity. Each
audit sees only the source and the two actual alternatives. Model judgments
remain fallible; abstention preserves the retrieval incumbent.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time

from .grounded import GroundedEnv, focused_state, object_schema
from .revision import provenance_row, unique_choice
from .routing import select_modality

AUDIT_MODES = ('direct', 'scope', 'symmetric', 'active')


class AnchoredEnv(GroundedEnv):
    def __init__(self, row, index, compact, asset_dir, proposal):
        self.pair_uris = [row['source'], row['baseline']['rank'][0], proposal['rank'][0]]
        assert len(set(self.pair_uris)) == 3
        super().__init__(provenance_row(row), index, compact, asset_dir, novel_only=True)
        aliases = {self.reverse[u]: alias for alias, u in zip(('SRC', 'A', 'B'), self.pair_uris)}
        self.records = {aliases[h]: r for h, r in self.records.items() if h in aliases}
        self.assets = {aliases[h]: p for h, p in self.assets.items() if h in aliases}
        self.handles = dict(zip(('SRC', 'A', 'B'), self.pair_uris))
        self.reverse = {u: h for h, u in self.handles.items()}
        self.priors = {}; self.hide_priors = True
        ledger = {}
        for ref, item in self.ledger.items():
            if item['entity'] not in aliases: continue
            new_h = aliases[item['entity']]
            ledger[new_h + ref[ref.index('.'):]] = dict(item, entity=new_h)
        self.ledger = ledger
        overview = []
        for item in self.overview:
            if item['entity'] not in aliases: continue
            item = deepcopy(item); old_h = item['entity']; item['entity'] = aliases[old_h]
            item.pop('prior', None)
            for field in item['fields'].values():
                if field['ref']: field['ref'] = aliases[old_h] + field['ref'][len(old_h):]
            if item['image_ref']: item['image_ref'] = aliases[old_h] + item['image_ref'][len(old_h):]
            overview.append(item)
        self.overview = sorted(overview, key=lambda x: ('SRC', 'A', 'B').index(x['entity']))
        # Import only actually observed raw pages, never proposal explanations or
        # generated reflection claims. Saved IDs get their own namespace.
        for i, item in enumerate(proposal.get('evidence', {}).values()):
            if item['uri'] not in self.reverse or item['preview']: continue
            h = self.reverse[item['uri']]
            if item['modality'] == 'image':
                if h not in self.assets: continue
                ref = h + '.image.full'; self.recent_images.append(h)
            else:
                ref = f"{h}.{item['modality']}.saved{i}"
            self.ledger[ref] = dict(item, entity=h)
        self.recent_images = list(dict.fromkeys(self.recent_images))

    def _sheet(self, asset_dir):
        from PIL import Image, ImageDraw
        assets_by_uri = {self.handles[h]: p for h, p in self.assets.items()}
        subset = {u: assets_by_uri[u] for u in self.pair_uris if u in assets_by_uri}
        if not subset: return None
        asset_dir = Path(asset_dir); asset_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps([self.pair_uris, subset], sort_keys=True).encode()).hexdigest()
        path = asset_dir / (key + '.jpg')
        if path.exists(): return str(path)
        sheet = Image.new('RGB', (480, 180), 'white'); draw = ImageDraw.Draw(sheet)
        for i, (h, uri) in enumerate(zip(('SRC', 'A', 'B'), self.pair_uris)):
            draw.text((i*160+4, 2), h, fill='black')
            if uri in subset:
                with Image.open(subset[uri]) as src:
                    im = src.convert('RGB'); im.thumbnail((152, 152)); sheet.paste(im, (i*160+4, 22)); im.close()
            else: draw.text((i*160+4, 50), 'image unavailable', fill='gray')
        with tempfile.NamedTemporaryFile(dir=asset_dir, suffix='.jpg', delete=False) as stream: temp=Path(stream.name)
        try: sheet.save(temp, format='JPEG', quality=90); temp.replace(path)
        finally: temp.unlink(missing_ok=True); sheet.close()
        return str(path)

    def images(self):
        images = [('SOURCE AND TWO ALTERNATIVES: SRC, A, B', self.sheet)] if self.sheet else []
        images += [(f'ORIGINAL IMAGE {h}; evidence {h}.image.full', self.assets[h]) for h in self.recent_images]
        return images


def scope_choice(obj, visible, env):
    """Derive the decision from the role judgments, not a contradictory choice field."""
    for h, field in [('SRC', 'source_ref'), ('A', 'a_ref'), ('B', 'b_ref')]:
        ref = obj.get(field)
        if ref not in visible or env.ledger[ref]['entity'] != h: return 'U'
    if obj.get('source_scope') != 'self': return 'U'
    same = [h for h in ('A', 'B') if obj.get(h.lower()+'_relation') == 'same_entity'
            and obj.get(h.lower()+'_scope') == 'self']
    if len(same) != 1: return 'U'
    chosen = same[0]; other = 'B' if chosen == 'A' else 'A'
    # Absence of support for the incumbent is not a contradiction.
    if obj.get(other.lower()+'_relation') not in ('related_entity', 'different_entity'): return 'U'
    return chosen


def compare(model, env, mode, order=('A', 'B')):
    state, refs = focused_state(env, model, ['SRC'] + list(order))
    instruction = (
        '\nIdentify the SUBJECT denoted by SRC. A and B are actual alternative records, not components of SRC. '
        'Evaluate both alternatives against SRC using the original evidence. Neither alternative has a privileged rank. '
        'Facts can describe an associated entity: authors are not their works; offices are not their holders; '
        'a topic or aspect is not automatically the entire object; a part is not its whole. '
        'A matching word or shared image symbol establishes association, not necessarily identity. '
        'A textual predicate, directed relation, name/alias, or visible image identifier can establish identity '
        'only with compatible subject roles and granularity. Use all modalities equally. '
        'Missing evidence is unknown. Do not invent equivalence, translations or relations. '
        'The broad entity type alone cannot establish identity. No facial identification. ')
    if mode == 'direct':
        scores, meta = model.choose(state + instruction +
            'Return the single matching record label A or B only if it is distinguished from the other. '
            'Return U for unresolved, conflicting, or neither. Output only A, B, or U.', env.images(), list(order)+['U'])
        choice = unique_choice(scores)
        return choice, {'kind': 'anchored_comparison', 'action': {'choice': choice},
            'order': list(order), 'scores': scores, 'model': meta, 'error': None}
    fields = {'source_subject': {'type': 'string', 'maxLength': 96}}
    for h in order: fields[h.lower()+'_subject'] = {'type':'string','maxLength':96}
    fields['source_ref'] = {'enum': sorted(r for r in refs if env.ledger[r]['entity']=='SRC')}
    fields['source_scope'] = {'enum': ['self', 'related_entity', 'unclear']}
    for h in order:
        prefix = h.lower()
        candidates = sorted(r for r in refs if env.ledger[r]['entity']==h)
        if not candidates: return 'U', {'kind':'anchored_comparison','action':{'choice':'U'},'reason':'Missing alternative evidence','error':None}
        fields[prefix+'_ref'] = {'enum': candidates}
        fields[prefix+'_scope'] = {'enum': ['self', 'related_entity', 'unclear']}
        fields[prefix+'_relation'] = {'enum': ['same_entity', 'related_entity', 'different_entity', 'unknown']}
    if not fields['source_ref']['enum']:
        return 'U', {'kind':'anchored_comparison','action':{'choice':'U'},'reason':'Missing source evidence','error':None}
    prompt = state + instruction + (
        'First describe the three record subjects from the supplied observations, preserving their roles and granularity. '
        'Cite one visible reference for each. self means the cited observation identifies that record subject, '
        'related_entity means it identifies an associated object, unclear means ownership is unresolved. '
        'For EACH candidate classify its relationship to the SOURCE SUBJECT: same_entity requires distinguishing '
        'same-subject evidence; related_entity means associated but distinct; different_entity means distinct; '
        'unknown means insufficient evidence. Relations must agree with your subject descriptions. '
        'Do not collapse subject identity to a shared proper noun. Return only the required JSON; there is no separate choice field.')
    from .environment import parse_object
    raw, meta = model.generate(prompt, env.images(), schema=object_schema(fields), max_tokens=320)
    try: obj=parse_object(raw); error=None
    except (ValueError,TypeError) as exc: obj={}; error=str(exc)
    choice='U' if error else scope_choice(obj, refs, env)
    return choice, {'kind':'anchored_comparison','action':obj,'derived_choice':choice,
        'order':list(order),'raw':raw,'model':meta,'error':error}


def audit_override(row, index, model, asset_dir, proposal, mode, budget=4):
    assert mode in AUDIT_MODES
    assert proposal['source']==row['source'] and proposal['baseline_rank']==row['baseline']['rank']
    result=deepcopy(proposal); original=row['baseline']['rank'][0]
    result.update(audit_mode=mode,discovery_cached=True,audit_calls=0,audit_inspections=0)
    if proposal['rank'][0]==original:
        result['audit_outcome']='no_override'; return result
    assert original in proposal['rank'][:10], 'Restoring original rank must preserve proposal top10 membership'
    env=AnchoredEnv(row,index,model.compact,asset_dir,proposal)
    start=time.time();begin=model.calls;inp=model.input_tokens;out=model.output_tokens
    previous=getattr(model,'unicode_json',False);model.unicode_json=True
    trace=[];inspections=0;choice='U';checks=0
    try:
        for step in range((budget if mode=='active' else 0)+1):
            choice,t=compare(model,env,'direct' if mode=='direct' else 'scope');trace.append(t);checks+=1
            if mode in ('symmetric','active'):
                reverse,rt=compare(model,env,'scope',('B','A'));trace.append(rt);checks+=1
                if choice!=reverse:choice='U'
            if choice!='U' or mode!='active' or step==budget:break
            feedback={'audit':'incumbent and challenger not yet distinguished','comparisons':[t['action'] for t in trace[-2:]]}
            item,pt,decision=select_modality(model,env,['SRC','A','B'],'B','A',feedback,'grounded_route_pages',set())
            trace.append(pt)
            if decision!='inspect':break
            action={'action':'inspect','entities':item['entities'],'modalities':[item['modality']],
                'question':'Distinguish the SRC subject identity between A and B using original '+item['modality'],
                'expected':'A fact about the subject that discriminates the alternatives','offset':0}
            observation=env.inspect(action);inspections+=1
            trace.append({'kind':'tool','action':action,'observation':observation})
            if observation.get('error'):break
    finally:model.unicode_json=previous
    if choice=='B':
        result['audit_outcome']='confirmed_override'
    else:
        result['rank']=[original]+[u for u in proposal['rank'] if u!=original]
        result['status']='anchored_retained';result['audit_outcome']='restored_original' if choice=='A' else 'unresolved_restore_original'
    result['pre_audit_rank']=list(proposal['rank'])
    result['trace']=deepcopy(proposal['trace'])+[dict(t,stage='anchored_audit') for t in trace]
    result['audit_evidence']=env.ledger;result['audit_handle_map']=env.handles
    result['audit_calls']=model.calls-begin;result['audit_inspections']=inspections
    for key,value in [('calls',model.calls-begin),('input_tokens',model.input_tokens-inp),('output_tokens',model.output_tokens-out),
                      ('seconds',time.time()-start),('inspections',inspections),('verification_calls',checks)]:result[key]+=value
    result['accounting']='End-to-end includes cached discovery cost plus all anchored comparisons and evidence queries'
    assert set(result['rank'][:10])==set(proposal['rank'][:10])
    return result
