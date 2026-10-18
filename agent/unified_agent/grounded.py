"""Content-aware evidence acquisition and focused identity verification.

All generated enums/IDs are ASCII. Original multilingual predicates stay in
the environment and are addressed through a query-local catalog, never rewritten.
"""
import json
import re
import time
from .environment import EvidenceEnv,parse_object
from .engine import render


def normalized(text):return re.sub(r'\s+',' ',text).strip()


class GroundedEnv(EvidenceEnv):
    def unread(self,h,m):
        if m=='image':return super().unread(h,m)
        values=self.records[h][m];full='; '.join(values)
        preview=self.ledger.get(f'{h}.{m}.preview',{}).get('text','')
        # Mark only complete original values inside a genuine prefix, not arbitrary substrings.
        limit=0
        for actual,shown in zip(full,preview):
            if actual!=shown:break
            limit+=1
        covered=set();position=0
        for value in values:
            end=position+len(value)
            if end<=limit:covered.add(normalized(self.compact(value,96)))
            position=end+2
        for eid,item in self.ledger.items():
            if item['entity']==h and item['modality']==m and not item['preview']:
                covered.add(normalized(item['text']))
        # Equivalent original values at multiple indices count as already observed content.
        result=[]
        for i,value in enumerate(values):
            key=normalized(self.compact(value,96))
            if key not in covered:result.append(i);covered.add(key)
        return result

    def inspect(self,action):
        before={(v['entity'],v['modality'],normalized(v['text'])) for v in self.ledger.values() if not v['preview']}
        result=super().inspect(action)
        if 'error' not in result:
            result['new_distinct_values']=len({(v['entity'],v['modality'],normalized(v['text'])) for v in result['evidence']}-before)
        return result


def object_schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def ask_json(model,env,prompt,schema,kind):
    raw,meta=model.generate(prompt,env.images(),schema=schema,max_tokens=192)
    try:obj=parse_object(raw);error=None
    except (ValueError,TypeError) as exc:obj={};error=str(exc)
    return obj,{'kind':kind,'action':obj,'raw':raw,'model':meta,'error':error}


def focused_state(env,model,focus):
    state,visible=render(env,model.compact)
    lines=['FOCUSED ENTITY RECORDS. SRC is always the source to identify. Each other handle is a distinct candidate record.']
    refs=set()
    for h in focus:
        lines.append('\nRECORD '+h+' (source)' if h=='SRC' else '\nRECORD '+h+' (candidate)')
        # Preserve an unambiguous subject URI as provenance, not a gold alignment assertion.
        lines.append('Record URI: '+env.handles[h])
        for eid,item in env.ledger.items():
            if item['entity']==h and eid in visible:
                lines.append('['+eid+'] '+item['text']);refs.add(eid)
    lines.append('Relation direction: "out p: neighbor" means this record is subject of p; '
                 '"in p: neighbor" means the neighbor is subject and this record is object. '
                 'A neighbor name, relative, associated work or object in a picture is not automatically the identity of this record.')
    return '\n'.join(lines),refs


def verify(model,env,best,rival,strict=True):
    focus=list(dict.fromkeys(['SRC',best,rival,env.reverse[env.baseline[0]]]))
    state,refs=focused_state(env,model,focus)
    source_refs=sorted(r for r in refs if env.ledger[r]['entity']=='SRC')
    target_refs=sorted(r for r in refs if env.ledger[r]['entity']==best)
    if not source_refs or not target_refs:
        return False,{'kind':'verifier','action':{'verdict':'uncertain'},'error':'No visible source or target evidence'},'uncertain'
    # Decide identity before requesting any justification. Generating a support template first
    # was observed to make every pilot proposal appear supported, including wrong identities.
    def semantic_check(handle):
        pair_state,_=focused_state(env,model,['SRC',handle])
        prompt=(pair_state+f'\nEXACT IDENTITY CHECK: SRC and candidate {handle}. '
            f'The two subjects are SOURCE {env.handles["SRC"]} and TARGET {env.handles[handle]}. '
            'Only these two records are being compared; other entities on the image overview are not alternative subjects of this question. '
            'same means these two records identify exactly the same real-world entity. '
            'different means they identify distinct entities, including relatives, a song and its album, an organization and a member, or a subtype and its parent. '
            'unknown means the supplied evidence does not distinguish these possibilities. '
            'Compare the named subjects and the ownership of facts. A generic shared attribute or visual resemblance does not establish identity. '
            'Missing fields are unknown, not contradiction. Use all supplied modalities equally. '
            'Output only the semantic word same, different, or unknown. Do not output candidate handles, letters or JSON.')
        scores,meta=model.choose(prompt,env.images(),['same','different','unknown'])
        best_value=max(scores.values());winners=[k for k,v in scores.items() if v==best_value]
        return winners[0] if len(winners)==1 else 'unknown',scores,meta
    verdict,scores,semantic_meta=semantic_check(best)
    preliminary={'verdict':verdict}
    base_trace={'kind':'verifier','action':preliminary,'hypothesis':best,'rival':rival,'strict_scope':strict,
        'semantic_scores':scores,'semantic_model':semantic_meta,'semantic_score_note':'uncalibrated next-token scores; exact ties abstain'}
    if verdict!='same':
        base_trace['accepted']=False
        return False,base_trace,verdict
    rival_verdict,rival_scores,rival_meta=semantic_check(rival)
    base_trace.update(rival_verdict=rival_verdict,rival_scores=rival_scores,rival_model=rival_meta)
    if rival_verdict=='same':
        base_trace.update(accepted=False,reason='More than one distinct candidate judged identical; query a distinguishing fact')
        return False,base_trace,'uncertain'
    if not strict:
        base_trace['accepted']=True
        return True,base_trace,verdict
    fields={'source_ref':{'enum':source_refs},'target_ref':{'enum':target_refs}}
    if strict:
        fields.update({'source_scope':{'enum':['self','related_entity','unclear']},
            'target_scope':{'enum':['self','related_entity','unclear']},
            'comparison':{'enum':['identifying_agreement','identity_conflict','shared_nondistinctive','insufficient']}})
    fields['verdict']={'enum':['same','different','uncertain']}
    task=(f'\nIDENTITY QUESTION: Does SRC denote the same real-world entity as candidate {best}, '
        f'rather than {rival} or the other displayed alternatives? SRC stays the source throughout. '
        'Choose the most diagnostic supplied evidence for SRC and for the proposed candidate, then judge exact identity. '
        'Missing information is unknown. A song is not its album; a member is not their organization; a relative is not the person. '
        'Shared broad categories, generic visual appearance, years without the same predicate meaning, and presentation metadata cannot alone establish identity. '
        'Names, attributes, relations and images have equal standing. A strong identifying observation from any source may suffice. '
        'Use visible text, logos or non-face image details only; facial resemblance cannot establish identity. '
        'Do not treat URI provenance as a known alignment. Conflicting dates can reflect data noise and require contextual interpretation. '
        'Use only existing references in the corresponding records. Return JSON with the supplied fields; no extra prose.')
    if strict:
        task+=(' For each cited item, self means the identifying fact or name refers to that record itself; '
            'related_entity means it only describes a neighbor, family member, work or associated object; unclear means attribution cannot be established. '
            'identifying_agreement requires evidence that distinguishes this identity from the competitor; '
            'shared_nondistinctive means the observation fits multiple distinct candidates. '
            'Return same only for identifying agreement of facts attributable to both entities themselves.')
    obj,trace=ask_json(model,env,state+task,object_schema(fields),'verifier')
    valid=obj.get('source_ref') in source_refs and obj.get('target_ref') in target_refs
    accepted=not trace['error'] and valid and obj.get('verdict')=='same'
    if strict:accepted=accepted and obj.get('source_scope')==obj.get('target_scope')=='self' and obj.get('comparison')=='identifying_agreement'
    trace.update(hypothesis=best,rival=rival,accepted=bool(accepted),evidence_refs_valid=valid,strict_scope=strict)
    trace.update(semantic_scores=scores,semantic_model=semantic_meta,rival_verdict=rival_verdict,rival_scores=rival_scores,rival_model=rival_meta)
    return bool(accepted),trace,obj.get('verdict','uncertain')


def catalog(env,focus,contrastive=False,exclude_modalities=()):
    items=[]
    for modality in ('name','attribute','relation','image'):
        if modality in exclude_modalities:continue
        if modality in ('name','image'):
            handles=[h for h in focus if env.unread(h,modality)]
            if handles and (not contrastive or len(handles)>=2):
                items.append({'modality':modality,'entities':handles,'predicate':None})
            continue
        fields={}
        for h in focus:
            for i in env.unread(h,modality):
                value=env.records[h][modality][i];predicate,sep,_=value.partition(': ')
                if sep and predicate:fields.setdefault(predicate,[]).append(h)
        for predicate,handles in fields.items():
            handles=list(dict.fromkeys(handles))
            if not contrastive or len(handles)>=2:
                items.append({'modality':modality,'entities':handles,'predicate':predicate})
    # Catalog order affects IDs only. IDs have no action or truth semantics.
    return {f'field_{i:04d}':item for i,item in enumerate(items)}


def select_query(model,env,focus,best,rival,feedback,contrastive=False,exclude_modalities=()):
    choices=catalog(env,focus,contrastive=contrastive,exclude_modalities=exclude_modalities)
    if not choices:return None,{'kind':'planner','action':{'decision':'abstain'},'reason':'No unread evidence'},'abstain'
    state,_=focused_state(env,model,focus)
    schema=object_schema({'field_id':{'enum':list(choices)+['none']},
        'distinction':{'enum':['identity_name','entity_type_or_granularity','time_or_place','relation_role','visual_identifier','other_specific_fact','unresolvable']}})
    prompt=(state+'\nVERIFICATION FEEDBACK\n'+json.dumps(feedback,ensure_ascii=False)+
        f'\nNEXT EVIDENCE: SRC might denote {best} or {rival}. The identity verifier has not established a sufficient distinction. '
        'Choose one unread evidence item that could resolve the specific identity disagreement. '
        'Choose the field_id whose observation is most useful for distinguishing the two identities, then specify the kind of distinction. '
        'The field catalog lists availability only, not observed answers. Choose the property or relation whose original values could favor one hypothesis over the other. '
        +('Every listed field covers at least two of SRC, the proposal and the rival; prefer evidence that directly contrasts those records. ' if contrastive else '')+
        'An image query opens original pixels for inspecting readable identifiers or non-face details. '
        'Do not request a generic shared category or formatting metadata simply because it is available. '
        'No modality has priority. Choose none only if no remaining field or original image can help distinguish the identities; use unresolvable for its distinction. '
        'The selected field directly determines the next action: a catalog ID requests its original evidence, whereas none stops without changing the original ranking. '
        'Return only the structured JSON.\nCATALOG\n'+json.dumps(choices,ensure_ascii=False))
    obj,trace=ask_json(model,env,prompt,schema,'planner')
    if trace['error']:return None,trace,'invalid_plan'
    if obj.get('field_id')=='none':
        trace['action']['decision']='abstain'
        return None,trace,'abstain'
    item=choices.get(obj.get('field_id'))
    if not item:return None,trace,'invalid_plan'
    trace['action']['decision']='inspect'
    trace['resolved_original_field']=item
    return item,trace,'inspect'


def solve_grounded(row,index,model,asset_dir,mode,budget):
    from .routing import ROUTING_MODES,select_modality
    from .revision import REVISION_MODES,provenance_row,verify_competing
    revised=mode in REVISION_MODES
    if 'provenance' in mode:row=provenance_row(row)
    env=GroundedEnv(row,index,model.compact,asset_dir,novel_only=True)
    clock=time.time();begin=model.calls;tokens_in=model.input_tokens;tokens_out=model.output_tokens
    trace=[];feedback=[];inspections=0;verification_calls=0;accepted='U';status='budget_exhausted'
    strict=mode!='grounded_plain';fixed=mode=='grounded_fixed';contrastive=mode=='grounded_contrastive';diverse=mode=='grounded_diverse';fixed_used=set();used_modalities=set();examined=set()
    handles=[h for h in env.handles if h!='SRC'];state,_=render(env,model.compact)
    schema=object_schema({'hypothesis':{'enum':handles},'rival':{'enum':handles}})
    for turn in range(budget+1):
        state,_=render(env,model.compact)
        turn_schema=schema
        task=('\nPropose the most plausible exact identity of SRC among ALL20 candidates and its strongest distinct rival. '
            'Candidates can be reconsidered after every observation. Ranks are fallible priors, not known matches. '
            'Use entity names and the facts about the entity itself; do not mistake an associated person, work or organization for the entity. '
            'Return only JSON with hypothesis and rival handles; these must be different.')
        if 'reflect' in mode and feedback:
            task+=(' The last proposal did not establish identity. First describe the SRC subject using supplied evidence, '
                'then identify the unresolved difference between candidate identities. Reconsider ALL20 candidates, '
                'including the former rival. Repeating a rejected hypothesis requires addressing its failed verification '
                'with available evidence; do not mistake a source-related entity for the source. '
                'Missing information is not evidence against a candidate. Do not invent facts. '
                'Return source_identity, unresolved_difference, hypothesis, rival in that order.')
            turn_schema=object_schema({'source_identity':{'type':'string','maxLength':160},
                'unresolved_difference':{'type':'string','maxLength':180},
                'hypothesis':{'enum':handles},'rival':{'enum':handles}})
        obj,t=ask_json(model,env,'STATE\n'+state+'\nLATEST FEEDBACK\n'+json.dumps(feedback[-2:],ensure_ascii=False)+task,turn_schema,'hypothesis')
        trace.append(t);best=obj.get('hypothesis');rival=obj.get('rival')
        if t['error'] or best not in handles or rival not in handles or best==rival:status='invalid_hypothesis';break
        verifier=verify_competing if revised and 'compare' in mode else verify
        ok,vt,verdict=verifier(model,env,best,rival,strict);verification_calls+=1;trace.append(vt)
        if ok:accepted=vt.get('accepted_handle',best);status='verified';break
        feedback.append({'hypothesis':best,'rival':rival,'verification':vt['action'],'accepted':ok,
            'rival_verdict':vt.get('rival_verdict'),'reason':vt.get('reason'),'error':vt.get('error')})
        if inspections>=budget:break
        focus=['SRC',best,rival]
        if mode=='grounded_fixed_cycle':
            from .routing import modality_options
            options=modality_options(env,focus)
            order=('name','attribute','relation','image')
            cursor=getattr(env,'fixed_cursor',0)
            eligible=[(cursor+i)%4 for i in range(4) if order[(cursor+i)%4] in options]
            if not eligible:status='no_unread_evidence';break
            position=eligible[0];env.fixed_cursor=(position+1)%4
            item={k:v for k,v in options[order[position]].items() if k!='unread_counts'}
            pt={'kind':'planner','action':{'decision':'inspect','distinction':'fixed_cycle_control'},'resolved_original_field':item};outcome='inspect'
        elif mode in ROUTING_MODES or revised:
            routing_mode=mode if mode in ROUTING_MODES else 'grounded_route_pages'
            item,pt,outcome=select_modality(model,env,focus,best,rival,feedback[-1],routing_mode,used_modalities)
            if outcome=='inspect' and mode=='grounded_route_fields' and item['modality'] in ('attribute','relation'):
                trace.append(pt)
                excluded=set(('name','attribute','relation','image'))-{item['modality']}
                item,pt,outcome=select_query(model,env,focus,best,rival,feedback[-1],exclude_modalities=excluded)
        elif fixed:
            available=env.available_tools();modality=next((m for m in ('name','attribute','relation','image') if m not in fixed_used and any(h in available[m] for h in focus)),None)
            if modality is None:status='no_unread_evidence';break
            fixed_used.add(modality);item={'modality':modality,'entities':[h for h in focus if h in available[modality]],'predicate':None}
            pt={'kind':'planner','action':{'decision':'inspect','distinction':'fixed_control'},'resolved_original_field':item};outcome='inspect'
        else:item,pt,outcome=select_query(model,env,focus,best,rival,feedback[-1],contrastive=contrastive,
                                           exclude_modalities=used_modalities if diverse else ())
        trace.append(pt)
        if outcome!='inspect':status='abstained' if outcome=='abstain' else outcome;break
        action={'action':'inspect','entities':item['entities'],'modalities':[item['modality']],
            'question':f'Distinguish SRC={best} from SRC={rival} using '+(item['predicate'] or item['modality']),
            'expected':'Original evidence resolving the competing identity interpretations','offset':0}
        if item['predicate'] is not None:action['predicate']=item['predicate']
        # Reloading previously viewed images is allowed by the base environment. This study instead stops exact repeated evidence actions.
        signature=json.dumps(item,sort_keys=True)
        # Text pages advance through unread values; repeating the same tool is not a repeated observation.
        # Preserve historical modes exactly for replication, fix pagination in the new routing arms.
        if signature in examined and ((mode not in ROUTING_MODES and not revised) or item['modality']=='image'):
            status='no_new_evidence';break
        examined.add(signature);observation=env.inspect(action);inspections+=1
        used_modalities.add(item['modality'])
        trace.append({'kind':'tool','action':action,'observation':observation})
        if observation.get('error'):status='tool_error';break
        feedback.append({'query':action,'returned_refs':[v['ref'] for v in observation['evidence']],
            'new_distinct_values':observation.get('new_distinct_values')})
    rank=env.rank(accepted)
    assert len(rank)==20 and set(rank)==set(env.baseline)
    return {'source':env.source,'baseline_rank':env.baseline,'rank':rank,'status':status,'trace':trace,
        'handle_map':env.handles,'evidence':env.ledger,'inspections':inspections,'verification_calls':verification_calls,
        'calls':model.calls-begin,'input_tokens':model.input_tokens-tokens_in,'output_tokens':model.output_tokens-tokens_out,
        'seconds':time.time()-clock,'verification_scope':'focused source/proposal/rival/baseline records; original image inputs preserved',
        'semantic_scope_is_model_judgment':True}
