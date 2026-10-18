"""Hypothesis-conditioned tool policy with grounded predicate selection and explicit stopping."""
import json
import time
from .environment import EvidenceEnv,parse_object
from .engine import render


def field_catalog(env,focus,modality):
    catalog={}
    for h in focus:
        for i in env.unread(h,modality):
            value=env.records[h][modality][i];predicate,sep,_=value.partition(': ')
            if sep and predicate:catalog.setdefault(predicate,[]).append(h)
    return {p:list(dict.fromkeys(hs)) for p,hs in catalog.items()}


def solve_discrete(row,index,model,asset_dir,mode,budget):
    env=EvidenceEnv(row,index,model.compact,asset_dir,novel_only=True)
    env.hide_priors=mode=='discrete_blind'
    begin=model.calls;tokens_in=model.input_tokens;tokens_out=model.output_tokens;start=time.time()
    trace=[];feedback=[];inspections=0;verification_calls=0;accepted='U';status='budget_exhausted'
    candidates=[h for h in env.handles if h!='SRC']
    # A declined verification may request additional evidence; all candidates are reconsidered each round.
    for step in range(budget+1):
        state,visible=render(env,model.compact)
        prompt=('STATE\n'+state+'\nRECENT FEEDBACK\n'+json.dumps(feedback[-2:],ensure_ascii=False)+
            '\nCURRENT TASK: Select the most plausible exact-identity hypothesis for SRC among ALL 20 candidates. '
            'Compare identity, entity granularity and attribute/relation ownership. This is a provisional hypothesis, not acceptance. '
            'All supplied modalities have equal standing. Return only the candidate handle.')
        scores,meta=model.choose(prompt,env.images(),candidates)
        ranked=sorted(scores,key=scores.get,reverse=True);best,rival=ranked[:2];focus=['SRC',best,rival]
        trace.append({'kind':'hypothesis','step':step,'action':{'hypothesis':best,'rival':rival},'scores':scores,'model':meta})
        availability=env.available_tools()
        catalogs={m:field_catalog(env,focus,m) for m in ('attribute','relation')}
        available={m:[h for h in focus if h in availability[m]] for m in availability}
        # No fabricated predicate when a record has no structured field separator.
        for m in catalogs:
            if not catalogs[m]:available[m]=[]
        options={'S':'Stop acquiring evidence and submit the current hypothesis for exact-identity verification.',
                 'U':'Abstain: available evidence cannot resolve exact identity.'}
        tool_codes={'N':'name','A':'attribute','R':'relation','I':'image'}
        if inspections<budget:
            for code,m in tool_codes.items():
                if available[m]:options[code]=f'Inspect unread {m} evidence for '+','.join(available[m])
        if len(options)>2:
            task=(f'\nCURRENT HYPOTHESES: SRC might denote {best}; the strongest competing candidate is {rival}. '
                'Select ONE next action that can resolve this comparison. If exact identity is already supported, choose S now. '
                'Do not spend budget simply because it remains. Query only evidence whose result could change the conclusion. '
                'Missing fields are unknown; images may resolve readable identifiers or non-face distinguishing details. '
                f'Remaining tool actions: {budget-inspections}.\n'+json.dumps(options,ensure_ascii=False)+
                '\nAvailable original predicates (field names and entity availability, not factual answers):\n'+json.dumps(catalogs,ensure_ascii=False)+
                '\nOutput one action label.')
            actions,ameta=model.choose('STATE\n'+state+'\nRECENT FEEDBACK\n'+json.dumps(feedback[-2:],ensure_ascii=False)+task,env.images(),list(options))
            selected=max(actions,key=actions.get)
            trace.append({'kind':'planner','step':step,'action':{'action':selected,'hypothesis':best,'rival':rival},'scores':actions,'model':ameta,'error':None})
        else:selected='S'
        if selected=='U':status='abstained';break
        if selected=='S':
            verdicts=[]
            # Symmetric verification changes only left/right roles, never modality priority.
            for left,right in [('SRC',best),(best,'SRC')]:
                verification=('STATE\n'+state+f'\nVERIFY: Do records {left} and {right} denote exactly the same entity? '
                    f'Consider {rival} and the other candidates as competing explanations. '
                    'A=exact identity supported by supplied distinguishing evidence. B=distinct or merely related entities. '
                    'U=insufficient evidence. Facts about members, authors, locations and depicted objects are not automatically facts identifying the entity itself. '
                    'A shared scene, clothing, generic category or a matching substring alone is insufficient. '
                    'Any of names, attributes, relations or images may support or refute identity; absence is unknown. '
                    'Output only A, B or U.')
                probs,vmeta=model.choose(verification,env.images(),['A','B','U']);verdict=max(probs,key=probs.get)
                verification_calls+=1;verdicts.append(verdict)
                trace.append({'kind':'verifier','action':{'choice':verdict,'left':left,'right':right},'scores':probs,'model':vmeta})
            if verdicts==['A','A']:
                accepted=best;status='verified';break
            feedback.append({'verification':'unresolved','hypothesis':best,'verdicts':verdicts})
            if inspections>=budget:break
            # The next policy sees this failed verification. No hand-authored alternative is substituted.
            continue
        modality=tool_codes[selected];entities=available[modality];predicate=None
        if modality in catalogs:
            schema={'type':'object','properties':{'predicate':{'enum':list(catalogs[modality])}},'required':['predicate'],'additionalProperties':False}
            prompt=('STATE\n'+state+f'\nChoose the ONE existing {modality} predicate whose values would best distinguish SRC={best} from SRC={rival}. '
                'Return only {"predicate":"exact field name"}. Choose an identifying relation or property, not merely a shared category. '
                'The catalog is schema availability, not observed values:\n'+json.dumps(catalogs[modality],ensure_ascii=False))
            raw,pmeta=model.generate(prompt,env.images(),max_tokens=128,schema=schema)
            trace.append({'kind':'predicate_selection','raw':raw,'model':pmeta})
            try:predicate=parse_object(raw)['predicate']
            except (ValueError,KeyError):status='invalid_predicate';break
            if predicate not in catalogs[modality]:status='invalid_predicate';break
            entities=catalogs[modality][predicate]
        question=f'Does SRC denote {best} rather than {rival}? Compare '+(predicate or modality)+' evidence.'
        action={'action':'inspect','entities':entities,'modalities':[modality],'question':question,
            'expected':'Original evidence supporting one exact identity or leaving the comparison unresolved','offset':0}
        if predicate is not None:action['predicate']=predicate
        obs=env.inspect(action);inspections+=1
        trace.append({'kind':'tool','action':action,'observation':obs})
        feedback.append({'question':question,'refs':[v['ref'] for v in obs.get('evidence',[])],
            'new_evidence_count':obs.get('new_evidence_count'),'error':obs.get('error')})
    rank=env.rank(accepted)
    assert set(rank)==set(env.baseline) and len(rank)==20
    # These are evidence present in context, explicitly not model-selected citations or a generated explanation.
    _,visible=render(env,model.compact)
    context_refs=[r for r in sorted(visible) if env.ledger[r]['entity'] in ['SRC',accepted]] if accepted!='U' else []
    return {'source':env.source,'baseline_rank':env.baseline,'rank':rank,'status':status,
        'calls':model.calls-begin,'input_tokens':model.input_tokens-tokens_in,'output_tokens':model.output_tokens-tokens_out,
        'seconds':time.time()-start,'inspections':inspections,'verification_calls':verification_calls,
        'handle_map':env.handles,'trace':trace,'evidence':env.ledger,'evidence_context_refs':context_refs,
        'evidence_context_note':'Available original evidence, not model-selected citations. Questions are templates conditioned on model-selected hypotheses and predicates.'}
