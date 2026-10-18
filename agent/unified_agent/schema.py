"""Decode only syntactically valid actions and citations; never constrain the answer by gold."""

def schema_for(env,visible,allow_inspect):
    alternatives=[]
    if allow_inspect and env.novel_only:
        for modality,handles in env.available_tools().items():
            if not handles:continue
            alternatives.append({'type':'object','properties':{
                'action':{'enum':['inspect']},
                'question':{'type':'string','minLength':1,'maxLength':140},
                'expected':{'type':'string','maxLength':100},
                'modalities':{'type':'array','items':{'enum':[modality]},'minItems':1,'maxItems':1},
                'entities':{'type':'array','items':{'enum':handles},'minItems':1,'maxItems':3},
                'offset':{'enum':[0]}},
                'required':['action','question','expected','modalities','entities','offset'],'additionalProperties':False})
    elif allow_inspect:
        alternatives.append({'type':'object','properties':{
            'action':{'enum':['inspect']},
            'entities':{'type':'array','items':{'enum':list(env.handles)},'minItems':1,'maxItems':3},
            'modalities':{'type':'array','items':{'enum':['name','attribute','relation','image']},'minItems':1,'maxItems':1},
            'question':{'type':'string','minLength':1,'maxLength':140},
            'expected':{'type':'string','maxLength':100},
            'offset':{'type':'integer','minimum':0,'maximum':512}},
            'required':['action','entities','modalities','question','expected','offset'],'additionalProperties':False})
    source_refs=sorted(r for r in visible if env.ledger[r]['entity']=='SRC')
    for handle in env.handles:
        if handle=='SRC':continue
        candidate_refs=sorted(r for r in visible if env.ledger[r]['entity']==handle)
        if not source_refs or not candidate_refs:continue
        alternatives.append({'type':'object','properties':{
            'action':{'enum':['finish']},'choice':{'enum':[handle]},
            'source_ref':{'enum':source_refs},'candidate_ref':{'enum':candidate_refs},
            'rationale':{'type':'string','maxLength':180}},
            'required':['action','choice','source_ref','candidate_ref','rationale'],'additionalProperties':False})
    alternatives.append({'type':'object','properties':{
        'action':{'enum':['finish']},'choice':{'enum':['U']},'rationale':{'type':'string','maxLength':180}},
        'required':['action','choice','rationale'],'additionalProperties':False})
    return {'anyOf':alternatives}


def normalize(obj):
    if obj.get('action')=='finish':
        obj=dict(obj)
        if obj.get('choice')=='U':obj['citations']=[]
        elif 'source_ref' in obj and 'candidate_ref' in obj:
            obj['citations']=[obj['source_ref'],obj['candidate_ref']]
    return obj
