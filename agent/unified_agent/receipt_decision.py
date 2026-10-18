"""Short subject-aware decisions and observable, two-query acquisition policies.

Every condition uses the same reader. Generated interpretations guide planning
only; the reader receives original observations, not a previous model answer.
"""
import time

from .acquisition146 import structured
from .acquisition_repair import EvidenceMemory, bind_request
from .active_top10 import options_for
from .ambiguity import make_env
from .grounded import object_schema
from .zh_active_evidence import state, state_hash, semantic_operations
import json

TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 120}
METHODS = ('no_acquisition', 'preplanned', 'adaptive')


def subject_reader(model, env):
    text, refs = state(env, model)
    def references(owner):
        items = sorted(r for r in refs if env.ledger[r]['entity'] == owner
                       and env.ledger[r]['location'] != 'record/uri')
        return items or [owner+'.record']
    schema = object_schema({
        'source_subject': TEXT,
        'source_ref': {'enum': references('SRC')},
        'identity_comparison': {'type': 'string', 'minLength': 1, 'maxLength': 160},
        'decision': {'anyOf': [object_schema({'choice': {'enum': [h]},
            'candidate_ref': {'enum': references(h)}}) for h in env.priors]}})
    prompt = (text + '\nRECORD SUBJECT ALIGNMENT. Determine what SOURCE SRC itself denotes, '
        'then identify the same subject among ALL TEN candidates. A record may describe '
        'a related participant, work or place, and its picture may depict an associated '
        'object. Read field ownership and relation direction before treating a shared '
        'name, fact or appearance as identifying agreement. Missing facts are unknown. '
        'Use actual observations and attached pixels; retrieval ranks can be wrong. '
        'Briefly describe SRC, cite one source observation, explain the decisive identity '
        'comparison, and select a candidate with an observation from its own record. '
        'Every candidate remains eligible. Return the required JSON.')
    result, trace = structured(model, prompt, env.images(), schema, 'subject_reader', max_tokens=512)
    choice = result['decision']['choice']
    assert result['source_ref'] in references('SRC')
    assert result['decision']['candidate_ref'] in references(choice)
    trace.update(choice=choice, visible_refs=sorted(refs), observation_state_sha256=state_hash(env, model))
    return choice, trace


def initial(row, index, model, folder):
    env = make_env(row, index, model, folder);env.source_last=True
    choice, trace = subject_reader(model, env)
    return {'source':row['source'],'choice':choice,'rank':env.rank(choice),'handle_map':env.handles,
        'trace':[trace],'initial_state_sha256':state_hash(env,model),'inspections':0,
        'candidate_count':10,'labels_loaded':False}


def query_schema(tools):
    operations=[object_schema({'tool':{'enum':[key]},'entities':{'type':'array',
        'items':{'enum':item['entities']},'minItems':1,'maxItems':min(3,len(item['entities']))}})
        for key,item in tools.items()]
    return object_schema({'identity_question':TEXT,'operation':{'anyOf':operations},
                          'expected_discrimination':TEXT})


def plan(model, memory, assessment, history, remaining, preplanned=False):
    tools=semantic_operations(options_for(memory.env,'adaptive',set()))
    if not tools:
        return [], {'kind':'planner','reason':'No unread operation exists','action':{'next':{'action':'stop'}}}
    stop=object_schema({'action':{'enum':['stop']},'reason':TEXT})
    query=query_schema(tools)
    if preplanned:
        inspect=object_schema({'action':{'enum':['inspect']},'queries':{'type':'array',
            'items':query,'minItems':1,'maxItems':remaining}})
    else:
        inspect=object_schema({'action':{'enum':['inspect']},**query['properties']})
    schema=object_schema({'next':{'anyOf':[stop,inspect]}})
    text,_=state(memory.view(),model)
    prompt=(text+'\nFALLIBLE LAST SUBJECT COMPARISON (interpretation, not extra evidence):\n'+
        json.dumps(assessment,ensure_ascii=False)+'\nEXECUTED QUERIES:\n'+json.dumps(history,ensure_ascii=False)+
        '\nAVAILABLE ORIGINAL FIELDS AND THEIR RECORD OWNERS:\n'+
        json.dumps({k:v['entities'] for k,v in tools.items()},ensure_ascii=False)+
        '\nDecide whether another original observation can resolve a concrete remaining '
        'identity distinction about SRC. Stop if observed evidence already distinguishes '
        'the identity, or if available observations cannot address the remaining question. '
        'Otherwise state the identity question, select the exact field or original image '
        'and its record owners, and say how different observations would distinguish the '
        'candidate identities. Do not query merely to confirm a resolved comparison. '
        'Previous subject and type interpretations can be wrong; reconsider any of the ten '
        'candidates. All four modalities are available without a fixed order. '
        f'Maximum remaining tool calls: {remaining}. ')
    if preplanned:
        prompt+=('Plan at most two queries now. They will be executed in this order without '
                 'replanning from their results, so choose complementary observations. ')
    result,trace=structured(model,prompt,memory.view().images(),schema,'planner',max_tokens=768)
    nxt=result['next']
    if nxt['action']=='stop':return [],trace
    queries=nxt['queries'] if preplanned else [{k:nxt[k] for k in query['properties']}]
    actions=[]
    for q in queries:
        action,request=bind_request(memory.env,tools,q['operation'],
            {'distinction':q['identity_question'],'needed_fact':q['expected_discrimination']})
        actions.append((action,request))
    trace['bound_requests']=[r for _,r in actions]
    return actions,trace


def solve(row,index,model,folder,baseline,method,budget=2):
    assert method in METHODS and 0<=budget<=2
    env=make_env(row,index,model,folder);env.source_last=True
    memory=EvidenceMemory(env)
    assert state_hash(memory.view(),model)==baseline['initial_state_sha256']
    current=baseline['choice'];assessment=baseline['trace'][0]['action']
    cache={baseline['initial_state_sha256']:(current,baseline['trace'][0])}
    trace=[];history=[];inspections=0;stopped=False
    counts=model.calls,model.input_tokens,model.output_tokens;started=time.perf_counter()
    pending=[]
    if method!='no_acquisition':
        while inspections<budget:
            if method=='adaptive' or not trace:
                pending,t=plan(model,memory,assessment,history,budget-inspections,method=='preplanned')
                trace.append(t)
                if not pending:stopped=True;break
            if not pending:break
            action,request=pending.pop(0)
            observation=memory.inspect(action)
            inspections+=1
            # Hypotheses never replace tool output: the common reader sees the
            # literal receipts, including potentially conflicting observations.
            memory.accept(observation,[e['ref'] for e in observation['evidence']])
            trace.append({'kind':'tool','action':action,'bound_request':request,'observation':observation})
            view=memory.view();digest=state_hash(view,model)
            if digest in cache:
                current,last=cache[digest]
                trace.append({'kind':'reader_cache','choice':current,'observation_state_sha256':digest})
            else:
                current,last=subject_reader(model,view);cache[digest]=(current,last);trace.append(last)
            assessment=last['action']
            history.append({'request':request,'returned_refs':[e['ref'] for e in observation['evidence']]})
            if method=='preplanned' and not pending:break
    assert len(env.rank(current))==10 and set(env.rank(current))==set(baseline['rank'])
    return {'source':row['source'],'choice':current,'rank':env.rank(current),'initial_choice':baseline['choice'],
        'handle_map':env.handles,'method':method,'trace':trace,'inspections':inspections,
        'status':'stopped' if stopped else 'budget_reached' if inspections==budget else 'plan_complete',
        'candidate_count':10,'labels_loaded':False,'seconds':time.perf_counter()-started,
        'calls':model.calls-counts[0],'input_tokens':model.input_tokens-counts[1],
        'output_tokens':model.output_tokens-counts[2],'accepted_refs':sorted(memory.accepted)}
