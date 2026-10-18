"""Jointly executable questions and observation-grounded identity decisions.

No label thresholds or entity-specific repairs. A question and exact original
tool are generated together; returned observations are checked for answerability
before replanning. The observation assessment never enters the factual ledger.
"""
from copy import deepcopy
import json
import time

from .ambiguity import make_env
from .active_top10 import options_for, verify as evidence_verify
from .grounded import object_schema
from .json_protocol import SHORT_TEXT
from .zh_active_evidence import (state, state_hash, ask, semantic_operations,
                                 semantic_verify, solve as legacy_solve)

MODES = ('single_active', 'bound_active', 'grounded_active', 'grounded_joint')
TEXT = SHORT_TEXT


def plan(model, env, feedback, history, allow_verify, remaining):
    tools = semantic_operations(options_for(env,'adaptive',set())) if remaining else {}
    operations = [object_schema({'tool':{'enum':[key]},
        'entities':{'type':'array','items':{'enum':item['entities']},
                    'minItems':1,'maxItems':min(3,len(item['entities']))}})
        for key,item in tools.items()]
    branches=[]
    if operations:
        branches.append(object_schema({'action':{'enum':['inspect']},
            'operation':{'anyOf':operations},'question':TEXT,'expected_observation':TEXT}))
    if allow_verify:
        branches.append(object_schema({'action':{'enum':['verify']},'basis':TEXT}))
    branches.append(object_schema({'action':{'enum':['stop']},'reason':TEXT}))
    schema=object_schema({'source_subject':TEXT,'ambiguity':TEXT,'next':{'anyOf':branches}})
    text,_=state(env,model)
    prompt=(text+'\nFALLIBLE ASSESSMENTS (not facts)\n'+json.dumps(feedback,ensure_ascii=False)+
        '\nEXECUTED QUESTIONS AND OBSERVATION ASSESSMENTS (interpretations, not new facts)\n'+
        json.dumps(history,ensure_ascii=False)+
        '\nAVAILABLE ORIGINAL TOOLS (addresses and owners only; unread values are hidden)\n'+
        json.dumps({k:v['entities'] for k,v in tools.items()},ensure_ascii=False)+
        '\nIdentify SRC among all TEN candidates. Briefly identify the source record subject and '
        'the unresolved distinction between plausible identities. Then select an executable next step. '
        'For inspect, bind an available tool and its record owners, then formulate a neutral factual '
        'question that this exact operation can answer and explain how alternative observations '
        'would distinguish the identities. Keep record subjects and predicate meaning consistent. '
        'Expected answers are hypotheses, never observations. Name, attribute, directed relation '
        'and original image are peers. Text properties return text; image: original_pixels opens pixels. '
        'Use verify if existing observations are sufficient, stop if no available observation helps. '
        'All ten candidates remain eligible. Use one concise sentence per text field. '
        +('Verification of unchanged observations is unavailable. ' if not allow_verify else '')+
        f'Remaining operations: {remaining}. Return source_subject, ambiguity and next. '
        'next is {"action":"inspect","operation":{"tool":"exact address","entities":["handle"]},'
        '"question":"fact to read","expected_observation":"how answers distinguish"}, '
        '{"action":"verify","basis":"observed basis"}, or {"action":"stop","reason":"limitation"}.')
    obj,trace=ask(model,env,prompt,schema,'planner')
    if trace['error']: return 'error',None,trace
    next_step=obj.get('next',{}); move=next_step.get('action')
    if move in ('verify','stop'):
        if move=='verify' and not allow_verify: trace['error']='Repeated unchanged-state verification'
        return move,None,trace
    operation=next_step.get('operation',{}); item=tools.get(operation.get('tool'))
    entities=operation.get('entities',[])
    if (move!='inspect' or not item or not 1<=len(entities)<=3 or
        any(h not in item['entities'] for h in entities) or
        not next_step.get('question') or not next_step.get('expected_observation')):
        trace['error']='Invalid tool-bound question';return 'error',None,trace
    action={'action':'inspect','entities':list(dict.fromkeys(entities)),
        'modalities':[item['modality']],'question':next_step['question'],
        'expected':next_step['expected_observation'],'offset':0}
    if item['predicate'] is not None: action['predicate']=item['predicate']
    trace['resolved_original_field']={**item,'entities':action['entities']}
    return 'inspect',action,trace


def assess_observation(model, env, action, observation):
    refs=[e['ref'] for e in observation['evidence']]
    schema=object_schema({'answer_status':{'enum':['answered','partial','unanswered']},
        'observation_refs':{'type':'array','items':{'enum':refs},'minItems':0,'maxItems':len(refs)},
        'observed_answer':TEXT,'remaining_question':TEXT})
    text,_=state(env,model,action['entities'])
    prompt=(text+'\nEXECUTED FACTUAL QUESTION\n'+action['question']+
        '\nNEW RAW OBSERVATIONS\n'+json.dumps(observation['evidence'],ensure_ascii=False)+
        '\nAssess whether these returned observations answer the question about the requested '
        'record subjects. State the observed answer and cite only returned references; actual pixels '
        'must be inspected for image references. partial or unanswered means the fact is still '
        'unresolved, not that a candidate is wrong. State a remaining question, or say none when '
        'answered. Do not decide alignment here or convert expected answers into facts. '
        'Return answer_status, observation_refs, observed_answer, remaining_question; keep prose concise.')
    obj,trace=ask(model,env,prompt,schema,'observation_assessment')
    if not trace['error'] and (obj.get('answer_status') not in ('answered','partial','unanswered') or
        any(r not in refs for r in obj.get('observation_refs',[]))):
        trace['error']='Invalid observation assessment'
    return obj,trace


def solve(row,index,model,folder,mode,budget=4):
    assert mode in MODES
    if mode=='single_active':return legacy_solve(row,index,model,folder,mode,budget)
    env=make_env(row,index,model,folder)
    started=time.time();begin=(model.calls,model.input_tokens,model.output_tokens)
    previous=getattr(model,'unicode_json',False);model.unicode_json=True
    trace=[];seen=set();history=[];feedback={};inspections=0;choice='U'
    status='unresolved';no_query_rank=None;must_verify=(mode=='grounded_joint')
    try:
        while True:
            before=state_hash(env,model)
            if must_verify or inspections>=budget:
                move='verify';action=None
            else:
                move,action,t=plan(model,env,feedback,history,before not in seen,budget-inspections)
                trace.append(t)
                if t['error']:status='invalid_plan';break
                if move=='stop':break
            if move=='verify':
                if before in seen:status='no_observation_progress';break
                seen.add(before);must_verify=False
                if mode=='bound_active':
                    choice,checks=semantic_verify(model,env,folder/'focused',unified_decision=True)
                else:
                    choice,t=evidence_verify(model,env,evidence_first=True);checks=[t]
                for t in checks:
                    if t['kind']=='verifier':t['observation_state_sha256']=before
                trace.extend(checks);feedback=checks[-1]['action']
                if inspections==0:no_query_rank=env.rank(choice)
                if any(t.get('error') for t in checks):status='invalid_verification';choice='U';break
                if choice!='U':status='verified';break
                if mode=='grounded_joint' or inspections>=budget:break
                continue
            observation=env.inspect(action);inspections+=1
            trace.append({'kind':'tool','action':action,'observation':observation})
            if observation.get('error'):status='tool_error';break
            if state_hash(env,model)==before:status='no_observation_progress';break
            assessment,t=assess_observation(model,env,action,observation);trace.append(t)
            if t['error']:status='invalid_observation_assessment';break
            history.append({'question':action['question'],'operation':{k:v for k,v in action.items()
                if k not in ('question','expected')},'assessment':assessment})
            # Verification receives only the updated raw ledger, never this model's
            # answer summary or its anticipated values. Unanswered queries replan.
            must_verify=assessment['answer_status'] in ('answered','partial')
        result={'source':env.source,'baseline_rank':list(env.baseline),'rank':env.rank(choice),
            'status':status,'mode':mode,'trace':trace,'handle_map':dict(env.handles),
            'evidence':deepcopy(env.ledger),'inspections':inspections,'no_query_rank':no_query_rank,
            'verification_calls':sum(t['kind']=='verifier' for t in trace),
            'calls':model.calls-begin[0],'input_tokens':model.input_tokens-begin[1],
            'output_tokens':model.output_tokens-begin[2],'seconds':time.time()-started,
            'candidate_count':10,'labels_loaded':False}
        assert len(result['rank'])==len(set(result['rank']))==10
        assert set(result['rank'])==set(row['baseline']['rank'])
        return result
    finally:model.unicode_json=previous
