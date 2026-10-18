import os
"""Paired modality agents: independent decision, bounded evidence, and joint routing."""
import argparse
import ctypes
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT=Path(os.environ.get('SGMEA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
BASE=Path(os.environ.get('SGMEA_BASE_ROOT', str(PROJECT/'runs')))
POOL=BASE
ROOT=Path(os.environ.get('SGMEA_V69_ROOT', str(PROJECT/'runs')))
FINAL_HEAD='same_planner'
WAIT_FOR=None
DEFAULT_METHODS=('source24_tok512_scope_grounded_notes3_full_budget2_allpixels_receipts', 'source24_tok1024_scope_grounded_notes3_full_budget2_allpixels_receipts')
_METHOD_FILTER=os.environ.get('SGMEA_V69_METHODS','').strip()
METHODS=tuple(x for x in DEFAULT_METHODS if not _METHOD_FILTER or x in _METHOD_FILTER.split(','))
GPU_PAIRS=((0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5), (6, 7), (6, 7))
RAW_KG_SHA256={'dataset/DBP15K/fr_en/fr_rel_triples': '8db3e44b24a77f153ce24558cb56eeb4c47c36dc3de72c10d18071552c8ad18d', 'dataset/DBP15K/fr_en/en_rel_triples': '1e4d5752fc6cf21f2347098dab4488e61b1573aec27b0230e375486fda1163db', 'dataset/DBP15K/ja_en/ja_rel_triples': 'bdc07af71b551cc87001ff024347f485cd79af2ad1981b9cd30497533de44684', 'dataset/DBP15K/ja_en/en_rel_triples': 'e43b85a974b01d4acf9922c5109924be6363638e87f84ee761df80bc38b2cc70', 'dataset/DBP15K/zh_en/zh_rel_triples': 'b2e5aa0ae41154cdec96ecc479c32d7a62be16d7970137364c6ddfc3370a8d08', 'dataset/DBP15K/zh_en/en_rel_triples': '86475df6f95dc87ea1a6712d30038ac40cb55ce13e9286120dcee7ba2757e22b'}
TYPES=['Person','Organization','Place','Creative Work','Event','Entity']
LABELS=list('ABCDEFGHIJ')
MAX_WORKER_RSS_GIB=18.0

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def trim_host_memory():
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (AttributeError, OSError):
        pass


def worker_rss_gib():
    fields=Path('/proc/self/statm').read_text().split()
    return int(fields[1])*os.sysconf('SC_PAGE_SIZE')/(1024**3)


def restart_if_host_memory_high(wid):
    trim_host_memory()
    rss=worker_rss_gib()
    dump(ROOT/f'host_memory/{wid}.json',{'rss_gib':rss,'restart_threshold_gib':MAX_WORKER_RSS_GIB,'time':time.time()})
    if rss < MAX_WORKER_RSS_GIB:
        return
    dump(ROOT/f'host_memory_restarts/{wid}_{int(time.time())}.json',{'rss_gib':rss,'worker':wid,'time':time.time()})
    os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--worker-id',str(wid)])


def receipt_block(receipts,candidate=None):
    lines=['EVIDENCE AGENT RECEIPTS (untrusted summaries; verify every claim against cited original records)']
    for item in receipts:
        result=item['result']
        if candidate is not None and isinstance(result,dict) and isinstance(result.get('observations'),list):
            result={'observations':[o for o in result['observations'] if o.get('candidate')==candidate]}
        lines.append(f"{item['agent'].upper()} RECEIPT")
        lines.append(json.dumps(result,ensure_ascii=False,separators=(',',':')))
    return '\n'.join(lines)


def helper():
    source=Path(__file__).resolve().with_name('evidence_helper.py')
    if not source.exists(): raise FileNotFoundError(source)
    spec=importlib.util.spec_from_file_location('evidence_helper',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def verify():
    for filename,base in [('code_sha256.json',ROOT/'code_snapshot'),('input_sha256.json',ROOT)]:
        for p,h in read(ROOT/filename).items():assert sha(base/p)==h,p


def wait_for_gpu_headroom(wid):
    pair=GPU_PAIRS[wid]
    while True:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(a):int(b) for a,b in (line.split(',') for line in output.strip().splitlines())}
        if all(free[g]>=10500 for g in pair):
            return
        dump(ROOT/f'waiting/{wid}.json',{'gpu_pair':pair,'free_mib':free,'time':time.time(),'reason':'wait for model/context headroom; do not stop unrelated jobs'})
        time.sleep(20)


def explicit_relations(rows, owner):
    for row in rows:
        direction=row.get('direction')
        if direction not in ['in','out']:continue
        row['triple']={'subject':owner if direction=='out' else row.get('neighbor',''),
            'predicate':row.get('predicate',''),
            'object':row.get('neighbor','') if direction=='out' else owner}


def planner_modality_choices(available, allow_initial_finish=True, initial_round=False):
    """Return the planner action set for the current adaptive protocol."""
    if initial_round and not allow_initial_finish and available:
        return list(available)
    return list(available) + ['finish']


def selected_tasks(tasks):
    """Keep a fixed prefix per dataset for small paired protocol pilots."""
    limit=int(os.environ.get('SGMEA_V69_TASK_LIMIT','0'))
    if limit <= 0:
        return tasks
    selected=[]
    counts={}
    for task in tasks:
        dataset=task.get('dataset',task.get('cohort',''))
        if counts.get(dataset,0) >= limit:
            continue
        selected.append(task)
        counts[dataset]=counts.get(dataset,0)+1
    return selected


def worker(wid):
    wait_for_gpu_headroom(wid)
    verify();sys.path.insert(0,str(Path(__file__).resolve().parent))
    import torch,jsonschema
    from transformers import Qwen3VLForConditionalGeneration
    from unified_agent.modality_agents_fixed import Evidence
    from unified_agent.modality_agents import enum,obj
    from unified_agent.modality_agents_states import RoleClient
    from unified_agent.model_backends import ComparisonVLM
    from decision_heads import blocks,FINAL,REASON
    from bounded_originals import bound_rows
    # The shared filesystem is nearly full.  Keep verbose per-call request and
    # model transcript artifacts on the node-local filesystem; final decisions
    # and traces remain under ROOT for resumability.
    local_call_root=Path(os.environ.get('SGMEA_LOCAL_CALL_ROOT', '/tmp/sgmea_v69_calls')) / str(wid)
    original_role_init=RoleClient.__init__
    def local_role_init(client, model_, prompts_, folder_, validate_):
        folder_=Path(folder_)
        try:
            relative=folder_.relative_to(ROOT)
        except ValueError:
            relative=Path(folder_.name)
        original_role_init(client, model_, prompts_, local_call_root/relative, validate_)
    RoleClient.__init__=local_role_init
    for gpu in range(2):torch.cuda.set_per_process_memory_fraction(0.52,gpu)
    original=Qwen3VLForConditionalGeneration.from_pretrained
    def load(*args,**kwargs):
        kwargs['device_map']={'model.visual':'cuda:0','model.language_model.embed_tokens':'cuda:0','model.language_model.rotary_emb':'cuda:0','model.language_model.norm':'cuda:1','lm_head':'cuda:1',**{f'model.language_model.layers.{j}':f'cuda:{int(j>=18)}' for j in range(36)}}
        return original(*args,**kwargs)
    Qwen3VLForConditionalGeneration.from_pretrained=load;torch.manual_seed(0)
    model=ComparisonVLM('qwen3',os.environ.get('SGMEA_VL_MODEL', read(ROOT/'runtime_model.json').get('path','models/Qwen3-VL-8B-Instruct')),Path(os.environ.get('SGMEA_AGENT_DEPENDENCY_ROOT', str(PROJECT/'agent_dependencies'))));m=helper()
    from bounded_token_outputs import mlp as bounded_mlp_output, norm as bounded_norm_output
    # Text FFN is independent per token: bound intermediates without truncating context.
    chunked_count=0
    for module in model.model.modules():
        if module.__class__.__name__=='Qwen3VLTextMLP':
            dense_forward=module.forward
            def bounded_forward(x,forward=dense_forward):
                return bounded_mlp_output(x,forward,2048)
            module.forward=bounded_forward
            chunked_count+=1
    for module in model.model.modules():
        if module.__class__.__name__=='Qwen3VLTextRMSNorm':
            dense_norm=module.forward
            def bounded_norm(x,forward=dense_norm):
                return bounded_norm_output(x,forward,2048)
            module.forward=bounded_norm
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from bounded_rotary import install
    install(modeling_qwen3_vl,2048)
    assert chunked_count==36,chunked_count
    dump(ROOT/f'memory_implementation/{wid}.json',{'text_ffn_layers':chunked_count,'chunk_tokens':2048,'weights_unchanged':True,'precision':'BF16','context_truncated':False,'placement':'two GPUs BF16 per worker','allocator_fraction':0.52,'time':time.time()})
    original_role_call=RoleClient.__call__
    def guarded_role_call(client,role,payload,images,schema):
        while True:
            # Clear only this process's unused cache, never stop another process.
            torch.cuda.empty_cache()
            free=[torch.cuda.mem_get_info(i)[0]/(1024**3) for i in range(2)]
            if min(free)>=2.0:break
            dump(ROOT/f'waiting_calls/{wid}.json',{'role':role,'free_gib':free,'time':time.time()})
            time.sleep(10)
        return original_role_call(client,role,payload,images,schema)
    RoleClient.__call__=guarded_role_call
    prompts={r:(ROOT/f'prompts/{r}.system.txt').read_text() for r in ['planner','planner_choice','attribute_scope_grounded','relation_scope_grounded','visual_scope_grounded']}
    tasks=selected_tasks(read(ROOT/'tasks.json'))
    allow_initial_finish=os.environ.get('SGMEA_V69_ALLOW_INITIAL_FINISH','1') != '0'
    claim_root=ROOT/'claims_8worker'
    claim_root.mkdir(exist_ok=True)
    conditions=[]
    if os.environ.get('SGMEA_V69_INCLUDE_BASELINE','1') != '0':
        conditions.append('no_acquisition')
        if os.environ.get('SGMEA_V69_INCLUDE_RETHINK','1') != '0':
            conditions.append('no_acquisition_rethink')
    conditions += list(METHODS)
    while True:
        t=None; claim=None
        for candidate_task in tasks:
            cohort0,k0=candidate_task['cohort'],candidate_task['key']
            if all((ROOT/f'runs/{cohort0}/{k0}/{condition}/decision.json').exists() for condition in conditions):
                continue
            candidate_claim=claim_root/f'{cohort0}__{k0}.lock'
            try:
                candidate_claim.mkdir()
            except FileExistsError:
                continue
            (candidate_claim/'owner').write_text(str(os.getpid()))
            t=candidate_task; claim=candidate_claim
            break
        if t is None:
            break
        try:
            cohort,k=t['cohort'],t['key'];ev=Evidence(read(ROOT/f'inputs/{cohort}/{k}.json'),read(ROOT/f'image_indices/{cohort}.json'));pv=read(ROOT/f'previews/{cohort}/{k}.json')
            dump(ROOT/f'original_relation_provenance/{cohort}/{k}.json', {'source':'four-view candidate input relations; raw relation audit unavailable for Freebase'})
            priors=read(ROOT/f'priors/{cohort}/{k}.json')
            for h in ['SRC']+LABELS:explicit_relations(pv['random'][h]['relations'],h)
            subjects={h:{'uri':ev.handles[h],**({'retrieval_prior':priors[ev.handles[h]]} if h!='SRC' else {})} for h in LABELS+['SRC']}
            catalog={}
            for role,comparable in ev.available().items():
                if role=='visual':
                    catalog[role]={'source_images':int('SRC' in ev.images),'candidate_images':comparable}
                else:
                    field='value' if role=='attribute' else 'neighbor'
                    valid={h:[r for r in ev.indices[h,role].rows if str(r[field]).strip()] for h in ['SRC']+LABELS}
                    catalog[role]={'source_record_count':len(valid['SRC']),'source_predicates':sorted({r['predicate'] for r in valid['SRC']}),'candidate_record_counts':{h:len(valid[h]) for h in comparable}}
            reader_cache={};route_cache={};choice_cache={}
            def emit(condition,event):
                folder=ROOT/f'runs/{cohort}/{k}/{condition}';folder.mkdir(parents=True,exist_ok=True)
                with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                dump(ROOT/f'progress/{wid}.json',{**t,'condition':condition,'phase':event['kind'],'time':time.time()})
            conditions=[]
            if os.environ.get('SGMEA_V69_INCLUDE_BASELINE','1') != '0':
                conditions.append('no_acquisition')
                if os.environ.get('SGMEA_V69_INCLUDE_RETHINK','1') != '0':
                    conditions.append('no_acquisition_rethink')
            conditions += list(METHODS)
            for condition in conditions:
                eliminated=ROOT/'eliminated_methods.json'
                if condition in METHODS and eliminated.exists() and condition in read(eliminated)['methods']:continue
                folder=ROOT/f'runs/{cohort}/{k}/{condition}'
                if (folder/'decision.json').exists():continue
                calls=[];acquired=[];receipts=[];used=set();current=None;judgment_count=0;planning_images=list(pv['images'])
                budget=1 if condition.endswith('budget1') else 2
                client=RoleClient(model,prompts,folder/'calls',jsonschema.validate)
                try:
                    while True:
                        available=m.available(ev,pv['random'],used) if len(used)<budget else []
                        if condition=='no_acquisition_rethink' and judgment_count:available=[]
                        schema=obj({'source_subject':{'type':'string','maxLength':100},'source_type':enum(TYPES),
                            'assessment':{'type':'string','maxLength':256},
                            'modality':enum(planner_modality_choices(
                                available, allow_initial_finish, initial_round=(judgment_count == 0)))})
                        payload={'ENTITY_EVIDENCE':blocks(pv['random'],subjects,acquired,handles=LABELS+['SRC']),
                            'RECEIPTS':receipts,'AVAILABLE_MODALITIES':available,'TOOL_CATALOG':catalog}
                        signature=hashlib.sha256(json.dumps([payload,schema,planning_images],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        if signature not in route_cache:
                            rc=RoleClient(model,prompts,folder/f'routes/{len(used)}',jsonschema.validate)
                            out=rc('planner',payload,planning_images,schema)
                            route_cache[signature]=(out,deepcopy(rc.calls))
                        route,charged=route_cache[signature];route=deepcopy(route);calls.extend(deepcopy(charged))
                        choice_payload={'ENTITY_EVIDENCE':payload['ENTITY_EVIDENCE'],'RECEIPTS':receipts}
                        choice_schema=obj({'uri':enum([ev.handles[h] for h in LABELS])})
                        choice_signature=hashlib.sha256(json.dumps([choice_payload,choice_schema,planning_images],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        if choice_signature not in choice_cache:
                            cc=RoleClient(model,prompts,folder/f'choices/{judgment_count}',jsonschema.validate)
                            choice_result=cc('planner_choice',choice_payload,planning_images,choice_schema)
                            choice_cache[choice_signature]=(choice_result,deepcopy(cc.calls))
                        choice_result,choice_calls=choice_cache[choice_signature];calls.extend(deepcopy(choice_calls))
                        route['choice']=next(h for h in LABELS if ev.handles[h]==choice_result['uri'])
                        emit(condition,{'kind':'judgment','result':route,'acquired_modalities':sorted(used)})
                        judgment_count+=1
                        current={'choice':route['choice'],'assessment':route['assessment'],'source_type':route['source_type']}
                        # Baseline is exactly the initial judgment of this same chain; no different prompt or head.
                        if condition=='no_acquisition':break
                        if condition=='no_acquisition_rethink':
                            if judgment_count>=3:break
                            continue
                        if route['modality']=='finish':break
                        role=route['modality'];assert role in available and role not in used
                        types={h:(route['source_type'] if h=='SRC' else 'Entity') for h in ['SRC']+LABELS}
                        p,imgs=m.acquire(ev,role,types)
                        # These record identifiers were already supplied to the initial planner.
                        p['TASK']['source']['subject_uri']=ev.handles['SRC']
                        for candidate in p['TASK']['candidates']:
                            candidate['subject_uri']=ev.handles[candidate['handle']]
                        reader_role=role+'_scope' if '_scope_' in condition else role
                        if '_grounded_' in condition:reader_role+='_grounded'
                        p['TASK']['instruction']='Compare SRC with all A-J and explain up to '+str(3 if '_notes3_' in condition else 2)+' candidate observations. The Planning Agent makes the final alignment choice.'
                        if '_factual_' in condition:
                            reader_role+='_factual'
                            p['TASK']['instruction']='Explain the received observations about the record subjects and connected objects. Preserve uncertainty. The Planning Agent decides alignment.'

                        if role!='visual':
                            query=' '.join(ev.names['SRC'])+' '+json.dumps(pv['random']['SRC'],ensure_ascii=False)
                            limit=12 if condition.startswith('source12_') else 24
                            source_limit=48 if condition.startswith('source48_') else limit
                            field='value' if role=='attribute' else 'neighbor'
                            p['RECORDS']={}
                            for h in ['SRC']+LABELS:
                                ranked=ev.indices[h,role].search(query,len(ev.indices[h,role].rows))[0]
                                usable=[r for r in ranked if str(r[field]).strip()]
                                p['RECORDS'][h]=[ev.public(r) for r in usable[:source_limit if h=='SRC' else limit]]
                        if role=='relation':
                            for h in ['SRC']+LABELS:explicit_relations(p['RECORDS'][h],h)
                        if role!='visual':
                            token_budget=512 if '_tok512_' in condition else 1024
                            budget_audit={}
                            for h,rows in p['RECORDS'].items():
                                p['RECORDS'][h],budget_audit[h]=bound_rows(rows,model.tokenizer,token_budget,role,h)
                            emit(condition,{'kind':'observation_budget','agent':role,'entities':budget_audit})
                        p['RECORD_COUNTS']={h:len(rows) for h,rows in p['RECORDS'].items()}
                        p['COMPARABLE_CANDIDATES']=[h for h in LABELS if p['RECORDS']['SRC'] and p['RECORDS'][h]]
                        signature=hashlib.sha256(json.dumps([reader_role,p],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        if signature not in reader_cache:
                            rc=RoleClient(model,prompts,folder/f'readers/{role}',jsonschema.validate)
                            branches=[]
                            for h in LABELS:
                                usable=bool(p['RECORDS']['SRC'] and p['RECORDS'][h])
                                branches.append(obj({'candidate':enum([h]),'subject':{'type':'string','maxLength':60},
                                    'note':{'type':'string','maxLength':140},
                                    'status':enum(['identity','association','contradiction','uncertain'] if usable else ['uncertain']),
                                    'refs':obj({'source':enum(['']+[r['id'] for r in p['RECORDS']['SRC']]),
                                                'candidate':enum(['']+[r['id'] for r in p['RECORDS'][h]])})}))
                            if '_factual_' in condition:
                                for branch in branches:
                                    branch['properties']['note']['maxLength']=240
                                    del branch['properties']['status']
                                    branch['required'].remove('status')
                            schema=obj({'source_scope':{'type':'string','maxLength':80},
                                'source_observation':{'type':'string','maxLength':100},
                                'source_refs':{'type':'array','maxItems':2,'items':enum(['']+[r['id'] for r in p['RECORDS']['SRC']])},
                                'observations':{'type':'array','maxItems':3 if '_notes3_' in condition else 2,'items':{'anyOf':branches}}})
                            out=rc(reader_role,p,imgs,schema)
                            reader_cache[signature]=(out,deepcopy(rc.calls))
                        receipt,charged=reader_cache[signature];calls.extend(deepcopy(charged));used.add(role)
                        cited_ids=set(receipt['source_refs'])
                        for observation in receipt['observations']:cited_ids.update(observation['refs'].values())
                        cited_records={h:[row for row in rows if row['id'] in cited_ids] for h,rows in p['RECORDS'].items()}
                        acquired.append({'agent':role,'records':p['RECORDS'] if '_full_' in condition else cited_records})
                        if '_ownerbound_' in condition:
                            receipt=deepcopy(receipt)
                            for observation in receipt['observations']:
                                h=observation['candidate']
                                observation['original_evidence']={
                                    'source':{'record_subject_uri':ev.handles['SRC'],'names':ev.names['SRC'],
                                        'records':[row for row in p['RECORDS']['SRC'] if row['id']==observation['refs']['source']]},
                                    'candidate':{'record_subject_uri':ev.handles[h],'names':ev.names[h],
                                        'records':[row for row in p['RECORDS'][h] if row['id']==observation['refs']['candidate']]}}
                        receipts.append({'agent':role,'result':receipt})
                        for image_pair in imgs:
                            if ('_allpixels_' in condition or image_pair[0] in cited_ids) and image_pair not in planning_images:planning_images.append(image_pair)
                        emit(condition,{'kind':'read','agent':role,'payload':p,'result':receipt})
                    choice=current['choice'];folder.mkdir(parents=True,exist_ok=True)
                    dump(folder/'decision.json',{'choice':choice,'uri':ev.handles[choice],'inspections':len(used),
                        'acquired':acquired,'receipts':receipts,'calls':calls,'labels_loaded':False,
                        'initial_choice_is_paired_baseline':True,'current_judgment':current})
                except Exception:dump(ROOT/f'failures/{cohort}/{k}/{condition}.json',{'error':traceback.format_exc(),'calls':calls+client.calls});torch.cuda.empty_cache()
                model.records.clear()
        finally:
            if claim is not None:
                shutil.rmtree(claim,ignore_errors=True)
        restart_if_host_memory_high(wid)
    dump(ROOT/f'done/{wid}.json',{'time':time.time()})


def launch():
    verify();assert not (ROOT/'launch.json').exists();jobs=[];(ROOT/'logs').mkdir(exist_ok=True)
    python_bin=os.environ.get('SGMEA_PYTHON', sys.executable)
    for wid,pair in enumerate(GPU_PAIRS):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
        with (ROOT/f'logs/{wid}.log').open('a') as f:
            p=subprocess.Popen([python_bin,'-u',str(Path(__file__).resolve()),'worker','--worker-id',str(wid)],cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append({'worker':wid,'pid':p.pid})
    dump(ROOT/'launch.json',{'jobs':jobs,'time':time.time()});print(jobs,flush=True)


def evaluate():
    labels_by_dataset={ds:read(ROOT/f'labels/{ds}.json') for ds in ['zh_en','ja_en','fr_en','FBDB15K','FBYG15K']};report={}
    tasks=selected_tasks(read(ROOT/'tasks.json'))
    for cohort in {t['cohort'] for t in tasks}:
        labels=labels_by_dataset[next(ds for ds in labels_by_dataset if cohort.startswith(ds+'_'))]
        report[cohort]={};outputs={}
        conditions=[]
        if os.environ.get('SGMEA_V69_INCLUDE_BASELINE','1') != '0':
            conditions.append('no_acquisition')
            if os.environ.get('SGMEA_V69_INCLUDE_RETHINK','1') != '0':
                conditions.append('no_acquisition_rethink')
        conditions += list(METHODS)
        for c in conditions:
            cases={}
            for t in tasks:
                if t['cohort']!=cohort:continue
                k=t['key'];p=ROOT/f'runs/{cohort}/{k}/{c}/decision.json'
                if not p.exists():continue
                d=read(p);gold=labels[read(ROOT/f'inputs/{cohort}/{k}.json')['source']]
                cases[t['index']]={'correct':d['uri']==gold,'uri':d['uri'],'gold':gold,'seconds':sum(x['seconds'] for x in d['calls']),'inspections':d['inspections']}
            outputs[c]=cases
            report[cohort][c]={'completed':len(cases),'correct':sum(x['correct'] for x in cases.values()),'mean_seconds':sum(x['seconds'] for x in cases.values())/len(cases) if cases else None,'cases':cases}
        for c in METHODS:
            baseline='no_acquisition_reason' if c.endswith('_reason') else 'no_acquisition'
            common=outputs[c].keys()&outputs[baseline].keys();report[cohort][c]['paired']={'n':len(common),'baseline':baseline,
                'fixed':[i for i in common if not outputs[baseline][i]['correct'] and outputs[c][i]['correct']],
                'broken':[i for i in common if outputs[baseline][i]['correct'] and not outputs[c][i]['correct']]}
            if 'no_acquisition_rethink' in outputs:
                rethink_common=outputs[c].keys()&outputs['no_acquisition_rethink'].keys()
                report[cohort][c]['paired_rethink']={'n':len(rethink_common),'fixed':[i for i in rethink_common if not outputs['no_acquisition_rethink'][i]['correct'] and outputs[c][i]['correct']],'broken':[i for i in rethink_common if outputs['no_acquisition_rethink'][i]['correct'] and not outputs[c][i]['correct']]}
    dump(ROOT/'report.json',report);print(json.dumps({g:{c:{k:v for k,v in x.items() if k!='cases'} for c,x in rows.items()} for g,rows in report.items()},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['launch','worker','evaluate']);ap.add_argument('--worker-id',type=int,default=0);a=ap.parse_args()
    if a.mode=='worker':worker(a.worker_id)
    else:globals()[a.mode]()
