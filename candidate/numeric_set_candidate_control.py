"""Frozen predicate semantics plus numeric/date set matching, without labels.

The attribute view compares original values rather than only embedded number
strings. Calendar formats are normalized to month intervals; ordinary scalar
values use a relative Gaussian kernel. This is a uniformly applied control,
not a hand-written entity matcher or a fitted multimodal weighting model.
"""
from __future__ import annotations
import os

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from sweep_unified_fusion import load_views
from unified_raw_qwen_candidates import DATASETS, VIEWS, dump, read_jsonl, visual_indexes
from write_unified_final import metric, rank_csls_matrix, write_rankings

MODEL = os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE')


def key_text(key):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', key).replace('.', ' ').replace('_', ' ')


def parsed_value(raw, is_date):
    raw=raw.strip()
    # Explicit ISO dates, including a year or month only.
    iso=re.fullmatch(r'(-?\d{3,4})-(\d{1,2})(?:-(\d{1,2}))?',raw)
    if iso:
        y,m=int(iso[1]),int(iso[2])
        if 1<=m<=12:
            return y+(m-1)/12,y+m/12,1,1/12
        return None
    try: value=float(raw)
    except ValueError: return None
    if not math.isfinite(value): return None
    if is_date and 500<=value<=3000:
        year=int(value); fraction=value-year
        if fraction<1e-8:
            return year,year+1,1,1.0
        # MMKB includes decimal-year months and compact YYYY.MMDD calendars.
        digits=raw.split('.',1)[1] if '.' in raw else ''
        encoded=digits.ljust(4,'0') if len(digits)<=4 else ''
        if encoded and 1<=int(encoded[:2])<=12 and 1<=int(encoded[2:])<=31:
            month=int(encoded[:2])
            return year+(month-1)/12,year+month/12,1,1/12
        month=round(fraction*12)
        if 1<=month<=12 and abs(fraction-month/12)<1e-6:
            return year+(month-1)/12,year+month/12,1,1/12
        # Unclear fractions stay scalar observations, not invented dates.
    digits=len(raw.rsplit('.',1)[1]) if '.' in raw and 'e' not in raw.lower() else 0
    precision=10.0**(-min(digits,12))
    return value,value,0,precision


def parsed_value_v2(raw,is_date):
    # Partial calendars retain the explicitly known year/month. Placeholder
    # symbols do not erase an observed year, and a missing month is not guessed.
    iso=re.fullmatch(r'(-?\d{1,4})-(\d{1,2}|##|\?\?)(?:-(\d{1,2}|##|\?\?))?',raw.strip())
    if iso:
        year=int(iso[1]);month=iso[2]
        if month in ('0','00','##','??'):
            return year,year+1,1,1.0
        month=int(month)
        if 1<=month<=12:return year+(month-1)/12,year+month/12,1,1/12
        return None
    result=parsed_value(raw,is_date)
    # In decimal-year month encodings December may equal the next integer
    # year. Keep this ambiguity alongside the literal year interpretation.
    if is_date:
        try:value=float(raw)
        except ValueError:return result
        if math.isfinite(value) and value.is_integer() and -5000<=value<=9999:
            return value-1/12,value+1,1,13/12
    return result


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--relative-bandwidth',type=float,default=0.001)
    parser.add_argument('--schema-encoder',choices=['labse','qwen'],default='labse')
    parser.add_argument('--kernels-only',action='store_true')
    parser.add_argument('--calendar-version',type=int,choices=[1,2],default=1)
    parser.add_argument('--datatype-root')
    parser.add_argument('--property-phrases')
    parser.add_argument('--calendar-kernel',choices=['distance','overlap'],default='distance')
    parser.add_argument('--unit-predictions')
    parser.add_argument('--unit-round',type=int,default=8)
    args=parser.parse_args()
    root,out=Path(args.root),Path(args.out)
    torch.set_num_threads(2)
    if args.schema_encoder=='qwen':
        from unified_raw_qwen_candidates import QwenEncoder
        model=QwenEncoder()
    else:
        model=SentenceTransformer(MODEL,device='cuda',local_files_only=True).eval()
        model.requires_grad_(False)
    datatypes={}
    phrases={}
    if args.property_phrases:
        predictions=read_jsonl(Path(args.property_phrases)/'predictions.jsonl')
        assert all(r['valid'] for r in predictions),'Invalid semantic property description'
        phrases={r['predicate']:r['phrase'] for r in predictions}
    if args.datatype_root:
        predictions=read_jsonl(Path(args.datatype_root)/'predictions.jsonl')
        assert all(r['valid'] for r in predictions),'Invalid datatype prediction; do not silently substitute'
        datatypes={r['predicate']:r['datatype']=='calendar' for r in predictions}
    for dataset in args.datasets:
        started=time.perf_counter()
        rows=read_jsonl(root/'inputs'/f'{dataset}.jsonl')
        n=sum(r['role']=='query' for r in rows)
        predicates=sorted({a.split(': ',1)[0] for row in rows for a in row['attribute'] if ': ' in a})
        texts=[phrases.get(key,key_text(key)) for key in predicates]
        prototypes=['a calendar date or year','a numerical measurement or count']
        if args.schema_encoder=='qwen':
            embeddings=model.encode(texts+prototypes,['target']*(len(texts)+2))
        else:
            embeddings=model.encode(texts+prototypes,batch_size=128,normalize_embeddings=True,
                                    convert_to_numpy=True,show_progress_bar=False)
        date_kind=(embeddings[:-2]@embeddings[-2:].T).argmax(1)==0
        if datatypes:
            from classify_numeric_schemas import numeric_literal
            numeric_keys={a.split(': ',1)[0] for row in rows for a in row['attribute'] if ': ' in a and numeric_literal(a.split(': ',1)[1])}
            assert all(k in datatypes for k in numeric_keys),'Incomplete datatype predictions'
            date_kind=np.asarray([datatypes.get(k,bool(kind)) for k,kind in zip(predicates,date_kind)])
        predicate_ids={key:i for i,key in enumerate(predicates)}
        dump(out/'schemas'/f'{dataset}.json',[{'predicate':k,'readable':t,'numeric_date_hint':bool(kind)}
             for k,t,kind in zip(predicates,texts,date_kind)])
        facts=[[],[]]
        for i,row in enumerate(rows):
            side=0 if i<n else 1; entity=i if side==0 else i-n
            for a in sorted(set(row['attribute'])):
                if ': ' not in a: continue
                key,value=a.split(': ',1); pid=predicate_ids[key]
                parsed=(parsed_value_v2 if args.calendar_version==2 else parsed_value)(value,date_kind[pid])
                if parsed is not None: facts[side].append((entity,pid,*parsed))
        if not facts[0] or not facts[1]:
            raise ValueError((dataset,'no numeric facts on a side'))
        source,target=[np.asarray(x,np.float32) for x in facts]
        unit_scale=np.ones((len(predicates),len(predicates)),np.float32)
        if args.unit_predictions:
            # Estimate multiplicative unit differences using this method's
            # own predicted subjects. No entity names or true links are used.
            predicted=json.loads((Path(args.unit_predictions)/'predicted_correspondences'/f'{dataset}_round{args.unit_round}.json').read_text())
            assert predicted['shape']==[n,len(rows)-n] and not predicted['labels_read']
            pairmap={a:b for a,b,confidence in predicted['pairs']}
            by_target=defaultdict(list)
            for entity,pid,lo,hi,kind,precision in facts[1]:
                if not kind and lo>0:by_target[entity].append((pid,lo))
            ratios=defaultdict(list)
            semantics=embeddings[:-2]@embeddings[:-2].T
            for entity,pid,lo,hi,kind,precision in facts[0]:
                if kind or lo<=0 or entity not in pairmap:continue
                for target_pid,value in by_target[pairmap[entity]]:
                    ratios[pid,target_pid].append(math.log(value/lo))
            source_pids=sorted({int(x[1]) for x in facts[0] if not x[4]})
            target_pids=sorted({int(x[1]) for x in facts[1] if not x[4]})
            candidates=semantics[np.ix_(source_pids,target_pids)]
            forward=np.argsort(-candidates,axis=1)[:,:min(3,len(target_pids))]
            backward=np.argsort(-candidates,axis=0)[:min(3,len(source_pids)),:]
            estimates=[]
            for i,near in enumerate(forward):
                for j in near:
                    p,q=source_pids[i],target_pids[j];values=ratios[p,q]
                    if i not in backward[:,j] or len(values)<3:continue
                    scale=math.exp(float(np.median(values)))
                    unit_scale[p,q]=scale
                    estimates.append({'source_property':predicates[p],'target_property':predicates[q],
                           'pairs':len(values),'scale':scale,'log_ratio_mad':float(np.median(np.abs(np.asarray(values)-np.median(values))))})
            dump(out/'unit_estimates'/f'{dataset}.json',{'method':'median log ratio of own predicted corresponding subjects, reciprocal top3 property retrieval',
                 'labels_read':False,'source':args.unit_predictions,'round':args.unit_round,'estimates':estimates})
        unit_scale=torch.tensor(unit_scale,device='cuda')
        source_entities=torch.tensor(source[:,0],device='cuda',dtype=torch.long)
        target_entities=torch.tensor(target[:,0],device='cuda',dtype=torch.long)
        sp,tp=torch.tensor(source[:,1],device='cuda',dtype=torch.long),torch.tensor(target[:,1],device='cuda',dtype=torch.long)
        source_values=torch.tensor(source[:,2:],device='cuda'); target_values=torch.tensor(target[:,2:],device='cuda')
        predicate_kernel=torch.tensor(np.maximum(embeddings[:-2]@embeddings[:-2].T,0)**2,device='cuda')
        t=target_values; target_count=len(rows)-n
        counts_s=np.bincount(source[:,0].astype(int),minlength=n)
        counts_t=np.bincount(target[:,0].astype(int),minlength=target_count)
        folder=out/'kernels';folder.mkdir(parents=True,exist_ok=True)
        # Bound memory by source subjects, not total graph size. This is the
        # identical fact kernel; no records or target facts are truncated.
        kernels={key:np.lib.format.open_memmap(folder/f'{dataset}_{key}.npy',mode='w+',
                 dtype=np.float32,shape=(n,target_count)) for key in ('forward','symmetric')}
        for begin in range(0,n,512):
            end=min(begin+512,n);batch=end-begin
            chosen=np.flatnonzero((source[:,0]>=begin)&(source[:,0]<end))
            source_best=torch.zeros((batch,target_count),device='cuda')
            target_best=torch.zeros((batch,len(target)),device='cuda')
            for offset in range(0,len(chosen),128):
                selected=torch.tensor(chosen[offset:offset+128],device='cuda',dtype=torch.long)
                s=source_values[selected];count=len(s)
                type_valid=s[:,2,None]==t[None,:,2]
                factor=unit_scale[sp[selected,None],tp[None,:]]
                converted=s[:,0,None]*factor
                difference=(converted-t[None,:,0]).abs()
                bandwidth=torch.maximum(s[:,3,None]*factor,t[None,:,3])
                scale=torch.maximum(converted.abs(),t[None,:,0].abs()).clamp_min(1.0)
                bandwidth=torch.maximum(bandwidth,args.relative_bandwidth*scale)
                numeric=torch.exp(-0.5*(difference/bandwidth).square())
                distance=torch.maximum(s[:,0,None]-t[None,:,1],t[None,:,0]-s[:,1,None]).clamp_min(0)
                widths_s=(s[:,1]-s[:,0]).clamp_min(1e-8)
                widths_t=(t[:,1]-t[:,0]).clamp_min(1e-8)
                calendar=torch.exp(-distance/torch.maximum(widths_s[:,None],widths_t[None,:]))
                calendar*=torch.sqrt(torch.minimum(widths_s[:,None],widths_t[None,:])/torch.maximum(widths_s[:,None],widths_t[None,:]))
                if args.calendar_kernel=='overlap':
                    # Affinity of two uniform distributions over the actually
                    # observed calendar intervals; touching endpoints have no
                    # common probability mass, missing precision stays missing.
                    overlap=(torch.minimum(s[:,1,None],t[None,:,1])-torch.maximum(s[:,0,None],t[None,:,0])).clamp_min(0)
                    calendar=overlap/torch.sqrt(widths_s[:,None]*widths_t[None,:])
                    del overlap
                values=torch.where(s[:,2,None]>0,calendar,numeric)
                kernel=predicate_kernel[sp[selected,None],tp[None,:]]*values*type_valid
                best=torch.zeros((count,target_count),device='cuda')
                best.scatter_reduce_(1,target_entities[None,:].expand(count,-1),kernel,reduce='amax',include_self=True)
                source_best.index_add_(0,source_entities[selected]-begin,best)
                target_best.scatter_reduce_(0,(source_entities[selected]-begin)[:,None].expand(-1,len(target)),kernel,reduce='amax',include_self=True)
                del selected,s,type_valid,factor,converted,difference,bandwidth,scale,numeric,distance,widths_s,widths_t,calendar,values,kernel,best
            forward=source_best/torch.tensor(np.maximum(counts_s[begin:end],1),device='cuda')[:,None]
            backward=torch.zeros((batch,target_count),device='cuda')
            backward.scatter_add_(1,target_entities[None,:].expand(batch,-1),target_best)
            backward/=torch.tensor(np.maximum(counts_t,1),device='cuda')[None,:]
            kernels['forward'][begin:end]=forward.cpu().numpy()
            kernels['symmetric'][begin:end]=torch.sqrt(forward*backward).cpu().numpy()
            del source_best,target_best,forward,backward
            dump(out/'status'/f'{dataset}.json',{'source_subjects_done':end,'total':n,'seconds':time.perf_counter()-started})
            print(dataset,'source subjects',end,'/',n,flush=True)
        for value in kernels.values():value.flush()
        has_s,has_t=counts_s>0,counts_t>0
        np.savez(out/'kernels'/f'{dataset}_availability.npz',source_has=has_s,target_has=has_t)
        if args.kernels_only:
            dump(out/'protocol'/f'{dataset}.json',{'parent':str(root.resolve()),'training':False,
                 'task_checkpoint':None,'schema_encoder':('frozen Qwen3-Embedding-8B' if args.schema_encoder=='qwen' else 'frozen LaBSE'),'ranking_reads_labels':False,
                 'input_node_order_sha256':hashlib.sha256(json.dumps([r['uri'] for r in rows]).encode()).hexdigest(),
                 'property_texts_sha256':hashlib.sha256(json.dumps(texts+prototypes,ensure_ascii=False).encode()).hexdigest(),
                 'source_facts':len(source),'target_facts':len(target),'source_entities':n,
                 'target_entities':target_count,'relative_bandwidth':args.relative_bandwidth,
                 'calendar_version':args.calendar_version,
                 'datatype_root':args.datatype_root,
                 'calendar_kernel':args.calendar_kernel,
                 'unit_predictions':args.unit_predictions,'unit_round':args.unit_round,
                 'property_phrases':args.property_phrases,
                 'seconds':time.perf_counter()-started,'stage':'kernel construction only'})
            dump(out/'done'/f'{dataset}.json',{'complete':True})
            continue
        _,_,scores,valid,source_has,target_has=load_views(root,dataset)
        sv,tv=visual_indexes(dataset)
        images=np.stack(list(sv.values())+list(tv.values()))
        mean=images.mean(0)
        def normalize(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        centered_s=np.stack([normalize(sv[r['uri']]-mean) if r['uri'] in sv else np.zeros(images.shape[1],np.float32) for r in rows[:n]])
        centered_t=np.stack([normalize(tv[r['uri']]-mean) if r['uri'] in tv else np.zeros(images.shape[1],np.float32) for r in rows[n:]])
        centered_image=centered_s@centered_t.T
        report={}
        for kind,kernel in kernels.items():
            sh=dict(source_has); th=dict(target_has); vm=dict(valid)
            # Equal dense and structured blocks inside the attribute view.
            attr=(np.where(valid['attribute'],scores['attribute'],0)+kernel)/np.maximum(
                 np.sqrt((source_has['attribute'].astype(np.float32)+has_s)[:,None]*
                         (target_has['attribute'].astype(np.float32)+has_t)[None,:]),1)
            sh['attribute']=sh['attribute']|has_s; th['attribute']=th['attribute']|has_t
            vm['attribute']=sh['attribute'][:,None]&th['attribute'][None,:]
            changed=dict(scores,attribute=attr)
            for visual in ['raw','centered']:
                changed['image']=scores['image'] if visual=='raw' else centered_image
                for mode in ['all','no_image']:
                    variant=f'{kind}_{visual}_{mode}'
                    matrix=rank_csls_matrix((changed,vm,sh,th),dict.fromkeys(VIEWS,1.0),use_image=mode=='all')
                    ranked=write_rankings(root,dataset,matrix,out/'rankings'/variant/dataset)
                    labels={r['source']:r['gold'] for r in read_jsonl(root/'labels'/f'{dataset}.jsonl')}
                    report[variant]=metric(ranked,labels)
                    dump(out/'metrics'/f'{dataset}.json',report)
                    print(dataset,variant,report[variant],flush=True)
        dump(out/'protocol'/f'{dataset}.json',{'parent':str(root.resolve()),'training':False,'task_checkpoint':None,
             'schema_encoder':'frozen LaBSE','predicate_kernel':'positive cosine squared',
             'numeric_kernel':'relative Gaussian; original scalar precision preserved','relative_bandwidth':args.relative_bandwidth,
             'calendar_kernel':'month intervals; precision ratio and interval distance',
             'attribute':'equal dense and numeric-set kernels','modalities':list(VIEWS),'modal_weights':dict.fromkeys(VIEWS,1.0),
             'image_gate':False,'ranking_reads_labels':False,'source_facts':len(source),'target_facts':len(target),
             'queries':n,'seconds':time.perf_counter()-started})
        dump(out/'done'/f'{dataset}.json',{'complete':True})
        del scores,kernels,changed,images,centered_s,centered_t


if __name__=='__main__':
    main()
