"""Frozen feature propagation on entire original graphs, without anchors.

Uses SEU's relation-frequency graph weights and depth-averaged propagated
cosines. Both complete raw graphs participate; evaluation still uses the
same fixed 1000 queries and full normalized target catalog. Each ranking is
persisted before the evaluation labels are read for its metrics.
"""
from __future__ import annotations
import os

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from prepare_semantic_relation_control import readable
from unified_raw_qwen_candidates import (DATASETS,VIEWS,dataset_dir,dump,label,
    parse_fact,read_jsonl,read_map,translated_names,visual_indexes,write_jsonl)
from write_unified_final import metric,write_rankings

MODEL=os.environ.get("SGMEA_TEXT_MODEL", "models/LaBSE")


def prepare(parent,audit,out,dataset,name_mode='translated'):
    base=dataset_dir(dataset)
    maps=[read_map(base/f'ent_ids_{i}') for i in (1,2)]
    names=translated_names(dataset) if name_mode=='translated' else {}
    uris=[list(m.values()) for m in maps]
    attributes=defaultdict(set)
    for file in ('att_triples1','att_triples2'):
        with (base/file).open() as f:
            for line in f:
                fact=parse_fact(line)
                if fact:attributes[fact[0]].add(f'{label(fact[1]) or fact[1]}: {fact[2]}')
    name_by_uri={uri:names.get(eid,'') or label(uri) for m in maps for eid,uri in m.items()}
    predicates=json.loads((audit/f'{dataset}_id2predicate.json').read_text())
    edges=defaultdict(set)
    for side,(m,u) in enumerate(zip(maps,uris),1):
        index={uri:i for i,uri in enumerate(u)}
        triples=[]
        with (base/f'triples_{side}').open() as f:
            for line in f:
                parts=line.split()
                if len(parts)!=3:continue
                h,r,t=parts;h,t=m.get(h,h),m.get(t,t)
                if h not in index or t not in index:continue
                triples.append((index[h],r,index[t]))
                pred=readable(predicates[r]) if r in predicates else 'unknown predicate'
                def describe(uri):
                    return name_by_uri.get(uri,'') or ('['+'; '.join(sorted(attributes[uri])[:3])+']' if attributes[uri] else '[unidentified entity]')
                edges[h].add(f'out {pred}: {describe(t)}')
                edges[t].add(f'in {pred}: {describe(h)}')
        frequencies=Counter(t[1] for t in triples)
        left,right,weights=[],[],[]
        for h,r,t in triples:
            weight=np.log(max(len(triples),1)/frequencies[r])
            left.extend((h,t));right.extend((t,h));weights.extend((weight,weight))
        self_weight=np.log(max(len(triples)/len(u),1.00001))
        left.extend(range(len(u)));right.extend(range(len(u)));weights.extend([self_weight]*len(u))
        graph=sp.csr_matrix((np.asarray(weights,np.float32),(left,right)),shape=(len(u),len(u)))
        graph.sum_duplicates()
        # Positive symmetric graph; row scaling disappears after L2 node norm.
        sp.save_npz(out/'graphs'/f'{dataset}_side{side}.npz',graph)
    rows=read_jsonl(parent/'inputs'/f'{dataset}.jsonl')
    query_uris=[r['uri'] for r in rows if r['role']=='query']
    df=Counter(word for uri in edges for word in set(' '.join(edges[uri]).lower().split()))
    def info(record):
        words=set(record.lower().split())
        return math.fsum(float(np.log((1+len(edges))/(1+df[w]))) for w in sorted(words))/max(len(words),1)
    records=[]
    for side,u in enumerate(uris):
        for uri in u:
            attr=sorted(attributes[uri])
            rel=sorted(edges[uri],key=lambda e:(-info(e),e))[:12]
            records.append({'uri':uri,'side':side,'name':name_by_uri[uri],
                'attribute_text':'\n'.join(attr),'relation_text':'\n'.join(rel)})
    write_jsonl(out/'inputs'/f'{dataset}.jsonl',records)
    dump(out/'manifests'/f'{dataset}.json',{'source_count':len(uris[0]),'target_count':len(uris[1]),
        'queries':query_uris,'targets':[r['uri'] for r in rows if r['role']=='target'],
        'parent':str(parent.resolve()),'labels_read':False})
    return records


def encode(model,records,out,dataset):
    for field in ('name','attribute','relation'):
        path=out/'embeddings'/f'{dataset}_{field}.npy'
        if path.exists():continue
        texts=[r['name'] if field=='name' else r[f'{field}_text'] for r in records]
        unique=list(dict.fromkeys(texts));mapping={t:i for i,t in enumerate(unique)}
        values=np.zeros((len(unique),768),np.float32)
        # Complete token chunks, same budget on both sides and all datasets.
        for start in range(0,len(unique),256):
            tokens=[model.tokenizer.encode(t,add_special_tokens=False) for t in unique[start:start+256]]
            chunks,owner,weight=[],[],[]
            for i,ids in enumerate(tokens):
                for begin in range(0,len(ids),126):
                    piece=ids[begin:begin+126]
                    chunks.append(model.tokenizer.decode(piece,skip_special_tokens=True));owner.append(i);weight.append(len(piece))
            if chunks:
                vectors=model.encode(chunks,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
                pooled=np.zeros((len(tokens),768),np.float32)
                np.add.at(pooled,owner,vectors*np.asarray(weight,np.float32)[:,None])
                pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
                values[start:start+len(tokens)]=pooled
            dump(out/'status'/f'{dataset}.json',{'phase':'encode','field':field,'unique_done':start+len(tokens),'unique_total':len(unique)})
        vectors=values[[mapping[t] for t in texts]]
        assert (np.linalg.norm(vectors[np.array([bool(t.strip()) for t in texts])],axis=1)>.99).all()
        with path.with_suffix('.tmp').open('wb') as f:np.save(f,vectors)
        path.with_suffix('.tmp').replace(path)
        print(dataset,field,'encoded',len(texts),flush=True)


def evaluate(parent,out,dataset,max_depth=2):
    import torch
    started=time.perf_counter()
    manifest=json.loads((out/'manifests'/f'{dataset}.json').read_text())
    records=read_jsonl(out/'inputs'/f'{dataset}.jsonl')
    ns=manifest['source_count'];nt=manifest['target_count']
    source_index={r['uri']:i for i,r in enumerate(records[:ns])}
    target_index={r['uri']:i for i,r in enumerate(records[ns:])}
    qi=np.array([source_index[u] for u in manifest['queries']]);ti=np.array([target_index[u] for u in manifest['targets']])
    graphs=[]
    for side in (1,2):
        g=sp.load_npz(out/'graphs'/f'{dataset}_side{side}.npz').tocsr()
        graphs.append(torch.sparse_csr_tensor(torch.tensor(g.indptr,device='cuda'),torch.tensor(g.indices,device='cuda'),torch.tensor(g.data,device='cuda'),size=g.shape))
    accum={'all':torch.zeros((ns,nt),device='cuda'),'no_image':torch.zeros((ns,nt),device='cuda')}
    available={}
    features={}
    sv,tv=visual_indexes(dataset)
    for field in VIEWS:
        if field=='image':
            dim=len(next(iter(sv.values())))
            x=np.stack([(sv if i<ns else tv).get(r['uri'],np.zeros(dim,np.float32)) for i,r in enumerate(records)])
        else:x=np.load(out/'embeddings'/f'{dataset}_{field}.npy')
        features[field]=[torch.tensor(x[:ns],device='cuda'),torch.tensor(x[ns:],device='cuda')]
    labels=None;report={}
    for depth in range(max_depth+1):
        available={f:[(x.square().sum(1)>1e-8) for x in part] for f,part in features.items()}
        for mode in accum:
            fields=list(VIEWS if mode=='all' else VIEWS[:-1])
            counts_s=sum(available[f][0].float() for f in fields)
            counts_t=sum(available[f][1].float() for f in fields)
            for begin in range(0,ns,256):
                end=min(begin+256,ns)
                score=sum(features[f][0][begin:end]@features[f][1].T for f in fields)
                score/=torch.sqrt(counts_s[begin:end,None]*counts_t[None,:]).clamp_min(1)
                accum[mode][begin:end]+=score
            current=accum[mode]/(depth+1)
            for transform in ('cosine','csls10','sinkhorn50'):
                if transform=='cosine':matrix=current
                elif transform=='csls10':
                    matrix=2*current-current.topk(10,dim=1).values.mean(1)[:,None]-current.topk(10,dim=0).values.mean(0)[None,:]
                else:
                    matrix=torch.exp((current-current.max(1).values[:,None])*50)
                    for _ in range(10):
                        matrix/=matrix.sum(1,keepdim=True).clamp_min(1e-12)
                        matrix/=matrix.sum(0,keepdim=True).clamp_min(1e-12)
                variant=f'{mode}_depth{depth}_{transform}'
                cross=matrix[qi][:,ti].cpu().numpy()
                ranks=write_rankings(parent,dataset,cross,out/'rankings'/variant/dataset)
                if labels is None:labels={r['source']:r['gold'] for r in read_jsonl(parent/'labels'/f'{dataset}.jsonl')}
                report[variant]=metric(ranks,labels)
                dump(out/'metrics'/f'{dataset}.json',report)
                print(dataset,variant,report[variant],flush=True)
                del cross,matrix
            del current
        if depth<max_depth:
            for field in features:
                features[field]=[torch.nn.functional.normalize(torch.sparse.mm(g,x),dim=1) for g,x in zip(graphs,features[field])]
        dump(out/'status'/f'{dataset}.json',{'phase':'rank','depth':depth,'max_depth':max_depth,'seconds':time.perf_counter()-started})
    dump(out/'done'/f'{dataset}.json',{'complete':True,'seconds':time.perf_counter()-started})


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--name-mode',choices=['translated','raw'],default='translated')
    parser.add_argument('--encode-only',action='store_true')
    parser.add_argument('--reuse-root',help='Reuse a frozen LaBSE field only when its entire ordered input is identical')
    args=parser.parse_args()
    parent,audit,out=Path(args.parent),Path(args.audit),Path(args.out)
    for folder in ('graphs','embeddings'): (out/folder).mkdir(parents=True,exist_ok=True)
    dump(out/'protocol.json',{'algorithm':'SEU-style frozen feature propagation and optional Sinkhorn retrieval',
        'source_reference':'https://github.com/MaoXinn/SEU','graph':'same-side original weighted adjacency, both directions and self edge',
        'encoder':'frozen public LaBSE, 126-token complete chunks','relation':'original semantic predicates and neighbor facts, top12 by corpus IDF',
        'name_input':args.name_mode,'external_name_lookup':False,'opaque_names':'missing',
        'node_universe':'all normalized benchmark ent_ids_1 and ent_ids_2 nodes; not restricted to gold-pair targets',
        'record_selection':'top12 complete relations by corpus IDF; sorted-word math.fsum and lexical tie break',
        'depths':[0,1,2],'transforms':['cosine','csls10','sinkhorn50'],'weights':'all available modalities equal',
        'training':False,'task_checkpoint':None,'anchors':None,'image_gate':False,'ranking_reads_labels':False})
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model=SentenceTransformer(MODEL,device='cuda',local_files_only=True).eval()
    model.requires_grad_(False)
    for dataset in args.datasets:
        if (out/'done'/f'{dataset}.json').exists():continue
        if (out/'inputs'/f'{dataset}.jsonl').exists():records=read_jsonl(out/'inputs'/f'{dataset}.jsonl')
        else:records=prepare(parent,audit,out,dataset,args.name_mode)
        if args.reuse_root:
            donor=Path(args.reuse_root)
            old=read_jsonl(donor/'inputs'/f'{dataset}.jsonl')
            same_nodes=[r['uri'] for r in records]==[r['uri'] for r in old]
            reused=[]
            if same_nodes:
                for field in ('name','attribute','relation'):
                    key='name' if field=='name' else field+'_text'
                    dst=out/'embeddings'/f'{dataset}_{field}.npy'
                    if not dst.exists() and all(a[key]==b[key] for a,b in zip(records,old)):
                        os.link(donor/'embeddings'/dst.name,dst);reused.append(field)
            dump(out/'cache_reuse'/f'{dataset}.json',{'donor':str(donor),'same_node_order':same_nodes,
                 'reused_fields':reused,'rule':'entire ordered raw field strings must be identical; same frozen encoder and chunk pooling',
                 'labels_read':False})
        encode(model,records,out,dataset)
        if args.encode_only:dump(out/'done'/f'{dataset}.json',{'complete':True,'encoding_only':True})
        else:evaluate(parent,out,dataset)
        torch.cuda.empty_cache()


if __name__=='__main__':main()
