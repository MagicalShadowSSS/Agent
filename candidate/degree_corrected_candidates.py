import os
"""Degree-corrected neighbor evidence in one correspondence update.

Positive Poisson deviance discounts overlap expected from graph degree alone.
It is a compatibility statistic, not a calibrated identity probability.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import scipy.sparse as sp
import torch

from collective_candidate_control import fuse_evidence
from simple_typed_transport import prepare_views
from structure_consistency import transport
from typed_relation_consistency import TypedRelationConsistency
from unified_raw_qwen_candidates import DATASETS, VIEWS, dump, read_jsonl
from write_unified_final import metric, write_rankings


def positive_deviance(observed, expected):
    safe=np.maximum(expected,1e-12)
    value=observed*np.log(np.maximum(observed,1e-12)/safe)-observed+expected
    return np.where(observed>expected,np.maximum(value,0),0)


class DegreeCorrectedOperator(TypedRelationConsistency):
    def __init__(self,*args,graph_root,dataset,**kwargs):
        super().__init__(*args,**kwargs)
        self.ordinary=[]
        for side in (1,2):
            graph=sp.load_npz(Path(graph_root)/'graphs'/f'{dataset}_side{side}.npz').tocsr()
            graph.setdiag(0);graph.eliminate_zeros();graph.data[:]=1
            self.ordinary.append(graph)

    def deviation(self, observed, expected):
        return positive_deviance(observed, expected)

    def channel(self,groups,sparse):
        result=np.zeros(self.shape,np.float32)
        marginal=getattr(self,'null_model','uniform')=='marginal'
        if marginal:
            source_mass=np.asarray(sparse.sum(1)).reshape(-1)
            target_mass=np.asarray(sparse.sum(0)).reshape(-1)
            total_mass=max(float(source_mass.sum()),1e-12)
        for a,b,weight in groups:
            if getattr(self,'neighbor_weight','none')!='none':
                if not hasattr(self,'neighbor_information'):
                    degrees=[np.maximum(np.asarray(g.sum(1)).reshape(-1),1) for g in self.ordinary]
                    if self.neighbor_weight=='adamic':
                        degrees=[np.log1p(degree) for degree in degrees]
                    self.neighbor_information=[1/np.sqrt(degree) for degree in degrees]
                # Weight the supporting neighbors, rather than the subjects.
                # Paired sqrt factors give inverse degree when the two
                # neighbor degrees agree, with symmetric graph-side roles.
                a=a.multiply(self.neighbor_information[0][None,:]).tocsr()
                b=b.multiply(self.neighbor_information[1][None,:]).tocsr()
            current=(a@sparse@b.T).tocsr()
            current.sum_duplicates()
            current=current.tocoo()
            da=np.asarray(a.sum(1)).reshape(-1)
            db=np.asarray(b.sum(1)).reshape(-1)
            if marginal:
                exposure_s=np.asarray(a@source_mass).reshape(-1)
                exposure_t=np.asarray(b@target_mass).reshape(-1)
                expected=exposure_s[current.row]*exposure_t[current.col]/total_mass
            else:
                expected=da[current.row]*db[current.col]/self.shape[1]
            contribution=weight*self.deviation(current.data,expected)
            # CSR canonicalization makes every COO coordinate unique.
            result[current.row,current.col]+=contribution
        if getattr(self, 'channel_normalization', 'row_max') == 'saturating':
            # A bounded compatibility transform. This is not a probability;
            # weak absolute support is not promoted to a row maximum of one.
            result /= 1 + result
        else:
            result/=np.maximum(result.max(1,keepdims=True),1e-12)
        return result

    def propagate(self,coupling,k=20):
        self.update_schema(coupling)
        values,columns=coupling.topk(min(k,coupling.shape[1]),dim=1)
        policy=getattr(self,'support_policy','soft')
        if policy=='mutual':
            inverse=coupling.argmax(0)
            valid=inverse[columns[:,0]]==torch.arange(len(columns),device=coupling.device)
            columns=columns[:,:1];values=valid.float()[:,None]
        elif policy=='mass':
            values/=coupling.sum(1,keepdim=True).clamp_min(1e-12)
        else:
            values/=values.sum(1,keepdim=True).clamp_min(1e-12)
        values,columns=values.cpu().numpy(),columns.cpu().numpy()
        sparse=sp.csr_matrix((values.reshape(-1),columns.reshape(-1),
            np.arange(0,values.size+1,values.shape[1])),shape=self.shape)
        scope=getattr(self,'support_scope','all')
        if scope=='ordinary':
            result=self.channel([(*self.ordinary,1.)],sparse)
        elif scope=='typed':
            result=self.channel(self.groups,sparse)
        else:
            result=self.channel(self.groups,sparse)
            result+=self.channel([(*self.ordinary,1.)],sparse)
            result*=.5
        return torch.tensor(result,device=coupling.device)


def run(args):
    from sentence_transformers import SentenceTransformer
    started=time.perf_counter();out=Path(args.out)
    evidence,has,qi,ti,manifest=prepare_views(args,args.dataset)
    fields=[f for f in VIEWS if f!=args.omit_view]
    prior=fuse_evidence(evidence,has,fields,'log_mean','source')
    counts=sum(has[f][0].float() for f in fields).clamp_min(1)
    del evidence
    torch.cuda.empty_cache()
    if getattr(args,'predicate_mode','lexical')=='topology':
        from topology_predicate_candidates import TopologyPredicateOperator
        operator=TopologyPredicateOperator(args.dataset,args.graph_root,args.typed_audit,
            out,args.steps,topk=args.predicate_k,freeze=args.freeze_predicates)
    else:
        model=SentenceTransformer(os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE'),
            device='cuda',local_files_only=True).eval()
        model.requires_grad_(False)
        operator=DegreeCorrectedOperator(args.dataset,Path(args.graph_root),
            Path(args.typed_audit),None,out,model,directional=True,infer_schema=True,
            normalize_neighbors=False,audit_rounds=set(args.steps),
            graph_root=args.graph_root,dataset=args.dataset)
        del model
    operator.neighbor_weight=args.neighbor_weight
    operator.null_model=args.null_model
    operator.support_policy=args.support_policy
    def project(score):
        if args.transport_mode=='row':
            value=torch.exp((score-score.max(1).values[:,None])/args.temperature)
            return value/value.sum(1,keepdim=True).clamp_min(1e-12)
        return transport(score,args.temperature,args.transport_mode)
    coupling=project(prior)
    rankings,timings={},{}

    def write(name,matrix):
        rankings[name]=write_rankings(Path(args.parent),args.dataset,
            matrix[qi][:,ti].cpu().numpy(),out/'rankings'/name)

    write('initial',prior);preparation=time.perf_counter()-started;core=0.
    for step in range(1,max(args.steps)+1):
        tick=time.perf_counter()
        support=operator.propagate(coupling,args.support_k)
        local=prior+support/counts[:,None]
        if args.update=='proximal':
            coupling=project(args.temperature*coupling.clamp_min(1e-30).log()
                +support/counts[:,None])
        elif args.update=='entropic':
            coupling=project(local)
        else:
            proposal=project(local)
            gamma=2/(step+2)
            coupling.mul_(1-gamma).add_(proposal,alpha=gamma)
            del proposal
        torch.cuda.synchronize();core+=time.perf_counter()-tick
        if step in args.steps:
            write(f'step{step}_coupling',coupling);write(f'step{step}_local',local)
            timings[str(step)]=core
            dump(out/'status.json',{'step':step,'core_seconds':core,
                'elapsed_seconds':time.perf_counter()-started})
            print(args.dataset,args.update,step,flush=True)
        del support,local
    labels={row['source']:row['gold'] for row in read_jsonl(
        Path(args.parent)/'labels'/f'{args.dataset}.jsonl')}
    dump(out/'metrics.json',{name:metric(rows,labels) for name,rows in rankings.items()})
    dump(out/'timings.json',{'prepare_seconds':preparation,'core_seconds':timings,
        'total_seconds':time.perf_counter()-started,
        'cuda_max_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    dump(out/'done.json',{'complete':True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--update',choices=['proximal','conditional_gradient','entropic'],default='proximal')
    parser.add_argument('--omit-view',choices=VIEWS)
    parser.add_argument('--center-embeddings',action='store_true')
    parser.add_argument('--predicate-mode',choices=['lexical','topology'],default='lexical')
    parser.add_argument('--predicate-k',type=int,default=1)
    parser.add_argument('--freeze-predicates',action='store_true')
    parser.add_argument('--neighbor-weight',choices=['none','resource','adamic'],default='none')
    parser.add_argument('--null-model',choices=['uniform','marginal'],default='uniform')
    parser.add_argument('--support-policy',choices=['soft','mass','mutual'],default='soft')
    parser.add_argument('--transport-mode',choices=['balanced','unbalanced','row'],default='balanced')
    parser.add_argument('--temperature',type=float,default=.05)
    parser.add_argument('--support-k',type=int,default=20)
    parser.add_argument('--steps',nargs='+',type=int,default=[2,4,8,16])
    parser.add_argument('--parent',default='output/unified_raw_qwen_1000_20261006_repaired_duplicates')
    parser.add_argument('--graph-root',default='output/full_graph_labse_stable_raw_20261006')
    parser.add_argument('--literal-root',default='output/nonnumeric_literal_attributes12_stable_20261006')
    parser.add_argument('--kernel-root',default='/tmp/sgmea_candidates_20261006/numeric_v3')
    parser.add_argument('--typed-audit',default='output/raw_relation_semantics_audit_20261006')
    args=parser.parse_args();args.fact_kernel_root,args.numeric_kernel=None,'forward'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if (out/'done.json').exists():raise FileExistsError(str(out))
    source=Path(__file__);(out/'code').mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/'code'/source.name)
    snapshots={}
    for name in ['degree_corrected_candidates.py','topology_predicate_candidates.py',
                 'simple_typed_transport.py','typed_relation_consistency.py',
                 'collective_candidate_control.py','structure_consistency.py']:
        path=source.parent/name
        shutil.copy2(path,out/'code'/name)
        snapshots[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out/'protocol.json',dict(vars(args),training=False,task_checkpoint=None,
        algorithm_stages=1,image_gate=False,top_k=20,gold_read_after_rankings=True,
        validation_role='reused development queries',main_ranking=f'step{max(args.steps)}_coupling',
        code_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    dump(out/'dependency_sha256.json',snapshots)
    run(args)


if __name__=='__main__':main()
