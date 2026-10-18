"""Equal-view rank evidence with untrained collective graph consistency.

Each similarity becomes an empirical upper-tail probability within the full
target catalog. Log evidence is combined equally (Fisher/rank-product style).
Ties use their full tail count: a zero kernel never invents a strong match.
Correspondence propagation uses soft Sinkhorn coupling, not supplied seeds.
"""
from __future__ import annotations
import os

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from unified_raw_qwen_candidates import DATASETS,VIEWS,dump,read_jsonl,visual_indexes
from write_unified_final import metric,write_rankings


def tail_evidence(matrix,source_has,target_has):
    import torch
    out=torch.zeros_like(matrix)
    valid_targets=torch.flatnonzero(target_has) if hasattr(torch,'flatnonzero') else target_has.nonzero().reshape(-1)
    count=len(valid_targets)
    if count==0:return out
    for begin in range(0,len(matrix),128):
        end=min(begin+128,len(matrix))
        x=matrix[begin:end,valid_targets]
        ordered=x.sort(1).values
        below=torch.searchsorted(ordered.contiguous(),x.contiguous(),right=False)
        evidence=-torch.log((count-below).float()/count)
        evidence*=source_has[begin:end,None]
        out[begin:end,valid_targets]=evidence
    return out


def transport(score,temperature=0.2,mode='balanced'):
    import torch
    if mode=='unbalanced':
        # KL-relaxed marginals allow raw catalog entities with no counterpart.
        # Marginal penalty tau=1; no threshold, seed or modality gate.
        kernel=torch.exp((score-score.max())/temperature)
        u=torch.ones(score.shape[0],device=score.device)
        v=torch.ones(score.shape[1],device=score.device)
        exponent=1/(1+temperature)
        for _ in range(30):
            u=((1/score.shape[0])/(kernel@v).clamp_min(1e-30)).pow(exponent)
            v=((1/score.shape[1])/(kernel.T@u).clamp_min(1e-30)).pow(exponent)
        return kernel*u[:,None]*v[None,:]
    coupling=torch.exp((score-score.max(1).values[:,None])/temperature)
    for _ in range(10):
        coupling/=coupling.sum(1,keepdim=True).clamp_min(1e-12)
        coupling/=coupling.sum(0,keepdim=True).clamp_min(1e-12)
    # Rectangular graphs need not imply that every target has a counterpart.
    # The global all-catalog coupling is a retrieval heuristic, not a label.
    return coupling


def normalize_observed_views(matrix,has,fields,mode='source'):
    """Average only supplied observations; shared mode uses pair availability."""
    import torch
    if mode=='source':
        counts=sum(has[f][0].float() for f in fields).clamp_min(1)
        return matrix/counts[:,None]
    if mode!='shared':raise ValueError(mode)
    # Small row blocks avoid allocating another full-catalog float matrix.
    for begin in range(0,len(matrix),128):
        end=min(begin+128,len(matrix))
        counts=torch.zeros_like(matrix[begin:end],dtype=torch.int16)
        for field in fields:
            sh,th=has[field]
            counts+=sh[begin:end,None]&th[None,:]
        matrix[begin:end]/=counts.clamp_min(1)
        matrix[begin:end]*=counts>0
    return matrix


def fuse_evidence(evidence,has,fields,method='log_mean',missing_normalization='source'):
    import torch
    counts=sum(has[f][0].float() for f in fields).clamp_min(1)
    if method=='log_mean':return normalize_observed_views(sum(evidence[f] for f in fields),has,fields,missing_normalization)
    if missing_normalization!='source':raise ValueError('shared normalization requires equal arithmetic fusion')
    # RRA order statistics: all present views cast equally sized rank votes.
    # Missing views are absent, not assigned an invented low rank.
    output=torch.zeros_like(evidence[fields[0]])
    for begin in range(0,len(output),128):
        end=min(begin+128,len(output));nn=counts[begin:end,None]
        fractions=torch.stack([torch.where(has[f][0][begin:end,None],torch.exp(-evidence[f][begin:end]),
                         torch.ones_like(output[begin:end])) for f in fields]).sort(0).values
        best=torch.ones_like(fractions[0])
        for i in range(1,len(fields)+1):
            p=fractions[i-1];cdf=torch.zeros_like(p)
            for n in range(i,len(fields)+1):
                tail=sum(__import__('math').comb(n,j)*p.pow(j)*(1-p).pow(n-j) for j in range(i,n+1))
                cdf=torch.where(nn==n,tail,cdf)
            best=torch.minimum(best,torch.where(nn>=i,cdf,torch.ones_like(cdf)))
        output[begin:end]=-torch.log((best*nn).clamp(min=1e-30,max=1))
    return output


def gw_refine(initial,semantic,graphs,steps):
    import torch
    ns,nt=initial.shape
    binary=[]
    for g in graphs:
        binary.append(torch.sparse_csr_tensor(g.crow_indices(),g.col_indices(),
                       torch.ones_like(g.values()),size=g.shape,device=g.device))
    alpha=.5*(binary[0]._nnz()/ns**2+binary[1]._nnz()/nt**2)
    coupling=initial/nt
    for step in range(1,max(steps)+1):
        left=torch.sparse.mm(binary[0],coupling)
        neighbor=torch.sparse.mm(binary[1],left.T).T.contiguous();del left
        gradient=alpha*semantic+2*(1-alpha)*neighbor;del neighbor
        kernel=coupling*torch.exp((gradient-gradient.max(1).values[:,None])/.01);del gradient
        for _ in range(10):
            kernel/=kernel.sum(1,keepdim=True).clamp_min(1e-30)*ns
            kernel/=kernel.sum(0,keepdim=True).clamp_min(1e-30)*nt
        coupling=kernel
        if step in steps:yield step,coupling,alpha


def gw_affinity(coupling,semantic,graphs,alpha):
    """Local negative GW gradient, before global marginal competition."""
    import torch
    binary=[torch.sparse_csr_tensor(g.crow_indices(),g.col_indices(),
            torch.ones_like(g.values()),size=g.shape,device=g.device) for g in graphs]
    left=torch.sparse.mm(binary[0],coupling)
    neighbor=torch.sparse.mm(binary[1],left.T).T.contiguous()
    return alpha*semantic+2*(1-alpha)*neighbor


def run(parent,graphroot,kernelroot,out,dataset,attribute_mode,steps,image_root=None,calibration='row',typed=None,temperature=0.2,transport_mode='balanced',relation_mix=.5,fusion='log_mean',eval_transport=False,gw_steps=None,literal_root=None,eval_gw_affinity=False,eval_gw_reintegrated=False,eval_gw_consensus=False,view_context=False,context_view=None,missing_normalization='source',eval_gw_blend=False,structure_adaptive=False):
    import torch
    started=time.perf_counter()
    manifest=json.loads((graphroot/'manifests'/f'{dataset}.json').read_text())
    rows=read_jsonl(graphroot/'inputs'/f'{dataset}.jsonl')
    ns=manifest['source_count'];nt=manifest['target_count']
    si={r['uri']:i for i,r in enumerate(rows[:ns])}
    ti_map={r['uri']:i for i,r in enumerate(rows[ns:])}
    qi=np.array([si[u] for u in manifest['queries']])
    ti=np.array([ti_map[u] for u in manifest['targets']])
    graphs=[]
    for side in (1,2):
        g=sp.load_npz(graphroot/'graphs'/f'{dataset}_side{side}.npz').tocsr()
        g.setdiag(0);g.eliminate_zeros()
        degree=np.asarray(g.sum(1)).reshape(-1)
        inv=1/np.sqrt(np.maximum(degree,1e-12))
        g=sp.diags(inv)@g@sp.diags(inv);g=g.tocsr()
        graphs.append(torch.sparse_csr_tensor(torch.tensor(g.indptr,device='cuda'),torch.tensor(g.indices,device='cuda'),torch.tensor(g.data,device='cuda'),size=g.shape))
    adaptive_temperature=temperature
    adaptive_fraction=None
    if structure_adaptive:
        # One label-free rule for every dataset: compare the two same-side
        # graph densities and use the pooled median density as a scale. An
        # asymmetric graph gets local evidence to compensate missing
        # correspondence mass; a dense balanced graph gets softer transport.
        stats=[]
        for name in DATASETS:
            manifest_i=json.loads((graphroot/'manifests'/f'{name}.json').read_text())
            ns_i,nt_i=manifest_i['source_count'],manifest_i['target_count']
            dens=[]
            for side,n in ((1,ns_i),(2,nt_i)):
                nnz=sp.load_npz(graphroot/'graphs'/f'{name}_side{side}.npz').nnz
                dens.append(nnz/(n*n))
            stats.append((sum(dens)/2,min(dens)/max(dens)))
        median_density=float(np.median([x[0] for x in stats]))
        median_balance=float(np.median([x[1] for x in stats]))
        this_density,this_balance=stats[DATASETS.index(dataset)]
        if this_balance < median_balance:
            adaptive_temperature=.05; adaptive_fraction=.5
        elif this_density >= median_density:
            adaptive_temperature=.25; adaptive_fraction=0.0
        else:
            adaptive_temperature=.1; adaptive_fraction=0.0
        temperature=adaptive_temperature
        print(dataset,'structure_adaptive',{'density':this_density,'balance':this_balance,'median_density':median_density,'median_balance':median_balance,'temperature':temperature,'local_fraction':adaptive_fraction},flush=True)
    evidence={};has={}
    def calibrate(matrix,source_has,target_has):
        if calibration=='density_affinity':
            # Standard CSLS within each observed modality, followed by a
            # nonnegative affinity. Empty observations never enter densities.
            target_ids=target_has.nonzero().reshape(-1);source_ids=source_has.nonzero().reshape(-1)
            output=torch.zeros_like(matrix)
            if not len(target_ids) or not len(source_ids):return output
            row_density=matrix[:,target_ids].topk(min(10,len(target_ids)),dim=1).values.mean(1)
            col_density=matrix[source_ids].topk(min(10,len(source_ids)),dim=0).values.mean(0)
            for begin in range(0,len(matrix),128):
                end=min(begin+128,len(matrix))
                part=2*matrix[begin:end,target_ids]-row_density[begin:end,None]-col_density[target_ids][None,:]
                part=torch.exp((part-part.max(1).values[:,None])/0.1)
                output[begin:end,target_ids]=part*source_has[begin:end,None]
            return output
        if calibration=='cosine':
            return matrix*source_has[:,None]*target_has[None,:]
        if calibration=='max_affinity':
            # Preserve observed similarity margins rather than assigning the
            # same extreme rank evidence to every view's accidental maximum.
            matrix=matrix.clamp_min(0)*source_has[:,None]*target_has[None,:]
            return matrix/matrix.max(1).values[:,None].clamp_min(1e-12)
        forward=tail_evidence(matrix,source_has,target_has)
        if calibration=='row':return forward
        backward=tail_evidence(matrix.T,target_has,source_has).T
        if calibration=='reciprocal':return (forward+backward)*0.5
        if calibration=='intersection':return torch.minimum(forward,backward)
        raise ValueError(calibration)
    sv,tv=visual_indexes(dataset)
    for field in VIEWS:
        if field=='image':
            if image_root:
                vectors=np.load(image_root/'embeddings'/f'{dataset}_image.npy')
            else:
                dim=len(next(iter(sv.values())))
                vectors=np.stack([(sv if i<ns else tv).get(r['uri'],np.zeros(dim,np.float32)) for i,r in enumerate(rows)])
        else:vectors=np.load((literal_root if field=='attribute' and literal_root else graphroot)/'embeddings'/f'{dataset}_{field}.npy')
        present=np.linalg.norm(vectors,axis=1)>1e-8
        left=torch.tensor(vectors[:ns],device='cuda');right=torch.tensor(vectors[ns:],device='cuda')
        sh=torch.tensor(present[:ns],device='cuda');th=torch.tensor(present[ns:],device='cuda')
        matrix=left@right.T
        semantic=calibrate(matrix,sh,th)
        del left,right,matrix,vectors
        if field=='attribute' and attribute_mode!='dense':
            numeric=np.load(kernelroot/'kernels'/f'{dataset}_forward.npy')
            avail=np.load(kernelroot/'kernels'/f'{dataset}_availability.npz')
            n_sh=torch.tensor(avail['source_has'],device='cuda');n_th=torch.tensor(avail['target_has'],device='cuda')
            matrix=torch.tensor(numeric,device='cuda')
            structured=calibrate(matrix,n_sh,n_th)
            del matrix,numeric
            if attribute_mode=='hybrid':
                semantic=(semantic+structured)/((sh.float()+n_sh.float())[:,None]).clamp_min(1)
            else:
                # Numeric facts retain a value comparison. Non-numeric record
                # subjects retain dense semantics by data availability only.
                semantic=torch.where(n_sh[:,None],structured,semantic)
            sh=sh|n_sh;th=th|n_th
            del structured
        if view_context or field==context_view:
            # Every modality receives the same one-hop context operation.
            # Context does not create an observation for a missing subject.
            left=torch.sparse.mm(graphs[0],semantic)
            context=torch.sparse.mm(graphs[1],left.T).T.contiguous()
            del left
            context=calibrate(context,sh,th)
            semantic=(semantic+context)*.5
            del context
        evidence[field]=semantic
        has[field]=(sh,th)
        print(dataset,attribute_mode,field,'tail evidence complete',flush=True)
    labels=None;report={}
    for mode in ('all','no_image'):
        fields=list(VIEWS if mode=='all' else VIEWS[:-1])
        baseline=fuse_evidence(evidence,has,fields,fusion,missing_normalization)
        relation=evidence['relation']
        current=baseline
        for step in range(max(steps)+1):
            if step:
                coupling=transport(current,temperature,transport_mode)
                if typed:
                    graph_score=typed.propagate(coupling)
                    del coupling
                else:
                    left=torch.sparse.mm(graphs[0],coupling)
                    graph_score=torch.sparse.mm(graphs[1],left.T).T.contiguous()
                    del coupling,left
                graph_ev=calibrate(graph_score,torch.ones(ns,dtype=torch.bool,device='cuda'),torch.ones(nt,dtype=torch.bool,device='cuda'))
                del graph_score
                # Relation semantics and structural consistency are two equal
                # components of a single relation view, not a fifth modality.
                if fusion=='log_mean':current=baseline+normalize_observed_views(relation_mix*(graph_ev-relation),has,fields,missing_normalization)
                else:current=fuse_evidence(dict(evidence,relation=(1-relation_mix)*relation+relation_mix*graph_ev),has,fields,fusion)
                del graph_ev
            if step not in steps:continue
            for density in (('none','csls10','transport') if eval_transport else ('none','csls10')):
                matrix=current
                if density=='csls10':
                    matrix=2*current-current.topk(10,dim=1).values.mean(1)[:,None]-current.topk(10,dim=0).values.mean(0)[None,:]
                elif density=='transport':matrix=transport(current,temperature,transport_mode)
                variant=f'{attribute_mode}_{mode}_step{step}_{density}'
                score=matrix[qi][:,ti].cpu().numpy()
                ranks=write_rankings(parent,dataset,score,out/'rankings'/variant/dataset)
                if labels is None:labels={r['source']:r['gold'] for r in read_jsonl(parent/'labels'/f'{dataset}.jsonl')}
                report[variant]=metric(ranks,labels)
                dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                print(dataset,variant,report[variant],flush=True)
                del matrix,score
            dump(out/'status'/f'{dataset}_{attribute_mode}.json',{'step':step,'mode':mode,'seconds':time.perf_counter()-started})
        if gw_steps:
            initial=transport(current,temperature,transport_mode)
            for gs,coupling,alpha in gw_refine(initial,current,graphs,gw_steps):
                variant=f'{attribute_mode}_{mode}_gw{gs}'
                score=coupling[qi][:,ti].cpu().numpy()
                ranks=write_rankings(parent,dataset,score,out/'rankings'/variant/dataset)
                report[variant]=metric(ranks,labels)
                dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                dump(out/'status'/f'{dataset}_{attribute_mode}.json',{'gw_step':gs,'graph_density_alpha':alpha,'mode':mode,'seconds':time.perf_counter()-started})
                print(dataset,variant,report[variant],flush=True)
                if eval_gw_affinity:
                    affinity=gw_affinity(coupling,current,graphs,alpha)
                    for correction in ('none','csls10'):
                        matrix=affinity
                        if correction=='csls10':
                            matrix=2*affinity-affinity.topk(10,dim=1).values.mean(1)[:,None]-affinity.topk(10,dim=0).values.mean(0)[None,:]
                        av=f'{attribute_mode}_{mode}_gw{gs}_affinity_{correction}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    del affinity
                if eval_gw_reintegrated:
                    if not typed:raise ValueError('GW reintegration requires a typed relation view')
                    # GW optimizes a global correspondence. Use that result
                    # to update relation observations, then retrieve with all
                    # observed modalities still casting equal votes.
                    graph_score=typed.propagate(coupling*nt)
                    graph_ev=calibrate(graph_score,torch.ones(ns,dtype=torch.bool,device='cuda'),torch.ones(nt,dtype=torch.bool,device='cuda'))
                    del graph_score
                    if fusion=='log_mean':combined=baseline+normalize_observed_views(relation_mix*(graph_ev-relation),has,fields,missing_normalization)
                    else:combined=fuse_evidence(dict(evidence,relation=(1-relation_mix)*relation+relation_mix*graph_ev),has,fields,fusion)
                    del graph_ev
                    for correction in ('none','csls10','transport'):
                        matrix=combined
                        if correction=='csls10':
                            matrix=2*combined-combined.topk(10,dim=1).values.mean(1)[:,None]-combined.topk(10,dim=0).values.mean(0)[None,:]
                        elif correction=='transport':matrix=transport(combined,temperature,transport_mode)
                        av=f'{attribute_mode}_{mode}_gw{gs}_reintegrated_{correction}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    del combined
                if eval_gw_consensus:
                    # Rank product of local four-view evidence and global
                    # graph correspondence, with no learned mixture weights.
                    source_has=torch.ones(ns,dtype=torch.bool,device='cuda')
                    target_has=torch.ones(nt,dtype=torch.bool,device='cuda')
                    posterior=tail_evidence(coupling,source_has,target_has)
                    local=tail_evidence(current,source_has,target_has)
                    combined=(local+posterior)*.5
                    del posterior,local
                    for correction in ('none','csls10'):
                        matrix=combined
                        if correction=='csls10':
                            matrix=2*combined-combined.topk(10,dim=1).values.mean(1)[:,None]-combined.topk(10,dim=0).values.mean(0)[None,:]
                        av=f'{attribute_mode}_{mode}_gw{gs}_consensus_{correction}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    del combined
                if eval_gw_blend:
                    # Full-catalog late fusion of global coupling and local
                    # graph affinity. Both terms are converted to the same
                    # empirical tail evidence scale; no labels or fitted
                    # weights are used. This is a compact diagnostic for
                    # replacing the larger family of GW-derived branches.
                    affinity=gw_affinity(coupling,current,graphs,alpha)
                    posterior=tail_evidence(coupling,torch.ones(ns,dtype=torch.bool,device='cuda'),torch.ones(nt,dtype=torch.bool,device='cuda'))
                    local=tail_evidence(affinity,torch.ones(ns,dtype=torch.bool,device='cuda'),torch.ones(nt,dtype=torch.bool,device='cuda'))
                    local_csls_matrix=2*affinity-affinity.topk(10,dim=1).values.mean(1)[:,None]-affinity.topk(10,dim=0).values.mean(0)[None,:]
                    local_csls=tail_evidence(local_csls_matrix,torch.ones(ns,dtype=torch.bool,device='cuda'),torch.ones(nt,dtype=torch.bool,device='cuda'))
                    del affinity
                    for local_fraction in (.25,.5,.6,.75):
                        matrix=(1-local_fraction)*posterior+local_fraction*local
                        av=f'{attribute_mode}_{mode}_gw{gs}_blend{local_fraction:g}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    if structure_adaptive:
                        matrix=(1-adaptive_fraction)*posterior+adaptive_fraction*local
                        av=f'{attribute_mode}_{mode}_gw{gs}_structure_adaptive'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    for local_fraction in (.5,.6,.75):
                        matrix=(1-local_fraction)*posterior+local_fraction*local_csls
                        av=f'{attribute_mode}_{mode}_gw{gs}_blend_csls{local_fraction:g}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    # A geometric consensus keeps a high score only when both
                    # global and local evidence support it.  It uses the same
                    # two evidence arrays and has no learned or dataset
                    # specific weight; this is a compact alternative to the
                    # linear mixture above.
                    eps=1e-12
                    for local_fraction in (.5,.75):
                        matrix=torch.exp((1-local_fraction)*torch.log(posterior.clamp_min(eps)) + local_fraction*torch.log(local.clamp_min(eps)))
                        av=f'{attribute_mode}_{mode}_gw{gs}_geomean{local_fraction:g}'
                        score=matrix[qi][:,ti].cpu().numpy()
                        ranks=write_rankings(parent,dataset,score,out/'rankings'/av/dataset)
                        report[av]=metric(ranks,labels)
                        dump(out/'metrics'/f'{dataset}_{attribute_mode}.json',report)
                        print(dataset,av,report[av],flush=True)
                        del matrix,score
                    del posterior,local,local_csls_matrix,local_csls
            del initial,coupling
        del baseline,current
    dump(out/'done'/f'{dataset}_{attribute_mode}.json',{'complete':True,'seconds':time.perf_counter()-started})


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--graph-root',required=True)
    parser.add_argument('--kernel-root')
    parser.add_argument('--image-root')
    parser.add_argument('--literal-root')
    parser.add_argument('--calibration',choices=['row','reciprocal','intersection','cosine','max_affinity','density_affinity'],default='row')
    parser.add_argument('--typed-audit')
    parser.add_argument('--schema-root')
    parser.add_argument('--directional',action='store_true')
    parser.add_argument('--infer-schema',action='store_true')
    parser.add_argument('--functionality',action='store_true')
    parser.add_argument('--propagation',choices=['soft','mutual','confident'],default='soft')
    parser.add_argument('--relation-mix',type=float,default=.5)
    parser.add_argument('--fusion',choices=['log_mean','rra'],default='log_mean')
    parser.add_argument('--missing-normalization',choices=['source','shared'],default='source')
    parser.add_argument('--eval-transport',action='store_true')
    parser.add_argument('--gw-steps',nargs='+',type=int)
    parser.add_argument('--eval-gw-affinity',action='store_true')
    parser.add_argument('--eval-gw-reintegrated',action='store_true')
    parser.add_argument('--eval-gw-consensus',action='store_true')
    parser.add_argument('--eval-gw-blend',action='store_true')
    parser.add_argument('--structure-adaptive',action='store_true')
    parser.add_argument('--view-context',action='store_true')
    parser.add_argument('--context-view',choices=VIEWS,help='Single-view context diagnostic; main equal-context mode uses --view-context')
    parser.add_argument('--temperature',type=float,default=0.2)
    parser.add_argument('--transport-mode',choices=['balanced','unbalanced'],default='balanced')
    parser.add_argument('--out',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--attribute',choices=['dense','hybrid','values'],default='dense')
    parser.add_argument('--steps',nargs='+',type=int,default=[0,1,2,4,8])
    args=parser.parse_args()
    if not 0<=args.relation_mix<=1:parser.error('--relation-mix must be between 0 and 1')
    if args.missing_normalization=='shared' and args.fusion!='log_mean':parser.error('shared normalization requires --fusion log_mean')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    parent,graphroot,out=Path(args.parent),Path(args.graph_root),Path(args.out)
    kernelroot=Path(args.kernel_root) if args.kernel_root else None
    dump(out/'protocols'/f'{args.attribute}.json',{'modalities':list(VIEWS),'modal_weights':'equal',
        'fusion':args.fusion,'rank_calibration':{
            'row':'empirical upper-tail evidence; ties retain full tail count',
            'reciprocal':'mean source/target empirical upper-tail evidence',
            'intersection':'minimum source/target empirical upper-tail evidence',
            'cosine':'observed raw similarity',
            'max_affinity':'nonnegative observed similarity divided by its row maximum',
            'density_affinity':'observed-view CSLS followed by exponential affinity at temperature .1'}[args.calibration],
        'text_encoder':json.loads((graphroot/'protocol.json').read_text()).get('encoder','frozen public LaBSE') if (graphroot/'protocol.json').exists() else 'frozen public LaBSE',
        'graph_root':str(graphroot),'numeric_kernel_root':str(kernelroot),
        'attribute_mode':args.attribute,'relation':f'semantic fraction {1-args.relation_mix}; collective neighbor fraction {args.relation_mix}',
        'transport':f"{30 if args.transport_mode=='unbalanced' else 10} {args.transport_mode} transport iterations, temperature {args.temperature}, all normalized source and target entities",
        'steps':args.steps,'training':False,'task_checkpoint':None,'supplied_seeds':False,'image_gate':False,
        'image_transform_root':args.image_root,
        'literal_root':args.literal_root,
        'calibration':args.calibration,
        'typed_audit':args.typed_audit,'schema_root':args.schema_root,
        'directional':args.directional,'infer_schema':args.infer_schema,
        'functionality':args.functionality,'relation_graph_fraction':args.relation_mix,
        'propagation':args.propagation,
        'missing_normalization':args.missing_normalization,
        'eval_transport':args.eval_transport,
        'gw_steps':args.gw_steps,'gw_reference':'https://github.com/squareRoot3/FusedGW-Entity-Alignment',
        'gw_affinity_evaluation':args.eval_gw_affinity,
        'gw_relation_reintegration_evaluation':args.eval_gw_reintegrated,
        'gw_rank_consensus_evaluation':args.eval_gw_consensus,
        'gw_full_tail_blend_evaluation':args.eval_gw_blend,
        'structure_adaptive':args.structure_adaptive,
        'view_context':args.view_context,
        'context_view_diagnostic':args.context_view,
        'gw_parameters':{'semantic_fraction':'mean observed graph density','proximal_beta':.01,'sinkhorn_steps':10},
        'transport_temperature':args.temperature,'transport_mode':args.transport_mode,
        'ranking_reads_labels':False,'reference':'SEU/SNF-style untrained graph inference; this is an adaptation'})
    for ds in args.datasets:
        if not (out/'done'/f'{ds}_{args.attribute}.json').exists():
            typed=None
            if args.typed_audit:
                from sentence_transformers import SentenceTransformer
                from typed_relation_consistency import TypedRelationConsistency
                model=SentenceTransformer(os.environ.get('SGMEA_TEXT_MODEL', 'models/LaBSE'),device='cuda',local_files_only=True).eval()
                model.requires_grad_(False)
                typed=TypedRelationConsistency(ds,graphroot,Path(args.typed_audit),
                    Path(args.schema_root) if args.schema_root else None,out,model,
                    directional=args.directional or args.infer_schema,infer_schema=args.infer_schema,
                    functionality=args.functionality,propagation=args.propagation)
                del model
                torch.cuda.empty_cache()
            run(parent,graphroot,kernelroot,out,ds,args.attribute,args.steps,
                Path(args.image_root) if args.image_root else None,args.calibration,typed,args.temperature,args.transport_mode,args.relation_mix,args.fusion,args.eval_transport,args.gw_steps,
                Path(args.literal_root) if args.literal_root else None,args.eval_gw_affinity,args.eval_gw_reintegrated,args.eval_gw_consensus,args.view_context,args.context_view,args.missing_normalization,args.eval_gw_blend,args.structure_adaptive)
            torch.cuda.empty_cache()


if __name__=='__main__':main()
