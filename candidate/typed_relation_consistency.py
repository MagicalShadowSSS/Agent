"""Label-free predicate retrieval and sparse typed-neighbor propagation."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from prepare_semantic_relation_control import readable
from unified_raw_qwen_candidates import dataset_dir,dump,read_jsonl,read_map


class TypedRelationConsistency:
    def __init__(self,dataset,graphroot,audit,schema,out,model,topk=3,directional=False,infer_schema=False,functionality=False,propagation='soft',audit_rounds=None,normalize_neighbors=True,schema_reciprocal=True,schema_attention=False):
        self.dataset=dataset;self.out=out;self.topk=topk
        self.directional=directional;self.infer_schema=infer_schema;self.round=0
        self.functionality=functionality
        self.propagation=propagation
        self.audit_rounds=audit_rounds
        maps=[read_map(dataset_dir(dataset)/f'ent_ids_{i}') for i in (1,2)]
        uris=[list(m.values()) for m in maps]
        index=[{u:i for i,u in enumerate(us)} for us in uris]
        predicates=json.loads((audit/f'{dataset}_id2predicate.json').read_text())
        texts={}
        if schema:
            for path in (schema/'predictions').glob('*.jsonl'):
                for row in read_jsonl(path):
                    if row['format_valid']:texts[row['predicate']]=row['phrase']
        edges=[defaultdict(list),defaultdict(list)]
        for side,m in enumerate(maps):
            with (dataset_dir(dataset)/f'triples_{side+1}').open() as f:
                for line in f:
                    h,r,t=line.split()
                    if r not in predicates:continue
                    hu,tu=m.get(h,h),m.get(t,t)
                    if hu in index[side] and tu in index[side]:
                        edges[side][r].append((index[side][hu],index[side][tu]))
        ids=[sorted(e) for e in edges]
        phrases=[[texts.get(predicates[r],readable(predicates[r])) for r in ii] for ii in ids]
        enc=[model.encode(p,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False) for p in phrases]
        similarity=enc[0]@enc[1].T
        ks,kt=min(topk,len(ids[1])),min(topk,len(ids[0]))
        forward=np.argsort(-similarity,axis=1)[:,:ks]
        backward=np.argsort(-similarity,axis=0)[:kt,:]
        matches=[]
        for i,near in enumerate(forward):
            # Preserve the original scalar rounding for existing recipes.
            weights=np.array([float(max(similarity[i,j],0)**2) for j in near])
            if schema_attention:
                attention=np.exp((similarity[i,near]-similarity[i,near].max())/.05)
                weights=attention/attention.sum()*weights.max()
            matches.extend((i,int(j),float(weight)) for j,weight in zip(near,weights)
                if not schema_reciprocal or i in backward[:,j])
        # Degree normalization uses selected raw edges; no observed correspondence.
        self.groups=[];degrees=[np.zeros(len(u),np.float32) for u in uris]
        selected=[set(),set()]
        for i,j,_ in matches:selected[0].add(ids[0][i]);selected[1].add(ids[1][j])
        matrices=[{},{}]
        for side in (0,1):
            for r in selected[side]:
                pairs=np.asarray(edges[side][r]);h,t=pairs[:,0],pairs[:,1]
                a=sp.csr_matrix((np.ones(2*len(h),np.float32),(np.r_[h,t],np.r_[t,h])),shape=(len(uris[side]),)*2)
                a.sum_duplicates();a.data[:]=1
                degrees[side]+=np.asarray(a.sum(1)).reshape(-1)
                matrices[side][r]=a
        norm=[sp.diags(1/np.sqrt(np.maximum(d,1))) if normalize_neighbors else sp.eye(len(d),format='csr') for d in degrees]
        for side in (0,1):
            matrices[side]={r:(norm[side]@a@norm[side]).tocsr() for r,a in matrices[side].items()}
        for i,j,weight in matches:
            self.groups.append((matrices[0][ids[0][i]],matrices[1][ids[1][j]],weight))
        self.shape=(len(uris[0]),len(uris[1]))
        if directional:
            # Keep the observed subject/object roles. One group for outgoing
            # neighbors and one for incoming neighbors, with equal mass.
            degrees=[np.zeros(len(u),np.float32) for u in uris]
            self.directed=[{},{}]
            for side in (0,1):
                for r,pairs in edges[side].items():
                    pairs=np.asarray(pairs);h,t=pairs[:,0],pairs[:,1]
                    a=sp.csr_matrix((np.ones(len(h),np.float32),(h,t)),shape=(len(uris[side]),)*2)
                    a.sum_duplicates();a.data[:]=1
                    degrees[side]+=np.asarray(a.sum(1)).reshape(-1)+np.asarray(a.sum(0)).reshape(-1)
                    self.directed[side][r]=a
            norm=[sp.diags(1/np.sqrt(np.maximum(d,1))) if normalize_neighbors else sp.eye(len(d),format='csr') for d in degrees]
            self.directed=[{r:(norm[side]@a@norm[side]).tocsr() for r,a in mm.items()} for side,mm in enumerate(self.directed)]
            self.ids=ids;self.edges=edges;self.predicates=predicates
            self.functional=[{},{}]
            for side in (0,1):
                for r,pairs in edges[side].items():
                    pp=np.asarray(pairs)
                    self.functional[side][r]=(len(np.unique(pp[:,0]))/len(pp),len(np.unique(pp[:,1]))/len(pp))
            self.semantic_matches=matches
            self.groups=self.oriented_groups([(i,j,w,False) for i,j,w in matches])
            if infer_schema:
                self.target_edge_types=defaultdict(list)
                for j,r in enumerate(ids[1]):
                    for h,t in edges[1][r]:self.target_edge_types[h,t].append(j)
        assert self.shape==(json.loads((graphroot/'manifests'/f'{dataset}.json').read_text())['source_count'],
                            json.loads((graphroot/'manifests'/f'{dataset}.json').read_text())['target_count'])
        dump(out/'predicate_matches'/f'{dataset}.json',{'topk':topk,
            'rule':'reciprocal schema nearest neighbors' if schema_reciprocal else 'source-to-target schema nearest neighbors, many-to-one allowed',
            'schema_attention':schema_attention,
            'schema_verbalization':str(schema) if schema else None,'training':False,'labels_read':False,
            'source_predicates_used':len(selected[0]),'target_predicates_used':len(selected[1]),
            'matches':[{'source':predicates[ids[0][i]],'target':predicates[ids[1][j]],
                'source_phrase':phrases[0][i],'target_phrase':phrases[1][j],'weight':w} for i,j,w in matches]})

    def oriented_groups(self,matches):
        groups=[]
        for i,j,w,inverse in matches:
            a=self.directed[0][self.ids[0][i]];b=self.directed[1][self.ids[1][j]]
            if inverse:b=b.T.tocsr()
            forward=backward=1.0
            if getattr(self,'functionality',False):
                fs=self.functional[0][self.ids[0][i]];ft=self.functional[1][self.ids[1][j]]
                if inverse:ft=ft[::-1]
                # Subject equality inferred through an object uses inverse
                # functionality; the incoming counterpart uses functionality.
                forward=np.sqrt(fs[1]*ft[1]);backward=np.sqrt(fs[0]*ft[0])
            groups.extend([(a,b,w*.5*forward),(a.T.tocsr(),b.T.tocsr(),w*.5*backward)])
        return groups

    def update_schema(self,coupling):
        # Mutual maxima are this iteration's predictions, never supplied
        # anchors. Count directed edge overlap only among these correspondences.
        mapping=coupling.argmax(1).cpu().numpy()
        inverse=coupling.argmax(0).cpu().numpy()
        valid=inverse[mapping]==np.arange(self.shape[0])
        if getattr(self,'propagation','soft')=='confident':
            mass=(coupling.max(1).values/coupling.sum(1).clamp_min(1e-12)).cpu().numpy()
            valid&=mass>.5
        covered=np.zeros(self.shape[1],bool);covered[mapping[valid]]=True
        counts=np.zeros((len(self.ids[0]),len(self.ids[1]),2),np.float64)
        source_counts=np.zeros(len(self.ids[0]));target_counts=np.zeros(len(self.ids[1]))
        for j,r in enumerate(self.ids[1]):
            pp=np.asarray(self.edges[1][r]);target_counts[j]=np.sum(covered[pp[:,0]]&covered[pp[:,1]])
        for i,r in enumerate(self.ids[0]):
            for h,t in self.edges[0][r]:
                if not (valid[h] and valid[t]):continue
                source_counts[i]+=1
                mh,mt=int(mapping[h]),int(mapping[t])
                for j in self.target_edge_types.get((mh,mt),()):counts[i,j,0]+=1
                for j in self.target_edge_types.get((mt,mh),()):counts[i,j,1]+=1
        dice=2*counts/np.maximum(source_counts[:,None,None]+target_counts[None,:,None],1)
        flat=dice.reshape(len(self.ids[0]),-1)
        matches=[];supported=set()
        selection = getattr(self, 'schema_selection', 'reciprocal')
        if selection == 'positive':
            # Keep every observed predicate overlap; no nearest-neighbor
            # count or reciprocal-rank filter at the relation level.
            for i, col in zip(*np.nonzero(flat > 0)):
                j,orientation=divmod(int(col),2)
                matches.append((int(i),j,float(dice[i,j,orientation]),bool(orientation)))
                supported.add(int(i))
        else:
            forward=np.argsort(-flat,axis=1)[:,:min(self.topk,flat.shape[1])]
            backward=np.argsort(-flat,axis=0)[:min(self.topk,len(flat)),:]
            for i,near in enumerate(forward):
                for col in near:
                    j,orientation=divmod(int(col),2)
                    if counts[i,j,orientation] and i in backward[:,col]:
                        matches.append((i,j,float(dice[i,j,orientation]),bool(orientation)));supported.add(i)
        # Retain lexical schema retrieval only for predicates with no observed
        # cross-graph edge overlap. This follows observation availability.
        matches.extend((i,j,w,False) for i,j,w in self.semantic_matches if i not in supported)
        self.groups=self.oriented_groups(matches)
        self.round+=1
        if self.audit_rounds is not None and self.round not in self.audit_rounds:
            return
        mass=(coupling.max(1).values/coupling.sum(1).clamp_min(1e-12)).cpu().numpy()
        dump(self.out/'predicted_correspondences'/f'{self.dataset}_round{self.round}.json',{
            'labels_read':False,'supplied_anchors':False,'shape':list(self.shape),
            'pairs':[[int(i),int(mapping[i]),float(mass[i])] for i in np.flatnonzero(valid)]})
        dump(self.out/'inferred_predicate_matches'/f'{self.dataset}_round{self.round}.json',{
            'schema_selection':selection,
            'mutual_predictions':int(valid.sum()),'source_predicates_with_overlap':len(supported),
            'labels_read':False,'training':False,'supplied_anchors':False,
            'matches':[{'source':self.predicates[self.ids[0][i]],'target':self.predicates[self.ids[1][j]],
                        'inverse':inv,'weight':w,'observed_overlap':float(counts[i,j,int(inv)])}
                       for i,j,w,inv in matches]})

    def propagate(self,coupling,k=20,preserve_mass=False):
        import torch
        if self.infer_schema:self.update_schema(coupling)
        values,columns=coupling.topk(min(k,coupling.shape[1]),dim=1)
        if not preserve_mass:values/=values.sum(1,keepdim=True).clamp_min(1e-12)
        values=values.cpu().numpy();columns=columns.cpu().numpy()
        if getattr(self,'propagation','soft') in ('mutual','confident'):
            inverse=coupling.argmax(0).cpu().numpy()
            valid=inverse[columns[:,0]]==np.arange(len(columns))
            if self.propagation=='confident':
                mass=(coupling.max(1).values/coupling.sum(1).clamp_min(1e-12)).cpu().numpy()
                valid&=mass>.5
            values=valid.astype(np.float32)[:,None];columns=columns[:,:1]
        y=sp.csr_matrix((values.reshape(-1),columns.reshape(-1),np.arange(0,values.size+1,values.shape[1])),shape=self.shape)
        result=np.zeros(self.shape,np.float32)
        for a,b,weight in self.groups:
            score=(a@y@b.T).tocoo()
            np.add.at(result,(score.row,score.col),score.data*weight)
        return torch.tensor(result,device=coupling.device)
