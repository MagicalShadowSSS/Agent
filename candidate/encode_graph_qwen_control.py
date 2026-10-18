"""Frozen Qwen text-encoder control on identical normalized entity records."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from classify_numeric_schemas import numeric_literal
from unified_raw_qwen_candidates import DATASETS, QwenEncoder, dump, read_jsonl


def literal_texts(rows, count=12):
    lists=[sorted({a for a in r['attribute'] if ': ' in a and not numeric_literal(a.split(': ',1)[1])}) for r in rows]
    df=Counter(word for records in lists for word in set(' '.join(records).lower().split()))
    def score(record):
        words=set(record.lower().split())
        return math.fsum(float(np.log((1+len(rows))/(1+df[w]))) for w in sorted(words))/max(len(words),1)
    return ['\n'.join(sorted(aa,key=lambda a:(-score(a),a))[:count]) for aa in lists]


def encode_dataset(encoder,args,ds):
    graphroot,out=Path(args.graph_root),Path(args.out)
    field=args.field
    path=out/'embeddings'/f'{ds}_{field}.npy'
    if path.exists():return
    started=time.perf_counter()
    rows=read_jsonl(graphroot/'inputs'/f'{ds}.jsonl')
    manifest=json.loads((graphroot/'manifests'/f'{ds}.json').read_text())
    if field=='attribute':
        raw=read_jsonl(Path(args.numeric_input)/'inputs'/f'{ds}.jsonl')
        assert [r['uri'] for r in raw]==[r['uri'] for r in rows]
        texts=literal_texts(raw)
    else:texts=[r['name'] if field=='name' else r['relation_text'] for r in rows]
    digest=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    ns=manifest['source_count'];roles=['query']*ns+['target']*(len(rows)-ns)
    encoder.cache.clear()
    resume=out/'resume'/f'{ds}_{field}.json'
    saved=json.loads(resume.read_text()) if resume.exists() else None
    completed=0
    with tempfile.TemporaryDirectory(prefix='sgmea_qwen_field_') as directory:
        local=Path(directory)/'vectors.npy'
        vectors=np.lib.format.open_memmap(local,mode='w+',dtype=np.float32,shape=(len(rows),encoder.dim))
        if saved:
            assert saved['input_strings_sha256']==digest
            previous=np.load(saved['vectors'],mmap_mode='r')
            assert previous.shape==vectors.shape
            completed=saved['done'];vectors[:completed]=previous[:completed]
            for i in range(completed):
                key=(roles[i],hashlib.sha256(texts[i].encode('utf-8')).hexdigest())
                if key not in encoder.cache:encoder.cache[key]=vectors[i].copy()
            del previous
        for start in range(completed,len(rows),256):
            end=min(start+256,len(rows))
            vectors[start:end]=encoder.encode(texts[start:end],roles[start:end])
            dump(out/'status'/f'{ds}_{field}.json',{'done':end,'total':len(rows),'field':field,'seconds':time.perf_counter()-started,'resumed_prefix':completed})
            print(ds,field,end,'/',len(rows),flush=True)
        observed=np.array([bool(t.strip()) for t in texts])
        norms=np.linalg.norm(vectors,axis=1)
        assert (norms[observed]>.99).all() and not (norms[~observed]>0).any()
        vectors.flush();del vectors
        path.parent.mkdir(parents=True,exist_ok=True)
        import shutil
        staging=path.with_suffix('.tmp')
        shutil.copyfile(local,staging);staging.replace(path)
    dump(out/'field_protocols'/f'{ds}_{field}.json',{
        'encoder':'frozen public Qwen3-Embedding-8B','field':field,'training':False,
        'labels_read':False,'external_lookup':False,'name_inference':False,
        'opaque_names':'missing','observed_nodes':int(observed.sum()),'nodes':len(rows),
        'input_strings_sha256':digest,'source_graph_root':str(graphroot),
        'input': 'same at most 12 nonnumeric complete IDF attribute records' if field=='attribute' else 'unchanged stable raw name or relation text',
        'pooling':'last-token normalized vectors; token-weighted complete chunk mean; existing QwenEncoder',
        'roles':'retrieval instruction on all source nodes; unprefixed targets',
        'resumed_prefix':completed,'batch_sequences':12,'batch_tokens':1536,
        'seconds':time.perf_counter()-started})
    dump(out/'done'/f'{ds}_{field}.json',{'complete':True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--graph-root',required=True)
    parser.add_argument('--numeric-input',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--field',choices=['name','attribute','relation'],required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    args=parser.parse_args();out=Path(args.out)
    import torch
    torch.set_num_threads(2)
    encoder=QwenEncoder()
    import fcntl
    locks=Path('/tmp/sgmea_qwen_field_locks');locks.mkdir(exist_ok=True)
    for ds in args.datasets:
        path=out/'embeddings'/f'{ds}_{args.field}.npy'
        if path.exists():continue
        key=hashlib.sha256(f'{out.resolve()}:{ds}:{args.field}'.encode()).hexdigest()
        with (locks/key).open('a') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            encode_dataset(encoder,args,ds)


if __name__=='__main__':main()
