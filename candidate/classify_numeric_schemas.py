import os
"""Frozen datatype inference from local predicate labels, no entity input."""
import argparse
import json
from pathlib import Path
import time
import re

from unified_raw_qwen_candidates import DATASETS,dump,read_jsonl

MODEL=os.environ.get("SGMEA_CLASSIFIER_MODEL", "models/Qwen2.5-VL-7B-Instruct")
SYSTEM='You identify the datatype of a knowledge-graph property. Return calendar for a date, year, or dated event; return number for a measurement, coordinate, count, amount, duration, or numeric code. Use the property meaning. Return only calendar or number.'
SYSTEM_V2='Classify a knowledge-graph property by its meaning. Return calendar for dates or calendar years, including birth year, founding year, and start/end year. Return number for measurements, coordinates, counts, amounts, codes, or durations such as a number of years. A calendar year is not a duration. Return only calendar or number.'
SYSTEM_V3='You classify what a knowledge-graph property measures. Use time for a point in calendar time, including a date or the year of an event. Use quantity for a measurement, coordinate, count, amount, code, or elapsed duration. Examples: birth year -> time; active years start year -> time; duration in years -> quantity; height in meters -> quantity. Reply only time or quantity.'


def numeric_literal(value):
    try:float(value);return True
    except ValueError:return bool(re.fullmatch(r'-?\d{1,4}-(?:\d{1,2}|##|\?\?)(?:-(?:\d{1,2}|##|\?\?))?',value.strip()))


def main():
    import torch
    from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--version',type=int,choices=[1,2,3],default=1)
    args=parser.parse_args();root,out=Path(args.root),Path(args.out)
    predicates=sorted({a.split(': ',1)[0] for ds in DATASETS for r in read_jsonl(root/'inputs'/f'{ds}.jsonl') for a in r['attribute'] if ': ' in a and (args.version==1 or numeric_literal(a.split(': ',1)[1]))})
    system={1:SYSTEM,2:SYSTEM_V2,3:SYSTEM_V3}[args.version]
    dump(out/'protocol.json',{'model':MODEL,'prompt':system,'input':'original local property label only',
         'version':args.version,
         'entity_inputs':False,'labels_read':False,'external_lookup':False,'training':False,'count':len(predicates)})
    path=out/'predictions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    old=read_jsonl(path) if path.exists() else []
    assert [r['predicate'] for r in old]==predicates[:len(old)]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
    torch.set_num_threads(2)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
          torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    model.requires_grad_(False);started=time.perf_counter()
    if args.version==3:
        fixtures={'birth year':'calendar','death year':'calendar','active years start year':'calendar','date of birth':'calendar',
                  'area land':'number','height meters':'number','number of years':'number','latitude':'number'}
        texts=[tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in fixtures]
        batch=tokenizer(texts,padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():result=model.generate(**batch,max_new_tokens=6,do_sample=False)
        answers=tokenizer.batch_decode(result[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
        mapped=[{'time':'calendar','quantity':'number'}.get(a.strip().lower().rstrip('.'),'invalid') for a in answers]
        passed=all(a==expected for a,expected in zip(mapped,fixtures.values()))
        dump(out/'semantic_fixtures.json',{'passed':passed,'results':[{'property':p,'expected':expected,'answer':answer,'raw':raw} for (p,expected),answer,raw in zip(fixtures.items(),mapped,answers)]})
        if not passed:raise ValueError('Temporal/quantity fixtures failed; do not launch full classification')
    for offset in range(len(old),len(predicates),48):
        selected=predicates[offset:offset+48]
        from numeric_set_candidate_control import key_text
        texts=[tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':p if args.version==1 else key_text(p)}],tokenize=False,add_generation_prompt=True) for p in selected]
        batch=tokenizer(texts,padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():result=model.generate(**batch,max_new_tokens=6,do_sample=False)
        answers=tokenizer.batch_decode(result[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
        with path.open('a') as stream:
            for p,raw in zip(selected,answers):
                answer=raw.strip().lower().rstrip('.')
                if args.version==3:answer={'time':'calendar','quantity':'number'}.get(answer,'invalid')
                stream.write(json.dumps({'predicate':p,'datatype':answer,'raw':raw,'valid':answer in ('calendar','number')},ensure_ascii=False)+'\n')
        dump(out/'status.json',{'done':offset+len(selected),'total':len(predicates),'seconds':time.perf_counter()-started})
        print(offset+len(selected),'/',len(predicates),flush=True)
    dump(out/'done.json',{'complete':True})


if __name__=='__main__':main()
