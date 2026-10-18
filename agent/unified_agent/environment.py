"""Label-free, replayable evidence environment. No model or evaluation imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile

MODALITIES=('name','attribute','relation','image')


def parse_object(text):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    obj=json.loads(text)
    if not isinstance(obj,dict):raise ValueError('Expected one JSON object')
    return obj


class EvidenceEnv:
    def __init__(self,row,image_index,compact,asset_dir,novel_only=False,candidate_count=20):
        self.source=row['source'];self.baseline=list(row['baseline']['rank']);self.compact=compact
        self.novel_only=novel_only
        candidates=list(row['candidates'])
        assert 1 <= candidate_count <= 20
        assert len(candidates)==candidate_count and len(set(self.baseline))==candidate_count
        assert set(self.baseline)=={c['uri'] for c in candidates}
        random.Random(int(hashlib.sha256(self.source.encode()).hexdigest()[:16],16)).shuffle(candidates)
        self.handles={'SRC':self.source};self.records={'SRC':self._record(row['record'])}
        self.priors={};self.assets={};self.ledger={};self.history=[];self.seen=set();self.recent_images=[]
        for h,c in zip('ABCDEFGHIJKLMNOPQRST',candidates):
            self.handles[h]=c['uri'];self.records[h]=self._record(c['record'])
            self.priors[h]={'initial_rank':self.baseline.index(c['uri'])+1,
                'view_ranks':{k:v.get('rank') for k,v in c['evidence'].items()}}
        self.reverse={u:h for h,u in self.handles.items()}
        for h,u in self.handles.items():
            if u in image_index and Path(image_index[u]).is_file():self.assets[h]=image_index[u]
        self.overview=[]
        for h,rec in self.records.items():
            fields={}
            for m in MODALITIES[:-1]:
                eid=f'{h}.{m}.preview';value=compact('; '.join(rec[m]),48)
                if value:
                    self.ledger[eid]={'entity':h,'modality':m,'uri':self.handles[h],
                        'location':f'record/{m}/prefix','text':value,'preview':True}
                fields[m]={'ref':eid if value else None,'text':value,'total_values':len(rec[m])}
            image_ref=f'{h}.image.preview' if h in self.assets else None
            if image_ref:self.ledger[image_ref]={'entity':h,'modality':'image','uri':self.handles[h],
                'location':self.assets[h],'preview':True,'text':'Labeled thumbnail in overview sheet; not an identity assertion.'}
            self.overview.append({'entity':h,'fields':fields,'image_ref':image_ref,
                                  'prior':self.priors.get(h)})
        self.sheet=self._sheet(asset_dir)

    def unread(self,h,m):
        if m=='image':return [0] if h in self.assets and h not in self.recent_images else []
        # A complete preview already exposes the field. Otherwise retrieve values not yet in the ledger.
        text='; '.join(self.records[h][m])
        if self.compact(text,48)==text:return []
        return [i for i in range(len(self.records[h][m])) if f'{h}.{m}.{i}' not in self.ledger]

    def available_tools(self):
        return {m:[h for h in self.handles if self.unread(h,m)] for m in MODALITIES}

    @staticmethod
    def _record(rec):
        return {m:[str(x) for x in rec.get(m,[])] for m in MODALITIES[:-1]}

    def _sheet(self,asset_dir):
        if not self.assets:return None
        from PIL import Image,ImageDraw
        asset_dir=Path(asset_dir);asset_dir.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256((self.source+json.dumps(self.assets,sort_keys=True)).encode()).hexdigest()
        path=asset_dir/(digest+'.jpg')
        if path.exists():return str(path)
        # Every retained candidate and source is visible, independent of labels.
        sheet=Image.new('RGB',(5*160,((len(self.handles)+4)//5)*180),'white');draw=ImageDraw.Draw(sheet)
        for i,h in enumerate(self.handles):
            x=(i%5)*160;y=(i//5)*180
            draw.text((x+4,y+2),h,fill='black')
            if h in self.assets:
                with Image.open(self.assets[h]) as src:
                    im=src.convert('RGB');im.thumbnail((152,152));sheet.paste(im,(x+4,y+22));im.close()
            else:draw.text((x+4,y+50),'image unavailable',fill='gray')
        with tempfile.NamedTemporaryFile(dir=asset_dir,suffix='.jpg',delete=False) as stream:
            temp=Path(stream.name)
        try:sheet.save(temp,format='JPEG',quality=90);temp.replace(path)
        finally:
            temp.unlink(missing_ok=True);sheet.close()
        return str(path)

    def inspect(self,action):
        allowed={'action','entities','modalities','question','expected','offset','predicate'}
        if set(action)-allowed:return {'error':'unknown action keys','allowed':sorted(allowed)}
        entities=action.get('entities');modalities=action.get('modalities');question=action.get('question')
        if not isinstance(entities,list) or not entities or len(entities)>3 or any(not isinstance(h,str) or h not in self.handles for h in entities):
            return {'error':'entities must contain 1 to 3 known handles, including SRC when comparing source evidence'}
        if not isinstance(modalities,list) or not modalities or any(m not in MODALITIES for m in modalities):
            return {'error':'modalities must be chosen from name, attribute, relation, image'}
        if not isinstance(question,str) or not question.strip():return {'error':'provide a concrete identity-discriminating question'}
        offset=action.get('offset',0)
        if type(offset) is not int or offset<0:return {'error':'offset must be a nonnegative integer'}
        if self.novel_only:
            if offset!=0:return {'error':'unread retrieval starts at zero; successive calls advance automatically'}
            if any(not self.unread(h,m) for h in entities for m in modalities):
                return {'error':'field already fully visible or unavailable; select from available_tools'}
        key=json.dumps([sorted(set(entities)),sorted(set(modalities)),question.casefold(),offset])
        if key in self.seen and not self.novel_only:return {'error':'duplicate query: revise the question, entities, modalities or offset'}
        self.seen.add(key);result={'question':question,'evidence':[],'coverage':[]}
        old_refs=set(self.ledger)
        tokens=set(re.findall(r'\w+',question.casefold()))
        if 'image' in modalities:self.recent_images=[]
        for h in dict.fromkeys(entities):
            for m in dict.fromkeys(modalities):
                if m=='image':
                    if h not in self.assets:
                        result['coverage'].append({'entity':h,'modality':m,'status':'missing, not contradictory'});continue
                    eid=f'{h}.image.full';self.ledger[eid]={'entity':h,'modality':m,'uri':self.handles[h],
                        'location':self.assets[h],'text':'Original image attached. Answer the query using visible evidence, not imagined contents.','preview':False}
                    self.recent_images.append(h);result['evidence'].append({'ref':eid,**self.ledger[eid]});continue
                values=self.records[h][m]
                pool=[(i,values[i]) for i in self.unread(h,m)] if self.novel_only else list(enumerate(values))
                if action.get('predicate') is not None:
                    pool=[(i,v) for i,v in pool if v.partition(': ')[0]==action['predicate']]
                scored=sorted(pool,key=lambda iv:(-len(tokens&set(re.findall(r'\w+',iv[1].casefold()))),iv[0]))
                selected=scored[offset:offset+6]
                for i,value in selected:
                    eid=f'{h}.{m}.{i}';item={'entity':h,'modality':m,'uri':self.handles[h],
                        'location':f'record/{m}/{i}','text':self.compact(value,96),'preview':False}
                    self.ledger[eid]=item;result['evidence'].append({'ref':eid,**item})
                result['coverage'].append({'entity':h,'modality':m,'total':len(values),'returned':len(selected),
                    'offset':offset,'next_offset':offset+6 if offset+6<len(values) else None,
                    'retrieval':'query token overlap; ties retain original field order',
                    'unread_remaining':len(self.unread(h,m)) if self.novel_only else None})
        self.history.append({'action':deepcopy(action),'observation':deepcopy(result)})
        result['new_evidence_count']=sum(x['ref'] not in old_refs for x in result['evidence'])
        if not result['new_evidence_count']:
            result['notice']='No new evidence. These images or values are already in state; reason over them or query a different field, entity or page.'
        return result

    def visible_refs(self):
        return {eid for eid,item in self.ledger.items() if item['modality']!='image' or item['preview'] or item['entity'] in self.recent_images}

    def validate_finish(self,obj):
        if obj.get('action')!='finish':return 'action must be finish'
        choice=obj.get('choice')
        if choice=='U':return None
        if not isinstance(choice,str) or choice not in self.handles or choice=='SRC':return 'choice must be candidate handle or U'
        refs=obj.get('citations')
        if not isinstance(refs,list) or not refs or any(not isinstance(r,str) or r not in self.visible_refs() for r in refs):
            return 'citations must reference evidence currently visible in the state'
        entities={self.ledger[r]['entity'] for r in refs}
        if 'SRC' not in entities or choice not in entities:return 'cite evidence for both SRC and the chosen candidate'
        return None

    def context(self):
        detailed=[{'ref':eid,**item} for eid,item in self.ledger.items() if not item['preview']]
        # Earlier images remain referenced, but only explicitly attached images may be cited as full-resolution evidence.
        return {'source':'SRC','overview':self.overview,'retrieved_evidence':detailed,
                'visible_refs':sorted(self.visible_refs()),'queried_images':self.recent_images}

    def images(self):
        images=[]
        if self.sheet:images.append(('OVERVIEW: labeled source and all candidate thumbnails',self.sheet))
        for h in self.recent_images:images.append((f'ORIGINAL IMAGE {h}, evidence {h}.image.full',self.assets[h]))
        return images

    def rank(self,choice):
        uri=self.handles[choice] if choice in self.handles and choice!='SRC' else self.baseline[0]
        return [uri]+[x for x in self.baseline if x!=uri]
