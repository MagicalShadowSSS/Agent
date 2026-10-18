"""Bounded planner + three independent modality readers; no labels in inference."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import unquote

LABELS = list('ABCDEFGHIJ')
TYPES = ['Person', 'Organization', 'Place', 'Creative Work', 'Event', 'Entity']
ROLES = ['planner', 'attribute', 'relation', 'visual']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def obj(fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def enum(values):
    return {'type': 'string', 'enum': list(values)}


def refs_schema(refs):
    return {'type': 'array', 'items': enum(sorted(refs)) if refs else {'type': 'string'},
            'minItems': 0, 'maxItems': min(4, len(refs))}


def planner_schema(refs, allow_inspect):
    finish = obj({'action': enum(['finish']), 'choice': enum(LABELS),
                  'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240}, 'refs': refs_schema(refs)})
    if not allow_inspect:
        return finish
    inspect = obj({'action': enum(['inspect']), 'agent': enum(ROLES[1:]),
                   'types': obj({h: enum(TYPES) for h in ['SRC']+LABELS}),
                   'candidates': {'type': 'array', 'items': enum(LABELS), 'minItems': 1, 'maxItems': 3},
                   'question': {'type': 'string', 'minLength': 1, 'maxLength': 240}})
    return {'anyOf': [inspect, finish]}


def reader_schema(candidates, records):
    branches = []
    for h in candidates:
        refs = [x['id'] for owner in ['SRC', h] for x in records[owner]]
        branches.append(obj({'candidate': enum([h]),
            'judgment': enum(['same', 'related', 'different', 'unknown']),
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240}, 'refs': refs_schema(refs)}))
    return obj({'status': enum(['answered', 'insufficient', 'question_mismatch']),
                'findings': {'type': 'array', 'items': {'anyOf': branches},
                             'minItems': len(candidates), 'maxItems': len(candidates)}})


def tokens(text):
    """Fixed lexical tokenizer: Latin words + CJK characters/bigrams; no translation."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = ''.join(c for c in unicodedata.normalize('NFKD', unquote(text)).casefold()
                   if not unicodedata.combining(c)).replace('_', ' ')
    result = []
    for word in re.findall(r'[^\W_]+', text, flags=re.UNICODE):
        for part in re.findall(r'[\u3040-\u30ff\u3400-\u9fff]+|[^\u3040-\u30ff\u3400-\u9fff]+', word):
            if re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fff]+', part):
                result.extend(part)
                result.extend(part[i:i+2] for i in range(len(part)-1))
            else:
                result.append(part)
    return result


class InvertedIndex:
    def __init__(self, rows):
        self.rows = rows
        self.postings = defaultdict(list)
        self.lengths = []
        for i, row in enumerate(rows):
            counts = Counter(tokens(row['raw']))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.avg = sum(self.lengths)/len(rows) if rows else 1

    def search(self, query, limit):
        scores = [0.0]*len(self.rows)
        for term in sorted(set(tokens(query))):
            postings = self.postings.get(term, [])
            idf = math.log(1+(len(self.rows)-len(postings)+0.5)/(len(postings)+0.5))
            for i, tf in postings:
                norm = 1.2*(1-0.75+0.75*self.lengths[i]/max(self.avg, 1))
                scores[i] += idf*tf*2.2/(tf+norm)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:limit]
        return [deepcopy(self.rows[i]) for i in order], {
            'query': query, 'total': len(scores), 'lexical_hit': any(s > 0 for s in scores),
            'hits': [{'id': self.rows[i]['id'], 'score': scores[i]} for i in order]}


class Evidence:
    def __init__(self, row, image_index):
        self.rank = list(row['baseline']['rank'][:10])
        if len(set(self.rank)) != 10:
            raise ValueError('Expected ten distinct fixed candidates')
        candidates = {c['uri']: c['record'] for c in row['candidates']}
        self.handles = dict(zip(LABELS, self.rank))
        self.handles['SRC'] = row['source']
        self.names, self.indices, self.images, self.originals = {}, {}, {}, {}
        for h, uri in self.handles.items():
            record = row['record'] if h == 'SRC' else candidates[uri]
            names = record.get('name', [])
            if isinstance(names, str):
                names = [names]
            display_names = list(names)
            if not uri.startswith('/m/'):
                display_names.append(unquote(uri.rsplit('/', 1)[-1]).replace('_', ' '))
            self.names[h] = list(dict.fromkeys(display_names))
            if uri in image_index and Path(image_index[uri]).is_file():
                self.images[h] = str(Path(image_index[uri]).resolve())
            for modality in ('attribute', 'relation'):
                records = []
                for i, raw in enumerate(record.get(modality, [])):
                    left, sep, right = raw.partition(': ')
                    if not sep:
                        raise ValueError(f'Unparseable original {h}/{modality}/{i}: {raw!r}')
                    item = {'id': f'{h}.{modality}.{i}', 'raw': raw}
                    if modality == 'attribute':
                        item.update(predicate=left, value=right)
                    else:
                        direction, space, predicate = left.partition(' ')
                        if not space or direction not in ('in', 'out'):
                            raise ValueError(f'Invalid relation direction: {raw!r}')
                        item.update(direction=direction, predicate=predicate, neighbor=right)
                    self.originals[item['id']] = item
                    records.append(item)
                self.indices[h, modality] = InvertedIndex(records)

    def preview(self, folder):
        from PIL import Image, ImageDraw
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        records, audit = {}, {}
        query = ' '.join(n for names in self.names.values() for n in names)
        for h in self.handles:
            record = {'name': self.names[h]}
            for modality, key in [('attribute', 'attributes'), ('relation', 'relations')]:
                selected, audit[h+'.'+modality] = self.indices[h, modality].search(query, 3)
                record[key] = [self.public(item) for item in selected]
            record['images'] = [{'id': h+'.image.preview'}] if h in self.images else []
            records[h] = record
        images = []
        if self.images:
            sheet = Image.new('RGB', (4*256, 3*284), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, h in enumerate(self.handles):
                x, y = i%4*256, i//4*284
                draw.text((x+5, y+5), h+'.image.preview', fill='black')
                if h in self.images:
                    with Image.open(self.images[h]) as source:
                        tile = source.convert('RGB')
                        tile.thumbnail((250, 250))
                        sheet.paste(tile, (x+(256-tile.width)//2, y+28))
                        tile.close()
                else:
                    draw.text((x+5, y+40), 'No image', fill='black')
            path = folder/'overview.png'
            sheet.save(path)
            sheet.close()
            images = [('Labeled preview images for A-J and SRC; IDs match RECORDS.', str(path.resolve()))]
        return records, images, audit

    @staticmethod
    def public(item):
        return {k: v for k, v in item.items() if k != 'raw'}

    def retrieve(self, action):
        owners = ['SRC']+action['candidates']
        role = action['agent']
        task = {'question': action['question'], 'source': {
            'handle': 'SRC', 'name': self.names['SRC'], 'type': action['types']['SRC']},
            'candidates': [{'handle': h, 'name': self.names[h], 'type': action['types'][h]}
                           for h in action['candidates']]}
        query = ' '.join(n for h in owners for n in self.names[h])+' '+action['question']
        records, audit, images = {}, {}, []
        for h in owners:
            if role == 'visual':
                records[h] = [{'id': h+'.image.full'}] if h in self.images else []
                if h in self.images:
                    images.append((h+'.image.full', self.images[h]))
            else:
                selected, audit[h] = self.indices[h, role].search(query, 10)
                records[h] = [self.public(item) for item in selected]
        return {'TASK': task, 'RECORDS': records}, images, audit


def validate_reader(result, candidates, records):
    actual = [item['candidate'] for item in result['findings']]
    if len(actual) != len(set(actual)) or set(actual) != set(candidates):
        raise ValueError('Reader did not return each requested candidate exactly once')
    for finding in result['findings']:
        allowed = {r['id'] for h in ['SRC', finding['candidate']] for r in records[h]}
        if not set(finding['refs']) <= allowed:
            raise ValueError('Reader cited evidence outside the compared pair')
        if finding['judgment'] != 'unknown' and not finding['refs']:
            raise ValueError('Non-unknown finding has no evidence reference')
    resolved = any(f['judgment'] != 'unknown' for f in result['findings'])
    if (result['status'] == 'answered') != resolved:
        raise ValueError('Reader status disagrees with findings')


def solve(evidence, preview, images, client, max_inspections=2, on_event=None):
    if max_inspections < 0:
        raise ValueError('Negative inspection budget')
    results, trace, extra_images = [], [], []
    visible = {r['id']: deepcopy(r) for record in preview.values()
               for key in ('attributes', 'relations', 'images') for r in record[key]}
    read_ids = set(visible)
    inspections = 0

    def emit(event):
        trace.append(event)
        if on_event:
            on_event(event)

    while True:
        allow = inspections < max_inspections
        payload = {'RECORDS': preview, 'CURRENT_CHOICE': None, 'AGENT_RESULTS': results,
                   'AVAILABLE_ACTIONS': {'inspect': allow, 'finish': True}}
        action = client('planner', payload, images+extra_images, planner_schema(visible, allow))
        emit({'kind': 'planner', 'action': action})
        if action['action'] == 'finish':
            if action['choice'] not in LABELS or not set(action['refs']) <= set(visible):
                raise ValueError('Invalid final choice or evidence references')
            choice = action['choice']
            target = evidence.handles[choice]
            return {'choice': choice, 'rank': [target]+[u for u in evidence.rank if u != target],
                    'handle_map': evidence.handles, 'inspections': inspections, 'trace': trace,
                    'evidence': visible, 'initial_preview_sha256': digest(preview),
                    'max_inspections': max_inspections, 'labels_loaded': False}
        # This check is required even when structured decoding already excludes inspect.
        if not allow:
            raise ValueError('Inspection budget exceeded')
        if action['agent'] not in ROLES[1:] or not 1 <= len(action['candidates']) <= 3:
            raise ValueError('Invalid inspection')
        if len(set(action['candidates'])) != len(action['candidates']):
            raise ValueError('Duplicate inspection candidates')
        payload, acquired_images, retrieval = evidence.retrieve(action)
        inspections += 1  # Count attempted reader calls, including failures.
        supplied = {r['id']: r for records in payload['RECORDS'].values() for r in records}
        new_ids = set(supplied)-read_ids
        emit({'kind': 'retrieval', 'agent': action['agent'], 'inspection': inspections,
              'query': action['question'], 'records': payload['RECORDS'], 'retrieval': retrieval,
              'new_evidence_ids': sorted(new_ids), 'images': acquired_images})
        result = client(action['agent'], payload, acquired_images,
                        reader_schema(action['candidates'], payload['RECORDS']))
        validate_reader(result, action['candidates'], payload['RECORDS'])
        cited = set(r for f in result['findings'] for r in f['refs'])
        visible.update({r: supplied[r] for r in cited})
        read_ids.update(supplied)
        for label, path in acquired_images:
            if label in cited and (label, path) not in extra_images:
                extra_images.append((label, path))
        receipt = {'agent': action['agent'], 'candidates': action['candidates'],
                   'question': action['question'], 'result': result,
                   'evidence': [supplied[r] for r in sorted(cited)]}
        results.append(receipt)
        emit({'kind': 'reader', **receipt})


class RoleClient:
    """Uses actual per-role system messages without modifying the historical backend."""
    def __init__(self, model, prompts, folder, validate):
        self.model, self.prompts, self.folder, self.validate = model, prompts, Path(folder), validate
        self.folder.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def __call__(self, role, payload, images, schema):
        import torch
        from PIL import Image
        from .engine import constrained_decoder
        number = len(self.calls)
        stem = self.folder/f'{number:02d}_{role}'
        system = self.prompts[role]
        prompt = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        stem.with_suffix('.system.txt').write_text(system)
        stem.with_suffix('.user.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        stem.with_suffix('.schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
        content, opened = [], []
        started = time.perf_counter()
        call = {'role': role, 'images': images, 'started_at': time.time()}
        self.calls.append(call)
        try:
            for label, path in images:
                content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
                with Image.open(path) as source:
                    opened.append(source.convert('RGB'))
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
            formatted = self.model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            stem.with_suffix('.formatted.txt').write_text(formatted)
            batch = self.model.processor(text=[formatted], images=opened or None,
                return_tensors='pt', padding=True).to('cuda:0')
            n = batch.input_ids.shape[1]
            max_tokens = 768
            if n+max_tokens > 16000:
                raise ValueError(f'Context overflow {n}+{max_tokens}; original records not truncated')
            decoder = constrained_decoder(self.model.format_tokens, schema, unicode_json=True)
            with torch.inference_mode():
                output = self.model.model.generate(**batch, max_new_tokens=max_tokens, do_sample=False,
                    use_cache=True, pad_token_id=self.model.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=decoder)
            tail = output[0, n:]
            raw = self.model.tokenizer.decode(tail, skip_special_tokens=True)
            stem.with_suffix('.output.txt').write_text(raw)
            call.update(input_tokens=n, output_tokens=len(tail), prompt_sha256=digest(formatted))
            result = json.loads(raw)
            self.validate(result, schema)
            call['valid_json_schema'] = True
            return result
        except Exception as exc:
            call['error'] = str(exc)
            raise
        finally:
            call['seconds'] = time.perf_counter()-started
            stem.with_suffix('.meta.json').write_text(json.dumps(call, ensure_ascii=False, indent=2))
            for image in opened:
                image.close()
