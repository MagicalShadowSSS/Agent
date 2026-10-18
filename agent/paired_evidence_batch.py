"""Independent SRC/candidate evidence calls, batched without sharing contexts.

This changes only how one modality Evidence Agent examines the ten records.
There are no additional roles, candidate filters, gold labels or voting rules.
Each pair has its own prompt and received-ID schema. Batch wall time is charged
once, with the individual inputs and outputs retained for inspection.
"""
from copy import deepcopy
import json
import time


def read_pairs(client, role, payload, images, obj, enum, batch_size=4, grounded=False):
    import torch
    from PIL import Image
    from unified_agent.engine import constrained_decoder

    model = client.model
    assert model.backend == 'qwen3'
    tasks, missing = [], []
    source = payload['RECORDS']['SRC']
    for candidate in payload['TASK']['candidates']:
        handle = candidate['handle']
        rows = payload['RECORDS'][handle]
        if not source or not rows:
            missing.append({'candidate': handle, 'status': 'uncertain',
                            'note': 'No supplied observation on one side.',
                            'refs': {'source': '', 'candidate': ''}})
            continue
        context = {'TASK': {**payload['TASK'], 'candidates': [candidate]},
                   'RECORDS': {'SRC': source, handle: rows}}
        attached = [pair for pair in images
                    if pair[0].startswith('SRC.') or pair[0].startswith(handle + '.')]
        schema = obj({'note': {'type': 'string', 'minLength': 1, 'maxLength': 120},
                      'status': enum(['support', 'conflict', 'mixed', 'uncertain']),
                      'refs': obj({'source': enum([''] + [row['id'] for row in source]),
                                   'candidate': enum([''] + [row['id'] for row in rows])})})
        if grounded:
            schema['properties']={'subjects':obj({'source':{'type':'string','maxLength':60},'candidate':{'type':'string','maxLength':60}}),**schema['properties']}
            schema['required']=['subjects','note','status','refs']
            schema['properties']['status']=enum(['identity','association','contradiction','uncertain'])
        tasks.append((handle, context, attached, schema))
    results = []
    for offset in range(0, len(tasks), batch_size):
        group = tasks[offset:offset + batch_size]
        while True:
            torch.cuda.empty_cache()
            free = [torch.cuda.mem_get_info(device)[0] / 1024**3 for device in range(2)]
            if min(free) >= 2.0:
                break
            (client.folder / 'waiting_for_memory.json').write_text(json.dumps(
                {'free_gib': free, 'time': time.time(), 'pair_offset': offset}, indent=2))
            time.sleep(10)
        started = time.perf_counter()
        batch_index = len(client.calls)
        call = {'role': role, 'batch_size': len(group), 'independent_contexts': True,
                'started_at': time.time(), 'backend': model.backend}
        client.calls.append(call)
        opened, formatted, schemas, stems = [], [], [], []
        old_padding = model.tokenizer.padding_side
        try:
            model.tokenizer.padding_side = 'left'
            for handle, context, attached, schema in group:
                stem = client.folder / f'{batch_index:02d}_{role}_{handle}'
                stems.append(stem)
                system = client.prompts[role]
                content = []
                for label, path in attached:
                    content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
                    with Image.open(path) as image:
                        opened.append(image.convert('RGB'))
                content.append({'type': 'text', 'text': json.dumps(context, ensure_ascii=False)})
                messages = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': content}]
                text = model.processor.apply_chat_template(messages, tokenize=False,
                                                           add_generation_prompt=True)
                formatted.append(text)
                schemas.append(schema)
                stem.with_suffix('.system.txt').write_text(system)
                stem.with_suffix('.user.json').write_text(json.dumps(context, ensure_ascii=False, indent=2))
                stem.with_suffix('.schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
                stem.with_suffix('.formatted.txt').write_text(text)
            batch = model.processor(text=formatted, images=opened or None,
                                    padding=True, return_tensors='pt').to('cuda:0')
            n = batch.input_ids.shape[1]
            assert n + 512 <= 262144, 'No silent context truncation.'
            decoders = [constrained_decoder(model.format_tokens, schema, unicode_json=True)
                        for schema in schemas]
            decoding = {'seconds': 0.0, 'calls': 0}

            def allowed(batch_id, tokens):
                tick = time.perf_counter()
                try:
                    return decoders[batch_id](0, tokens)
                finally:
                    decoding['seconds'] += time.perf_counter() - tick
                    decoding['calls'] += 1

            call['input_preparation_seconds'] = time.perf_counter() - started
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = model.model.generate(**dict(batch), logits_to_keep=1,
                    max_new_tokens=512, do_sample=False, use_cache=True,
                    pad_token_id=model.tokenizer.eos_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=allowed)
            call.update(generation_seconds=time.perf_counter() - generation_started,
                        constraint_callback_seconds=decoding['seconds'],
                        constraint_callback_calls=decoding['calls'],
                        input_tokens=int(batch.attention_mask.sum()),
                        padded_input_tokens=int(batch.input_ids.numel()),
                        batch_input_width=n, images=[a for task in group for a in task[2]])
            assert torch.equal(output[:, :n], batch.input_ids)
            output_count = 0
            for index, (handle, context, attached, schema) in enumerate(group):
                tail = output[index, n:]
                output_count += int((tail != model.tokenizer.eos_token_id).sum())
                raw = model.tokenizer.decode(tail, skip_special_tokens=True)
                stems[index].with_suffix('.output.txt').write_text(raw)
                result = json.loads(raw)
                client.validate(result, schema)
                received = {row['id'] for rows in context['RECORDS'].values() for row in rows}
                assert all(not ref or ref in received for ref in result['refs'].values())
                results.append({'candidate': handle, **result})
            call['output_tokens'] = output_count
            call['valid_json_schema'] = True
        except Exception as error:
            call['error'] = repr(error)
            raise
        finally:
            call['seconds'] = time.perf_counter() - started
            model.tokenizer.padding_side = old_padding
            for stem in stems:
                stem.with_suffix('.meta.json').write_text(json.dumps(call, ensure_ascii=False, indent=2))
            for image in opened:
                image.close()
    all_results = results + missing
    assert {x['candidate'] for x in all_results} == set('ABCDEFGHIJ')
    return {'source_observation': 'Source context and original records supplied to each independent comparison.',
            'source_refs': [], 'observations': all_results}
