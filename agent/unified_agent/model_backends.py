"""Native local VLM adapters with a common constrained decoding contract.

InternVL preprocessing and visual-token replacement follow its downloaded
official README and modeling_internvl_chat.py. Both backends use real logits
for label decisions and the same JSON grammar; no parsed-label fallback.
"""
from collections import OrderedDict
import hashlib
import sys
import time
from pathlib import Path

from .engine import LocalVLM, SYSTEM, constrained_decoder


class ComparisonVLM(LocalVLM):
    def __init__(self, backend, model_path, dependency_path):
        sys.path.insert(0, str(Path(dependency_path).resolve()))
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
        torch.set_num_threads(2)
        self.backend = backend
        self.calls = self.input_tokens = self.output_tokens = 0
        self.records = []
        self.unicode_json = True
        self.feature_cache = OrderedDict()
        if backend == 'qwen3':
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                min_pixels=256*32*32, max_pixels=512*32*32)
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
                local_files_only=True, torch_dtype=torch.bfloat16, device_map={'':'cuda:0'},
                attn_implementation='sdpa').eval()
        elif backend == 'internvl3':
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=True, use_fast=False)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=False).eval().cuda()
            # The official wrapper defaults to eager attention without flash-attn.
            # Its Qwen2 language model supports PyTorch SDPA on the installed version.
            self.model.language_model.set_attn_implementation('sdpa')
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        else:
            raise ValueError(backend)
        self.model.requires_grad_(False)
        sys.path.insert(0, str(Path(dependency_path).resolve()))
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.format_tokens = build_token_enforcer_tokenizer_data(self.tokenizer)

    def _features(self, path):
        import torch
        from PIL import Image
        from torchvision import transforms as T
        if path in self.feature_cache:
            self.feature_cache.move_to_end(path)
            return self.feature_cache[path]
        size = 448
        with Image.open(path) as original:
            image = original.convert('RGB')
        width, height = image.size
        ratios = sorted({(i, j) for i in range(1, 3) for j in range(1, 3)
                         if i*j <= 2}, key=lambda r:(r[0]*r[1], r))
        ratio = (1, 1); distance = float('inf')
        for candidate in ratios:
            delta = abs(width/height-candidate[0]/candidate[1])
            if delta < distance or (delta == distance and width*height >
                                   0.5*size*size*candidate[0]*candidate[1]):
                ratio, distance = candidate, delta
        resized = image.resize((size*ratio[0], size*ratio[1]))
        tiles = [resized.crop(((i%ratio[0])*size, (i//ratio[0])*size,
                               (i%ratio[0]+1)*size, (i//ratio[0]+1)*size))
                 for i in range(ratio[0]*ratio[1])]
        if len(tiles) != 1:
            tiles.append(image.resize((size, size)))
        transform = T.Compose([T.Resize((size,size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        pixels = torch.stack([transform(tile) for tile in tiles]).to('cuda:0',torch.bfloat16)
        with torch.inference_mode():
            features = self.model.extract_feature(pixels).detach()
        image.close(); resized.close()
        for tile in tiles: tile.close()
        self.feature_cache[path] = features
        if len(self.feature_cache) > 24: self.feature_cache.popitem(last=False)
        return features

    def _internvl_input(self, prompt, images, system):
        import torch
        module = sys.modules[type(self.model).__module__]
        template = module.get_conv_template(self.model.template)
        template.system_message = system
        blocks=[]; features=[]
        for label, path in images:
            feature = self._features(path)
            features.append(feature)
            blocks.append(label+'\n<img>'+'<IMG_CONTEXT>'*(feature.shape[0]*self.model.num_image_token)+'</img>')
        blocks.append(prompt)
        template.append_message(template.roles[0], '\n'.join(blocks))
        template.append_message(template.roles[1], None)
        formatted = template.get_prompt()
        batch = self.tokenizer(formatted,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            embeds = self.model.language_model.get_input_embeddings()(batch.input_ids)
            if features:
                mask = batch.input_ids == self.model.img_context_token_id
                visual = torch.cat(features).reshape(-1,embeds.shape[-1])
                assert int(mask.sum()) == visual.shape[0], 'Image-to-token mapping mismatch'
                embeds[mask] = visual
        return batch, embeds, formatted, self.tokenizer.convert_tokens_to_ids(template.sep.strip())

    def _record(self, n, out, started, formatted, images, **extra):
        self.calls += 1; self.input_tokens += n; self.output_tokens += out
        meta = {'input_tokens':n, 'output_tokens':out, 'seconds':time.time()-started,
                'prompt_sha256':hashlib.sha256(formatted.encode()).hexdigest(),
                'images':[p for _,p in images], 'backend':self.backend, **extra}
        self.records.append(meta)
        return meta

    def choose(self, prompt, images, labels):
        if self.backend == 'qwen3':
            return super().choose(prompt,images,labels)
        import torch
        system = SYSTEM.split('Return one valid JSON')[0]+'Return only one of the requested action labels.'
        started=time.time()
        batch, embeds, formatted, _ = self._internvl_input(prompt,images,system)
        n=batch.input_ids.shape[1]
        if n+1>16000: raise ValueError('Context budget exceeded; no silent truncation')
        codes=[self.tokenizer.encode(label,add_special_tokens=False) for label in labels]
        assert all(len(code)==1 for code in codes), (labels,codes)
        with torch.inference_mode():
            logits=self.model.language_model(inputs_embeds=embeds,attention_mask=batch.attention_mask,
                use_cache=False,logits_to_keep=1).logits[0,-1].float()
            values=logits[[code[0] for code in codes]].softmax(0).cpu().tolist()
        meta=self._record(n,1,started,formatted,images,decision_method='argmax over allowed one-token labels; not calibrated confidence')
        return dict(zip(labels,values)),meta

    def generate(self, prompt, images, max_tokens=224, schema=None):
        if self.backend == 'qwen3':
            return super().generate(prompt,images,max_tokens,schema)
        import torch
        started=time.time()
        batch, embeds, formatted, eos=self._internvl_input(prompt,images,SYSTEM)
        n=batch.input_ids.shape[1]
        if n+max_tokens>16000: raise ValueError(f'Context budget exceeded: {n}+{max_tokens}')
        decoding={}
        if schema:
            decoding['prefix_allowed_tokens_fn']=constrained_decoder(self.format_tokens,schema,self.unicode_json)
        with torch.inference_mode():
            result=self.model.language_model.generate(input_ids=batch.input_ids,inputs_embeds=embeds,
                attention_mask=batch.attention_mask,max_new_tokens=max_tokens,do_sample=False,
                use_cache=True,eos_token_id=eos,pad_token_id=eos,**decoding)
        assert torch.equal(result[0,:n],batch.input_ids[0]), 'Unexpected generation prefix'
        tail=result[0,n:]
        raw=self.tokenizer.decode(tail,skip_special_tokens=True)
        return raw,self._record(n,len(tail),started,formatted,images,decoder='unicode_json_v8_boundaries')
