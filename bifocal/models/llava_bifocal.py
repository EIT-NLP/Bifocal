import copy
import warnings
from datetime import timedelta
from typing import List, Tuple

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

from bifocal.config import BifocalConfig
from bifocal.llava import apply, set_query
from bifocal.query import query_embeddings

torch.backends.cuda.matmul.allow_tf32 = True
warnings.filterwarnings("ignore")


@register_model("llava_bifocal")
class LlavaBifocal(lmms):
    def __init__(self, pretrained, budget=128, conv_template="vicuna_v1", use_cache=True,
                 attn_implementation="sdpa", batch_size=1, **kwargs):
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("Bifocal requires batch_size=1, use multiple processes for data parallelism")
        self.bifocal = BifocalConfig(budget=int(budget))
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))])
        self.accelerator = accelerator
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        self._tokenizer, model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, None, get_model_name_from_path(pretrained), device_map=f"cuda:{accelerator.local_process_index}",
            attn_implementation=attn_implementation)
        self._model = apply(model, self.bifocal)
        self._config = self._model.config
        self._model.eval()
        self._model.tie_weights()
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.batch_size_per_gpu = 1
        if accelerator.num_processes > 1:
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.process_index
            self._world_size = accelerator.num_processes
        else:
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        encoding = self.tokenizer.encode(string, add_special_tokens=bool(add_special_tokens))
        return encoding[-left_truncate_len:] if left_truncate_len else encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens if isinstance(tokens, list) else [tokens])

    def flatten(self, items):
        if not items or any(i is None for i in items):
            return []
        return [x for group in items if group for x in group]

    def _set_query(self, context):
        if not self.bifocal.baseline:
            set_query(self.model, query_embeddings(self.model, self.tokenizer, self.device, context))

    def _clear_query(self):
        if not self.bifocal.baseline:
            set_query(self.model, None)

    def _images(self, visuals):
        if not visuals:
            return None
        images = process_images(visuals, self._image_processor, self._config)
        if isinstance(images, list):
            return [image.to(dtype=torch.float16, device=self.device) for image in images]
        return images.to(dtype=torch.float16, device=self.device)

    def _prompt(self, question):
        conv = (copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template
                else conv_templates[self.conv_template].copy())
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        progress = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [r.args for r in requests]:
            doc = self.task_dict[task][split][doc_id]
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(doc)
            visuals = self.flatten([doc_to_visual(doc)])
            image = self._images(visuals)
            question = contexts[0] if isinstance(contexts, list) else contexts
            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in question:
                question = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals)) + "\n" + question
            conv = self._prompt(question)
            context_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                                return_tensors="pt").unsqueeze(0).to(self.device)
            conv.messages[1][1] = continuation
            input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            labels[0, : context_ids.shape[1]] = -100
            self._set_query(question)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True,
                                     image_sizes=[[v.size[0], v.size[1]] for v in visuals])
            self._clear_query()
            greedy = outputs["logits"].argmax(dim=-1)[:, context_ids.shape[1]: input_ids.shape[1]]
            target = input_ids[:, context_ids.shape[1]:]
            results.append((float(outputs["loss"].item()), bool((greedy == target).all())))
            progress.update(1)
        progress.close()
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        ordered = utils.Collator([r.args for r in requests], lambda x: (-len(self.tok_encode(x[0])), x[0]),
                                 grouping=True)
        progress = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for chunk in ordered.get_batched(n=1, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][i]) for i in doc_id]
            flat_visuals = self.flatten(visuals)
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.pop("until", [self.tok_decode(self.eot_token_id)])
            until = [until] if isinstance(until, str) else until
            if "image_aspect_ratio" in gen_kwargs and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
            image = self._images(flat_visuals)
            context = contexts[0]
            question = context
            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                count = len(visuals[0]) if isinstance(visuals[0], list) else 1
                question = " ".join([DEFAULT_IMAGE_TOKEN] * count) + "\n" + context
            prompt = self._prompt(question).get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(self.device)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            temperature = gen_kwargs.get("temperature", 0)
            self._set_query(context)
            output = self.model.generate(
                input_ids,
                attention_mask=input_ids.ne(pad_id),
                pad_token_id=pad_id,
                images=image,
                image_sizes=[v.size for v in flat_visuals],
                do_sample=temperature > 0,
                top_p=gen_kwargs.get("top_p", None),
                num_beams=gen_kwargs.get("num_beams", 1),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                use_cache=self.use_cache,
            )
            self._clear_query()
            text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            results.extend(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            progress.update(1)
        progress.close()
        return ordered.get_original(results)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError
