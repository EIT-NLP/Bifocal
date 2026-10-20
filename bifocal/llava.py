import torch
import torch.nn as nn
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .config import DENSITY_STRENGTH, NUM_PATCHES, STAGE2_LAYER
from .selection import select_tokens


def clip_shape(self, tensor, seq_len, bsz):
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


def clip_attention_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None,
                           output_attentions=False):
    bsz, tgt_len, embed_dim = hidden_states.size()
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)
    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    if causal_attention_mask is not None:
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    if attention_mask is not None:
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None
    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_probs, value_states)
    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    return self.out_proj(attn_output), attn_weights_reshaped


def clip_encoder_layer_forward(self, hidden_states, attention_mask, causal_attention_mask,
                               output_attentions=False):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = residual + self.mlp(hidden_states)
    if output_attentions:
        return hidden_states, attn_weights
    return (hidden_states,)


@torch.no_grad()
def vision_tower_forward(self, images):
    pixels = images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype)
    outputs = self.vision_tower(pixels, output_hidden_states=True, output_attentions=True)
    patch = outputs.hidden_states[self.select_layer][:, 1:, :]
    cls_attention = outputs.attentions[self.select_layer][:, :, 0, 1:].sum(dim=1)
    tokens, index = select_tokens(patch, cls_attention, self.bifocal_config,
                                  self.bifocal_projector, self.bifocal_words)
    return tokens.to(images.dtype), index.squeeze(0)


def llava_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                  inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                  output_hidden_states=None, images=None, image_sizes=None, return_dict=None,
                  cache_position=None, **kwargs):
    if inputs_embeds is None:
        (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds,
         labels) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels, images,
            image_sizes=image_sizes)
    for key, value in (("output_attentions", output_attentions),
                       ("output_hidden_states", output_hidden_states),
                       ("return_dict", return_dict)):
        if value is not None:
            kwargs[key] = value
    return LlamaForCausalLM.forward(
        self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
        use_cache=use_cache, cache_position=cache_position, **kwargs)


def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask,
                                         past_key_values, labels, images, image_sizes=None):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels
    tokens, keep_index = vision_tower.forward(images)
    image_features = self.get_model().mm_projector(tokens)

    original_labels = labels
    original_position_ids = position_ids
    original_attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
    labels = [lab[mask] for lab, mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    visual_starts = []
    image_index = 0
    for batch_index, ids in enumerate(input_ids):
        num_images = int((ids == IMAGE_TOKEN_INDEX).sum())
        if num_images == 0:
            features = image_features[image_index]
            embeds = self.get_model().embed_tokens(ids)
            new_input_embeds.append(torch.cat([embeds, features[0:0]], dim=0))
            new_labels.append(labels[batch_index])
            image_index += 1
            continue
        boundaries = [-1] + torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist() + [ids.shape[0]]
        sample_labels = labels[batch_index]
        text_ids = [ids[boundaries[i] + 1:boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        text_labels = [sample_labels[boundaries[i] + 1:boundaries[i + 1]]
                       for i in range(len(boundaries) - 1)]
        split_sizes = [x.shape[0] for x in text_labels]
        text_embeds = torch.split(self.get_model().embed_tokens(torch.cat(text_ids)), split_sizes, dim=0)
        pieces = []
        piece_labels = []
        for i in range(num_images + 1):
            pieces.append(text_embeds[i])
            piece_labels.append(text_labels[i])
            if i < num_images:
                features = image_features[image_index]
                if batch_index == 0:
                    visual_starts.append(sum(x.shape[0] for x in pieces))
                image_index += 1
                pieces.append(features)
                piece_labels.append(torch.full((features.shape[0],), IGNORE_INDEX,
                                               device=sample_labels.device, dtype=sample_labels.dtype))
        new_input_embeds.append(torch.cat([x.to(self.device) for x in pieces]))
        new_labels.append(torch.cat(piece_labels))

    if len(visual_starts) > 1:
        raise ValueError("Bifocal supports a single image per sample")
    max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if max_length is not None:
        new_input_embeds = [x[:max_length] for x in new_input_embeds]
        new_labels = [x[:max_length] for x in new_labels]
    self.get_model().bifocal_visual = (
        (visual_starts[0], keep_index, NUM_PATCHES) if visual_starts else None)

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)
    padded_embeds = []
    padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                               device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype,
                               device=position_ids.device)
    left_padding = getattr(self.config, "tokenizer_padding_side", "right") == "left"
    for i, (embeds, sample_labels) in enumerate(zip(new_input_embeds, new_labels)):
        length = embeds.shape[0]
        padding = torch.zeros((max_len - length, embeds.shape[1]), dtype=embeds.dtype, device=embeds.device)
        positions = torch.arange(0, length, dtype=position_ids.dtype, device=position_ids.device)
        if left_padding:
            padded_embeds.append(torch.cat((padding, embeds), dim=0))
            if length > 0:
                padded_labels[i, -length:] = sample_labels
                attention_mask[i, -length:] = True
                position_ids[i, -length:] = positions
        else:
            padded_embeds.append(torch.cat((embeds, padding), dim=0))
            if length > 0:
                padded_labels[i, :length] = sample_labels
                attention_mask[i, :length] = True
                position_ids[i, :length] = positions
    new_input_embeds = torch.stack(padded_embeds, dim=0)

    new_labels = None if original_labels is None else padded_labels
    attention_mask = None if original_attention_mask is None else attention_mask.to(
        dtype=original_attention_mask.dtype)
    position_ids = None if original_position_ids is None else position_ids
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def apply(model, config):
    LlavaLlamaForCausalLM.forward = llava_forward
    language_model = model.get_model()
    language_model.bifocal_visual = None
    language_model.bifocal_stage2 = None
    if config.baseline:
        return model
    if not getattr(modeling_llama, "BIFOCAL_PATCH", False):
        raise RuntimeError("transformers is missing the Bifocal modeling_llama patch, run `bash install.sh`")
    if not hasattr(CLIPAttention, "_shape"):
        CLIPAttention._shape = clip_shape
    CLIPAttention.forward = clip_attention_forward
    CLIPEncoderLayer.forward = clip_encoder_layer_forward
    CLIPVisionTower.forward = vision_tower_forward
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
    language_model.bifocal_stage2 = (STAGE2_LAYER, config.stage2_tokens, DENSITY_STRENGTH)
    tower = language_model.get_vision_tower()
    tower.bifocal_config = config
    tower.bifocal_projector = language_model.mm_projector
    tower.bifocal_words = None
    return model


def set_query(model, word_embeds):
    model.get_model().get_vision_tower().bifocal_words = word_embeds
