import re

import torch

REC_PATTERN = re.compile(r"bounding box coordinate of the region this sentence describes\s*:\s*(.+)",
                         re.IGNORECASE | re.DOTALL)
IMAGE_PATTERN = re.compile(r"<image>", re.IGNORECASE)
INSTRUCTION_SUFFIXES = (
    "\nanswer the question using a single word or phrase.",
    "\nyour response must be concise.",
    "\nprovide a short and direct response.",
    "\nanswer with the option's letter from the given choices directly.",
)
PUNCTUATION = ".,!?;:\"'()[]{}<>`"


def extract_query(context):
    if not isinstance(context, str):
        return None
    query = IMAGE_PATTERN.sub(" ", context).strip()
    match = REC_PATTERN.search(query)
    if match:
        return match.group(1).strip().strip('"').strip()
    lowered = query.lower()
    for suffix in INSTRUCTION_SUFFIXES:
        position = lowered.rfind(suffix)
        if position >= 0:
            query = query[:position]
            lowered = query.lower()
    return query.strip()


@torch.no_grad()
def word_embeddings(tokenizer, embed_tokens, device, text):
    words = [w for w in (raw.strip(PUNCTUATION) for raw in str(text).lower().split()) if w]
    vectors = []
    for word in words:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if ids:
            tokens = torch.as_tensor(ids, device=device).unsqueeze(0)
            vectors.append(embed_tokens(tokens)[0].mean(dim=0))
    if not vectors:
        return None
    return torch.stack(vectors, dim=0)


def query_embeddings(model, tokenizer, device, context):
    text = extract_query(context) or context
    return word_embeddings(tokenizer, model.get_model().embed_tokens, device, text)
