import torch
import torch.nn.functional as F

from .config import BLOCK, COARSE_SIDE, COARSE_TOKENS, GRID


def smooth(maps):
    n = maps.shape[0]
    pooled = F.avg_pool2d(maps.view(n, 1, GRID, GRID), 3, stride=1, padding=1,
                          count_include_pad=True)
    return pooled.view(n, GRID * GRID)


def min_max(maps):
    low = maps.min(dim=1, keepdim=True).values
    high = maps.max(dim=1, keepdim=True).values
    return (maps - low) / (high - low + 1e-8)


def norm_map(patch_llm):
    return smooth(patch_llm[0].float().norm(dim=-1).view(1, -1))


def residualize(y, x):
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    denom = xc.pow(2).sum(dim=1, keepdim=True)
    slope = torch.where(denom > 1e-12, (xc * yc).sum(dim=1, keepdim=True) / denom,
                        torch.zeros_like(denom))
    return yc - slope * xc


def image_saliency(cls_attention, norms):
    return residualize(smooth(cls_attention.float().view(1, -1)), norms)


def word_maps(patch_llm, word_embeds):
    patches = F.normalize(patch_llm[0].float(), dim=-1)
    words = F.normalize(word_embeds.to(device=patch_llm.device, dtype=torch.float32), dim=-1)
    return smooth(words @ patches.t())


def query_map(patch_llm, word_embeds, saliency, eps=1e-6):
    sims = word_maps(patch_llm, word_embeds)
    centered_saliency = saliency - saliency.mean(dim=1, keepdim=True)
    centered_sims = sims - sims.mean(dim=1, keepdim=True)
    corr = (centered_sims * centered_saliency).sum(dim=1) / (
        centered_sims.norm(dim=1) * centered_saliency.norm(dim=1) + 1e-12)
    weights = corr.clamp_min(0.0) + 1e-6
    weights = weights / weights.sum()
    normalized = min_max(sims)
    return torch.exp((weights.view(-1, 1) * torch.log(normalized + eps)).sum(dim=0)).view(1, -1)


def fuse(relevance, saliency, norms, eps=1e-6):
    signals = torch.cat([relevance.float().view(1, -1), saliency, -norms], dim=0)
    return torch.exp(torch.log(min_max(signals) + eps).mean(dim=0)).view(1, -1)


def query_relevance(patch, cls_attention, projector, word_embeds):
    rows = []
    for b in range(patch.shape[0]):
        patch_llm = projector(patch[b:b + 1])
        norms = norm_map(patch_llm)
        saliency = image_saliency(cls_attention[b:b + 1], norms)
        relevance = query_map(patch_llm, word_embeds, saliency)
        rows.append(fuse(relevance, saliency, norms))
    return torch.cat(rows, dim=0)


def _block_view(x, batch):
    x = x.view(batch, COARSE_SIDE, BLOCK, COARSE_SIDE, BLOCK)
    return x.permute(0, 1, 3, 2, 4).reshape(batch, COARSE_SIDE, COARSE_SIDE, BLOCK * BLOCK)


def _block_view_features(x, batch, dim):
    x = x.view(batch, COARSE_SIDE, BLOCK, COARSE_SIDE, BLOCK, dim)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, COARSE_SIDE, COARSE_SIDE, BLOCK * BLOCK, dim)


def _weighted_blocks(patch, weights):
    batch, _, dim = patch.shape
    w = _block_view(weights, batch)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    blocks = _block_view_features(patch, batch, dim)
    return (blocks * w.unsqueeze(-1)).sum(dim=3).reshape(batch, COARSE_TOKENS, dim), w


def coarse_pool(patch, cls_attention):
    batch, _, dim = patch.shape
    tokens, w = _weighted_blocks(patch, cls_attention)
    device = patch.device
    in_row = torch.arange(BLOCK, device=device).repeat_interleave(BLOCK).float()
    in_col = torch.arange(BLOCK, device=device).repeat(BLOCK).float()
    block_row = torch.arange(COARSE_SIDE, device=device).view(1, -1, 1) * BLOCK
    block_col = torch.arange(COARSE_SIDE, device=device).view(1, 1, -1) * BLOCK
    wf = w.float()
    row = torch.floor((wf * in_row).sum(-1) + 0.5).long().clamp(0, BLOCK - 1)
    col = torch.floor((wf * in_col).sum(-1) + 0.5).long().clamp(0, BLOCK - 1)
    index = ((block_row + row) * GRID + (block_col + col)).reshape(batch, COARSE_TOKENS)
    order = index.argsort(dim=1)
    return (torch.gather(tokens, 1, order.unsqueeze(-1).expand(-1, -1, dim)),
            torch.gather(index, 1, order))


def residual_pool(patch, cls_attention, coarse_tokens, coarse_index, fine_index):
    batch, _, dim = patch.shape
    weights = cls_attention.clone()
    weights.scatter_(1, fine_index, 0.0)
    tokens, _ = _weighted_blocks(patch, weights)
    remaining = torch.ones_like(cls_attention)
    remaining.scatter_(1, fine_index, 0.0)
    counts = _block_view(remaining, batch).reshape(batch, COARSE_TOKENS, BLOCK * BLOCK).sum(-1)
    block_id = (coarse_index // GRID // BLOCK) * COARSE_SIDE + (coarse_index % GRID // BLOCK)
    pooled = torch.gather(tokens, 1, block_id.unsqueeze(-1).expand(-1, -1, dim))
    empty = torch.gather(counts == 0, 1, block_id)
    return torch.where(empty.unsqueeze(-1), coarse_tokens, pooled)


def facility_location(features, saliency, blocked, seeds, k):
    x = features.float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    similarity = x @ x.t()
    s = saliency.float()
    s = s / s.max().clamp_min(1e-8)
    coverage = similarity[seeds].max(dim=0).values.clamp_min(0.0)
    free = ~blocked
    selected = torch.empty(k, dtype=torch.long, device=features.device)
    floor = torch.full_like(s, -1.0)
    s_row = s.unsqueeze(0)
    for i in range(k):
        gain = (torch.relu(similarity - coverage.unsqueeze(0)) * s_row).sum(dim=1)
        pick = torch.argmax(torch.where(free, gain, floor))
        selected[i] = pick
        free.index_fill_(0, pick.view(1), False)
        coverage = torch.maximum(coverage, similarity.index_select(0, pick.view(1)).squeeze(0))
    return selected


def select_fine(relevance, cls_attention, coarse_index, patch, config, has_query):
    priority = relevance.clone()
    priority.scatter_(1, coarse_index, float("-inf"))
    if not has_query:
        return torch.topk(priority, config.fine_tokens, dim=1).indices
    rows = []
    for b in range(relevance.shape[0]):
        seeds = torch.cat([coarse_index[b], torch.topk(priority[b], config.query_tokens).indices])
        blocked = torch.zeros(relevance.shape[1], dtype=torch.bool, device=relevance.device)
        blocked.index_fill_(0, seeds, True)
        rows.append(facility_location(patch[b], cls_attention[b], blocked, seeds,
                                      config.intrinsic_tokens))
    intrinsic = torch.stack(rows, dim=0)
    priority.scatter_(1, intrinsic, float("-inf"))
    query = torch.topk(priority, config.query_tokens, dim=1).indices
    return torch.cat([intrinsic, query], dim=1)


def merge_sorted(coarse_tokens, coarse_index, fine_tokens, fine_index):
    index = torch.cat([coarse_index, fine_index], dim=1)
    tokens = torch.cat([coarse_tokens, fine_tokens], dim=1)
    order = index.argsort(dim=1)
    return (torch.gather(tokens, 1, order.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])),
            torch.gather(index, 1, order))


def select_tokens(patch, cls_attention, config, projector=None, word_embeds=None):
    dim = patch.shape[-1]
    has_query = projector is not None and word_embeds is not None and word_embeds.shape[0] > 0
    coarse_tokens, coarse_index = coarse_pool(patch, cls_attention)
    if has_query:
        relevance = query_relevance(patch, cls_attention, projector, word_embeds)
    else:
        relevance = cls_attention.clone()
    fine_index = select_fine(relevance, cls_attention, coarse_index, patch, config, has_query)
    coarse_tokens = residual_pool(patch, cls_attention, coarse_tokens, coarse_index, fine_index)
    fine_tokens = torch.gather(patch, 1, fine_index.unsqueeze(-1).expand(-1, -1, dim))
    return merge_sorted(coarse_tokens, coarse_index, fine_tokens, fine_index)
