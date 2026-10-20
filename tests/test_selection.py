import pytest
import torch

from bifocal.config import COARSE_TOKENS, NUM_PATCHES, BifocalConfig
from bifocal.selection import coarse_pool, select_tokens

EXPECTED = {192: (336, 48, 136), 128: (224, 32, 80), 64: (112, 16, 24)}


def random_inputs(seed, with_query=True):
    g = torch.Generator().manual_seed(seed)
    patch = torch.randn(1, NUM_PATCHES, 64, generator=g)
    cls_attention = torch.rand(1, NUM_PATCHES, generator=g) ** 4
    projector = torch.nn.Linear(64, 32)
    words = torch.randn(5, 32, generator=g) if with_query else None
    return patch, cls_attention, projector, words


@pytest.mark.parametrize("budget", sorted(EXPECTED))
def test_budget_rule(budget):
    config = BifocalConfig(budget)
    stage1, stage2, intrinsic = EXPECTED[budget]
    assert (config.stage1_tokens, config.stage2_tokens, config.intrinsic_tokens) == (stage1, stage2, intrinsic)
    assert config.fine_tokens == stage1 - COARSE_TOKENS
    assert config.average_tokens == budget


def test_baseline_and_invalid_budgets():
    assert BifocalConfig(NUM_PATCHES).baseline
    with pytest.raises(ValueError):
        BifocalConfig(130)
    with pytest.raises(ValueError):
        BifocalConfig(332)


@pytest.mark.parametrize("budget", sorted(EXPECTED))
@pytest.mark.parametrize("with_query", [True, False])
def test_selection_is_a_sorted_set(budget, with_query):
    patch, cls_attention, projector, words = random_inputs(budget, with_query)
    with torch.no_grad():
        tokens, index = select_tokens(patch, cls_attention, BifocalConfig(budget), projector, words)
    count = BifocalConfig(budget).stage1_tokens
    assert tokens.shape == (1, count, patch.shape[-1])
    assert index.shape == (1, count)
    assert index.unique().numel() == count
    assert bool((index[0, 1:] > index[0, :-1]).all())
    assert 0 <= int(index.min()) and int(index.max()) < NUM_PATCHES


def test_coarse_tokens_cover_every_block():
    patch, cls_attention, _, _ = random_inputs(0)
    _, index = coarse_pool(patch, cls_attention)
    blocks = (index // 24 // 3) * 8 + (index % 24 // 3)
    assert sorted(blocks[0].tolist()) == list(range(COARSE_TOKENS))
