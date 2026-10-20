import math
from dataclasses import dataclass

GRID = 24
NUM_PATCHES = GRID * GRID
NUM_LAYERS = 32
COARSE_TOKENS = 64
COARSE_SIDE = math.isqrt(COARSE_TOKENS)
BLOCK = GRID // COARSE_SIDE
STAGE2_LAYER = 16
DENSITY_STRENGTH = 1.0


@dataclass(frozen=True)
class BifocalConfig:
    budget: int = 128

    def __post_init__(self):
        if self.baseline:
            return
        if self.budget % 4:
            raise ValueError(f"budget must be a multiple of 4, got {self.budget}")
        if not COARSE_TOKENS < self.stage1_tokens <= NUM_PATCHES:
            raise ValueError(f"budget {self.budget} gives {self.stage1_tokens} Stage-1 tokens, "
                             f"which must lie in ({COARSE_TOKENS}, {NUM_PATCHES}]")

    @property
    def baseline(self):
        return self.budget >= NUM_PATCHES

    @property
    def stage1_tokens(self):
        return self.budget * 7 // 4

    @property
    def stage2_tokens(self):
        return self.budget // 4

    @property
    def fine_tokens(self):
        return self.stage1_tokens - COARSE_TOKENS

    @property
    def intrinsic_tokens(self):
        return self.fine_tokens // 2

    @property
    def query_tokens(self):
        return self.fine_tokens - self.intrinsic_tokens

    @property
    def average_tokens(self):
        if self.baseline:
            return float(NUM_PATCHES)
        return (STAGE2_LAYER * self.stage1_tokens
                + (NUM_LAYERS - STAGE2_LAYER) * self.stage2_tokens) / NUM_LAYERS
