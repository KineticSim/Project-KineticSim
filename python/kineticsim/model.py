from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict

NOISE = 0
MOMENTUM = 1
MAKER = 2
FUNDAMENTAL = 3
N_STRATEGIES = 4

@dataclass(frozen=True)
class SimConfig:

    n_markets: int = 1024
    n_agents: int = 256
    n_levels: int = 128
    n_steps: int = 1000
    seed: int = 0

    init_price: int = 64
    init_depth: float = 8.0

    frac_noise: float = 0.70
    frac_momentum: float = 0.15
    frac_maker: float = 0.15
    frac_fundamental: float = 0.0

    max_order_qty: float = 4.0
    noise_spread: int = 3
    market_order_prob: float = 0.10
    maker_half_spread: int = 1

    def as_dict(self) -> Dict:
        return asdict(self)

    def validate(self) -> None:
        assert self.n_levels >= 8, "need at least 8 price levels"
        assert (self.n_levels & (self.n_levels - 1)) == 0, \
            "n_levels must be a power of two (warp/scan friendliness)"
        assert 0 < self.init_price < self.n_levels - 1
        frac_sum = (self.frac_noise + self.frac_momentum + self.frac_maker
                    + self.frac_fundamental)
        assert abs(frac_sum - 1.0) < 1e-6, "agent fractions must sum to 1"

SWEEP_MARKETS = [64, 256, 1024, 4096, 16384]
SWEEP_AGENTS = [16, 64, 256, 1024]
DEFAULT_STEPS = 500
