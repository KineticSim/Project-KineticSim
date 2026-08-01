from __future__ import annotations

from typing import Dict

import numpy as np

from .model import SimConfig, NOISE, MOMENTUM, MAKER, FUNDAMENTAL

def agent_types_for(cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    return rng.choice(
        [NOISE, MOMENTUM, MAKER, FUNDAMENTAL],
        size=cfg.n_markets * cfg.n_agents,
        p=[cfg.frac_noise, cfg.frac_momentum, cfg.frac_maker,
           cfg.frac_fundamental],
    ).astype(np.int32)

class _CudaBase:
    name = "cuda"
    _module_name = ""

    def __init__(self, cfg: SimConfig):
        cfg.validate()
        self.cfg = cfg
        self._mod = None

    def _module(self):
        if self._mod is None:
            self._mod = __import__(self._module_name)
        return self._mod

    def run(self, n_steps: int | None = None, record_prices: bool = False) -> Dict:
        cfg = self.cfg
        n = n_steps if n_steps is not None else cfg.n_steps
        atype = agent_types_for(cfg)
        args = (
            cfg.n_markets, cfg.n_agents, cfg.n_levels, n, int(cfg.seed),
            cfg.init_price, float(cfg.init_depth), float(cfg.max_order_qty),
            cfg.noise_spread, float(cfg.market_order_prob), cfg.maker_half_spread,
            atype,
        )
        r = self._module().simulate(*args, True) if record_prices \
            else self._module().simulate(*args)
        self.price_history = np.asarray(r["price_history"]) if record_prices else None
        self.last_price = np.asarray(r["last_price"])
        self.total_volume = np.asarray(r["total_volume"])
        self.n_trades = np.asarray(r["n_trades"])
        events = cfg.n_markets * cfg.n_agents * n
        dt = float(r["elapsed_s"])
        return {
            "backend": self.name,
            "n_markets": cfg.n_markets,
            "n_agents": cfg.n_agents,
            "n_levels": cfg.n_levels,
            "n_steps": n,
            "wall_time_s": dt,
            "events": events,
            "events_per_s": events / dt if dt > 0 else float("nan"),
            "steps_per_s": n / dt if dt > 0 else float("nan"),
            "gpu_mem_gb": float(r["gpu_mem_gb"]),
            "mean_last_price": float(self.last_price.mean()),
            "std_last_price": float(self.last_price.std()),
            "mean_volume_per_market": float(self.total_volume.mean()),
            "mean_trades_per_market": float(self.n_trades.mean()),
        }

class KineticSimCUDA(_CudaBase):
    name = "kineticsim_cuda"
    _module_name = "kineticsim_cuda"

class KineticSimNaiveCUDA(_CudaBase):
    name = "naive_cuda"
    _module_name = "kineticsim_naive"
