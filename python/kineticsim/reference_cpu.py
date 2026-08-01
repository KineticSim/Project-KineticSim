from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .model import SimConfig, NOISE, MOMENTUM, MAKER, FUNDAMENTAL

def _best_bid_ask(bid: np.ndarray, ask: np.ndarray):
    L = bid.shape[1]
    has_bid = (bid > 0).any(axis=1)
    has_ask = (ask > 0).any(axis=1)
    best_bid = (L - 1) - np.argmax((bid > 0)[:, ::-1], axis=1)
    best_ask = np.argmax(ask > 0, axis=1)
    return best_bid, best_ask, has_bid, has_ask

def call_auction(BUY: np.ndarray, SELL: np.ndarray):
    M, L = BUY.shape
    p = np.arange(L)[None, :]

    Dcum = np.flip(np.cumsum(np.flip(BUY, axis=1), axis=1), axis=1)
    Scum = np.cumsum(SELL, axis=1)

    match = np.minimum(Dcum, Scum)
    pstar = np.argmax(match, axis=1)
    volume = match[np.arange(M), pstar]
    V = volume[:, None]

    demand_above = Dcum - BUY
    traded_buy = np.clip(V - demand_above, 0.0, BUY)
    supply_below = Scum - SELL
    traded_sell = np.clip(V - supply_below, 0.0, SELL)

    new_bid = BUY - traded_buy
    new_ask = SELL - traded_sell
    return new_bid, new_ask, pstar, volume

class KineticSimCPU:

    name = "cpu_numpy"

    def __init__(self, cfg: SimConfig):
        cfg.validate()
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels
        self.rng = np.random.default_rng(cfg.seed)

        self.bid = np.zeros((M, L), dtype=np.float32)
        self.ask = np.zeros((M, L), dtype=np.float32)
        self.bid[:, cfg.init_price - 1] = cfg.init_depth
        self.ask[:, cfg.init_price + 1] = cfg.init_depth

        self.last_price = np.full(M, float(cfg.init_price), dtype=np.float32)
        self.prev_mid = np.full(M, float(cfg.init_price), dtype=np.float32)

        probs = [cfg.frac_noise, cfg.frac_momentum, cfg.frac_maker,
                 cfg.frac_fundamental]
        self.agent_type = self.rng.choice(
            [NOISE, MOMENTUM, MAKER, FUNDAMENTAL], size=(M, A), p=probs
        ).astype(np.int32)

        self.step_idx = 0
        self.total_volume = np.zeros(M, dtype=np.float64)
        self.n_trades = np.zeros(M, dtype=np.int64)

    def step(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels
        bid, ask = self.bid, self.ask

        bb, ba, has_b, has_a = _best_bid_ask(bid, ask)
        mid = np.where(has_b & has_a, 0.5 * (bb + ba), self.last_price)
        ret = np.sign(mid - self.prev_mid)
        self.prev_mid = mid.copy()

        midA = mid[:, None]
        retA = ret[:, None]
        typ = self.agent_type

        u_side = self.rng.random((M, A))
        u_mkt = self.rng.random((M, A))
        qty = self.rng.integers(1, cfg.max_order_qty + 1, size=(M, A)).astype(np.float32)
        noise_off = self.rng.integers(-cfg.noise_spread, cfg.noise_spread + 1, size=(M, A))

        side = np.where(u_side < 0.5, 1, -1).astype(np.int32)
        price = np.rint(midA + noise_off).astype(np.int32)

        mom_side = np.where(retA != 0, retA, side).astype(np.int32)
        is_mom = typ == MOMENTUM
        side = np.where(is_mom, mom_side, side)
        price = np.where(is_mom, np.rint(midA + mom_side).astype(np.int32), price)

        dev = float(cfg.init_price) - midA
        fund_side = np.where(dev > 0, 1, np.where(dev < 0, -1, side)).astype(np.int32)
        is_fund = typ == FUNDAMENTAL
        side = np.where(is_fund, fund_side, side)
        price = np.where(is_fund, np.rint(midA + fund_side).astype(np.int32), price)

        is_mm = typ == MAKER
        agent_ids = np.arange(A)[None, :]
        mm_buy = ((agent_ids + self.step_idx) % 2) == 0
        mm_side = np.where(mm_buy, 1, -1).astype(np.int32)
        mm_price = np.rint(midA + np.where(mm_buy, -cfg.maker_half_spread,
                                           cfg.maker_half_spread)).astype(np.int32)
        side = np.where(is_mm, mm_side, side)
        price = np.where(is_mm, mm_price, price)

        is_market = (u_mkt < cfg.market_order_prob) & (~is_mm)
        price = np.where(is_market & (side > 0), L - 1, price)
        price = np.where(is_market & (side < 0), 0, price)

        price = np.clip(price, 0, L - 1)

        incoming_buy = np.zeros((M, L), dtype=np.float32)
        incoming_sell = np.zeros((M, L), dtype=np.float32)
        rows = np.repeat(np.arange(M), A)
        flat_price = price.reshape(-1)
        flat_qty = qty.reshape(-1)
        flat_side = side.reshape(-1)
        bmask = flat_side > 0
        smask = ~bmask
        np.add.at(incoming_buy, (rows[bmask], flat_price[bmask]), flat_qty[bmask])
        np.add.at(incoming_sell, (rows[smask], flat_price[smask]), flat_qty[smask])

        BUY = bid + incoming_buy
        SELL = ask + incoming_sell

        new_bid, new_ask, pstar, volume = call_auction(BUY, SELL)

        self.bid, self.ask = new_bid, new_ask
        traded = volume > 0
        self.last_price = np.where(traded, pstar.astype(np.float32), self.last_price)
        self.total_volume += volume
        self.n_trades += traded.astype(np.int64)
        self.step_idx += 1

    def run(self, n_steps: int | None = None) -> Dict:
        n = n_steps if n_steps is not None else self.cfg.n_steps
        t0 = time.perf_counter()
        for _ in range(n):
            self.step()
        dt = time.perf_counter() - t0
        return self.stats(n, dt)

    def stats(self, n_steps: int, wall_time: float) -> Dict:
        cfg = self.cfg
        events = cfg.n_markets * cfg.n_agents * n_steps
        return {
            "backend": self.name,
            "n_markets": cfg.n_markets,
            "n_agents": cfg.n_agents,
            "n_levels": cfg.n_levels,
            "n_steps": n_steps,
            "wall_time_s": wall_time,
            "events": events,
            "events_per_s": events / wall_time if wall_time > 0 else float("nan"),
            "steps_per_s": n_steps / wall_time if wall_time > 0 else float("nan"),
            "mean_last_price": float(self.last_price.mean()),
            "std_last_price": float(self.last_price.std()),
            "mean_volume_per_market": float(self.total_volume.mean()),
            "mean_trades_per_market": float(self.n_trades.mean()),
        }
