from __future__ import annotations

import time
from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp

from .model import SimConfig, NOISE, MOMENTUM, MAKER, FUNDAMENTAL

def call_auction_step(cfg_max_order_qty: float, cfg_noise_spread: int,
                      cfg_market_order_prob: float, cfg_maker_half_spread: int,
                      cfg_init_price: int,
                      agent_type: jnp.ndarray, arange_M: jnp.ndarray, arange_A: jnp.ndarray,
                      state: tuple, step_idx: int):
    bid, ask, last_price, prev_mid, total_volume, n_trades, key = state
    M, L = bid.shape
    A = agent_type.shape[1]

    has_b = (bid > 0).any(axis=1)
    has_a = (ask > 0).any(axis=1)

    best_bid = (L - 1) - jnp.argmax((bid > 0)[:, ::-1], axis=1)
    best_ask = jnp.argmax(ask > 0, axis=1)

    mid = jnp.where(has_b & has_a, 0.5 * (best_bid + best_ask), last_price)
    ret = jnp.sign(mid - prev_mid)

    key, key_side, key_mkt, key_qty, key_noise = jax.random.split(key, 5)

    u_side = jax.random.uniform(key_side, shape=(M, A))
    u_mkt = jax.random.uniform(key_mkt, shape=(M, A))
    qty = jax.random.randint(key_qty, shape=(M, A), minval=1, maxval=int(cfg_max_order_qty) + 1).astype(jnp.float32)
    noise_off = jax.random.randint(key_noise, shape=(M, A), minval=-cfg_noise_spread, maxval=cfg_noise_spread + 1)

    midA = mid[:, None]
    retA = ret[:, None]

    side = jnp.where(u_side < 0.5, 1, -1).astype(jnp.int32)
    price = jnp.round(midA + noise_off).astype(jnp.int32)

    is_mom = agent_type == MOMENTUM
    mom_side = jnp.where(retA != 0, retA.astype(jnp.int32), side)
    side = jnp.where(is_mom, mom_side, side)
    price = jnp.where(is_mom, jnp.round(midA + mom_side.astype(jnp.float32)).astype(jnp.int32), price)

    is_fund = agent_type == FUNDAMENTAL
    dev_f = float(cfg_init_price) - midA
    fund_side = jnp.where(dev_f > 0, 1, jnp.where(dev_f < 0, -1, side)).astype(jnp.int32)
    side = jnp.where(is_fund, fund_side, side)
    price = jnp.where(is_fund, jnp.round(midA + fund_side.astype(jnp.float32)).astype(jnp.int32), price)

    is_mm = agent_type == MAKER
    mm_buy = ((arange_A[None, :] + step_idx) % 2) == 0
    mm_side = jnp.where(mm_buy, 1, -1).astype(jnp.int32)
    mm_price = jnp.round(
        midA + jnp.where(mm_buy, -float(cfg_maker_half_spread), float(cfg_maker_half_spread))
    ).astype(jnp.int32)
    side = jnp.where(is_mm, mm_side, side)
    price = jnp.where(is_mm, mm_price, price)

    is_market = (u_mkt < cfg_market_order_prob) & (~is_mm)
    price = jnp.where(is_market & (side > 0), L - 1, price)
    price = jnp.where(is_market & (side < 0), 0, price)
    price = jnp.clip(price, 0, L - 1)

    buy_mask = (side > 0).astype(jnp.float32)
    sell_mask = (side <= 0).astype(jnp.float32)

    row_idx = arange_M[:, None]

    incoming_buy = jnp.zeros((M, L)).at[row_idx, price].add(qty * buy_mask)
    incoming_sell = jnp.zeros((M, L)).at[row_idx, price].add(qty * sell_mask)

    BUY = bid + incoming_buy
    SELL = ask + incoming_sell

    Dcum = jnp.flip(jnp.cumsum(jnp.flip(BUY, axis=1), axis=1), axis=1)
    Scum = jnp.cumsum(SELL, axis=1)
    match = jnp.minimum(Dcum, Scum)
    pstar = jnp.argmax(match, axis=1)
    volume = match[arange_M, pstar]
    V = volume[:, None]

    traded_buy = jnp.minimum(jnp.maximum(V - (Dcum - BUY), 0.0), BUY)
    traded_sell = jnp.minimum(jnp.maximum(V - (Scum - SELL), 0.0), SELL)

    new_bid = BUY - traded_buy
    new_ask = SELL - traded_sell

    traded = volume > 0
    new_last_price = jnp.where(traded, pstar.astype(jnp.float32), last_price)
    new_total_volume = total_volume + volume
    new_n_trades = n_trades + traded.astype(jnp.int32)

    new_state = (new_bid, new_ask, new_last_price, mid, new_total_volume, new_n_trades, key)
    return new_state, new_last_price

_JAX_COMPILE_CACHE = {}

def make_run_sim(M, A, L, n, max_order_qty, noise_spread, market_order_prob, maker_half_spread, init_price):
    @jax.jit
    def run_sim(bid, ask, last_price, prev_mid, total_volume, n_trades, key, agent_type, arange_M, arange_A):
        init_state = (bid, ask, last_price, prev_mid, total_volume, n_trades, key)

        def f(state, step_idx):
            next_state, px = call_auction_step(
                max_order_qty, noise_spread, market_order_prob, maker_half_spread,
                init_price, agent_type, arange_M, arange_A, state, step_idx
            )
            return next_state, px

        final_state, _ = jax.lax.scan(f, init_state, jnp.arange(n))
        return final_state
    return run_sim

class KineticSimJax:
    name = "jax_gpu"

    def __init__(self, cfg: SimConfig):
        cfg.validate()
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels

        bid_np = np.zeros((M, L), dtype=np.float32)
        ask_np = np.zeros((M, L), dtype=np.float32)
        bid_np[:, cfg.init_price - 1] = cfg.init_depth
        ask_np[:, cfg.init_price + 1] = cfg.init_depth

        self.bid = jnp.array(bid_np)
        self.ask = jnp.array(ask_np)
        self.last_price = jnp.full((M,), float(cfg.init_price), dtype=jnp.float32)
        self.prev_mid = jnp.full((M,), float(cfg.init_price), dtype=jnp.float32)

        rng = np.random.default_rng(cfg.seed)
        probs = [cfg.frac_noise, cfg.frac_momentum, cfg.frac_maker,
                 cfg.frac_fundamental]
        self.agent_type = jnp.array(
            rng.choice([NOISE, MOMENTUM, MAKER, FUNDAMENTAL],
                       size=(M, A), p=probs).astype(np.int32)
        )

        self.total_volume = jnp.zeros(M, dtype=jnp.float32)
        self.n_trades = jnp.zeros(M, dtype=jnp.int32)

        self.key = jax.random.PRNGKey(cfg.seed)

        self.arange_M = jnp.arange(M)
        self.arange_A = jnp.arange(A)

    def run(self, n_steps: int | None = None) -> Dict:
        cfg = self.cfg
        n = n_steps if n_steps is not None else cfg.n_steps
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels

        cache_key = (M, A, L, n, cfg.max_order_qty, cfg.noise_spread, cfg.market_order_prob, cfg.maker_half_spread, cfg.init_price)

        if cache_key not in _JAX_COMPILE_CACHE:
            run_sim = make_run_sim(
                M, A, L, n,
                cfg.max_order_qty, cfg.noise_spread, cfg.market_order_prob, cfg.maker_half_spread, cfg.init_price
            )
            _ = run_sim(
                self.bid, self.ask, self.last_price, self.prev_mid,
                self.total_volume, self.n_trades, self.key,
                self.agent_type, self.arange_M, self.arange_A
            )
            _[0].block_until_ready()
            _JAX_COMPILE_CACHE[cache_key] = run_sim

        run_sim = _JAX_COMPILE_CACHE[cache_key]

        t0 = time.perf_counter()
        final_state = run_sim(
            self.bid, self.ask, self.last_price, self.prev_mid,
            self.total_volume, self.n_trades, self.key,
            self.agent_type, self.arange_M, self.arange_A
        )
        final_state[0].block_until_ready()
        dt = time.perf_counter() - t0

        bid, ask, last_price, prev_mid, total_volume, n_trades, key = final_state
        self.bid = bid
        self.ask = ask
        self.last_price = last_price
        self.prev_mid = prev_mid
        self.total_volume = total_volume
        self.n_trades = n_trades
        self.key = key

        events = cfg.n_markets * cfg.n_agents * n
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
            "gpu_mem_gb": 0.0,
            "mean_last_price": float(self.last_price.mean().item()),
            "std_last_price": float(self.last_price.std().item()),
            "mean_volume_per_market": float(self.total_volume.mean().item()),
            "mean_trades_per_market": float(self.n_trades.mean().item()),
        }
