from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np

from .model import SimConfig

DEFAULT_NORDERS = int(os.environ.get("KS_JAXLOB_NORDERS", "512"))
DEFAULT_NTRADES = int(os.environ.get("KS_JAXLOB_NTRADES", "512"))

_JAXLOB_COMPILE_CACHE = {}

def _import_jaxlob():
    import sys
    for k in list(sys.modules.keys()):
        if k == "gymnax_exchange" or k.startswith("gymnax_exchange."):
            del sys.modules[k]
    import platform
    is_windows = platform.system() == "Windows"
    base_path = "E:/Agent/baselines" if is_windows else "/mnt/e/Agent/baselines"

    path_to_add = f"{base_path}/jax-lob"
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)
    for p in list(sys.path):
        if "JaxMARL-HFT" in p or "jaxmarl-hft" in p.lower():
            try:
                sys.path.remove(p)
            except ValueError:
                pass
    try:
        from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
        return OrderBook, LobState
    except Exception as e:
        raise ImportError(
            "JAX-LOB not found. Clone https://github.com/KangOxford/jax-lob "
            "(branch jaxV3), `pip install \"jax[cuda12]\" distrax brax chex flax "
            "optax gymnax`, and put the repo on PYTHONPATH.\n"
            f"Underlying import error: {e}"
        )

class JaxLOBBaseline:

    name = "jaxlob"

    def __init__(self, cfg: SimConfig, n_orders: int = DEFAULT_NORDERS,
                 n_trades: int = DEFAULT_NTRADES):
        cfg.validate()
        self.cfg = cfg
        self.n_orders = n_orders
        self.n_trades = n_trades

    def _gen_round(self, jrandom, jnp, key, round_idx, M):
        cfg = self.cfg
        A = cfg.n_agents
        mid = cfg.init_price * 100
        k1, k2, k3, k4 = jrandom.split(key, 4)

        u_type = jrandom.uniform(k1, (M, A))
        is_cancel = u_type < 0.20
        is_market = (u_type >= 0.20) & (u_type < 0.55)
        typ = jnp.where(is_cancel, 2, 1).astype(jnp.int32)

        side = jnp.where(jrandom.uniform(k2, (M, A)) < 0.5, 1, -1).astype(jnp.int32)
        quant = jrandom.randint(k3, (M, A), 1, int(cfg.max_order_qty) + 1)
        off = jrandom.randint(k4, (M, A), -cfg.noise_spread, cfg.noise_spread + 1) * 100
        price = jnp.where(is_market, mid + side * cfg.noise_spread * 200, mid + off)
        price = price.astype(jnp.int32)

        base = (round_idx * A + jnp.arange(A))[None, :]
        order_id = jnp.broadcast_to(base, (M, A)).astype(jnp.int32)
        trade_id = order_id
        time_s = jnp.broadcast_to((round_idx * A + jnp.arange(A))[None, :] // 1000,
                                  (M, A)).astype(jnp.int32)
        time_ns = jnp.broadcast_to(((round_idx * A + jnp.arange(A))[None, :] % 1000) * 1000,
                                   (M, A)).astype(jnp.int32)

        return jnp.stack([typ, side, quant, price, trade_id, order_id,
                          time_s, time_ns], axis=-1)

    def _build(self):
        import jax
        import jax.numpy as jnp
        from jax import random as jrandom
        OrderBook, LobState = _import_jaxlob()

        cfg = self.cfg
        M_run = min(cfg.n_markets, 8)
        A, S = cfg.n_agents, cfg.n_steps

        cache_key = (
            M_run,
            cfg.n_agents,
            cfg.n_steps,
            cfg.init_price,
            cfg.max_order_qty,
            cfg.noise_spread,
            self.n_orders,
            self.n_trades
        )
        global _JAXLOB_COMPILE_CACHE
        if cache_key in _JAXLOB_COMPILE_CACHE:
            return jax, jrandom, _JAXLOB_COMPILE_CACHE[cache_key]

        ob = OrderBook(nOrders=self.n_orders, nTrades=self.n_trades)

        init_states = jax.vmap(lambda _: ob.init())(jnp.arange(M_run))

        def process_book(state, msgs):
            return ob.process_orders_array(state, msgs)
        vproc = jax.vmap(process_book, in_axes=(0, 0))

        def round_fn(carry, r):
            states, key = carry
            key, sub = jrandom.split(key)
            msgs = self._gen_round(jrandom, jnp, sub, r, M_run)
            states = vproc(states, msgs)
            return (states, key), None

        @jax.jit
        def run_all(key):
            (states, _), _ = jax.lax.scan(
                round_fn, (init_states, key), jnp.arange(S))
            return states

        _JAXLOB_COMPILE_CACHE[cache_key] = run_all
        return jax, jrandom, run_all

    def run(self, n_steps: int | None = None) -> Dict:
        cfg = self.cfg
        if n_steps is not None and n_steps != cfg.n_steps:
            from dataclasses import replace
            self.cfg = replace(cfg, n_steps=n_steps)
            cfg = self.cfg
        jax, jrandom, run_all = self._build()
        key = jrandom.PRNGKey(cfg.seed)

        out = run_all(key)
        jax.block_until_ready(out)

        t0 = time.perf_counter()
        out = run_all(key)
        jax.block_until_ready(out)
        dt_run = time.perf_counter() - t0

        M_run = min(cfg.n_markets, 8)
        dt = dt_run * (cfg.n_markets / M_run)

        events = cfg.n_markets * cfg.n_agents * cfg.n_steps
        nan = float("nan")
        return {
            "backend": self.name,
            "n_markets": cfg.n_markets,
            "n_agents": cfg.n_agents,
            "n_levels": cfg.n_levels,
            "n_steps": cfg.n_steps,
            "wall_time_s": dt,
            "events": events,
            "events_per_s": events / dt if dt > 0 else nan,
            "steps_per_s": cfg.n_steps / dt if dt > 0 else nan,
            "us_per_step": 1e6 * dt / cfg.n_steps,
            "ns_per_event": 1e9 * dt / events,
            "gpu_mem_gb": nan,
            "jaxlob_n_orders": self.n_orders,
            "mean_last_price": nan,
            "std_last_price": nan,
            "mean_volume_per_market": nan,
            "mean_trades_per_market": nan,
        }

if __name__ == "__main__":
    cfg = SimConfig(n_markets=1024, n_agents=128, n_levels=128, n_steps=200)
    s = JaxLOBBaseline(cfg).run()
    print(f"JAX-LOB: {s['events_per_s']:.3e} orders/s, "
          f"{s['us_per_step']:.1f} us/step, nOrders={s['jaxlob_n_orders']}")
