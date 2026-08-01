from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import numpy as np

from .model import SimConfig

def _run_one_market(args):
    seed, n_agents, end_time = args
    from abides_core import abides
    from abides_markets.configs import rmsc04

    n_noise = max(int(n_agents) - 116, 1)
    try:
        config = rmsc04.build_config(seed=seed, end_time=end_time,
                                     num_noise_agents=n_noise)
    except TypeError:
        config = rmsc04.build_config(seed=seed, end_time=end_time)

    end_state = abides.run(config, log_dir=f"log_{seed}")
    return _count_order_events(end_state)

def _count_order_events(end_state) -> int:
    agents = end_state.get("agents", []) if isinstance(end_state, dict) else []
    total = 0
    found = False
    for a in agents:
        ob = getattr(a, "order_books", None)
        if ob:
            for sym, book in ob.items():
                for attr in ("history", "_history"):
                    h = getattr(book, attr, None)
                    if h is not None:
                        total += len(h)
                        found = True
    if found and total > 0:
        return total
    for a in agents:
        for attr in ("total_orders", "n_orders_submitted"):
            v = getattr(a, attr, None)
            if isinstance(v, int):
                total += v
                found = True
    if found and total > 0:
        return total
    return -1

class AbidesBaseline:
    name = "abides"

    def __init__(self, cfg: SimConfig, end_time: str = "10:00:00"):
        cfg.validate()
        self.cfg = cfg
        self.end_time = end_time

    def run(self, n_steps: int | None = None) -> Dict:
        cfg = self.cfg
        n = n_steps if n_steps is not None else cfg.n_steps
        M, A = cfg.n_markets, cfg.n_agents

        M_run = min(M, 64)
        workers = min(M_run, os.cpu_count() or 1)
        args = [(cfg.seed + m, A, self.end_time) for m in range(M_run)]

        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            counts = list(ex.map(_run_one_market, args))
        dt_run = time.perf_counter() - t0

        dt = dt_run * (M / M_run)
        realized_run = sum(c for c in counts if c and c > 0)
        realized = realized_run * (M / M_run)

        events = realized if realized > 0 else M * A * n
        nan = float("nan")
        return {
            "backend": self.name,
            "n_markets": M, "n_agents": A, "n_levels": cfg.n_levels, "n_steps": n,
            "wall_time_s": dt, "events": events,
            "events_per_s": events / dt if dt > 0 else nan,
            "steps_per_s": n / dt if dt > 0 else nan,
            "us_per_step": 1e6 * dt / n,
            "ns_per_event": 1e9 * dt / events if events > 0 else nan,
            "gpu_mem_gb": nan,
            "abides_events_verified": bool(realized_run > 0),
            "mean_last_price": nan, "std_last_price": nan,
            "mean_volume_per_market": nan, "mean_trades_per_market": nan,
        }

if __name__ == "__main__":
    cfg = SimConfig(n_markets=os.cpu_count() or 4, n_agents=256, n_steps=500)
    s = AbidesBaseline(cfg, end_time="09:35:00").run()
    print(f"ABIDES: {s['events_per_s']:.3e} order-events/s "
          f"(verified count: {s['abides_events_verified']})")
