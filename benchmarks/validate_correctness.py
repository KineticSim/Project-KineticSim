#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import numpy as np

from kineticsim.model import SimConfig
from kineticsim.reference_cpu import call_auction, KineticSimCPU

def check_clearing_engine(n: int = 5000) -> None:
    rng = np.random.default_rng(123)
    for _ in range(n):
        M, L = rng.integers(1, 6), 16
        B = rng.integers(0, 12, (M, L)).astype(np.float32)
        S = rng.integers(0, 12, (M, L)).astype(np.float32)
        nb, na, pstar, vol = call_auction(B, S)
        assert (nb >= -1e-3).all() and (na >= -1e-3).all(), "negative residual"
        tb, ts = (B - nb).sum(1), (S - na).sum(1)
        assert np.allclose(tb, ts, atol=1e-2), "buy/sell volume mismatch"
        assert np.allclose(tb, vol, atol=1e-2), "reported volume mismatch"
    print(f"[1] clearing engine: PASS ({n} random books, conservation + nonneg)")

def _stats(res_dict):
    return np.array([res_dict["mean_last_price"], res_dict["std_last_price"],
                     res_dict["mean_volume_per_market"]])

def check_cuda_vs_naive() -> bool:
    try:
        import kineticsim_cuda, kineticsim_naive
        from kineticsim.cuda_backend import KineticSimCUDA, KineticSimNaiveCUDA, agent_types_for
    except Exception as e:
        print(f"[2] CUDA==naive: SKIPPED (modules not built: {e})")
        return False

    cfg = SimConfig(n_markets=512, n_agents=128, n_levels=128, n_steps=300, seed=7)
    a = KineticSimCUDA(cfg); a.run()
    b = KineticSimNaiveCUDA(cfg); b.run()
    ok = (np.array_equal(a.last_price, b.last_price)
          and np.allclose(a.total_volume, b.total_volume, rtol=0, atol=1e-3)
          and np.array_equal(a.n_trades, b.n_trades))
    print(f"[2] CUDA==naive (exact): {'PASS' if ok else 'FAIL'}")
    if not ok:
        d = np.abs(a.last_price - b.last_price)
        print(f"    max |Δ last_price| = {d.max()}, "
              f"max |Δ vol| = {np.abs(a.total_volume-b.total_volume).max()}")
    return ok

def check_cuda_vs_cpu() -> bool:
    try:
        import kineticsim_cuda
        from kineticsim.cuda_backend import KineticSimCUDA
    except Exception as e:
        print(f"[3] CUDA~=CPU: SKIPPED (optimized module not built: {e})")
        return False

    cfg = SimConfig(n_markets=2048, n_agents=128, n_levels=128, n_steps=400, seed=11)
    g = KineticSimCUDA(cfg); gs = g.run()
    c = KineticSimCPU(cfg); cs = c.run()
    gv, cv = _stats(gs), _stats(cs)
    rel = np.abs(gv - cv) / (np.abs(cv) + 1e-6)
    ok = (rel < 0.10).all()
    print(f"[3] CUDA~=CPU (statistical): {'PASS' if ok else 'FAIL'}")
    print(f"    {'stat':22s} {'cuda':>12s} {'cpu':>12s} {'rel.err':>9s}")
    for nm, a, b, r in zip(["mean_last_price", "std_last_price",
                            "mean_volume/mkt"], gv, cv, rel):
        print(f"    {nm:22s} {a:12.3f} {b:12.3f} {r:9.3%}")
    return ok

if __name__ == "__main__":
    check_clearing_engine()
    ok2 = check_cuda_vs_naive()
    ok3 = check_cuda_vs_cpu()
    if not (ok2 or ok3):
        print("\n(GPU checks skipped — build the CUDA modules with ./build.sh "
              "to run them on the RTX 5090.)")
