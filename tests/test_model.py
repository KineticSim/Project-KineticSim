import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from kineticsim.model import SimConfig
from kineticsim.reference_cpu import call_auction, KineticSimCPU, _best_bid_ask

def test_clearing_conservation():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        B = rng.integers(0, 12, (4, 16)).astype(np.float32)
        S = rng.integers(0, 12, (4, 16)).astype(np.float32)
        nb, na, pstar, vol = call_auction(B, S)
        assert (nb >= -1e-3).all() and (na >= -1e-3).all()
        np.testing.assert_allclose((B - nb).sum(1), vol, atol=1e-2)
        np.testing.assert_allclose((S - na).sum(1), vol, atol=1e-2)

def test_known_book():
    B = np.zeros((1, 8), np.float32); S = np.zeros((1, 8), np.float32)
    B[0, 4] = 5; B[0, 2] = 3; S[0, 3] = 4; S[0, 5] = 6
    nb, na, pstar, vol = call_auction(B, S)
    assert vol[0] == 4.0
    assert abs((B - nb).sum() - 4.0) < 1e-4
    assert abs((S - na).sum() - 4.0) < 1e-4

def test_no_persistent_cross():
    cfg = SimConfig(n_markets=128, n_agents=64, n_steps=200, seed=3)
    sim = KineticSimCPU(cfg); sim.run()
    bb, ba, hb, ha = _best_bid_ask(sim.bid, sim.ask)
    assert int(((bb > ba) & hb & ha).sum()) == 0
    assert (sim.bid >= 0).all() and (sim.ask >= 0).all()

def test_determinism():
    cfg = SimConfig(n_markets=64, n_agents=32, n_steps=120, seed=42)
    a = KineticSimCPU(cfg); a.run()
    b = KineticSimCPU(cfg); b.run()
    np.testing.assert_array_equal(a.last_price, b.last_price)
    np.testing.assert_allclose(a.total_volume, b.total_volume)

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all tests passed")
