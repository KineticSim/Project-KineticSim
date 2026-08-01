import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kineticsim import SimConfig, get_backend

FUND_FRACS = np.round(np.arange(0.00, 0.31, 0.05), 2)
MOM_FRACS = [0.45, 0.50, 0.55]
MAKER = 0.15
SEEDS = [0, 1, 2, 3, 7, 42]
N_MARKETS = 1024
N_AGENTS = 256
N_STEPS = 1000

def vol_for(f_mom, f_fund, seed):
    cfg = SimConfig(
        n_markets=N_MARKETS, n_agents=N_AGENTS, n_levels=128, n_steps=N_STEPS,
        seed=seed, frac_noise=round(1.0 - f_mom - MAKER - f_fund, 6),
        frac_momentum=f_mom, frac_maker=MAKER, frac_fundamental=f_fund,
    )
    eng = get_backend("cuda")(cfg)
    r = eng.run(record_prices=True)
    return float(eng.price_history.astype(np.float64).std(axis=0).mean()), r["wall_time_s"]

def main():
    gpu_total = 0.0
    print(f"{'mom':>5} {'seed':>5} | " +
          " ".join(f"f={f:.2f}" for f in FUND_FRACS) +
          "  | peak@   amp    shape")
    summary = {m: {"peak_locs": [], "amps": [], "nonmono": 0} for m in MOM_FRACS}
    for f_mom in MOM_FRACS:
        for seed in SEEDS:
            vols = []
            for f_fund in FUND_FRACS:
                v, t = vol_for(f_mom, f_fund, seed)
                vols.append(v)
                gpu_total += t
            vols = np.array(vols)
            k = int(np.argmax(vols))
            amp = vols[k] / vols[0]
            interior = 0 < k < len(FUND_FRACS) - 1
            falls = vols[-1] < vols[k]
            nonmono = interior and falls and amp > 1.2
            summary[f_mom]["peak_locs"].append(FUND_FRACS[k])
            summary[f_mom]["amps"].append(amp)
            summary[f_mom]["nonmono"] += int(nonmono)
            print(f"{f_mom:5.2f} {seed:5d} | " +
                  " ".join(f"{v:6.1f}" for v in vols) +
                  f"  | {FUND_FRACS[k]:.2f} {amp:6.2f}x  "
                  f"{'non-monotonic' if nonmono else 'MONOTONIC?'}")

    print(f"\nGPU kernel time total: {gpu_total:.2f}s "
          f"({len(MOM_FRACS)*len(SEEDS)*len(FUND_FRACS)} cells, "
          f"{len(MOM_FRACS)*len(SEEDS)*len(FUND_FRACS)*N_MARKETS*N_AGENTS*N_STEPS:.2e} events)")
    for f_mom in MOM_FRACS:
        s = summary[f_mom]
        locs, amps = np.array(s["peak_locs"]), np.array(s["amps"])
        print(f"mom={f_mom:.2f}: non-monotonic in {s['nonmono']}/{len(SEEDS)} seeds; "
              f"peak at fund={locs.min():.2f}-{locs.max():.2f}; "
              f"amplification {amps.min():.2f}-{amps.max():.2f}x "
              f"(mean {amps.mean():.2f}x)")

if __name__ == "__main__":
    main()
