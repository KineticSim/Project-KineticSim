import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import numpy as np
import scipy.stats as stats

from kineticsim import SimConfig, get_backend

M, A, S = 4096, 256, 500
SEEDS = [0, 1, 2]

def main():
    print(f"KS distributional test at M={M}, A={A}, S={S}, seeds={SEEDS}")
    worst = {"price": (0.0, 1.0), "volume": (0.0, 1.0)}
    for seed in SEEDS:
        cfg = SimConfig(n_markets=M, n_agents=A, n_levels=128, n_steps=S, seed=seed)
        cpu = get_backend("cpu")(cfg); cpu.run()
        gpu = get_backend("cuda")(cfg); gpu.run()

        ks_px = stats.ks_2samp(cpu.last_price, gpu.last_price)
        ks_vol = stats.ks_2samp(cpu.total_volume, gpu.total_volume)
        rel_px = abs(gpu.last_price.mean() - cpu.last_price.mean()) / cpu.last_price.mean()
        rel_vol = abs(gpu.total_volume.mean() - cpu.total_volume.mean()) / cpu.total_volume.mean()

        print(f"seed {seed}: price  D={ks_px.statistic:.4f} p={ks_px.pvalue:.3f} "
              f"(mean rel err {rel_px*100:.3f}%)")
        print(f"        volume D={ks_vol.statistic:.4f} p={ks_vol.pvalue:.3f} "
              f"(mean rel err {rel_vol*100:.3f}%)")

        if ks_px.statistic > worst["price"][0]:
            worst["price"] = (ks_px.statistic, ks_px.pvalue)
        if ks_vol.statistic > worst["volume"][0]:
            worst["volume"] = (ks_vol.statistic, ks_vol.pvalue)

    print(f"\nworst case over seeds: price D={worst['price'][0]:.4f} "
          f"(p={worst['price'][1]:.3f}), volume D={worst['volume'][0]:.4f} "
          f"(p={worst['volume'][1]:.3f})")

if __name__ == "__main__":
    main()
