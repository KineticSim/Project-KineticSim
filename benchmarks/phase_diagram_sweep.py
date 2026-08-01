import os
import sys
import time

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kineticsim import SimConfig, get_backend

MOM_FRACS = np.round(np.arange(0.00, 0.71, 0.05), 2)
MAKER_FRACS = np.round(np.arange(0.05, 0.31, 0.05), 2)
FUND_FRACS = np.round(np.arange(0.00, 0.31, 0.05), 2)
FIXED_MAKER_B = 0.15
N_MARKETS = 1024
N_AGENTS = 256
N_STEPS = 1000
SEED = 42
ACF_LAGS = np.arange(1, 21)
STD_MOM, STD_MAKER = 0.15, 0.15

def run_cell(f_mom: float, f_maker: float, f_fund: float):
    f_noise = round(1.0 - f_mom - f_maker - f_fund, 6)
    cfg = SimConfig(
        n_markets=N_MARKETS, n_agents=N_AGENTS, n_levels=128, n_steps=N_STEPS,
        seed=SEED, frac_noise=f_noise, frac_momentum=f_mom,
        frac_maker=f_maker, frac_fundamental=f_fund,
    )
    eng = get_backend("cuda")(cfg)
    res = eng.run(record_prices=True)
    prices = eng.price_history.astype(np.float64)
    returns = np.diff(prices, axis=0)

    vol = prices.std(axis=0).mean()
    kurts = np.zeros(N_MARKETS)
    live = returns.std(axis=0) > 0
    kurts[live] = stats.kurtosis(returns[:, live], axis=0, fisher=True)
    volume = float(np.mean(eng.total_volume) / N_STEPS)
    return {
        "frac_mom": f_mom, "frac_maker": f_maker, "frac_fund": f_fund,
        "volatility": float(vol), "kurtosis": float(kurts.mean()),
        "volume_per_step": volume, "gpu_s": float(res["wall_time_s"]),
        "returns": returns,
    }

def acf_panel_data(returns: np.ndarray):
    acf_raw = np.zeros(len(ACF_LAGS))
    acf_abs = np.zeros(len(ACF_LAGS))
    live = np.where(returns.std(axis=0) > 0)[0]
    for m in live:
        r = returns[:, m]
        for i, lag in enumerate(ACF_LAGS):
            acf_raw[i] += np.corrcoef(r[:-lag], r[lag:])[0, 1]
            acf_abs[i] += np.corrcoef(np.abs(r[:-lag]), np.abs(r[lag:]))[0, 1]
    return acf_raw / len(live), acf_abs / len(live)

def main():
    t0 = time.time()
    rows, gpu_total = [], 0.0
    grid_vol_A = np.full((len(MAKER_FRACS), len(MOM_FRACS)), np.nan)
    grid_kurt_A = np.full_like(grid_vol_A, np.nan)
    grid_vol_B = np.full((len(FUND_FRACS), len(MOM_FRACS)), np.nan)
    grid_kurt_B = np.full_like(grid_vol_B, np.nan)
    std_returns = None

    for i, f_maker in enumerate(MAKER_FRACS):
        for j, f_mom in enumerate(MOM_FRACS):
            cell = run_cell(f_mom, f_maker, 0.0)
            gpu_total += cell["gpu_s"]
            grid_vol_A[i, j] = cell["volatility"]
            grid_kurt_A[i, j] = cell["kurtosis"]
            if abs(f_maker - STD_MAKER) < 1e-9 and abs(f_mom - STD_MOM) < 1e-9:
                std_returns = cell["returns"]
            rows.append({k: v for k, v in cell.items() if k != "returns"})
            print(f"A maker={f_maker:.2f} mom={f_mom:.2f} | "
                  f"vol={cell['volatility']:8.3f} kurt={cell['kurtosis']:8.3f} "
                  f"volume={cell['volume_per_step']:7.1f}")

    n_skipped = 0
    for i, f_fund in enumerate(FUND_FRACS):
        for j, f_mom in enumerate(MOM_FRACS):
            if f_mom + f_fund + FIXED_MAKER_B > 1.0 + 1e-9:
                n_skipped += 1
                continue
            cell = run_cell(f_mom, FIXED_MAKER_B, f_fund)
            gpu_total += cell["gpu_s"]
            grid_vol_B[i, j] = cell["volatility"]
            grid_kurt_B[i, j] = cell["kurtosis"]
            rows.append({k: v for k, v in cell.items() if k != "returns"})
            print(f"B fund={f_fund:.2f} mom={f_mom:.2f} | "
                  f"vol={cell['volatility']:8.3f} kurt={cell['kurtosis']:8.3f} "
                  f"volume={cell['volume_per_step']:7.1f}")

    acf_raw, acf_abs = acf_panel_data(std_returns)
    n_cells = len(rows)
    events = n_cells * N_MARKETS * N_AGENTS * N_STEPS
    print(f"\n{n_cells} cells ({n_skipped} infeasible skipped), "
          f"{events:.3e} agent-events, GPU kernel time {gpu_total:.2f}s, "
          f"wall {time.time()-t0:.1f}s")
    print(f"ACF lag-1: r_t {acf_raw[0]:+.3f}, |r_t| {acf_abs[0]:+.3f}")

    import csv
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/phase_diagram.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    m_res_dir = os.path.join(os.path.dirname(__file__), "..", "manuscript", "result")
    os.makedirs(m_res_dir, exist_ok=True)
    extent_A = [MOM_FRACS[0], MOM_FRACS[-1], MAKER_FRACS[0], MAKER_FRACS[-1]]
    extent_B = [MOM_FRACS[0], MOM_FRACS[-1], FUND_FRACS[0], FUND_FRACS[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    cmap_v = plt.get_cmap("viridis").copy(); cmap_v.set_bad("lightgray")
    cmap_k = plt.get_cmap("magma").copy();   cmap_k.set_bad("lightgray")

    im0 = axes[0, 0].imshow(grid_vol_A, origin="lower", aspect="auto",
                            extent=extent_A, cmap=cmap_v)
    axes[0, 0].set_title("Volatility: Momentum $\\times$ Maker", fontsize=12,
                         fontweight="bold")
    axes[0, 0].set_xlabel("Momentum Agent Fraction $\\alpha_{mom}$", fontsize=10)
    axes[0, 0].set_ylabel("Market Maker Fraction $\\alpha_{maker}$", fontsize=10)
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(grid_kurt_A, origin="lower", aspect="auto",
                            extent=extent_A, cmap=cmap_k)
    axes[0, 1].set_title("Excess Kurtosis: Momentum $\\times$ Maker", fontsize=12,
                         fontweight="bold")
    axes[0, 1].set_xlabel("Momentum Agent Fraction $\\alpha_{mom}$", fontsize=10)
    axes[0, 1].set_ylabel("Market Maker Fraction $\\alpha_{maker}$", fontsize=10)
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(np.ma.masked_invalid(grid_vol_B), origin="lower",
                            aspect="auto", extent=extent_B, cmap=cmap_v)
    axes[1, 0].set_title("Volatility: Momentum $\\times$ Fundamentalist "
                         "($\\alpha_{maker}{=}0.15$)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Momentum Agent Fraction $\\alpha_{mom}$", fontsize=10)
    axes[1, 0].set_ylabel("Fundamentalist Fraction $\\alpha_{fund}$", fontsize=10)
    fig.colorbar(im2, ax=axes[1, 0])

    ax = axes[1, 1]
    ax.bar(ACF_LAGS - 0.2, acf_raw, width=0.4, color="#aec7e8", label="Returns $r_t$")
    ax.bar(ACF_LAGS + 0.2, acf_abs, width=0.4, color="#ffbb78", label="Abs returns $|r_t|$")
    ax.plot(ACF_LAGS, acf_abs, color="#ff7f0e", linestyle="-", alpha=0.8)
    ax.set_title("Return ACF ($\\alpha_{mom}{=}\\alpha_{maker}{=}0.15$)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (simulation steps)", fontsize=10)
    ax.set_ylabel("Autocorrelation Coefficient", fontsize=10)
    ax.set_xticks(np.arange(1, 21, 2))
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    for name in ("fig_market_sweep.pdf", "fig_market_sweep.png"):
        path = os.path.join(m_res_dir, name)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print("saved", path)
    plt.close()

if __name__ == "__main__":
    main()
