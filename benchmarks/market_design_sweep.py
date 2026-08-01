import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from kineticsim import SimConfig, KineticSimCPU

def main():
    print("Running market composition parameter sweep on CPU reference backend...")

    mom_fractions = np.arange(0.0, 0.71, 0.05)
    n_markets_per_config = 64
    n_steps = 1000

    volatilities = []
    kurtoses = []
    volumes = []

    target_f_mom = 0.15
    acf_lags = np.arange(1, 21)
    mean_acf_raw = np.zeros(len(acf_lags))
    mean_acf_abs = np.zeros(len(acf_lags))

    for f_mom in mom_fractions:
        f_maker = 0.15
        f_noise = 1.0 - f_maker - f_mom

        cfg = SimConfig(
            n_markets=n_markets_per_config,
            n_agents=256,
            n_levels=128,
            n_steps=n_steps,
            seed=42,
            frac_noise=f_noise,
            frac_momentum=f_mom,
            frac_maker=f_maker
        )

        sim = KineticSimCPU(cfg)
        sim.reset()

        prices = []
        for step in range(n_steps):
            sim.step()
            prices.append(sim.last_price.copy())

        prices = np.array(prices)

        returns = np.diff(prices, axis=0)

        m_vols = []
        m_kurts = []

        for m in range(n_markets_per_config):
            m_vols.append(np.std(prices[:, m]))
            m_ret = returns[:, m]
            if np.std(m_ret) > 0:
                m_kurts.append(stats.kurtosis(m_ret, fisher=True))
            else:
                m_kurts.append(0.0)

            if abs(f_mom - target_f_mom) < 1e-6:
                acf_raw = []
                acf_abs = []
                for lag in acf_lags:
                    r_t = m_ret[:-lag]
                    r_lag = m_ret[lag:]
                    acf_raw.append(np.corrcoef(r_t, r_lag)[0, 1])
                    acf_abs.append(np.corrcoef(np.abs(r_t), np.abs(r_lag))[0, 1])
                mean_acf_raw += np.array(acf_raw) / n_markets_per_config
                mean_acf_abs += np.array(acf_abs) / n_markets_per_config

        m_vols_mkt = sim.total_volume / n_steps

        volatilities.append(np.mean(m_vols))
        kurtoses.append(np.mean(m_kurts))
        volumes.append(np.mean(m_vols_mkt))

        print(f"Mom Frac: {f_mom:.2f} | Volatility: {volatilities[-1]:.4f} | Excess Kurtosis: {kurtoses[-1]:.4f} | Vol/step: {volumes[-1]:.4f}")

    out_dir = "benchmarks/figures"
    os.makedirs(out_dir, exist_ok=True)
    m_res_dir = "../manuscript/result"
    os.makedirs(m_res_dir, exist_ok=True)

    for target_dir, name in [(out_dir, "fig_market_sweep.png"), (m_res_dir, "fig_market_sweep.png"), (m_res_dir, "fig_market_sweep.pdf")]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        axes[0, 0].plot(mom_fractions, volatilities, marker='o', color='#d62728', linewidth=2)
        axes[0, 0].set_title("Price Volatility (std dev)", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Momentum Agent Fraction", fontsize=10)
        axes[0, 0].set_ylabel("Standard Deviation of Price", fontsize=10)
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

        axes[0, 1].plot(mom_fractions, kurtoses, marker='s', color='#1f77b4', linewidth=2)
        axes[0, 1].set_title("Excess Kurtosis of Returns", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Momentum Agent Fraction", fontsize=10)
        axes[0, 1].set_ylabel("Kurtosis (Fisher)", fontsize=10)
        axes[0, 1].axhline(y=0.0, color='gray', linestyle=':')
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

        axes[1, 0].plot(mom_fractions, volumes, marker='^', color='#2ca02c', linewidth=2)
        axes[1, 0].set_title("Mean Volume per Step", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Momentum Agent Fraction", fontsize=10)
        axes[1, 0].set_ylabel("Trading Volume (units/step)", fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

        axes[1, 1].bar(acf_lags - 0.2, mean_acf_raw, width=0.4, color='#aec7e8', label='Returns $r_t$')
        axes[1, 1].bar(acf_lags + 0.2, mean_acf_abs, width=0.4, color='#ffbb78', label='Abs returns $|r_t|$')
        axes[1, 1].plot(acf_lags, mean_acf_abs, color='#ff7f0e', linestyle='-', alpha=0.8)
        axes[1, 1].set_title("Return Autocorrelation (ACF)", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Lag (simulation steps)", fontsize=10)
        axes[1, 1].set_ylabel("Autocorrelation Coefficient", fontsize=10)
        axes[1, 1].set_xticks(np.arange(1, 21, 2))
        axes[1, 1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if name.endswith(".pdf"):
            plt.savefig(os.path.join(target_dir, name), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(target_dir, name), dpi=300)
        plt.close()

    print(f"Parameter sweep and autocorrelation analysis complete. Plots saved to {out_dir}/fig_market_sweep.png and {m_res_dir}/fig_market_sweep.png/.pdf")

if __name__ == "__main__":
    main()
