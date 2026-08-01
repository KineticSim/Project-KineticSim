import numpy as np
import scipy.stats as stats
from kineticsim import SimConfig, KineticSimCPU

def main():
    cfg = SimConfig(
        n_markets=10,
        n_agents=256,
        n_levels=128,
        n_steps=5000,
        seed=42,
        frac_noise=0.70,
        frac_momentum=0.15,
        frac_maker=0.15
    )

    sim = KineticSimCPU(cfg)
    sim.reset()

    prices = []
    for step in range(cfg.n_steps):
        sim.step()
        prices.append(sim.last_price.copy())

    prices = np.array(prices)

    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)

    kurtoses = []
    autocorr_raw = []
    autocorr_abs = []

    for m in range(cfg.n_markets):
        m_ret = returns[:, m]
        k = stats.kurtosis(m_ret, fisher=True)
        kurtoses.append(k)

        raw_ac = []
        abs_ac = []
        for lag in [1, 2, 5, 10, 20]:
            r_t = m_ret[:-lag]
            r_lag = m_ret[lag:]

            corr_raw = np.corrcoef(r_t, r_lag)[0, 1]
            raw_ac.append(corr_raw)

            corr_abs = np.corrcoef(np.abs(r_t), np.abs(r_lag))[0, 1]
            abs_ac.append(corr_abs)

        autocorr_raw.append(raw_ac)
        autocorr_abs.append(abs_ac)

    print(f"Mean Excess Kurtosis of returns: {np.mean(kurtoses):.4f} (expected > 0 for fat tails)")

    mean_raw = np.mean(autocorr_raw, axis=0)
    mean_abs = np.mean(autocorr_abs, axis=0)

    print("\nLag Autocorrelation of returns (expected ~0):")
    for l, val in zip([1, 2, 5, 10, 20], mean_raw):
        print(f"  Lag {l:2d}: {val:.4f}")

    print("\nLag Autocorrelation of absolute returns (expected > 0 and slowly decaying):")
    for l, val in zip([1, 2, 5, 10, 20], mean_abs):
        print(f"  Lag {l:2d}: {val:.4f}")

if __name__ == "__main__":
    main()
