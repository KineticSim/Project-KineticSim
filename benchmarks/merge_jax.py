import pandas as pd
import numpy as np
import os

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    results_csv = os.path.join(HERE, "results.csv")
    jax_csv = os.path.join(HERE, "results_jax.csv")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Missing {results_csv}")
    if not os.path.exists(jax_csv):
        raise FileNotFoundError(f"Missing {jax_csv}")

    df_orig = pd.read_csv(results_csv)
    df_jax = pd.read_csv(jax_csv)

    df_orig = df_orig[df_orig.backend != "jax_gpu"]

    df = pd.concat([df_orig, df_jax], ignore_index=True)

    speedup_cols = [c for c in df.columns if c.startswith("speedup_vs_")]
    df = df.drop(columns=speedup_cols, errors="ignore")

    keys = ["sweep", "n_markets", "n_agents", "n_steps"]

    refnames = {
        "cpu_numpy": "cpu",
        "torch_gpu": "torch",
        "jax_gpu": "jax"
    }

    for refname in refnames:
        if (df.backend == refname).any():
            ref_t = df[df.backend == refname].set_index(keys)["wall_time_s"].to_dict()
            short = refnames[refname]
            df[f"speedup_vs_{short}"] = [
                (ref_t.get(tuple(r[x] for x in keys)) or np.nan) / r["wall_time_s"]
                if r["wall_time_s"] > 0 else np.nan
                for _, r in df.iterrows()
            ]

    df.to_csv(results_csv, index=False)
    print(f"Successfully merged JAX results. Recalculated speedups for {len(df)} rows -> {results_csv}")

if __name__ == "__main__":
    main()
