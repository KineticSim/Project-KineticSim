import pandas as pd
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
baselines_csv = os.path.join(HERE, "results_baselines.csv")
kineticsim_csv = os.path.join(HERE, "results_kineticsim.csv")
out_csv = os.path.join(HERE, "results.csv")

if not os.path.exists(baselines_csv):
    raise FileNotFoundError(f"Missing {baselines_csv}")
if not os.path.exists(kineticsim_csv):
    raise FileNotFoundError(f"Missing {kineticsim_csv}")

df_base = pd.read_csv(baselines_csv)
df_ks = pd.read_csv(kineticsim_csv)

df = pd.concat([df_base, df_ks], ignore_index=True)

speedup_cols = [c for c in df.columns if c.startswith("speedup_vs_")]
df = df.drop(columns=speedup_cols, errors="ignore")

keys = ["sweep", "n_markets", "n_agents", "n_steps"]
refnames = {
    "cpu_numpy": "cpu",
    "torch_gpu": "torch",
    "abides": "abides",
    "jaxlob": "jaxlob",
    "jaxmarl_hft": "jaxmarl_hft"
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

df.to_csv(out_csv, index=False)
print(f"Merged and calculated speedups for {len(df)} rows -> {out_csv}")
