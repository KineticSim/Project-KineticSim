import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("benchmarks/results.csv")
    print("Backend, M, A, S, Mean Time (s), Std Time (s), Mean Throughput, Std Throughput")
    for idx, row in df.iterrows():
        b = row["backend"]
        M = int(row["n_markets"])
        A = int(row["n_agents"])
        S = int(row["n_steps"])
        t_mean = row["wall_time_s"]
        t_std = row["wall_time_std"] if not pd.isna(row["wall_time_std"]) else 0.0
        events = row["events"]

        thr_mean = events / t_mean
        thr_std = thr_mean * (t_std / t_mean) if t_mean > 0 else 0.0

        print(f"{b}, M={M}, A={A}, S={S} | Time: {t_mean*1000:.3f} +- {t_std*1000:.3f} ms | Throughput: {thr_mean:.3e} +- {thr_std:.3e} ev/s")

if __name__ == "__main__":
    main()
