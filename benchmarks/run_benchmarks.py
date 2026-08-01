#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import replace
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import numpy as np

from kineticsim.model import SimConfig, SWEEP_MARKETS, SWEEP_AGENTS, DEFAULT_STEPS

def available_backends() -> Dict[str, type]:
    from kineticsim.reference_cpu import KineticSimCPU
    backends: Dict[str, type] = {"cpu": KineticSimCPU}

    try:
        import torch
        if torch.cuda.is_available():
            from kineticsim.torch_gpu import KineticSimTorch, KineticSimTorchGraph, KineticSimTorchCompile
            backends["torch"] = KineticSimTorch
            backends["torch_graph"] = KineticSimTorchGraph
            backends["torch_compile"] = KineticSimTorchCompile
    except Exception:
        pass

    try:
        import jax
        from kineticsim.jax_gpu import KineticSimJax
        backends["jax"] = KineticSimJax
    except Exception:
        pass

    try:
        import kineticsim_naive
        from kineticsim.cuda_backend import KineticSimNaiveCUDA
        backends["naive"] = KineticSimNaiveCUDA
    except Exception:
        pass

    try:
        import kineticsim_cuda
        from kineticsim.cuda_backend import KineticSimCUDA
        backends["cuda"] = KineticSimCUDA
    except Exception:
        pass

    return backends

def measure(backend_cls, cfg: SimConfig, trials: int, warmup: bool) -> Dict:
    try:
        import torch
        torch._dynamo.reset()
    except Exception:
        pass

    if warmup:
        try:
            backend_cls(replace(cfg, n_steps=min(cfg.n_steps, 20))).run(
                min(cfg.n_steps, 20))
        except Exception:
            pass

    times: List[float] = []
    last = None
    for _ in range(trials):
        eng = backend_cls(cfg)
        s = eng.run(cfg.n_steps)
        times.append(s["wall_time_s"])
        last = s

    times.sort()
    med = statistics.median(times)
    events = cfg.n_markets * cfg.n_agents * cfg.n_steps
    row = dict(last)
    row.update({
        "wall_time_s": med,
        "wall_time_min": times[0],
        "wall_time_max": times[-1],
        "wall_time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "trials": trials,
        "events_per_s": events / med if med > 0 else float("nan"),
        "steps_per_s": cfg.n_steps / med if med > 0 else float("nan"),
        "us_per_step": 1e6 * med / cfg.n_steps,
        "ns_per_event": 1e9 * med / events,
    })
    return row

def run_sweep(name, backends, configs, trials, cpu_max_events) -> List[Dict]:
    rows: List[Dict] = []
    for cfg in configs:
        events = cfg.n_markets * cfg.n_agents * cfg.n_steps
        for bname, bcls in backends.items():
            if bname == "cpu" and events > cpu_max_events:
                print(f"  [skip] cpu  M={cfg.n_markets} A={cfg.n_agents} "
                      f"({events:,} events > cpu cap)")
                continue
            print(f"  [{name}] {bname:13s} M={cfg.n_markets:6d} "
                  f"A={cfg.n_agents:5d} S={cfg.n_steps} ...", end="", flush=True)
            try:
                actual_trials = 1 if bname == "cpu" else trials
                r = measure(bcls, cfg, actual_trials, warmup=(bname != "cpu"))
            except Exception as e:
                print(f" FAILED ({e})")
                continue
            r["sweep"] = name
            rows.append(r)
            print(f" {r['events_per_s']:.3e} ev/s  "
                  f"({r['wall_time_s']*1e3:.1f} ms)")
    return rows

def main() -> None:
    ap = argparse.ArgumentParser(description="KineticSim benchmark driver")
    ap.add_argument("--all", action="store_true", help="run every sweep")
    ap.add_argument("--markets", action="store_true")
    ap.add_argument("--agents", action="store_true")
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--backends", default="auto",
                    help="comma list of cpu,torch,naive,cuda or 'auto'")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--markets-n", type=int, default=8192, help="fixed-mode markets")
    ap.add_argument("--agents-n", type=int, default=256, help="fixed-mode agents")
    ap.add_argument("--levels", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-max-events", type=float, default=2e9,
                    help="skip the (slow) CPU backend above this events count")
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    args = ap.parse_args()

    if not any([args.all, args.markets, args.agents, args.fixed, args.latency]):
        args.all = True

    backends = available_backends()
    if args.backends != "auto":
        want = [b.strip() for b in args.backends.split(",")]
        backends = {k: v for k, v in backends.items() if k in want}
    print("Backends:", ", ".join(backends) or "(none!)")
    if not backends:
        sys.exit("No backends available.")

    base = dict(n_levels=args.levels, n_steps=args.steps, seed=args.seed)
    all_rows: List[Dict] = []

    if args.all or args.markets:
        print("\n== markets sweep (A=%d, S=%d) ==" % (args.agents_n, args.steps))
        cfgs = [SimConfig(n_markets=m, n_agents=args.agents_n, **base)
                for m in SWEEP_MARKETS]
        all_rows += run_sweep("markets", backends, cfgs, args.trials, args.cpu_max_events)

    if args.all or args.agents:
        print("\n== agents sweep (M=%d, S=%d) ==" % (args.markets_n, args.steps))
        cfgs = [SimConfig(n_markets=args.markets_n, n_agents=a, **base)
                for a in SWEEP_AGENTS]
        all_rows += run_sweep("agents", backends, cfgs, args.trials, args.cpu_max_events)

    if args.all or args.fixed:
        print("\n== fixed workload ==")
        cfg = SimConfig(n_markets=args.markets_n, n_agents=args.agents_n, **base)
        all_rows += run_sweep("fixed", backends, [cfg], args.trials, args.cpu_max_events)

    if args.all or args.latency:
        print("\n== latency (repeated trials of one config) ==")
        cfg = SimConfig(n_markets=4096, n_agents=256, **base)
        all_rows += run_sweep("latency", backends, [cfg],
                              max(args.trials, 11), args.cpu_max_events)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    if not df.empty:
        keys = ["sweep", "n_markets", "n_agents", "n_steps"]
        def add_speedup(ref):
            ref_t = (df[df.backend.isin([ref])]
                     .set_index(keys)["wall_time_s"].to_dict())
            col = []
            for _, r in df.iterrows():
                k = tuple(r[x] for x in keys)
                t = ref_t.get(k)
                col.append(t / r["wall_time_s"] if t and r["wall_time_s"] > 0 else np.nan)
            df[f"speedup_vs_{ref}"] = col
        refnames = {"cpu_numpy": "cpu", "torch_gpu": "torch"}
        for refname in ("cpu_numpy", "torch_gpu"):
            if (df.backend == refname).any():
                ref_t = df[df.backend == refname].set_index(keys)["wall_time_s"].to_dict()
                short = refnames[refname]
                df[f"speedup_vs_{short}"] = [
                    (ref_t.get(tuple(r[x] for x in keys)) or np.nan) / r["wall_time_s"]
                    if r["wall_time_s"] > 0 else np.nan
                    for _, r in df.iterrows()
                ]
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(df)} rows -> {args.out}")
        cols = ["sweep", "backend", "n_markets", "n_agents", "events_per_s",
                "us_per_step"]
        cols += [c for c in df.columns if c.startswith("speedup_vs_")]
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(df[cols].to_string(index=False))
    else:
        print("No rows collected.")

if __name__ == "__main__":
    main()
