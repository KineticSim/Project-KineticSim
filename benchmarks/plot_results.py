#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 12,
    "font.family": "DejaVu Sans", "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12.5, "axes.labelweight": "medium", "axes.linewidth": 1.0,
    "axes.edgecolor": "#444444", "axes.grid": True, "grid.color": "#d9d9d9",
    "grid.linewidth": 0.8, "legend.fontsize": 10.5, "legend.frameon": True,
    "legend.framealpha": 0.95, "legend.edgecolor": "#cccccc",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})

STYLE = {
    "cpu_numpy":       ("#6b7280", "o", "CPU (NumPy)"),
    "torch_gpu":       ("#2563eb", "s", "PyTorch GPU"),
    "torch_graph":     ("#8b5cf6", "p", "PyTorch CUDA Graphs"),
    "torch_compile":   ("#ec4899", "*", "PyTorch compile"),
    "jax_gpu":         ("#10b981", "v", "JAX GPU"),
    "naive_cuda":      ("#f59e0b", "^", "Naive Custom CUDA"),
    "kineticsim_cuda": ("#dc2626", "D", "KineticSim"),
}
ORDER = ["cpu_numpy", "torch_gpu", "torch_graph", "torch_compile", "jax_gpu", "naive_cuda", "kineticsim_cuda"]
LW, MS = 2.6, 9

def _save(fig, out, name, rect=None):
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)

def _human(v):
    if v >= 1e9: return f"{v/1e9:.1f} G ev/s"
    if v >= 1e6: return f"{v/1e6:.0f} M ev/s"
    return f"{v:.0f} ev/s"

def _log2_xaxis(ax, vals):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"$2^{{{int(round(np.log2(x)))}}}$"))
    ax.set_xticks(sorted(set(vals)))

def _bold_kineticsim(legend):
    if legend is not None:
        for text in legend.get_texts():
            if "KineticSim" in text.get_text():
                text.set_weight("bold")

def plot_throughput(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    sub_m = df[df.sweep == "markets"]
    if not sub_m.empty:
        ax = axes[0]
        peak = None
        for b in ORDER:
            bb = sub_m[sub_m.backend == b].sort_values("n_markets")
            if bb.empty: continue
            c, mk, lab = STYLE[b]
            z = 5 if b == "kineticsim_cuda" else 3
            ax.plot(bb["n_markets"], bb["events_per_s"], marker=mk, color=c, label=lab,
                    lw=LW if b == "kineticsim_cuda" else 2.0, ms=MS,
                    markeredgecolor="white", markeredgewidth=1.1, zorder=z)
            if b == "kineticsim_cuda":
                ix = bb["events_per_s"].idxmax()
                peak = (bb.loc[ix, "n_markets"], bb.loc[ix, "events_per_s"])
        _log2_xaxis(ax, sub_m["n_markets"].unique())
        ax.set_yscale("log")
        ax.set_xlabel("Number of parallel markets ($M$)")
        ax.set_ylabel("Throughput (agent-events / s)")
        if peak is not None:
            ax.annotate(f"peak {_human(peak[1])}", xy=peak, xytext=(-12, -30),
                        textcoords="offset points", ha="right", va="top", fontsize=9.5,
                        fontweight="bold", color="#dc2626",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0))
        ax.grid(True, which="both", ls=":", alpha=0.55)
        ax.text(0.015, 0.97, "fixed: $A{=}256$, $S{=}500$, $L{=}128$", transform=ax.transAxes, va="top", ha="left",
                fontsize=9.0, style="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#dddddd"))

    sub_a = df[df.sweep == "agents"]
    if not sub_a.empty:
        ax = axes[1]
        peak = None
        for b in ORDER:
            bb = sub_a[sub_a.backend == b].sort_values("n_agents")
            if bb.empty: continue
            c, mk, lab = STYLE[b]
            z = 5 if b == "kineticsim_cuda" else 3
            ax.plot(bb["n_agents"], bb["events_per_s"], marker=mk, color=c, label=lab,
                    lw=LW if b == "kineticsim_cuda" else 2.0, ms=MS,
                    markeredgecolor="white", markeredgewidth=1.1, zorder=z)
            if b == "kineticsim_cuda":
                ix = bb["events_per_s"].idxmax()
                peak = (bb.loc[ix, "n_agents"], bb.loc[ix, "events_per_s"])
        _log2_xaxis(ax, sub_a["n_agents"].unique())
        ax.set_yscale("log")
        ax.set_xlabel("Agents per market ($A$)")
        ax.set_ylabel("Throughput (agent-events / s)")
        if peak is not None:
            ax.annotate(f"peak {_human(peak[1])}", xy=peak, xytext=(-12, -30),
                        textcoords="offset points", ha="right", va="top", fontsize=9.5,
                        fontweight="bold", color="#dc2626",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0))
        ax.grid(True, which="both", ls=":", alpha=0.55)
        ax.text(0.015, 0.97, "fixed: $M{=}8192$, $S{=}500$, $L{=}128$", transform=ax.transAxes, va="top", ha="left",
                fontsize=9.0, style="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#dddddd"))

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11.5, frameon=True)
    _bold_kineticsim(leg)
    _save(fig, out, "fig_throughput", rect=[0, 0, 1, 0.88])

def plot_speedup(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    sub_fixed = df[df.sweep == "fixed"]
    if not sub_fixed.empty:
        ax = axes[0]
        gpu = [b for b in ("torch_gpu", "jax_gpu", "naive_cuda", "kineticsim_cuda") if (sub_fixed.backend == b).any()]
        x = np.arange(len(gpu))
        w = 0.38
        series = [("speedup_vs_cpu", "vs CPU NumPy", "#475569"),
                  ("speedup_vs_torch", "vs PyTorch GPU", "#0ea5e9")]
        for i, (col, lab, col_c) in enumerate(series):
            vals = [float(sub_fixed[sub_fixed.backend == b][col].iloc[0]) for b in gpu]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lab, color=col_c,
                          edgecolor="white", linewidth=0.8, zorder=3)
            for rect, v in zip(bars, vals):
                if v >= 1.0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v * 1.05, f"{v:,.0f}x",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 1e4)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[b][2] if b != "naive_cuda" else "Naive CUDA" for b in gpu], fontsize=8.5)
        ax.set_ylabel("Speedup (x, log scale)")
        ax.axhline(1.0, color="#999999", ls="--", lw=1.0, zorder=1)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.55)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9.0)

    sub_markets = df[df.sweep == "markets"]
    if not sub_markets.empty:
        ax = axes[1]
        plots = [
            ("kineticsim_cuda", "speedup_vs_cpu",  "#dc2626", "D", "-",  "KineticSim vs CPU"),
            ("naive_cuda",      "speedup_vs_cpu",  "#f59e0b", "^", "-",  "Naive CUDA vs CPU"),
            ("kineticsim_cuda", "speedup_vs_torch","#dc2626", "D", "--", "KineticSim vs PyTorch"),
            ("naive_cuda",      "speedup_vs_torch","#f59e0b", "^", "--", "Naive CUDA vs PyTorch"),
            ("kineticsim_cuda", "speedup_vs_jax",   "#dc2626", "D", "-.", "KineticSim vs JAX"),
            ("naive_cuda",      "speedup_vs_jax",   "#f59e0b", "^", "-.", "Naive CUDA vs JAX"),
        ]
        for b, col, c, mk, ls, lab in plots:
            bb = sub_markets[sub_markets.backend == b].dropna(subset=[col]).sort_values("n_markets")
            if bb.empty: continue
            ax.plot(bb["n_markets"], bb[col], marker=mk, color=c, ls=ls, lw=2.2, ms=8,
                    markeredgecolor="white", markeredgewidth=1.0, label=lab)
        _log2_xaxis(ax, sub_markets["n_markets"].unique())
        ax.set_yscale("log")
        ax.set_xlabel("Number of parallel markets ($M$)")
        ax.set_ylabel("Speedup over baseline (x, log)")
        ax.axhline(1.0, color="#999999", ls=":", lw=1.0)
        ax.grid(True, which="both", ls=":", alpha=0.55)
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.8)
        _bold_kineticsim(leg)

    _save(fig, out, "fig_speedup", rect=[0, 0, 1, 0.88])

def plot_latency(df, out):
    sub = df[df.sweep == "latency"]
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    backs = [b for b in ORDER if (sub.backend == b).any() and b not in ("torch_graph", "torch_compile")]
    meds = [float(sub[sub.backend == b]["us_per_step"].iloc[0]) for b in backs]
    lo = [float(sub[sub.backend == b]["wall_time_min"].iloc[0]) /
          float(sub[sub.backend == b]["n_steps"].iloc[0]) * 1e6 for b in backs]
    hi = [float(sub[sub.backend == b]["wall_time_max"].iloc[0]) /
          float(sub[sub.backend == b]["n_steps"].iloc[0]) * 1e6 for b in backs]
    err = [[max(m - l, 0) for m, l in zip(meds, lo)],
           [max(h - m, 0) for h, m in zip(hi, meds)]]
    colors = [STYLE[b][0] for b in backs]
    x = np.arange(len(backs))
    bars = ax.bar(x, meds, yerr=err, capsize=6, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3, error_kw=dict(ecolor="#222222", lw=1.3))
    ks = meds[backs.index("kineticsim_cuda")]
    for rect, v, b in zip(bars, meds, backs):
        tag = f"{v:,.1f} us" + ("" if b == "kineticsim_cuda" else f"\n({v/ks:,.0f}x slower)")
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.25, tag,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(top=max(meds) * 6); ax.set_xticks(x)
    ax.set_xticklabels([STYLE[b][2] if b != "naive_cuda" else "Naive CUDA" for b in backs], fontsize=10.0, rotation=0, ha="center")
    ax.set_ylabel("Per-step latency (us, log scale)")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.55)
    _save(fig, out, "fig_latency")

def plot_memory(df, out):
    sub = df[(df.sweep == "markets") & df.gpu_mem_gb.notna()]
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for b in ("torch_gpu", "torch_graph", "torch_compile", "naive_cuda", "kineticsim_cuda"):
        bb = sub[sub.backend == b].sort_values("n_markets")
        if bb.empty: continue
        c, mk, lab = STYLE[b]
        ax.plot(bb["n_markets"], bb["gpu_mem_gb"] * 1024.0, marker=mk, color=c,
                lw=2.4 if b == "kineticsim_cuda" else 2.0, ms=8,
                markeredgecolor="white", markeredgewidth=1.0, label=lab)
    big = sub[sub.n_markets == sub.n_markets.max()]
    try:
        tm = float(big[big.backend == "torch_gpu"]["gpu_mem_gb"].iloc[0])
        km = float(big[big.backend == "kineticsim_cuda"]["gpu_mem_gb"].iloc[0])
        ax.annotate(f"{tm/km:.0f}x smaller\nfootprint",
                    xy=(big.n_markets.max(), km * 1024), xytext=(-10, 38),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color="#dc2626",
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
    except (IndexError, ValueError): pass
    _log2_xaxis(ax, sub["n_markets"].unique())
    ax.set_yscale("log")
    ax.set_xlabel("Number of parallel markets ($M$)")
    ax.set_ylabel("GPU global memory (MB, log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    _bold_kineticsim(leg)
    _save(fig, out, "fig_memory")

def plot_efficiency(df, out):
    sub = df[df.sweep == "markets"]
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for b in ORDER:
        bb = sub[sub.backend == b].sort_values("n_markets")
        if bb.empty: continue
        c, mk, lab = STYLE[b]
        ax.plot(bb["n_markets"], bb["ns_per_event"], marker=mk, color=c,
                lw=2.4 if b == "kineticsim_cuda" else 2.0, ms=8,
                markeredgecolor="white", markeredgewidth=1.0, label=lab)
    _log2_xaxis(ax, sub["n_markets"].unique())
    ax.set_yscale("log")
    ax.set_xlabel("Number of parallel markets ($M$)")
    ax.set_ylabel("Time per agent-event (ns, log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=8.5)
    _bold_kineticsim(leg)
    _save(fig, out, "fig_efficiency")

def plot_memory_efficiency(df, out):
    sub = df[df.sweep == "markets"]
    if sub.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    ax = axes[0]
    sub_m = sub[sub.gpu_mem_gb.notna()]
    if not sub_m.empty:
        for b in ("torch_gpu", "torch_graph", "torch_compile", "naive_cuda", "kineticsim_cuda"):
            bb = sub_m[sub_m.backend == b].sort_values("n_markets")
            if bb.empty: continue
            c, mk, lab = STYLE[b]
            ax.plot(bb["n_markets"], bb["gpu_mem_gb"] * 1024.0, marker=mk, color=c,
                    lw=2.4 if b == "kineticsim_cuda" else 2.0, ms=8,
                    markeredgecolor="white", markeredgewidth=1.0, label=lab)
        big = sub_m[sub_m.n_markets == sub_m.n_markets.max()]
        try:
            tm = float(big[big.backend == "torch_gpu"]["gpu_mem_gb"].iloc[0])
            km = float(big[big.backend == "kineticsim_cuda"]["gpu_mem_gb"].iloc[0])
            ax.annotate(f"{tm/km:.0f}x smaller\nfootprint",
                        xy=(big.n_markets.max(), km * 1024), xytext=(-10, 38),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        fontweight="bold", color="#dc2626",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
        except (IndexError, ValueError): pass
        _log2_xaxis(ax, sub["n_markets"].unique())
        ax.set_yscale("log")
        ax.set_xlabel("Number of parallel markets ($M$)")
        ax.set_ylabel("GPU global memory (MB, log scale)")
        ax.grid(True, which="both", ls=":", alpha=0.55)

    ax = axes[1]
    for b in ORDER:
        bb = sub[sub.backend == b].sort_values("n_markets")
        if bb.empty: continue
        c, mk, lab = STYLE[b]
        ax.plot(bb["n_markets"], bb["ns_per_event"], marker=mk, color=c,
                lw=2.4 if b == "kineticsim_cuda" else 2.0, ms=8,
                markeredgecolor="white", markeredgewidth=1.0, label=lab)
    _log2_xaxis(ax, sub["n_markets"].unique())
    ax.set_yscale("log")
    ax.set_xlabel("Number of parallel markets ($M$)")
    ax.set_ylabel("Time per agent-event (ns, log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.55)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11.5, frameon=True)
    _bold_kineticsim(leg)
    _save(fig, out, "fig_memory_efficiency", rect=[0, 0, 1, 0.88])

def plot_price_agreement(df, out):
    sub = df[df.sweep == "markets"]
    if sub.empty: return
    backs = [b for b in ORDER if (sub.backend == b).any()]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, metric, ylabel in (
        (axes[0], "mean_last_price", "Mean clearing price (tick)"),
        (axes[1], "mean_volume_per_market", "Mean volume / market"),
    ):
        for b in backs:
            bb = sub[sub.backend == b].sort_values("n_markets")
            c, mk, lab = STYLE[b]
            ax.plot(bb["n_markets"], bb[metric], marker=mk, color=c, lw=1.8, ms=8,
                    markeredgecolor="white", markeredgewidth=1.0, label=lab, alpha=0.9)
        _log2_xaxis(ax, sub["n_markets"].unique())
        ax.set_xlabel("Number of parallel markets ($M$)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls=":", alpha=0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11.5, frameon=True)
    _bold_kineticsim(leg)
    _save(fig, out, "fig_price_agreement", rect=[0, 0, 1, 0.88])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "results.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    plot_throughput(df, args.out)
    plot_speedup(df, args.out)
    plot_latency(df, args.out)
    plot_memory(df, args.out)
    plot_efficiency(df, args.out)
    plot_memory_efficiency(df, args.out)
    plot_price_agreement(df, args.out)
    print("Figures written to", args.out)

if __name__ == "__main__":
    main()
