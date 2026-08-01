#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "manuscript")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_diagram():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    c_cpu = "#eaeded"
    c_gpu_bg = "#f8f9f9"
    c_gpu_border = "#2c3e50"
    c_block = "#ebf5fb"
    c_block_border = "#2980b9"
    c_shmem = "#e8f8f5"
    c_shmem_border = "#16a085"
    c_thread = "#fef9e7"
    c_thread_border = "#f39c12"
    c_arrow = "#34495e"
    c_bid = "#2ecc71"
    c_ask = "#e74c3c"

    rect_cpu = patches.FancyBboxPatch(
        (0.2, 5.5), 9.6, 0.6, boxstyle="round,pad=0.02",
        facecolor=c_cpu, edgecolor="#7f8c8d", lw=1.0
    )
    ax.add_patch(rect_cpu)
    ax.text(0.4, 5.7, "HOST CPU (Ryzen 9950X3D)", fontsize=9, fontweight="bold", color="#7f8c8d")
    ax.text(9.6, 5.7, "Single Kernel Launch (1x overhead)", fontsize=8.5, color="#c0392b", ha="right", fontweight="bold")

    ax.annotate("", xy=(5.0, 5.0), xytext=(5.0, 5.5),
                arrowprops=dict(arrowstyle="fancy", color=c_gpu_border, lw=1.5, ls="-"))
    ax.text(5.1, 5.25, "simulate_kernel<<<M, L>>>", fontsize=8, fontfamily="monospace", fontweight="bold")

    rect_gpu = patches.FancyBboxPatch(
        (0.2, 0.1), 9.6, 4.8, boxstyle="round,pad=0.02",
        facecolor=c_gpu_bg, edgecolor=c_gpu_border, lw=1.5
    )
    ax.add_patch(rect_gpu)
    ax.text(0.4, 4.6, "GPU DEVICE (GeForce RTX 5090)", fontsize=10, fontweight="bold", color=c_gpu_border)

    for offset in [0.15, 0.075]:
        rect_shadow = patches.FancyBboxPatch(
            (0.5 + offset, 1.2 + offset), 3.8, 3.0, boxstyle="round,pad=0.02",
            facecolor="#eaeded", edgecolor="#bdc3c7", lw=1.0
        )
        ax.add_patch(rect_shadow)

    rect_active = patches.FancyBboxPatch(
        (0.5, 1.2), 3.8, 3.0, boxstyle="round,pad=0.02",
        facecolor=c_block, edgecolor=c_block_border, lw=1.5
    )
    ax.add_patch(rect_active)
    ax.text(0.7, 3.9, "CUDA Thread Block $m$ (Market $m$)", fontsize=9, fontweight="bold", color="#1b4f72")

    ax.text(0.7, 3.5, "Thread $t$ maps to Level $p = t$", fontsize=8, fontweight="bold", color=c_block_border)

    lob_y = 1.4
    lob_h = 1.9
    lob_x = 0.7
    lob_w = 3.4

    ax.add_patch(patches.Rectangle((lob_x, lob_y), lob_w, lob_h, facecolor="#ffffff", edgecolor="#d5dbdb", lw=1))

    n_ticks = 8
    tick_h = lob_h / n_ticks
    for i in range(n_ticks + 1):
        ax.plot([lob_x, lob_x + lob_w], [lob_y + i * tick_h, lob_y + i * tick_h], color="#e5e8e8", lw=0.8)

    ax.text(lob_x - 0.05, lob_y + 7.5 * tick_h, "$p_{L-1}$ (t=L-1)", fontsize=6.5, ha="right", va="center")
    ax.text(lob_x - 0.05, lob_y + 0.5 * tick_h, "$p_0$ (t=0)", fontsize=6.5, ha="right", va="center")
    ax.text(lob_x - 0.05, lob_y + 4.0 * tick_h, "... p ... (t)", fontsize=6.5, ha="right", va="center")

    bid_levels = [3, 4, 3, 0, 0, 0, 0, 0]
    ask_levels = [0, 0, 0, 0, 2, 4, 3, 2]
    for i in range(n_ticks):
        y_pos = lob_y + i * tick_h + tick_h/4.0
        if bid_levels[i] > 0:
            ax.add_patch(patches.Rectangle((lob_x + lob_w/2.0 - bid_levels[i]*0.3, y_pos), bid_levels[i]*0.3, tick_h/2.0, color=c_bid, alpha=0.8))
        if ask_levels[i] > 0:
            ax.add_patch(patches.Rectangle((lob_x + lob_w/2.0, y_pos), ask_levels[i]*0.3, tick_h/2.0, color=c_ask, alpha=0.8))

    ax.plot([lob_x + lob_w/2.0, lob_x + lob_w/2.0], [lob_y, lob_y + lob_h], color=c_block_border, ls="-.", lw=1.0)
    ax.text(lob_x + lob_w/2.0, lob_y + lob_h + 0.05, "Mid", fontsize=6.5, ha="center")
    ax.text(lob_x + 0.3, lob_y + 1.2 * tick_h, "Bids (Buy)", fontsize=7, color="#1e8449", fontweight="bold")
    ax.text(lob_x + lob_w - 0.3, lob_y + 6.2 * tick_h, "Asks (Sell)", fontsize=7, color="#922b21", fontweight="bold")

    ax.text(0.7, 0.9, "LOB persistent in Shared Memory\n(Zero global round-trips)",
            fontsize=8, fontweight="bold", color="#0e6251", style="italic")

    rect_pipeline = patches.FancyBboxPatch(
        (4.7, 0.4), 4.9, 4.0, boxstyle="round,pad=0.02",
        facecolor=c_shmem, edgecolor=c_shmem_border, lw=1.5
    )
    ax.add_patch(rect_pipeline)
    ax.text(4.9, 4.1, "Persistent Step Execution Pipeline", fontsize=9, fontweight="bold", color="#0e6251")

    p_x = 4.9
    p_w = 4.5
    step_names = [
        "1. Observe LOB Stats (Shared Mid & Return)",
        "2. Run Strategy RNG (SplitMix64 Counter)",
        "3. Scatter Agent Orders (Shared Atomics aggregation)",
        "4. Clear Auction (Parallel Scans & Reductions)",
        "5. Update Residual Book (Price-priority fills)"
    ]

    for i, step in enumerate(step_names):
        y_pos = 3.3 - i * 0.7
        rect_step = patches.FancyBboxPatch(
            (p_x, y_pos), p_w, 0.5, boxstyle="round,pad=0.02",
            facecolor="#ffffff", edgecolor=c_shmem_border if i != 3 else "#d68910",
            lw=1.0 if i != 3 else 1.5
        )
        ax.add_patch(rect_step)
        col = "#0e6251" if i != 3 else "#9a7d0a"
        ax.text(p_x + 0.2, y_pos + 0.2, step, fontsize=8, fontweight="bold" if i == 3 else "normal", color=col)

        if i > 0:
            ax.annotate("", xy=(p_x + p_w/2.0, y_pos + 0.5), xytext=(p_x + p_w/2.0, y_pos + 0.7),
                        arrowprops=dict(arrowstyle="<-", color=c_arrow, lw=0.8))

    ax.annotate("", xy=(p_x + p_w - 0.2, 3.55), xytext=(p_x + p_w - 0.2, 0.75),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.7", color=c_shmem_border, lw=1.5))
    ax.text(p_x + p_w + 0.1, 2.15, "Loop for S steps", fontsize=7.5, fontweight="bold", color="#117864", rotation=270, va="center")

    clear_x = 5.2
    clear_y = 1.3
    clear_w = 1.3
    clear_h = 0.9

    ax.add_patch(patches.Rectangle((clear_x, clear_y), clear_w, clear_h, facecolor="#fef9e7", edgecolor="#f5b041", lw=1.0))
    ax.text(clear_x + clear_w/2.0, clear_y + clear_h + 0.05, "Call Auction Clearing", fontsize=6.5, fontweight="bold", ha="center", color="#b7950b")

    curve_x = np.linspace(clear_x + 0.1, clear_x + clear_w - 0.1, 50)
    curve_dcum = clear_y + 0.1 + (clear_h - 0.2) * (1.0 / (1.0 + np.exp(10 * (curve_x - (clear_x + clear_w/2.0)))))
    curve_scum = clear_y + 0.1 + (clear_h - 0.2) * (1.0 / (1.0 + np.exp(-10 * (curve_x - (clear_x + clear_w/2.0)))))

    ax.plot(curve_x, curve_dcum, color="#1e8449", lw=1.2, label="Dcum")
    ax.plot(curve_x, curve_scum, color="#922b21", lw=1.2, label="Scum")

    p_star_x = clear_x + clear_w/2.0
    p_star_y = clear_y + clear_h/2.0
    ax.plot(p_star_x, p_star_y, marker="o", color="#d68910", ms=4)
    ax.plot([p_star_x, p_star_x], [clear_y, p_star_y], color="#7f8c8d", ls=":", lw=0.8)
    ax.text(p_star_x, clear_y + 0.05, "$p^*$", fontsize=7, color="#d68910", fontweight="bold", ha="center")
    ax.text(clear_x + 0.1, clear_y + 0.7, "$D_{cum}$", fontsize=5.5, color="#1e8449")
    ax.text(clear_x + clear_w - 0.35, clear_y + 0.7, "$S_{cum}$", fontsize=5.5, color="#922b21")

    ax.text(clear_x + clear_w + 0.1, clear_y + clear_h/2.0, "Parallel Hillis-Steele Scan\n+ Argmax reduction\n(synchronization-free)",
            fontsize=6.5, va="center", color="#78281f", fontweight="bold")

    ax.annotate("", xy=(4.65, 3.1), xytext=(4.35, 3.1),
                arrowprops=dict(arrowstyle="<->", color=c_arrow, lw=1.2))
    ax.text(4.5, 3.25, "Read mid / Write atomics", fontsize=7.5, color=c_arrow, fontweight="bold", ha="center")

    ax.text(2.4, 0.45, "M Markets in parallel", fontsize=8.5, fontweight="bold", color="#1b4f72", ha="center")

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"fig_methodology.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Wrote fig_methodology.{ext}")

    plt.close(fig)

if __name__ == "__main__":
    generate_diagram()
