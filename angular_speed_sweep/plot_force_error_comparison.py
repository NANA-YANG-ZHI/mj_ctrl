"""Compare avg force Z error across 4 control methods vs EE linear speed.

Produces a broken-axis plot:
  - Top panel  (short, compressed): y = 3–20 N
  - Bottom panel (tall, detailed) : y = 0–3.1 N
Extreme feedforward outliers (~932 N, ~667 N) are clipped to the top of the
upper panel and annotated with their actual values.

Usage:
    python angular_speed_sweep/plot_force_error_comparison.py

Output:
    angular_speed_sweep/plots/force_error_comparison.png
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
OUT_PATH = os.path.join(PLOTS_DIR, "force_error_comparison.png")

METHODS = [
    ("Feedforward",      os.path.join(PLOTS_DIR, "feedforward",      "sweep_results.csv"), "tab:blue",   "o"),
    ("Feedforward + PI", os.path.join(PLOTS_DIR, "feedforward_pi",   "sweep_results.csv"), "tab:orange", "s"),
    ("PD New Params",    os.path.join(PLOTS_DIR, "pd_newparameters", "sweep_results.csv"), "tab:green",  "^"),
    ("Paper",            os.path.join(PLOTS_DIR, "paper",            "sweep_results.csv"), "tab:red",    "D"),
]

# y-axis break: show 0..BOT_MAX in detail, BOT_MAX..TOP_MAX compressed.
BOT_MAX = 3.1
TOP_MAX = 20.0
BREAK_Y = 3.0


def load_csv(path):
    v, err = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v.append(float(row["ee_linear_speed_m_s"]))
                err.append(float(row["avg_force_z_error"]))
            except (KeyError, ValueError):
                continue
    return np.array(v), np.array(err)


def main():
    # ── Load all four datasets ────────────────────────────────────────────────
    datasets = []
    for name, path, color, marker in METHODS:
        v, err = load_csv(path)
        datasets.append((name, v, err, color, marker))

    # ── Figure with broken y-axis ─────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])  # y: BREAK_Y .. TOP_MAX
    ax_bot = fig.add_subplot(gs[1])  # y: 0 .. BOT_MAX

    # ── Plot all methods on both panels ───────────────────────────────────────
    for name, v, err, color, marker in datasets:
        # Clip values above TOP_MAX so they don't mess up axes (handled separately)
        err_clipped = np.where(err > TOP_MAX, np.nan, err)

        kw = dict(color=color, linewidth=1.5, markersize=4,
                  marker=marker, markerfacecolor=color)
        ax_top.plot(v, err_clipped, label=name, **kw)
        ax_bot.plot(v, err_clipped, label=name, **kw)

    # ── Annotate extreme feedforward outliers ─────────────────────────────────
    # feedforward: spikes at v≈1.508 (932 N) and v≈1.539 (667 N)
    ff_name, ff_v, ff_err, ff_color, _ = datasets[0]  # feedforward is first
    outlier_mask = ff_err > TOP_MAX
    for xi, yi in zip(ff_v[outlier_mask], ff_err[outlier_mask]):
        # Draw a clipped triangle marker pointing up at top of panel
        ax_top.plot(xi, TOP_MAX * 0.96, marker="^", color=ff_color,
                    markersize=9, clip_on=False, zorder=5)
        ax_top.text(xi, TOP_MAX * 0.85, f"{yi:.0f} N",
                    color=ff_color, fontsize=8, va="top", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ff_color,
                              lw=0.8, alpha=0.85))

    # ── Axis limits ───────────────────────────────────────────────────────────
    ax_top.set_ylim(BREAK_Y, TOP_MAX)
    ax_bot.set_ylim(0.0, BOT_MAX)

    # Share x-axis range
    x_min = 0.0
    x_max = max(v.max() for _, v, _, _, _ in datasets) * 1.02
    for ax in (ax_top, ax_bot):
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)

    # ── Broken-axis cosmetics ─────────────────────────────────────────────────
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Diagonal break marks at the boundary between the two panels
    d = 0.012  # fraction of axis size
    kw_break = dict(color="k", clip_on=False, linewidth=1, transform=None)

    for ax, ypos in [(ax_top, 0), (ax_bot, 1)]:
        kw_break["transform"] = ax.transAxes
        ax.plot((-d, +d), (ypos - d, ypos + d), **kw_break)
        ax.plot((1 - d, 1 + d), (ypos - d, ypos + d), **kw_break)

    # ── Labels and legend ─────────────────────────────────────────────────────
    xlabel = "EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)"
    ylabel = "Avg |Force Z Error| (N)"

    ax_bot.set_xlabel(xlabel, fontsize=11)
    ax_bot.set_ylabel(ylabel, fontsize=11)
    ax_top.set_ylabel(ylabel, fontsize=11)

    # Single legend on the bottom panel
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_bot.legend(handles, labels, loc="upper left", fontsize=10,
                  framealpha=0.85)

    fig.suptitle(
        "Force Error Comparison — All Methods vs. EE Linear Speed",
        fontsize=13, y=0.995,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
