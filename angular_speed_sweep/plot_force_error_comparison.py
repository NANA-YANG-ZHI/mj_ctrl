"""Compare avg force Z error across 4 control methods vs EE linear speed.

Produces a single-axis plot with a piecewise linear y-scale:
  - y = 0–3 N  : full linear detail  (normal slope, tall)
  - y = 3–20 N : compressed linear   (slope ÷17, ~25% of axis height)

This matches the visual emphasis of the previous broken-axis version while
keeping a single continuous y-axis.  Extreme feedforward outliers (~932 N,
~667 N) are clipped to TOP_MAX and annotated.

Usage:
    python angular_speed_sweep/plot_force_error_comparison.py

Output:
    angular_speed_sweep/plots/force_error_comparison.png
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
OUT_PATH = os.path.join(PLOTS_DIR, "force_error_comparison.png")

METHODS = [
    ("Feedforward",      os.path.join(PLOTS_DIR, "feedforward",      "sweep_results.csv"), "tab:blue",   "o"),
    ("Feedforward + PI", os.path.join(PLOTS_DIR, "feedforward_pi",   "sweep_results.csv"), "tab:orange", "s"),
    ("PD New Params",    os.path.join(PLOTS_DIR, "pd_newparameters", "sweep_results.csv"), "tab:green",  "^"),
    ("Paper",            os.path.join(PLOTS_DIR, "paper",            "sweep_results.csv"), "tab:red",    "D"),
]

# Piecewise linear scale parameters
BREAK    = 3.0   # N — inflection point
COMPRESS = 17.0  # above BREAK: each N takes 1/17 of a normal N's visual height
TOP_MAX  = 20.0  # N — axis ceiling (extreme outliers annotated separately)


def forward(y):
    """Data → display transform (piecewise linear)."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= BREAK, y, BREAK + (y - BREAK) / COMPRESS)


def inverse(z):
    """Display → data transform (inverse of forward)."""
    z = np.asarray(z, dtype=float)
    return np.where(z <= BREAK, z, BREAK + (z - BREAK) * COMPRESS)


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
    # ── Load data ─────────────────────────────────────────────────────────────
    datasets = []
    for name, path, color, marker in METHODS:
        v, err = load_csv(path)
        datasets.append((name, v, err, color, marker))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 11))

    # ── Plot all 4 methods (clip extreme outliers to NaN so they don't distort)
    for name, v, err, color, marker in datasets:
        err_plot = np.where(err > TOP_MAX, np.nan, err)
        ax.plot(v, err_plot, marker=marker, color=color, label=name,
                linewidth=1.5, markersize=4, markerfacecolor=color)

    # ── Apply piecewise linear y-scale ────────────────────────────────────────
    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_ylim(0.0, TOP_MAX)

    # ── Manual y-ticks: dense below BREAK, sparse above ───────────────────────
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 8, 11, 14, 17, 20])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # ── Inflection guide line at y = BREAK ────────────────────────────────────
    ax.axhline(BREAK, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.annotate(
        f"  scale ÷{COMPRESS:.0f} above",
        xy=(0.0, BREAK), xycoords=("axes fraction", "data"),
        fontsize=7.5, color="gray", style="italic", va="bottom",
    )

    # ── Annotate extreme feedforward outliers (>TOP_MAX) ──────────────────────
    ff_name, ff_v, ff_err, ff_color, _ = datasets[0]  # feedforward is first
    outlier_mask = ff_err > TOP_MAX
    for xi, yi in zip(ff_v[outlier_mask], ff_err[outlier_mask]):
        ax.plot(xi, TOP_MAX * 0.98, marker="^", color=ff_color,
                markersize=9, zorder=5, clip_on=False)
        ax.text(
            xi, TOP_MAX * 0.88, f"{yi:.0f} N",
            color=ff_color, fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=ff_color, lw=0.8, alpha=0.85),
        )

    # ── Labels, legend, grid ──────────────────────────────────────────────────
    ax.set_xlim(0.0, max(v.max() for _, v, _, _, _ in datasets) * 1.02)
    ax.set_xlabel("EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)", fontsize=11)
    ax.set_ylabel("Avg |Force Z Error| (N)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Force Error Comparison — All Methods vs. EE Linear Speed",
        fontsize=13,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
