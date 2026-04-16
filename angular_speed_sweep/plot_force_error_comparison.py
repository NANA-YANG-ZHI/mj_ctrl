"""Compare control-method metrics vs EE linear speed — 4 plots.

Each plot uses a piecewise linear y-scale:
  - y ≤ BREAK  : normal slope (detail zone, ~70 % of axis height)
  - y >  BREAK : compressed by COMPRESS factor (overview zone)

Extreme outliers (instability events) are clipped to TOP_MAX and annotated.

Plots generated (flat surface, no --slope-angle)
-------------------------------------------------
force_error_comparison.png     – avg_force_z_error  (N)
force_var_comparison.png       – var_force_z_error  (N²)
position_error_comparison.png  – avg_position_error (m)
position_var_comparison.png    – var_position_error (m²)

With --slope-angle 30 the output files are named with a _slope30 suffix.

Usage
-----
    python angular_speed_sweep/plot_force_error_comparison.py
    python angular_speed_sweep/plot_force_error_comparison.py --slope-angle 30
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

METHOD_KEYS = [
    ("Feedforward",      "feedforward",    "tab:blue",   "o"),
    ("Feedforward + PI", "feedforward_pi", "tab:orange", "s"),
    ("PD",               "pd",             "tab:green",  "^"),
    ("HFDC",             "paper",          "tab:red",    "D"),
    ("HFDC + PI",        "paper_pi",       "tab:purple", "P"),
]


def build_methods(slope_angle):
    """Return (label, data_dir, color, marker) list for the given slope angle.

    slope_angle=0.0 → flat-surface directories (feedforward/, paper_pi/, …)
    slope_angle=30  → slope directories        (slope30.0_feedforward/, …)
    """
    methods = []
    for label, key, color, marker in METHOD_KEYS:
        if slope_angle == 0.0:
            dir_name = key
        else:
            dir_name = f"slope{slope_angle}_{key}"
        data_dir = os.path.join(PLOTS_DIR, dir_name, "data")
        methods.append((label, data_dir, color, marker))
    return methods


def build_metrics(slope_angle):
    """Return METRICS list with output filenames adjusted for slope angle."""
    suffix = "" if slope_angle == 0.0 else f"_slope{slope_angle:g}"
    return [
        dict(
            col="avg_force_z_error",
            ylabel="Avg |Force Z Error| (N)",
            title="Avg Force Z Error",
            out=f"force_error_comparison{suffix}.png",
            break_y=3.0,
            compress=17.0,
            top_max=20.0,
            yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 8, 11, 14, 17, 20],
            fmt="{:.0f}",
        ),
        dict(
            col="var_force_z_error",
            ylabel="Var Force Z Error (N²)",
            title="Force Z Error Variance",
            out=f"force_var_comparison{suffix}.png",
            break_y=15.0,
            compress=15.0,
            top_max=120.0,
            yticks=[0, 3, 6, 9, 12, 15, 30, 50, 80, 100, 120],
            fmt="{:.0f}",
        ),
        dict(
            col="avg_position_error",
            ylabel="Avg Position Error (m)",
            title="Avg Position Error",
            out=f"position_error_comparison{suffix}.png",
            break_y=0.020,
            compress=25.0,
            top_max=0.50,
            yticks=[0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.10, 0.20, 0.35, 0.50],
            fmt="{:.3f}",
        ),
        dict(
            col="var_position_error",
            ylabel="Var Position Error (m²)",
            title="Position Error Variance",
            out=f"position_var_comparison{suffix}.png",
            break_y=2e-4,
            compress=30.0,
            top_max=0.055,
            yticks=[0, 5e-5, 1e-4, 1.5e-4, 2e-4, 5e-3, 0.01, 0.02, 0.035, 0.055],
            fmt="{:.4f}",
        ),
    ]




DT = 0.001                          # simulation timestep (ControllerConfig.dt)
FORCE_SKIP_SAMPLES = int(1.0 / DT)  # skip first 1 s for force metrics (= 1000 samples)


def load_all(data_dir):
    """Load per-speed .npz files and aggregate into metric arrays."""
    import glob
    npz_files = sorted(
        glob.glob(os.path.join(data_dir, "data_*.npz")),
        key=lambda p: float(re.match(r'data_([0-9.]+)', os.path.basename(p)).group(1))
    )
    rows = {k: [] for k in (
        "ee_linear_speed_m_s",
        "avg_force_z_error", "var_force_z_error",
        "avg_position_error", "var_position_error",
    )}
    for fpath in npz_files:
        d = np.load(fpath)
        rows["ee_linear_speed_m_s"].append(float(d["ee_linear_speed_m_s"]))

        # Force: skip first second
        fe = d["force_error"]
        fe_ss = fe[FORCE_SKIP_SAMPLES:]
        if len(fe_ss) > 0:
            rows["avg_force_z_error"].append(np.mean(np.abs(fe_ss)))
            rows["var_force_z_error"].append(np.var(fe_ss))
        else:
            rows["avg_force_z_error"].append(float("nan"))
            rows["var_force_z_error"].append(float("nan"))

        # Position: no skip
        pe = d["position_error"]
        if len(pe) > 0:
            rows["avg_position_error"].append(np.mean(pe))
            rows["var_position_error"].append(np.var(pe))
        else:
            rows["avg_position_error"].append(float("nan"))
            rows["var_position_error"].append(float("nan"))

    return {k: np.array(v) for k, v in rows.items()}


def make_piecewise(break_y, compress):
    """Return (forward, inverse) transform functions for set_yscale."""
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= break_y,
                        y,
                        break_y + (y - break_y) / compress)

    def inverse(z):
        z = np.asarray(z, dtype=float)
        return np.where(z <= break_y,
                        z,
                        break_y + (z - break_y) * compress)

    return forward, inverse


def make_plot(cfg, datasets, surface_label="flat surface"):
    col     = cfg["col"]
    break_y = cfg["break_y"]
    compress= cfg["compress"]
    top_max = cfg["top_max"]

    fig, ax = plt.subplots(figsize=(10, 11))

    # ── Plot each method ─────────────────────────────────────────────────────
    for name, data, color, marker in datasets:
        v   = data["ee_linear_speed_m_s"]
        err = data[col]
        err_plot = np.where(err > top_max, np.nan, err)
        ax.plot(v, err_plot, marker=marker, color=color, label=name,
                linewidth=1.5, markersize=4, markerfacecolor=color)

    # ── Piecewise y-scale ────────────────────────────────────────────────────
    fwd, inv = make_piecewise(break_y, compress)
    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(0.0, top_max)

    # ── Y-ticks ──────────────────────────────────────────────────────────────
    ax.set_yticks(cfg["yticks"])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")

    # ── Inflection guide line ────────────────────────────────────────────────
    ax.axhline(break_y, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.annotate(
        f"  scale ÷{compress:.0f} above",
        xy=(0.0, break_y), xycoords=("axes fraction", "data"),
        fontsize=7.5, color="gray", style="italic", va="bottom",
    )

    # ── Annotate outliers (per method) ───────────────────────────────────────
    y_annot = top_max * 0.92   # where to draw the triangle
    y_label = top_max * 0.81   # where to put the text box
    for name, data, color, marker in datasets:
        v   = data["ee_linear_speed_m_s"]
        err = data[col]
        mask = err > top_max
        if not mask.any():
            continue
        for xi, yi in zip(v[mask], err[mask]):
            ax.plot(xi, y_annot, marker="^", color=color,
                    markersize=8, zorder=5, clip_on=False)
            # format the label value
            lbl = _fmt_val(yi, cfg["col"])
            ax.text(xi, y_label, lbl, color=color, fontsize=7,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=color, lw=0.7, alpha=0.85))

    # ── Axes labels / legend / grid ──────────────────────────────────────────
    x_max = max(data["ee_linear_speed_m_s"].max() for _, data, _, _ in datasets)
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_xlabel("EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)", fontsize=11)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{cfg['title']} vs. EE Linear Speed  [{surface_label}]",
        fontsize=13,
    )

    out = os.path.join(PLOTS_DIR, cfg["out"])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {cfg['out']}")


def _fmt_val(v, col):
    """Format an outlier value with appropriate precision/units."""
    if col in ("avg_force_z_error", "var_force_z_error"):
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}k"
        return f"{v:.0f}"
    if col in ("avg_position_error", "var_position_error"):
        if abs(v) >= 0.1:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return f"{v:.3g}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare control-method metrics vs EE linear speed."
    )
    parser.add_argument(
        "--slope-angle",
        type=float,
        default=0.0,
        help="Slope angle in degrees (default: 0 = flat surface). "
             "Reads from slope<angle>_<method>/data directories and "
             "adds _slope<angle> suffix to output filenames.",
    )
    args = parser.parse_args()

    slope_angle = args.slope_angle
    surface_label = f"{slope_angle:g}° slope" if slope_angle != 0.0 else "flat surface"
    print(f"[INFO] Surface: {surface_label}")

    methods = build_methods(slope_angle)
    metrics = build_metrics(slope_angle)

    datasets = []
    for name, data_dir, color, marker in methods:
        if not os.path.isdir(data_dir):
            print(f"[SKIP] {name}: data dir not found ({data_dir})")
            continue
        datasets.append((name, load_all(data_dir), color, marker))

    if not datasets:
        print("No data found. Run experiments first.")
        return

    for cfg in metrics:
        make_plot(cfg, datasets, surface_label)


if __name__ == "__main__":
    main()
