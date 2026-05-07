import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Polygon

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.linewidth": 0.4,
    "lines.linewidth": 0.6,
})

# ── parameters ────────────────────────────────────────────────────────────────
R         = 1.0
THETA_DEG = 50
THETA     = np.radians(THETA_DEG)
ALEN      = 0.44

cp     = np.array([R * np.sin(THETA), R * np.cos(THETA)])
n_hat  = np.array([ np.sin(THETA),  np.cos(THETA)])
t2_hat = np.array([ np.cos(THETA), -np.sin(THETA)])

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 8), dpi=300)
ax.set_aspect('equal')
ax.axis('off')

# ── cylinder circle ───────────────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), R, fill=False, color='k', lw=0.7))

# ── helper ────────────────────────────────────────────────────────────────────
def arrow(origin, direction, color):
    tip = origin + ALEN * direction
    ax.annotate('', xy=tip, xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=0.8, mutation_scale=6))
    return tip

# ── n̂(θ) — red ───────────────────────────────────────────────────────────────
tip_n = arrow(cp, n_hat, 'red')
ax.text(tip_n[0] + 0.07, tip_n[1] + 0.02,
        r'$\hat{n}(\theta)$', color='red', fontsize=6, va='center', ha='left')

# ── t̂₂(θ) — blue ─────────────────────────────────────────────────────────────
tip_t2 = arrow(cp, t2_hat, '#1f77b4')
ax.text(tip_t2[0] + 0.06, tip_t2[1] - 0.03,
        r'$\hat{t}_2(\theta)$', color='#1f77b4', fontsize=6, va='top', ha='left')

# ── t̂₁ — green ⊙ ─────────────────────────────────────────────────────────────
ax.plot(*cp, 'o', ms=7, mfc='white', mec='green', mew=1.2, zorder=5)
ax.plot(*cp, '.', ms=2.5, color='green', zorder=6)
ax.text(cp[0] - 0.20, cp[1] + 0.13,
        r'$\hat{t}_1$', color='green', fontsize=6, va='center', ha='center')

# ── right-angle mark ──────────────────────────────────────────────────────────
sq = 0.052
ax.add_patch(Polygon(
    [cp + sq*t2_hat, cp + sq*t2_hat + sq*n_hat, cp + sq*n_hat],
    closed=False, fill=False, ec='dimgray', lw=0.45))

# ── radius line + R label ─────────────────────────────────────────────────────
ax.plot([0, cp[0]], [0, cp[1]], 'k--', lw=0.5, alpha=0.6)
perp = np.array([-n_hat[1], n_hat[0]])
ax.text(*(cp * 0.5 + 0.09 * perp), r'$R$', fontsize=6.5, ha='center', va='center')

# ── p_c centre ────────────────────────────────────────────────────────────────
ax.plot(0, 0, 'k.', ms=3)
ax.text(0.06, -0.09, r'$p_c$', fontsize=6.5)

# ── θ arc — purple ────────────────────────────────────────────────────────────
arc_r = 0.28
ax.add_patch(Arc((0, 0), 2*arc_r, 2*arc_r, angle=0,
                 theta1=90-THETA_DEG, theta2=90, color='purple', lw=0.7))
mid_mpl = np.radians(90 - THETA_DEG / 2)
lbl_r   = arc_r + 0.10
ax.text(lbl_r * np.cos(mid_mpl), lbl_r * np.sin(mid_mpl),
        r'$\theta$', color='purple', fontsize=6.5, ha='center', va='center')

# ── coordinate axes ───────────────────────────────────────────────────────────
ax_len = 1.35
ap = dict(arrowstyle='->', color='k', lw=0.7, mutation_scale=6)
ax.annotate('', xy=(ax_len, 0),  xytext=(-0.15, 0),  arrowprops=ap)
ax.annotate('', xy=(0, ax_len),  xytext=(0, -0.15),  arrowprops=ap)
ax.text(ax_len + 0.07, 0,         r'$y$', fontsize=7.5, va='center')
ax.text(0.04,          ax_len+0.08, r'$z$', fontsize=7.5)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc='red',      ec='red',      label=r'$\hat{n}(\theta)$: surface normal'),
    mpatches.Patch(fc='#1f77b4',  ec='#1f77b4',  label=r'$\hat{t}_2(\theta)$: circ. tangent'),
    mpatches.Patch(fc='green',    ec='green',    label=r'$\hat{t}_1$: axial (out of page)'),
    mpatches.Patch(fc='purple',   ec='purple',   label=r'$\theta$: sweep angle from $+z$'),
]
ax.legend(handles=legend_items, loc='lower left',
          fontsize=4.2, framealpha=0.9, edgecolor='lightgray',
          borderpad=0.5, labelspacing=0.3, handlelength=0.8, handletextpad=0.4)

ax.set_xlim(-1.65, 1.80)
ax.set_ylim(-1.55, 1.75)

fig.savefig('cylinder_cross_section.pdf', dpi=300)
fig.savefig('cylinder_cross_section.png', dpi=300)
plt.show()
