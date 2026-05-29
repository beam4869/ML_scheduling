"""
Rebuild the 6-panel Case A figure from the CSVs exported by
combined_comparison_caseA.jl (run that Julia file first).

    julia combined_comparison_caseA.jl     # -> results_caseA/*.csv
    python plot_caseA.py                    # -> combined_comparison_caseA_py.png
"""

import os
import tempfile
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_caseA")

# ---- shared style (mirrors the Julia colour / linestyle scheme) ----
COL4 = ["royalblue", "seagreen", "darkorange", "crimson"]
STY4 = ["-", "--", ":", "-."]
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 15, 15, 11, 10

plt.rcParams.update({
    "font.size": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEG,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.7,
    "grid.color": "0.75",
})

# ---- load data ----
overview = pd.read_csv(os.path.join(RES, "overview.csv"))
highlight = pd.read_csv(os.path.join(RES, "highlight.csv"))
price_em = pd.read_csv(os.path.join(RES, "price_emission.csv"))
ammo_par = pd.read_csv(os.path.join(RES, "ammonia_pareto.csv"))
chlor_par = pd.read_csv(os.path.join(RES, "chlor_pareto.csv"))
ammo_prof = pd.read_csv(os.path.join(RES, "ammonia_profiles.csv"))
chlor_prof = pd.read_csv(os.path.join(RES, "chlor_profiles.csv"))
labels = pd.read_csv(os.path.join(RES, "labels.csv"))
LAB4 = list(labels.sort_values("k")["label"])

AMMO_YTICKS = [0, 500, 1000]
AMMO_YLIM = (-30, 1180)
CHLOR_YTICKS = [1e5, 2e5, 3e5]
CHLOR_YLIM = (3e4, 3.8e5)


def role_point(df, role):
    """Return (cost, emission) of the Pareto row tagged with `role`."""
    r = df[df["role"] == role]
    return (r["cost"].iloc[0], r["emission"].iloc[0]) if len(r) else (None, None)


def format_objective_axes(ax):
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def add_case_label(ax, label, color, loc="left"):
    x, ha = (0.02, "left") if loc == "left" else (0.98, "right")
    ax.text(x, 0.80, label, transform=ax.transAxes, ha=ha, va="center",
            color=color, fontsize=FS_LEG + 1, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5))


# ---- figure scaffold: 2 rows x 3 cols, col 3 = 4-stack ----
fig = plt.figure(figsize=(22, 13.5))
outer = GridSpec(2, 3, figure=fig, width_ratios=[1.08, 1.08, 1.18],
                 wspace=0.36, hspace=0.34,
                 left=0.055, right=0.985, top=0.95, bottom=0.065)

# ===== (a) overview scatter =====
ax_a = fig.add_subplot(outer[0, 0])
ax_a.scatter(overview["chlor_strength"], overview["ammonia_strength"],
             s=9, c="lightgray", alpha=0.55, edgecolors="none",
             label="All 8000 windows")
ax_a.scatter(highlight["chlor_strength"], highlight["ammonia_strength"],
             s=240, marker="*", c="crimson", edgecolors="black",
             linewidths=0.6, label=str(highlight["label"].iloc[0]), zorder=5)
ax_a.axvline(0.9775, color="black", linestyle="--", linewidth=1.5,
             alpha=0.75, zorder=4, label="Grouping threshold (0.9775)")
ax_a.axhline(0.9775, color="black", linestyle="--", linewidth=1.5,
             alpha=0.75, zorder=4)
ax_a.set_xlabel("Chlor-Alkali Strength")
ax_a.set_ylabel("Ammonia Strength")
ax_a.set_title("(a) Selected Cases Overview")
ax_a.legend(loc="lower left", fontsize=FS_LEG + 3, markerscale=1.2)

# ===== (b) ammonia Pareto frontier =====
ax_b = fig.add_subplot(outer[0, 1])
ax_b.scatter(ammo_par["cost"], ammo_par["emission"],
             s=34, c="purple", label="Pareto points")
for k, lab in enumerate(LAB4):
    x, y = role_point(ammo_par, lab)
    if x is not None:
        ax_b.scatter([x], [y], s=95, marker="D", color=COL4[k],
                     edgecolors="black", linewidths=0.6, label=lab, zorder=5)
ax_b.set_xlabel("Cost Objective ($)")
ax_b.set_ylabel("Emission Objective (kg CO$_2$)")
ax_b.set_title("(b) Ammonia Pareto Frontier")
ax_b.legend(loc="upper right", fontsize=FS_LEG + 3)
format_objective_axes(ax_b)

# ===== (c) ammonia NH3 / H2 profiles — 4-stack =====
inner_c = outer[0, 2].subgridspec(4, 1, hspace=0.10)
t = ammo_prof["time"]
profile_handles = [
    Line2D([0], [0], color="0.20", lw=2.2, ls="-", label="NH$_3$"),
    Line2D([0], [0], color="0.20", lw=1.7, ls="--", label="H$_2$"),
]
for k in range(4):
    ax = fig.add_subplot(inner_c[k])
    ax.plot(t, ammo_prof[f"nh3_{k+1}"], color=COL4[k], lw=2,
            ls="-", label="NH$_3$")
    ax.plot(t, ammo_prof[f"h2_{k+1}"], color=COL4[k], lw=1.5,
            ls="--", label="H$_2$")
    ax.set_ylim(AMMO_YLIM)
    ax.set_yticks(AMMO_YTICKS)
    ax.grid(axis="y")
    add_case_label(ax, LAB4[k], COL4[k], loc="right")
    if k == 0:
        ax.set_title("(c) Ammonia Profiles")
        ax.legend(handles=profile_handles, loc="lower right", ncol=2,
                  fontsize=FS_LEG + 3, handlelength=2.4)
    if k == 1:
        ax.set_ylabel("kg/h")
    if k == 3:
        ax.set_xlabel("Time Step (h)")
    else:
        ax.set_xticklabels([])

# ===== (d) price & emission =====
inner_d = outer[1, 0].subgridspec(2, 1, hspace=0.10)
ax_d1 = fig.add_subplot(inner_d[0])
ax_d2 = fig.add_subplot(inner_d[1], sharex=ax_d1)
ax_d1.plot(price_em["time"], price_em["price"],
           color="steelblue", lw=2, label="Price")
ax_d1.set_ylabel("Price ($/kWh)")
ax_d1.set_title("(d) Case A: Price & Grid Emission")
ax_d1.legend(loc="upper left")
ax_d1.tick_params(labelbottom=False)
ax_d2.plot(price_em["time"], price_em["gridCI"],
           color="firebrick", lw=2, ls="--", label="Emission")
ax_d2.set_xlabel("Time Step (h)")
ax_d2.set_ylabel("Emission (kg CO$_2$/kWh)")
ax_d2.legend(loc="lower left")

# ===== (e) chlor-alkali Pareto frontier =====
ax_e = fig.add_subplot(outer[1, 1])
ax_e.scatter(chlor_par["cost"], chlor_par["emission"],
             s=34, c="purple", label="Pareto points")
for k, lab in enumerate(LAB4):
    x, y = role_point(chlor_par, lab)
    if x is not None:
        ax_e.scatter([x], [y], s=95, marker="D", color=COL4[k],
                     edgecolors="black", linewidths=0.6, label=lab, zorder=5)
ax_e.set_xlabel("Cost Objective ($)")
ax_e.set_ylabel("Emission Objective (kg CO$_2$)")
ax_e.set_title("(e) Chlor-Alkali Pareto Frontier")
ax_e.legend(loc="upper right", fontsize=FS_LEG + 3)
format_objective_axes(ax_e)

# ===== (f) chlor-alkali current profiles — 4-stack =====
inner_f = outer[1, 2].subgridspec(4, 1, hspace=0.10)
tc = chlor_prof["time"]
for k in range(4):
    ax = fig.add_subplot(inner_f[k])
    ax.plot(tc, chlor_prof[f"I_{k+1}"], color=COL4[k], lw=2,
            ls=STY4[k], label=LAB4[k])
    ax.set_ylim(CHLOR_YLIM)
    ax.set_yticks(CHLOR_YTICKS)
    ax.grid(axis="y")
    add_case_label(ax, LAB4[k], COL4[k], loc="right" if k < 2 else "left")
    if k == 0:
        ax.set_title("(f) Chlor-Alkali Profiles")
    if k == 1:
        ax.set_ylabel("I (A)")
    if k == 3:
        ax.set_xlabel("Time Step (h)")
    else:
        ax.set_xticklabels([])

out_path = os.path.join(HERE, "combined_comparison_caseA_py.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
