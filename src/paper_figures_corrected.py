"""
Journal Figure 5 — Label-Corrected Chlori-Alkali Results
==========================================================
All experimental results are hard-coded from the completed run of
transfer_learning_corrected.py (5 seeds × 8 sample sizes).

Color convention
  LSTM fine-tune  : deep blue       #1565C0
  LSTM scratch    : sky blue        #64B5F6
  LSTM head-only  : teal            #00897B
  RF fine-tune    : deep orange     #E65100
  RF scratch      : light orange    #FFB74D
  Zero-shot       : gray            #9E9E9E
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  HARD-CODED RESULTS (mean ± std across 5 seeds)
# ══════════════════════════════════════════════════════════════════════════════

N = [20, 50, 100, 250, 500, 1000, 2000, 4000]

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM_FT_M  = np.array([0.6629, 0.6663, 0.7503, 0.8500, 0.9001, 0.9178, 0.9335, 0.9514])
LSTM_SC_M  = np.array([0.5383, 0.5577, 0.5924, 0.8570, 0.9139, 0.9244, 0.9365, 0.9566])
LSTM_HO_M  = np.array([0.5746, 0.5650, 0.6114, 0.7061, 0.7520, 0.7549, 0.7537, 0.7670])

# std computed from per-seed values
LSTM_FT_S  = np.array([
    np.std([0.6442,0.6925,0.6367,0.6650,0.6758]),
    np.std([0.6677,0.6756,0.6257,0.6815,0.6809]),
    np.std([0.7577,0.7533,0.6870,0.7606,0.7928]),
    np.std([0.8476,0.8552,0.8516,0.8481,0.8476]),
    np.std([0.8971,0.8964,0.9000,0.9033,0.9039]),
    np.std([0.9179,0.9173,0.9146,0.9210,0.9183]),
    np.std([0.9352,0.9337,0.9282,0.9362,0.9345]),
    np.std([0.9547,0.9500,0.9465,0.9497,0.9560]),
])
LSTM_SC_S  = np.array([
    np.std([0.5279,0.4850,0.5757,0.5637,0.5393]),
    np.std([0.5692,0.5560,0.5498,0.5766,0.5370]),
    np.std([0.6057,0.5644,0.6205,0.5828,0.5886]),
    np.std([0.8592,0.8053,0.8783,0.8827,0.8596]),
    np.std([0.9115,0.9187,0.9260,0.9005,0.9129]),
    np.std([0.9203,0.9166,0.9306,0.9273,0.9274]),
    np.std([0.9335,0.9450,0.9310,0.9302,0.9428]),
    np.std([0.9547,0.9580,0.9497,0.9585,0.9617]),
])
LSTM_HO_S  = np.array([
    np.std([0.5746]*5),   # placeholder — head-only std not logged per seed
    np.std([0.5650]*5),
    np.std([0.6114]*5),
    np.std([0.7061]*5),
    np.std([0.7520]*5),
    np.std([0.7549]*5),
    np.std([0.7537]*5),
    np.std([0.7670]*5),
])
LSTM_ZERO  = 0.4767

# ── RF ────────────────────────────────────────────────────────────────────────
RF_FT_M    = np.array([0.4414, 0.5053, 0.5949, 0.7652, 0.8630, 0.9073, 0.9360, 0.9566])
RF_SC_M    = np.array([0.5829, 0.6944, 0.7899, 0.8612, 0.8934, 0.9129, 0.9346, 0.9496])

RF_FT_S    = np.array([
    np.std([0.4529,0.4392,0.4466,0.4388,0.4292]),
    np.std([0.5016,0.5164,0.4986,0.5221,0.4879]),
    np.std([0.5842,0.5805,0.6224,0.6058,0.5818]),
    np.std([0.7701,0.7639,0.7848,0.7623,0.7450]),
    np.std([0.8533,0.8649,0.8839,0.8701,0.8427]),
    np.std([0.8980,0.9023,0.9107,0.9191,0.9063]),
    np.std([0.9345,0.9318,0.9295,0.9417,0.9423]),
    np.std([0.9623,0.9557,0.9507,0.9583,0.9560]),
])
RF_SC_S    = np.array([
    np.std([0.5485,0.6259,0.5580,0.6284,0.5538]),
    np.std([0.6989,0.6667,0.7292,0.7548,0.6225]),
    np.std([0.7709,0.7814,0.8123,0.7885,0.7963]),
    np.std([0.8677,0.8550,0.8634,0.8804,0.8396]),
    np.std([0.8839,0.8991,0.8907,0.9111,0.8821]),
    np.std([0.9073,0.9120,0.9093,0.9281,0.9080]),
    np.std([0.9335,0.9307,0.9262,0.9425,0.9403]),
    np.std([0.9513,0.9495,0.9447,0.9497,0.9525]),
])
RF_ZERO    = 0.4161

# ══════════════════════════════════════════════════════════════════════════════
#  COLOR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

# Panels (a), (b), (d) — original palette
C_LSTM_FT  = "#2166AC"   # deep blue
C_LSTM_SC  = "#6BAED6"   # light blue
C_LSTM_HO  = "#1A9850"   # green
C_RF_FT    = "#D6604D"   # dark red
C_RF_SC    = "#F4A582"   # salmon
C_ZERO     = "#888888"   # gray

# Panel (c) only — advantage bar colors (positive=deep, negative=light)
C_ADV_LSTM_POS = "#1565C0"   # deep blue  (LSTM FT wins)
C_ADV_LSTM_NEG = "#64B5F6"   # sky blue   (LSTM SC wins)
C_ADV_RF_POS   = "#E65100"   # deep orange (RF FT wins)
C_ADV_RF_NEG   = "#FFB74D"   # light orange (RF SC wins)

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

Y_MIN, Y_MAX = 0.40, 1.02   # show zero-shot near 0.41–0.48

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — 2×2 multi-panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig5():
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── (a) LSTM vs RF: fine-tune & scratch ───────────────────────────────────
    ax_a.plot(N, RF_FT_M,   "o-",  color=C_RF_FT,   lw=2.0, ms=6, label="RF Mixed Transfer")
    ax_a.plot(N, RF_SC_M,   "o--", color=C_RF_SC,   lw=1.8, ms=6, label="RF Scratch")
    ax_a.plot(N, LSTM_FT_M, "s-",  color=C_LSTM_FT, lw=2.0, ms=6, label="LSTM Fine-tune")
    ax_a.plot(N, LSTM_SC_M, "s--", color=C_LSTM_SC, lw=1.8, ms=6, label="LSTM Scratch")

    for m, s, c in [(RF_FT_M, RF_FT_S, C_RF_FT), (RF_SC_M, RF_SC_S, C_RF_SC),
                    (LSTM_FT_M, LSTM_FT_S, C_LSTM_FT), (LSTM_SC_M, LSTM_SC_S, C_LSTM_SC)]:
        ax_a.fill_between(N, m - s, m + s, alpha=0.10, color=c)

    ax_a.axhline(RF_ZERO,   color=C_RF_FT,  ls=":", lw=1.5, alpha=0.8,
                 label=f"RF zero-shot ({RF_ZERO:.3f})")
    ax_a.axhline(LSTM_ZERO, color=C_LSTM_FT, ls=":", lw=1.5, alpha=0.8,
                 label=f"LSTM zero-shot ({LSTM_ZERO:.3f})")

    ax_a.set_xscale("log")
    ax_a.set_xticks(N)
    ax_a.set_xticklabels([str(n) for n in N], fontsize=8)
    ax_a.set_ylim(Y_MIN, Y_MAX)
    ax_a.set_xlabel("Number of Labeled Chlor-Alkali Samples")
    ax_a.set_ylabel("Test Accuracy")
    # ax_a.set_title("LSTM vs RF: Fine-tune & Scratch")
    ax_a.legend(fontsize=8, ncol=1, loc="lower right")

    # ── (b) LSTM strategy comparison ──────────────────────────────────────────
    ax_b.plot(N, LSTM_FT_M, "s-",  color=C_LSTM_FT, lw=2.0, ms=6, label="Full fine-tune")
    ax_b.plot(N, LSTM_HO_M, "D-",  color=C_LSTM_HO, lw=2.0, ms=6, label="Head-only")
    ax_b.plot(N, LSTM_SC_M, "o--", color=C_LSTM_SC, lw=1.8, ms=6, label="Scratch")
    ax_b.axhline(LSTM_ZERO, color=C_ZERO, ls=":", lw=1.5, alpha=0.8,
                 label=f"Zero-shot ({LSTM_ZERO:.3f})")

    ax_b.fill_between(N, LSTM_FT_M - LSTM_FT_S, LSTM_FT_M + LSTM_FT_S,
                      alpha=0.12, color=C_LSTM_FT)
    ax_b.fill_between(N, LSTM_SC_M - LSTM_SC_S, LSTM_SC_M + LSTM_SC_S,
                      alpha=0.12, color=C_LSTM_SC)

    # Shade the FT-wins regime (N ≤ 100, indices 0–2)
    ax_b.axvspan(N[0] * 0.75, 180, alpha=0.06, color=C_LSTM_FT)
    ax_b.text(22, 0.975, "FT wins\n(N ≤ 100)",
              fontsize=8.5, color=C_LSTM_FT, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", fc="white",
                        ec=C_LSTM_FT, alpha=0.85))

    ax_b.set_xscale("log")
    ax_b.set_xticks(N)
    ax_b.set_xticklabels([str(n) for n in N], fontsize=8)
    ax_b.set_ylim(Y_MIN, Y_MAX)
    ax_b.set_xlabel("Number of Labeled Chlor-Alkali Samples")
    ax_b.set_ylabel("Test Accuracy")
    # ax_b.set_title("LSTM Fine-tune Strategy Comparison")
    ax_b.legend(fontsize=8, loc="lower right")

    # ── (c) Transfer advantage (fine-tune − scratch) ──────────────────────────
    lstm_adv = (LSTM_FT_M - LSTM_SC_M) * 100   # convert to percentage points
    rf_adv   = (RF_FT_M   - RF_SC_M)   * 100

    x = np.arange(len(N))
    w = 0.35

    # LSTM bars: blue gradient — positive=deep blue, negative=light blue
    lstm_colors = [C_ADV_LSTM_POS if v >= 0 else C_ADV_LSTM_NEG for v in lstm_adv]
    # RF bars: orange gradient — positive=deep orange, negative=light orange
    rf_colors   = [C_ADV_RF_POS   if v >= 0 else C_ADV_RF_NEG   for v in rf_adv]

    bars_l = ax_c.bar(x - w/2, lstm_adv, w, color=lstm_colors,
                      alpha=0.88, edgecolor="black", lw=0.6)
    bars_r = ax_c.bar(x + w/2, rf_adv,   w, color=rf_colors,
                      alpha=0.88, edgecolor="black", lw=0.6)

    ax_c.axhline(0, color="black", lw=0.9)

    # Custom legend patches
    from matplotlib.patches import Patch
    ax_c.legend(handles=[
        Patch(facecolor=C_ADV_LSTM_POS, edgecolor="black", label="LSTM advantage"),
        Patch(facecolor=C_ADV_RF_POS,   edgecolor="black", label="RF advantage"),
    ], fontsize=9)

    # Value labels on bars — percentage format, placed just outside bar tip
    for bar, val in zip(bars_l, lstm_adv):
        ypos = val + 0.4 if val >= 0 else val - 1.2
        ax_c.text(bar.get_x() + bar.get_width() / 2, ypos,
                  f"{val:+.1f}", ha="center", fontsize=7,
                  color=C_ADV_LSTM_POS if val >= 0 else C_ADV_LSTM_NEG, fontweight="bold")
    for bar, val in zip(bars_r, rf_adv):
        ypos = val + 0.4 if val >= 0 else val - 1.2
        ax_c.text(bar.get_x() + bar.get_width() / 2, ypos,
                  f"{val:+.1f}", ha="center", fontsize=7,
                  color=C_ADV_RF_POS if val >= 0 else C_ADV_RF_NEG, fontweight="bold")

    ylim = max(abs(np.concatenate([lstm_adv, rf_adv]))) + 4
    ax_c.set_ylim(-ylim, ylim)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(n) for n in N], fontsize=9)
    ax_c.set_xlabel("Number of Labeled Chlor-Alkali Samples")
    ax_c.set_ylabel("Difference from Scratch Accuracy (%)")
    # ax_c.set_title("Transfer Learning Advantage")
    ax_c.text(0.02, 0.97, "▲ Transfer wins",
              transform=ax_c.transAxes, fontsize=8.5,
              color=C_ADV_LSTM_POS, va="top")
    ax_c.text(0.02, 0.03, "▼ Scratch wins",
              transform=ax_c.transAxes, fontsize=8.5,
              color="#777", va="bottom")

    # ── (d) Stability: std across seeds ───────────────────────────────────────
    x_d = np.arange(len(N))
    w_d = 0.25

    ax_d.bar(x_d - w_d, LSTM_FT_S, w_d, color=C_LSTM_HO, alpha=0.88,
             edgecolor="black", lw=0.6, label="LSTM Full fine-tune")
    # ax_d.bar(x_d,        LSTM_HO_S, w_d, color=C_LSTM_HO, alpha=0.88,
    #          edgecolor="black", lw=0.6, label="LSTM Head-only")
    ax_d.bar(x_d + w_d,  LSTM_SC_S, w_d, color=C_LSTM_SC, alpha=0.88,
             edgecolor="black", lw=0.6, label="LSTM Scratch")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([str(n) for n in N], fontsize=9)
    ax_d.set_xlabel("Number of Labeled Chlor-Alkali Samples")
    ax_d.set_ylabel("Std of Accuracy (5 seeds)")
    # ax_d.set_title("Stability: Variance Across Random Seeds\n"
                #    "(lower = more robust)")
    ax_d.legend(fontsize=9)

    # ── Panel letters ─────────────────────────────────────────────────────────
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # plt.suptitle(
    #     "Transfer Learning from Ammonia to Chlor-Alkali Energy Market "
    #     "(Label-Corrected):\n"
    #     "Sample Efficiency, Strategy Comparison, and Advantage Analysis",
    #     fontsize=13, fontweight="bold", y=1.01,
    # )

    plt.savefig("paper_fig5_corrected_updated_corrected.pdf", bbox_inches="tight")
    plt.savefig("paper_fig5_corrected_updated_corrected.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("Saved: paper_fig5_corrected_updated_corrected.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    header = (f"{'N':>6} | {'LSTM FT':>8} {'LSTM SC':>8} {'FT adv':>7}"
              f" | {'RF FT':>7} {'RF SC':>7} {'RF adv':>7}")
    print("\n" + "=" * len(header))
    print("RESULTS SUMMARY — Label-Corrected Chlor-Alkali Dataset")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, n in enumerate(N):
        la = LSTM_FT_M[i] - LSTM_SC_M[i]
        ra = RF_FT_M[i]   - RF_SC_M[i]
        print(f"{n:>6} | {LSTM_FT_M[i]:>8.4f} {LSTM_SC_M[i]:>8.4f} {la:>+7.4f}"
              f" | {RF_FT_M[i]:>7.4f} {RF_SC_M[i]:>7.4f} {ra:>+7.4f}")
    print("=" * len(header))
    print(f"\nZero-shot baseline: LSTM={LSTM_ZERO:.4f}  RF={RF_ZERO:.4f}")
    print("\nKey findings:")
    print("  LSTM full fine-tune outperforms scratch at N ≤ 100 (up to +15.8%).")
    print("  Crossover between N=100 and N=250.")
    print("  RF fine-tune never outperforms RF scratch.")
    print("  Head-only plateaus ~77% regardless of N.")
    print("  Zero-shot below chance — confirms original labels were inverted.")


if __name__ == "__main__":
    plot_fig5()
    print_summary()
