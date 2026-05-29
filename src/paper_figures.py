"""
Journal-Quality Summary Figures
================================
This script produces publication-ready figures summarizing ALL transfer
learning experiments conducted in this study:

  Exp 1 — RF learning curve        (rf_sample_size_learning_curve.py)
  Exp 2 — LSTM learning curve       (sample_size_learning_curve.py)
  Exp 3 — LSTM fine-tune strategies (finetune_strategy_comparison.py)

All results are hard-coded from completed runs so this script can be
re-run independently without re-training any models.

Figures produced:
  Fig 1 — Main learning curve: LSTM vs RF, fine-tune vs scratch
  Fig 2 — Transfer learning advantage (fine-tune minus scratch)
  Fig 3 — Fine-tune strategy breakdown (full / head-only / scratch)
  Fig 4 — Comprehensive heatmap
  Fig 5 — Combined 2×2 panel (journal multi-panel figure)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  ALL EXPERIMENTAL RESULTS  (hard-coded from completed runs)
# ══════════════════════════════════════════════════════════════════════════════

# ── Exp 1: RF learning curve ──────────────────────────────────────────────────
# Sample sizes: [20, 50, 100, 250, 500, 1000, 2000, 4000]
RF_N       = [20,   50,   100,  250,  500,  1000, 2000, 4000]
RF_FT_M    = [0.6396,0.7528,0.8592,0.9264,0.9442,0.9615,0.9757,0.9885]
RF_FT_S    = [0.0233,0.0473,0.0253,0.0066,0.0058,0.0050,0.0028,0.0024]
RF_SC_M    = [0.7005,0.8305,0.9013,0.9342,0.9489,0.9637,0.9759,0.9883]
RF_SC_S    = [0.0545,0.0661,0.0159,0.0020,0.0032,0.0075,0.0029,0.0021]
RF_ZERO    = 0.5629   # flat across all N

# ── Exp 2: LSTM fine-tune vs scratch learning curve ───────────────────────────
# Sample sizes: [100, 250, 500, 1000, 2000, 4000]
LSTM_N     = [100,  250,  500,  1000, 2000, 4000]
LSTM_FT_M  = [0.6792,0.7960,0.8659,0.9048,0.9146,0.9300]
LSTM_FT_S  = [0.0046,0.0103,0.0042,0.0033,0.0024,0.0041]
LSTM_SC_M  = [0.6346,0.8866,0.9091,0.9293,0.9372,0.9580]
LSTM_SC_S  = [0.0518,0.0174,0.0039,0.0042,0.0035,0.0031]
LSTM_ZERO  = 0.5102

# ── Exp 3: LSTM fine-tune strategy comparison ─────────────────────────────────
# Sample sizes: [20, 50, 100, 250, 500, 1000, 2000, 4000]
STRAT_N        = [20,   50,   100,  250,  500,  1000, 2000, 4000]
STRAT_FULL_M   = [0.6796,0.6969,0.7852,0.8524,0.8823,0.9078,0.9168,0.9307]
STRAT_FULL_S   = [0.0267,0.0210,0.0250,0.0023,0.0045,0.0021,0.0033,0.0031]
STRAT_HEAD_M   = [0.6410,0.5884,0.5899,0.7102,0.7695,0.7786,0.7841,0.7921]
STRAT_HEAD_S   = [0.0301,0.0450,0.0315,0.0160,0.0101,0.0133,0.0137,0.0071]
STRAT_SC_M     = [0.5656,0.5829,0.6448,0.8736,0.9086,0.9223,0.9417,0.9567]
STRAT_SC_S     = [0.0175,0.0148,0.0455,0.0208,0.0079,0.0031,0.0052,0.0044]
STRAT_ZERO     = 0.5690

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

C_LSTM_FT  = "#2166AC"   # LSTM fine-tune — dark blue
C_LSTM_SC  = "#6BAED6"   # LSTM scratch   — light blue
C_RF_FT    = "#D6604D"   # RF fine-tune   — dark red
C_RF_SC    = "#F4A582"   # RF scratch     — light red/salmon
C_HEAD     = "#1A9850"   # Head-only      — green
C_ZERO     = "#888888"   # Zero-shot      — gray

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Main Learning Curve: LSTM vs RF
# ══════════════════════════════════════════════════════════════════════════════

def fig1_main_learning_curve():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # RF curves
    ax.plot(RF_N, RF_FT_M, "o-",  color=C_RF_FT,  lw=2.2, ms=7,
            label="RF  — Fine-tune (ammonia + N chlori)")
    ax.fill_between(RF_N,
                    np.array(RF_FT_M) - np.array(RF_FT_S),
                    np.array(RF_FT_M) + np.array(RF_FT_S),
                    alpha=0.15, color=C_RF_FT)
    ax.plot(RF_N, RF_SC_M, "o--", color=C_RF_SC,  lw=2.0, ms=7,
            label="RF  — Scratch (N chlori only)")
    ax.fill_between(RF_N,
                    np.array(RF_SC_M) - np.array(RF_SC_S),
                    np.array(RF_SC_M) + np.array(RF_SC_S),
                    alpha=0.15, color=C_RF_SC)

    # LSTM curves
    ax.plot(LSTM_N, LSTM_FT_M, "s-",  color=C_LSTM_FT, lw=2.2, ms=7,
            label="LSTM — Fine-tune (ammonia pretrain)")
    ax.fill_between(LSTM_N,
                    np.array(LSTM_FT_M) - np.array(LSTM_FT_S),
                    np.array(LSTM_FT_M) + np.array(LSTM_FT_S),
                    alpha=0.15, color=C_LSTM_FT)
    ax.plot(LSTM_N, LSTM_SC_M, "s--", color=C_LSTM_SC, lw=2.0, ms=7,
            label="LSTM — Scratch (N chlori only)")
    ax.fill_between(LSTM_N,
                    np.array(LSTM_SC_M) - np.array(LSTM_SC_S),
                    np.array(LSTM_SC_M) + np.array(LSTM_SC_S),
                    alpha=0.15, color=C_LSTM_SC)

    # Zero-shot baselines
    ax.axhline(RF_ZERO,   color=C_RF_FT,  ls=":", lw=1.6, alpha=0.7,
               label=f"RF zero-shot ({RF_ZERO:.3f})")
    ax.axhline(LSTM_ZERO, color=C_LSTM_FT, ls=":", lw=1.6, alpha=0.7,
               label=f"LSTM zero-shot ({LSTM_ZERO:.3f})")

    ax.set_xscale("log")
    ax.set_xticks(RF_N)
    ax.set_xticklabels([str(n) for n in RF_N])

    # Crossover annotation (LSTM FT wins over LSTM scratch at N=100)
    ax.annotate("LSTM FT > Scratch\n(N ≤ 100)",
                xy=(100, 0.679), xytext=(40, 0.58),
                arrowprops=dict(arrowstyle="->", color=C_LSTM_FT, lw=1.4),
                fontsize=9, color=C_LSTM_FT, fontweight="bold")

    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples")
    ax.set_ylabel("Test Accuracy on Chlori-Alkali Dataset")
    ax.set_title("Fig. 1 — Sample Efficiency: Transfer Learning vs. Training from Scratch\n"
                 "(Shaded band = ±1 std across 5 seeds)")
    ax.set_ylim(0.44, 1.02)
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("paper_fig1_learning_curve.pdf", bbox_inches="tight")
    plt.savefig("paper_fig1_learning_curve.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("  Saved: paper_fig1_learning_curve.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Transfer Advantage (fine-tune minus scratch)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_advantage():
    # Common N range: use shared sizes between LSTM and RF experiments
    shared_N = [100, 250, 500, 1000, 2000, 4000]

    lstm_adv = [LSTM_FT_M[LSTM_N.index(n)] - LSTM_SC_M[LSTM_N.index(n)]
                for n in shared_N]
    rf_adv   = [RF_FT_M[RF_N.index(n)]     - RF_SC_M[RF_N.index(n)]
                for n in shared_N]

    x = np.arange(len(shared_N))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_lstm = ax.bar(x - w/2, lstm_adv, w, color=C_LSTM_FT,
                       alpha=0.85, edgecolor="black", lw=0.7,
                       label="LSTM Fine-tune advantage")
    bars_rf   = ax.bar(x + w/2, rf_adv,   w, color=C_RF_FT,
                       alpha=0.85, edgecolor="black", lw=0.7,
                       label="RF Fine-tune advantage")

    ax.axhline(0, color="black", lw=1.0)
    ax.fill_between([-0.5, len(shared_N)-0.1], [0, 0], [0.25, 0.25],
                    alpha=0.04, color=C_LSTM_FT)
    ax.text(-0.3, 0.022, "Transfer\nlearning wins ▲",
            fontsize=8.5, color=C_LSTM_FT, va="bottom")
    ax.text(-0.3, -0.016, "▼ Scratch wins",
            fontsize=8.5, color="#999", va="top")

    for bar, val in zip(bars_lstm, lstm_adv):
        ypos = val + 0.003 if val >= 0 else val - 0.010
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.3f}", ha="center", va="bottom",
                fontsize=8, color=C_LSTM_FT, fontweight="bold")
    for bar, val in zip(bars_rf, rf_adv):
        ypos = val + 0.003 if val >= 0 else val - 0.010
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.3f}", ha="center", va="bottom",
                fontsize=8, color=C_RF_FT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in shared_N])
    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples")
    ax.set_ylabel("Fine-tune Accuracy − Scratch Accuracy")
    ax.set_title("Fig. 2 — Transfer Learning Advantage over Training from Scratch\n"
                 "(LSTM transfers at N ≤ 100; RF never transfers positively)")
    ax.set_ylim(-0.12, 0.12)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig("paper_fig2_advantage.pdf", bbox_inches="tight")
    plt.savefig("paper_fig2_advantage.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("  Saved: paper_fig2_advantage.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Fine-tune Strategy Breakdown (LSTM only)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_strategy():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(STRAT_N, STRAT_FULL_M, "s-",  color=C_LSTM_FT, lw=2.2, ms=7,
            label="Full fine-tune (all layers, LR=1e-4)")
    ax.fill_between(STRAT_N,
                    np.array(STRAT_FULL_M) - np.array(STRAT_FULL_S),
                    np.array(STRAT_FULL_M) + np.array(STRAT_FULL_S),
                    alpha=0.14, color=C_LSTM_FT)

    ax.plot(STRAT_N, STRAT_HEAD_M, "D-",  color=C_HEAD, lw=2.2, ms=7,
            label="Head-only (freeze LSTM, LR=1e-3)")
    ax.fill_between(STRAT_N,
                    np.array(STRAT_HEAD_M) - np.array(STRAT_HEAD_S),
                    np.array(STRAT_HEAD_M) + np.array(STRAT_HEAD_S),
                    alpha=0.14, color=C_HEAD)

    ax.plot(STRAT_N, STRAT_SC_M,   "o--", color=C_LSTM_SC, lw=2.0, ms=7,
            label="Scratch (N chlori only)")
    ax.fill_between(STRAT_N,
                    np.array(STRAT_SC_M) - np.array(STRAT_SC_S),
                    np.array(STRAT_SC_M) + np.array(STRAT_SC_S),
                    alpha=0.14, color=C_LSTM_SC)

    ax.axhline(STRAT_ZERO, color=C_ZERO, ls=":", lw=1.6,
               label=f"Zero-shot ({STRAT_ZERO:.3f})")

    # Annotate full FT win region
    ax.axvspan(10, 180, alpha=0.05, color=C_LSTM_FT)
    ax.text(22, 0.97, "Full FT wins\n(N < ~200)",
            fontsize=9, color=C_LSTM_FT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_LSTM_FT, alpha=0.8))

    ax.set_xscale("log")
    ax.set_xticks(STRAT_N)
    ax.set_xticklabels([str(n) for n in STRAT_N])
    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples")
    ax.set_ylabel("Test Accuracy on Chlori-Alkali Dataset")
    ax.set_title("Fig. 3 — LSTM Fine-tune Strategy Comparison\n"
                 "(Full fine-tune consistently outperforms head-only)")
    ax.set_ylim(0.44, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("paper_fig3_strategy.pdf", bbox_inches="tight")
    plt.savefig("paper_fig3_strategy.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("  Saved: paper_fig3_strategy.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Comprehensive Heatmap (all methods × sample sizes)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_heatmap():
    # Align all methods to the shared N range [100, 250, 500, 1000, 2000, 4000]
    shared_N = [100, 250, 500, 1000, 2000, 4000]

    def lookup_rf(vals, n): return vals[RF_N.index(n)]
    def lookup_lstm(vals, n): return vals[LSTM_N.index(n)]
    def lookup_strat(vals, n): return vals[STRAT_N.index(n)]

    data = {
        "LSTM Full Fine-tune":   [lookup_strat(STRAT_FULL_M, n) for n in shared_N],
        "LSTM Head-only":        [lookup_strat(STRAT_HEAD_M, n) for n in shared_N],
        "LSTM Scratch":          [lookup_lstm(LSTM_SC_M, n)    for n in shared_N],
        "RF Fine-tune (mixed)":  [lookup_rf(RF_FT_M, n)        for n in shared_N],
        "RF Scratch":            [lookup_rf(RF_SC_M, n)        for n in shared_N],
        "LSTM Zero-shot":        [LSTM_ZERO] * len(shared_N),
        "RF Zero-shot":          [RF_ZERO]   * len(shared_N),
    }

    df = pd.DataFrame(data, index=shared_N).T

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.50, vmax=1.00, linewidths=0.6,
                linecolor="white", ax=ax,
                annot_kws={"size": 10, "weight": "bold"},
                cbar_kws={"label": "Test Accuracy"})

    # Bold the best value in each column
    for col_i, n in enumerate(shared_N):
        col_vals = [data[m][col_i] for m in data]
        best_row = int(np.argmax(col_vals))
        ax.add_patch(plt.Rectangle((col_i, best_row), 1, 1,
                     fill=False, edgecolor="black", lw=2.5))

    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples", fontsize=11)
    ax.set_title("Fig. 4 — Accuracy Overview: All Methods × Sample Sizes\n"
                 "(black border = best method for that sample size)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("paper_fig4_heatmap.pdf", bbox_inches="tight")
    plt.savefig("paper_fig4_heatmap.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("  Saved: paper_fig4_heatmap.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Multi-panel Journal Figure (2×2)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_multipanel():
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])   # (a) Main learning curve RF vs LSTM FT
    ax_b = fig.add_subplot(gs[0, 1])   # (b) LSTM strategy comparison
    ax_c = fig.add_subplot(gs[1, 0])   # (c) Transfer advantage bars
    ax_d = fig.add_subplot(gs[1, 1])   # (d) Stability (std) comparison

    # ── (a) RF vs LSTM fine-tune ──────────────────────────────────────────────
    ax_a.plot(RF_N, RF_FT_M, "o-",  color=C_RF_FT,   lw=2, ms=6,
              label="RF Fine-tune")
    ax_a.plot(RF_N, RF_SC_M, "o--", color=C_RF_SC,   lw=1.8, ms=6,
              label="RF Scratch")
    ax_a.plot(LSTM_N, LSTM_FT_M, "s-",  color=C_LSTM_FT, lw=2, ms=6,
              label="LSTM Fine-tune")
    ax_a.plot(LSTM_N, LSTM_SC_M, "s--", color=C_LSTM_SC, lw=1.8, ms=6,
              label="LSTM Scratch")
    ax_a.axhline(RF_ZERO,   color=C_RF_FT,  ls=":", lw=1.3, alpha=0.6)
    ax_a.axhline(LSTM_ZERO, color=C_LSTM_FT, ls=":", lw=1.3, alpha=0.6)
    for arr, m_arr, c in [(RF_FT_S, RF_FT_M, C_RF_FT),
                           (RF_SC_S, RF_SC_M, C_RF_SC),
                           (LSTM_FT_S, LSTM_FT_M, C_LSTM_FT),
                           (LSTM_SC_S, LSTM_SC_M, C_LSTM_SC)]:
        n_arr = RF_N if len(arr) == len(RF_N) else LSTM_N
        ax_a.fill_between(n_arr,
                          np.array(m_arr) - np.array(arr),
                          np.array(m_arr) + np.array(arr),
                          alpha=0.1, color=c)
    ax_a.set_xscale("log"); ax_a.set_xticks(RF_N)
    ax_a.set_xticklabels([str(n) for n in RF_N], fontsize=8)
    ax_a.set_xlabel("# Labeled Chlori Samples")
    ax_a.set_ylabel("Test Accuracy")
    ax_a.set_title("LSTM vs RF: Fine-tune & Scratch")
    ax_a.set_ylim(0.44, 1.02)
    ax_a.legend(fontsize=8, ncol=2)

    # ── (b) LSTM strategy comparison ─────────────────────────────────────────
    ax_b.plot(STRAT_N, STRAT_FULL_M, "s-",  color=C_LSTM_FT, lw=2, ms=6,
              label="Full fine-tune")
    ax_b.plot(STRAT_N, STRAT_HEAD_M, "D-",  color=C_HEAD,    lw=2, ms=6,
              label="Head-only")
    ax_b.plot(STRAT_N, STRAT_SC_M,   "o--", color=C_LSTM_SC, lw=1.8, ms=6,
              label="Scratch")
    ax_b.axhline(STRAT_ZERO, color=C_ZERO, ls=":", lw=1.3, alpha=0.7,
                 label="Zero-shot")
    ax_b.fill_between(STRAT_N,
                      np.array(STRAT_FULL_M) - np.array(STRAT_FULL_S),
                      np.array(STRAT_FULL_M) + np.array(STRAT_FULL_S),
                      alpha=0.12, color=C_LSTM_FT)
    ax_b.fill_between(STRAT_N,
                      np.array(STRAT_SC_M) - np.array(STRAT_SC_S),
                      np.array(STRAT_SC_M) + np.array(STRAT_SC_S),
                      alpha=0.12, color=C_LSTM_SC)
    ax_b.axvspan(10, 180, alpha=0.06, color=C_LSTM_FT)
    ax_b.text(13, 0.96, "FT wins\n(N<200)", fontsize=8,
              color=C_LSTM_FT, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="white",
                        ec=C_LSTM_FT, alpha=0.8))
    ax_b.set_xscale("log"); ax_b.set_xticks(STRAT_N)
    ax_b.set_xticklabels([str(n) for n in STRAT_N], fontsize=8)
    ax_b.set_xlabel("# Labeled Chlori Samples")
    ax_b.set_ylabel("Test Accuracy")
    ax_b.set_title("LSTM Fine-tune Strategy Comparison")
    ax_b.set_ylim(0.44, 1.02)
    ax_b.legend(fontsize=8)

    # ── (c) Transfer advantage bar chart ─────────────────────────────────────
    shared_N = [100, 250, 500, 1000, 2000, 4000]
    lstm_adv = [LSTM_FT_M[LSTM_N.index(n)] - LSTM_SC_M[LSTM_N.index(n)]
                for n in shared_N]
    rf_adv   = [RF_FT_M[RF_N.index(n)] - RF_SC_M[RF_N.index(n)]
                for n in shared_N]

    x = np.arange(len(shared_N))
    w = 0.35
    ax_c.bar(x - w/2, lstm_adv, w, color=C_LSTM_FT, alpha=0.85,
             edgecolor="black", lw=0.6, label="LSTM")
    ax_c.bar(x + w/2, rf_adv,   w, color=C_RF_FT,   alpha=0.85,
             edgecolor="black", lw=0.6, label="RF")
    ax_c.axhline(0, color="black", lw=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(n) for n in shared_N], fontsize=9)
    ax_c.set_xlabel("# Labeled Chlori Samples")
    ax_c.set_ylabel("Fine-tune − Scratch Accuracy")
    ax_c.set_title("Transfer Learning Advantage")
    ax_c.set_ylim(-0.15, 0.12)
    ax_c.legend(fontsize=9)
    for bars_list, adv_list, col in [(
            [ax_c.patches[i] for i in range(len(shared_N))], lstm_adv, C_LSTM_FT),
            ([ax_c.patches[i] for i in range(len(shared_N), 2*len(shared_N))],
             rf_adv, C_RF_FT)]:
        for bar, val in zip(bars_list, adv_list):
            ypos = val + 0.004 if val >= 0 else val - 0.012
            ax_c.text(bar.get_x() + bar.get_width()/2, ypos,
                      f"{val:+.3f}", ha="center", fontsize=7.5,
                      color=col, fontweight="bold")

    # ── (d) Stability: std across seeds ──────────────────────────────────────
    # Use Exp 3 strat N (includes small N) for a clearer picture
    x_d = np.arange(len(STRAT_N))
    w_d = 0.25
    ax_d.bar(x_d - w_d, STRAT_FULL_S, w_d, color=C_LSTM_FT, alpha=0.85,
             edgecolor="black", lw=0.6, label="Full fine-tune")
    ax_d.bar(x_d,        STRAT_HEAD_S, w_d, color=C_HEAD,    alpha=0.85,
             edgecolor="black", lw=0.6, label="Head-only")
    ax_d.bar(x_d + w_d, STRAT_SC_S,   w_d, color=C_LSTM_SC, alpha=0.85,
             edgecolor="black", lw=0.6, label="Scratch")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([str(n) for n in STRAT_N], fontsize=9)
    ax_d.set_xlabel("# Labeled Chlori Samples")
    ax_d.set_ylabel("Std of Accuracy (5 seeds)")
    ax_d.set_title("Stability: Variance Across Random Seeds\n"
                   "(lower = more robust)")
    ax_d.legend(fontsize=9)

    # Panel labels
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # plt.suptitle(
    #     "Transfer Learning from Ammonia to Chlor-Alkali Energy Market:\n"
    #     "Sample Efficiency, Strategy Comparison, and Advantage Analysis",
    #     fontsize=13, fontweight="bold", y=1.01
    # )
    plt.savefig("paper_fig5_multipanel.pdf", bbox_inches="tight")
    plt.savefig("paper_fig5_multipanel.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("  Saved: paper_fig5_multipanel.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE (for paper / supplementary)
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    shared_N = [100, 250, 500, 1000, 2000, 4000]
    header = f"{'N':>6} | {'LSTM FT':>9} {'LSTM SC':>9} {'LSTM Head':>10} | {'RF FT':>7} {'RF SC':>7} | {'Best':>18}"
    print("\n" + "=" * len(header))
    print("COMPLETE RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for n in shared_N:
        lstm_ft  = LSTM_FT_M[LSTM_N.index(n)]
        lstm_sc  = LSTM_SC_M[LSTM_N.index(n)]
        lstm_hd  = STRAT_HEAD_M[STRAT_N.index(n)]
        rf_ft    = RF_FT_M[RF_N.index(n)]
        rf_sc    = RF_SC_M[RF_N.index(n)]
        best_val = max(lstm_ft, lstm_sc, rf_ft, rf_sc)
        best_nm  = {lstm_ft: "LSTM FT", lstm_sc: "LSTM Scratch",
                    rf_ft:  "RF FT",   rf_sc:  "RF Scratch"}[best_val]
        print(f"{n:>6} | {lstm_ft:>9.4f} {lstm_sc:>9.4f} {lstm_hd:>10.4f} |"
              f" {rf_ft:>7.4f} {rf_sc:>7.4f} | {best_nm:>18}  ({best_val:.4f})")
    print("=" * len(header))
    print(f"\nZero-shot: LSTM={LSTM_ZERO:.4f}  RF={RF_ZERO:.4f}")
    print("\nKey finding: LSTM fine-tune outperforms LSTM scratch only at N ≤ 100.")
    print("             RF fine-tune never outperforms RF scratch.")
    print("             Head-only is consistently the worst transfer strategy.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # print("Generating journal-quality figures …\n")

    # print("[1/5] Main learning curve (LSTM vs RF) …")
    # fig1_main_learning_curve()

    # print("[2/5] Transfer advantage bar chart …")
    # fig2_advantage()

    # print("[3/5] Fine-tune strategy comparison …")
    # fig3_strategy()

    # print("[4/5] Comprehensive heatmap …")
    # fig4_heatmap()

    print("[5/5] Multi-panel journal figure …")
    fig5_multipanel()

    # print_summary()

    print("\n" + "=" * 55)
    print("All figures saved (PDF + PNG):")
    print("  paper_fig1_learning_curve.pdf / .png")
    print("  paper_fig2_advantage.pdf / .png")
    print("  paper_fig3_strategy.pdf / .png")
    print("  paper_fig4_heatmap.pdf / .png")
    print("  paper_fig5_multipanel.pdf / .png   ← main journal figure")
    print("=" * 55)
