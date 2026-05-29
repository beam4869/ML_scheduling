"""
RF Transfer Learning Sample Efficiency Study: Ammonia → Chlori-Alkali
======================================================================
Core question:
  How many labeled chlori-alkali samples does RF transfer learning need
  to outperform training from scratch?

Methods compared:
  1. RF Fine-tune (mixed)  : train RF on ALL ammonia + N chlori samples combined
  2. RF Chlori-only scratch: train RF only on N chlori samples
  3. RF Zero-shot          : train RF on ammonia only, test directly on chlori

Sample sizes swept: [20, 50, 100, 250, 500, 1000, 2000, 4000]
Results averaged over 5 random seeds per sample size.

Also produces a combined LSTM vs RF comparison plot using the
LSTM results from sample_size_learning_curve.py for direct comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SAMPLE_SIZES  = [20, 50, 100, 250, 500, 1000, 2000, 4000]
RANDOM_SEEDS  = [0, 7, 21, 38, 42]

# RF hyperparameters (consistent with transfer_learning_chlori.py)
N_ESTIMATORS  = 200
MAX_DEPTH     = None
MIN_SAMPLES_SPLIT = 2
N_JOBS        = -1     # use all CPU cores

# LSTM results from sample_size_learning_curve.py (for comparison plot)
# sample sizes: [100, 250, 500, 1000, 2000, 4000]
LSTM_SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 4000]
LSTM_FT_MEANS  = [0.6792, 0.7960, 0.8659, 0.9048, 0.9146, 0.9300]
LSTM_SC_MEANS  = [0.6346, 0.8866, 0.9091, 0.9293, 0.9372, 0.9580]
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════ DATA LOADING ════════════════════════════════════

def load_ammonia():
    df = pd.read_csv(
        "../ammonia_dataset/total dataset for ML with statistic numbers.csv"
    )
    # Use all 96 raw features + 9 stat features (cols 96–104) = 105 features
    # Label is at col 107
    X = df.iloc[:, :105].values
    y = df.iloc[:, 107].values.astype(int)
    return X, y


def load_chlori():
    df = pd.read_csv(
        "../chlori-alkali_dataset/Tokyo_ML_features_8000.csv"
    )
    X = df.iloc[:, :105].values
    y = df["label"].values.astype(int)
    return X, y


# ═══════════════════════════ RF BUILDERS ═════════════════════════════════════

def build_rf(random_state=42):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        n_jobs=N_JOBS,
        random_state=random_state,
    )


# ═══════════════════════════ SINGLE EXPERIMENT ═══════════════════════════════

def run_one(seed, n_train, X_am, y_am, X_cl, y_cl):
    """
    Train and evaluate RF for one (seed, n_train) combination.
    Returns dict with keys: 'finetune', 'scratch', 'zero_shot'
    """
    idx_all = np.arange(len(y_cl))
    idx_train, idx_test, _, y_te = train_test_split(
        idx_all, y_cl,
        train_size=n_train, random_state=seed, stratify=y_cl,
    )

    X_cl_train = X_cl[idx_train]
    X_cl_test  = X_cl[idx_test]
    y_cl_train = y_cl[idx_train]

    # ── Zero-shot: train RF on ALL ammonia, test on chlori ────────────────────
    rf_zero = build_rf(random_state=seed)
    rf_zero.fit(X_am, y_am)
    y_zero  = rf_zero.predict(X_cl_test)
    acc_zero = accuracy_score(y_te, y_zero)

    # ── Fine-tune (mixed): train RF on ALL ammonia + N chlori ────────────────
    X_mixed = np.vstack([X_am, X_cl_train])
    y_mixed = np.concatenate([y_am, y_cl_train])
    rf_ft = build_rf(random_state=seed)
    rf_ft.fit(X_mixed, y_mixed)
    y_ft   = rf_ft.predict(X_cl_test)
    acc_ft = accuracy_score(y_te, y_ft)

    # ── Chlori-only scratch: train RF on N chlori only ────────────────────────
    rf_sc = build_rf(random_state=seed)
    rf_sc.fit(X_cl_train, y_cl_train)
    y_sc   = rf_sc.predict(X_cl_test)
    acc_sc = accuracy_score(y_te, y_sc)

    return {"finetune": acc_ft, "scratch": acc_sc, "zero_shot": acc_zero}


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()
    print(f"Ammonia : {X_am.shape}  label: 0={(y_am==0).mean():.1%}  1={(y_am==1).mean():.1%}")
    print(f"Chlori  : {X_cl.shape}  label: 0={(y_cl==0).mean():.1%}  1={(y_cl==1).mean():.1%}")

    # ── Sweep sample sizes × seeds ────────────────────────────────────────────
    results = {
        "finetune":  {n: [] for n in SAMPLE_SIZES},
        "scratch":   {n: [] for n in SAMPLE_SIZES},
        "zero_shot": {n: [] for n in SAMPLE_SIZES},
    }

    total = len(SAMPLE_SIZES) * len(RANDOM_SEEDS)
    done  = 0
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*55}")
        print(f"Sample size N = {n_train}")
        for seed in RANDOM_SEEDS:
            res = run_one(seed, n_train, X_am, y_am, X_cl, y_cl)
            results["finetune"][n_train].append(res["finetune"])
            results["scratch"][n_train].append(res["scratch"])
            results["zero_shot"][n_train].append(res["zero_shot"])
            done += 1
            print(f"  seed={seed}  fine-tune={res['finetune']:.4f}"
                  f"  scratch={res['scratch']:.4f}"
                  f"  zero-shot={res['zero_shot']:.4f}"
                  f"  [{done}/{total}]")

    # ── Compute means / stds ──────────────────────────────────────────────────
    ft_means  = [np.mean(results["finetune"][n])  for n in SAMPLE_SIZES]
    ft_stds   = [np.std(results["finetune"][n])   for n in SAMPLE_SIZES]
    sc_means  = [np.mean(results["scratch"][n])   for n in SAMPLE_SIZES]
    sc_stds   = [np.std(results["scratch"][n])    for n in SAMPLE_SIZES]
    zs_means  = [np.mean(results["zero_shot"][n]) for n in SAMPLE_SIZES]
    advantages = [ft_means[i] - sc_means[i] for i in range(len(SAMPLE_SIZES))]

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'N':>6}  {'FT mean':>8}  {'FT std':>7}  "
          f"{'SC mean':>8}  {'SC std':>7}  {'Advantage':>9}")
    print("=" * 72)
    for i, n in enumerate(SAMPLE_SIZES):
        marker = " ◄ FT wins" if advantages[i] > 0 else ""
        print(f"{n:>6}  {ft_means[i]:>8.4f}  {ft_stds[i]:>7.4f}  "
              f"{sc_means[i]:>8.4f}  {sc_stds[i]:>7.4f}  "
              f"{advantages[i]:>+9.4f}{marker}")
    print("=" * 72)
    print(f"Zero-shot (flat): {np.mean(zs_means):.4f}")

    # ─────────────────────────── PLOTS ───────────────────────────────────────

    # ── Plot 1: RF learning curve ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(SAMPLE_SIZES, ft_means, "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, label="RF Fine-tune (ammonia + N chlori)")
    ax.fill_between(SAMPLE_SIZES,
                    np.array(ft_means) - np.array(ft_stds),
                    np.array(ft_means) + np.array(ft_stds),
                    alpha=0.18, color="#4C72B0")

    ax.plot(SAMPLE_SIZES, sc_means, "s--", color="#DD8452", linewidth=2.5,
            markersize=8, label="RF Chlori-only (from scratch)")
    ax.fill_between(SAMPLE_SIZES,
                    np.array(sc_means) - np.array(sc_stds),
                    np.array(sc_means) + np.array(sc_stds),
                    alpha=0.18, color="#DD8452")

    ax.axhline(np.mean(zs_means), color="gray", linestyle=":",
               linewidth=1.8, label=f"RF Zero-shot ({np.mean(zs_means):.3f})")

    # Shade fine-tune winning region
    for i in range(len(SAMPLE_SIZES) - 1):
        if advantages[i] > 0:
            ax.axvspan(SAMPLE_SIZES[i], SAMPLE_SIZES[i+1],
                       alpha=0.07, color="#4C72B0")

    # Annotate points
    for x, y_ft, y_sc in zip(SAMPLE_SIZES, ft_means, sc_means):
        ax.annotate(f"{y_ft:.3f}", (x, y_ft), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color="#2b5297",
                    fontweight="bold")
        ax.annotate(f"{y_sc:.3f}", (x, y_sc), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=9, color="#b35a1f",
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(n) for n in SAMPLE_SIZES], fontsize=11)
    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples (log scale)",
                  fontsize=13)
    ax.set_ylabel("Accuracy on Chlori-Alkali Test Set", fontsize=13)
    ax.set_title(
        "RF Sample Efficiency: Transfer Learning vs Training from Scratch\n"
        f"(N_estimators={N_ESTIMATORS}, {len(RANDOM_SEEDS)} seeds)",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rf_learning_curve.png", dpi=150)
    plt.show()

    # ── Plot 2: RF advantage bar ──────────────────────────────────────────────
    colors_adv = ["#4C72B0" if a > 0 else "#DD8452" for a in advantages]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(n) for n in SAMPLE_SIZES], advantages,
                  color=colors_adv, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_ylabel("Fine-tune Accuracy − Scratch Accuracy", fontsize=12)
    ax.set_title(
        "RF Transfer Learning Advantage over Training from Scratch\n"
        "(positive = fine-tune wins, negative = scratch wins)",
        fontsize=12,
    )
    for bar, val in zip(bars, advantages):
        offset = 0.003 if val >= 0 else -0.010
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + offset, f"{val:+.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("rf_learning_curve_advantage.png", dpi=150)
    plt.show()

    # ── Plot 3: RF vs LSTM side-by-side (shared sample sizes only) ───────────
    # Use only sizes present in both experiments
    shared_sizes = [n for n in SAMPLE_SIZES if n in LSTM_SAMPLE_SIZES]
    rf_ft_shared = [ft_means[SAMPLE_SIZES.index(n)]   for n in shared_sizes]
    rf_sc_shared = [sc_means[SAMPLE_SIZES.index(n)]   for n in shared_sizes]
    lstm_ft_sh   = [LSTM_FT_MEANS[LSTM_SAMPLE_SIZES.index(n)] for n in shared_sizes]
    lstm_sc_sh   = [LSTM_SC_MEANS[LSTM_SAMPLE_SIZES.index(n)] for n in shared_sizes]

    fig, ax = plt.subplots(figsize=(11, 6))

    # LSTM lines
    ax.plot(shared_sizes, lstm_ft_sh, "o-",  color="#4C72B0", linewidth=2.5,
            markersize=8, label="LSTM Fine-tune")
    ax.plot(shared_sizes, lstm_sc_sh, "o--", color="#4C72B0", linewidth=2.0,
            markersize=8, alpha=0.55, label="LSTM Scratch")

    # RF lines
    ax.plot(shared_sizes, rf_ft_shared, "s-",  color="#DD8452", linewidth=2.5,
            markersize=8, label="RF Fine-tune")
    ax.plot(shared_sizes, rf_sc_shared, "s--", color="#DD8452", linewidth=2.0,
            markersize=8, alpha=0.55, label="RF Scratch")

    # Zero-shot reference lines
    ax.axhline(np.mean(zs_means), color="#DD8452", linestyle=":",
               linewidth=1.5, alpha=0.7,
               label=f"RF Zero-shot ({np.mean(zs_means):.3f})")
    ax.axhline(0.510, color="#4C72B0", linestyle=":",
               linewidth=1.5, alpha=0.7,
               label="LSTM Zero-shot (~0.510)")

    # Annotate fine-tune values
    for x, y_lstm, y_rf in zip(shared_sizes, lstm_ft_sh, rf_ft_shared):
        ax.annotate(f"{y_lstm:.3f}", (x, y_lstm),
                    textcoords="offset points", xytext=(-18, 5),
                    ha="center", fontsize=8, color="#2b5297")
        ax.annotate(f"{y_rf:.3f}", (x, y_rf),
                    textcoords="offset points", xytext=(18, -12),
                    ha="center", fontsize=8, color="#b35a1f")

    ax.set_xscale("log")
    ax.set_xticks(shared_sizes)
    ax.set_xticklabels([str(n) for n in shared_sizes], fontsize=11)
    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples (log scale)",
                  fontsize=13)
    ax.set_ylabel("Accuracy on Chlori-Alkali Test Set", fontsize=13)
    ax.set_title(
        "LSTM vs RF: Transfer Learning (Fine-tune) and Scratch Comparison\n"
        f"Solid = fine-tune, Dashed = scratch, {len(RANDOM_SEEDS)} seeds",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rf_vs_lstm_learning_curve.png", dpi=150)
    plt.show()

    # ── Plot 4: Heatmap ───────────────────────────────────────────────────────
    heatmap_data = pd.DataFrame({
        "RF Fine-tune":  ft_means,
        "RF Scratch":    sc_means,
        "RF Zero-shot":  zs_means,
    }, index=SAMPLE_SIZES).T

    fig, ax = plt.subplots(figsize=(11, 3))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"size": 11})
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_title("RF Accuracy Heatmap: Method × Sample Size", fontsize=12)
    plt.tight_layout()
    plt.savefig("rf_learning_curve_heatmap.png", dpi=150)
    plt.show()

    # ── Print crossover analysis ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CROSSOVER ANALYSIS")
    print("=" * 55)
    print("RF transfer learning (fine-tune) advantage:")
    for i, n in enumerate(SAMPLE_SIZES):
        symbol = "✓ FT wins" if advantages[i] > 0 else "✗ Scratch wins"
        print(f"  N={n:<5}  {advantages[i]:+.4f}  {symbol}")

    print("\nLSTM transfer learning (fine-tune) advantage (from prev exp):")
    lstm_adv = [LSTM_FT_MEANS[i] - LSTM_SC_MEANS[i]
                for i in range(len(LSTM_SAMPLE_SIZES))]
    for n, adv in zip(LSTM_SAMPLE_SIZES, lstm_adv):
        symbol = "✓ FT wins" if adv > 0 else "✗ Scratch wins"
        print(f"  N={n:<5}  {adv:+.4f}  {symbol}")

    print("\nAll plots saved:")
    print("  rf_learning_curve.png")
    print("  rf_learning_curve_advantage.png")
    print("  rf_vs_lstm_learning_curve.png    ← main comparison figure")
    print("  rf_learning_curve_heatmap.png")
