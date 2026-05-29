"""
Transfer Learning Sample Efficiency Study: Ammonia → Chlori-Alkali
===================================================================
Core question:
  How many labeled chlori-alkali samples does each method need to
  achieve a given accuracy?

This is the most valuable transfer learning experiment — it shows
the regime where pre-training on ammonia data actually helps.

Methods compared:
  1. Standard Fine-tune  : pre-train on ALL ammonia → fine-tune on N chlori samples
  2. Chlori-only scratch : train LSTM from scratch on N chlori samples only
  3. Zero-shot           : ammonia model with NO adaptation (flat baseline)

Sample sizes swept: [100, 250, 500, 1000, 2000, 4000]
Results averaged over 5 random seeds per sample size.

Expected insight:
  - At small N (100–500): fine-tune should outperform scratch
    (ammonia pre-training acts as regularization / knowledge prior)
  - At large N (2000–4000): scratch catches up or surpasses fine-tune
    (enough chlori data to learn from scratch)
  - The crossover point reveals the practical value of transfer learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SAMPLE_SIZES    = [100, 250, 500, 1000, 2000, 4000]   # chlori training sizes
RANDOM_SEEDS    = [0, 7, 21, 38, 42]

LSTM_UNITS      = 64
DENSE_UNITS     = 64
DROPOUT         = 0.2
REC_DROPOUT     = 0.2

LR_PRETRAIN     = 5e-4
LR_FINETUNE     = 1e-4
EPOCHS_PRETRAIN = 30     # ammonia pre-training (run once)
EPOCHS_FINETUNE = 50     # fine-tune / scratch epochs for chlori
BATCH_SIZE      = 64
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════ DATA LOADING ════════════════════════════════════

def load_ammonia():
    df = pd.read_csv(
        "../ammonia_dataset/total dataset for ML with statistic numbers.csv"
    )
    X = df.iloc[:, :96].values.reshape(-1, 48, 2)
    y = df.iloc[:, 107].values.astype(int)
    return X, y


def load_chlori():
    df = pd.read_csv(
        "../chlori-alkali_dataset/Tokyo_ML_features_8000.csv"
    )
    X = df.iloc[:, :96].values.reshape(-1, 48, 2)
    y = df["label"].values.astype(int)
    return X, y


def make_scalers(X_am, X_cl):
    sc_am = StandardScaler()
    sc_cl = StandardScaler()
    X_am_s  = sc_am.fit_transform(X_am.reshape(-1, 96)).reshape(-1, 48, 2)
    X_cl_s  = sc_cl.fit_transform(X_cl.reshape(-1, 96)).reshape(-1, 48, 2)
    # Chlori scaled with AMMONIA scaler → for transfer methods
    X_cl_am = sc_am.transform(X_cl.reshape(-1, 96)).reshape(-1, 48, 2)
    return X_am_s, X_cl_s, X_cl_am


# ═══════════════════════════ MODEL BUILDERS ══════════════════════════════════

def build_lstm(lr):
    model = Sequential([
        Input(shape=(48, 2)),
        LSTM(LSTM_UNITS, recurrent_dropout=REC_DROPOUT),
        Dropout(DROPOUT),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def clone_all_weights(source_model, lr):
    """Copy ALL weights (LSTM + Dense head) from source into a new model."""
    new_model = build_lstm(lr)
    new_model.set_weights(source_model.get_weights())
    return new_model


# ═══════════════════════════ SINGLE EXPERIMENT ═══════════════════════════════

def run_one(seed, n_train, source_model,
            X_cl_am, X_cl_cl, y_cl):
    """
    Train and evaluate for one (seed, n_train) combination.
    Returns dict with keys: 'finetune', 'scratch', 'zero_shot'
    """
    # Always use the full remaining data as test set
    idx_all = np.arange(len(y_cl))
    idx_train, idx_test, _, y_te = train_test_split(
        idx_all, y_cl,
        train_size=n_train, random_state=seed, stratify=y_cl,
    )

    # ── Zero-shot (no adaptation at all) ─────────────────────────────────────
    y_zero = (source_model.predict(
        X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_zero = accuracy_score(y_te, y_zero)

    # ── Standard Fine-tune (all layers, ammonia pretrained init) ─────────────
    ft = clone_all_weights(source_model, LR_FINETUNE)
    ft.fit(X_cl_am[idx_train], y_cl[idx_train],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train),
           verbose=0)
    y_ft = (ft.predict(X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_ft = accuracy_score(y_te, y_ft)

    # ── Chlori-only from scratch ──────────────────────────────────────────────
    sc = build_lstm(LR_PRETRAIN)
    sc.fit(X_cl_cl[idx_train], y_cl[idx_train],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train),
           verbose=0)
    y_sc = (sc.predict(X_cl_cl[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_sc = accuracy_score(y_te, y_sc)

    return {"finetune": acc_ft, "scratch": acc_sc, "zero_shot": acc_zero}


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()
    X_am_s, X_cl_cl, X_cl_am = make_scalers(X_am, X_cl)
    print(f"Ammonia: {X_am_s.shape}   Chlori: {X_cl_cl.shape}")
    print(f"Chlori label distribution: "
          f"0={( y_cl==0).mean():.1%}  1={(y_cl==1).mean():.1%}")

    # ── Pre-train source LSTM once on ALL ammonia ─────────────────────────────
    print(f"\nPre-training LSTM on ALL ammonia ({EPOCHS_PRETRAIN} epochs) …")
    source_model = build_lstm(LR_PRETRAIN)
    pretrain_history = source_model.fit(
        X_am_s, y_am,
        epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE,
        validation_split=0.1, verbose=1,
    )
    am_acc = accuracy_score(
        y_am,
        (source_model.predict(X_am_s, verbose=0) > 0.5).astype(int).flatten()
    )
    print(f"Ammonia train accuracy: {am_acc:.4f}")

    # ── Sweep over sample sizes and seeds ────────────────────────────────────
    # results[method][n_train] = list of accuracies across seeds
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
            res = run_one(seed, n_train, source_model,
                          X_cl_am, X_cl_cl, y_cl)
            results["finetune"][n_train].append(res["finetune"])
            results["scratch"][n_train].append(res["scratch"])
            results["zero_shot"][n_train].append(res["zero_shot"])
            done += 1
            print(f"  seed={seed}  fine-tune={res['finetune']:.4f}"
                  f"  scratch={res['scratch']:.4f}"
                  f"  zero-shot={res['zero_shot']:.4f}"
                  f"  [{done}/{total}]")

    # ── Summary table ─────────────────────────────────────────────────────────
    ft_means  = [np.mean(results["finetune"][n])  for n in SAMPLE_SIZES]
    ft_stds   = [np.std(results["finetune"][n])   for n in SAMPLE_SIZES]
    sc_means  = [np.mean(results["scratch"][n])   for n in SAMPLE_SIZES]
    sc_stds   = [np.std(results["scratch"][n])    for n in SAMPLE_SIZES]
    zs_means  = [np.mean(results["zero_shot"][n]) for n in SAMPLE_SIZES]

    print("\n" + "=" * 72)
    print(f"{'N':>6}  {'FT mean':>8}  {'FT std':>7}  "
          f"{'SC mean':>8}  {'SC std':>7}  {'Advantage':>9}")
    print("=" * 72)
    for i, n in enumerate(SAMPLE_SIZES):
        advantage = ft_means[i] - sc_means[i]
        marker = " ◄ FT wins" if advantage > 0 else ""
        print(f"{n:>6}  {ft_means[i]:>8.4f}  {ft_stds[i]:>7.4f}  "
              f"{sc_means[i]:>8.4f}  {sc_stds[i]:>7.4f}  "
              f"{advantage:>+9.4f}{marker}")
    print("=" * 72)
    print(f"Zero-shot (flat): {np.mean(zs_means):.4f}")

    # ── Plot 1: Main learning curve ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(SAMPLE_SIZES, ft_means, "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, label="Standard Fine-tune (ammonia pretrained)")
    ax.fill_between(SAMPLE_SIZES,
                    np.array(ft_means) - np.array(ft_stds),
                    np.array(ft_means) + np.array(ft_stds),
                    alpha=0.18, color="#4C72B0")

    ax.plot(SAMPLE_SIZES, sc_means, "s--", color="#DD8452", linewidth=2.5,
            markersize=8, label="Chlori-only (from scratch)")
    ax.fill_between(SAMPLE_SIZES,
                    np.array(sc_means) - np.array(sc_stds),
                    np.array(sc_means) + np.array(sc_stds),
                    alpha=0.18, color="#DD8452")

    ax.axhline(np.mean(zs_means), color="gray", linestyle=":",
               linewidth=1.8, label=f"Zero-shot baseline ({np.mean(zs_means):.3f})")

    # Shade the region where fine-tune beats scratch
    for i in range(len(SAMPLE_SIZES) - 1):
        if ft_means[i] > sc_means[i]:
            ax.axvspan(SAMPLE_SIZES[i], SAMPLE_SIZES[i+1],
                       alpha=0.07, color="#4C72B0")

    # Annotate each point
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
        "Sample Efficiency: Transfer Learning vs Training from Scratch\n"
        f"(Ammonia pretrain={EPOCHS_PRETRAIN} ep, "
        f"fine-tune/scratch={EPOCHS_FINETUNE} ep, {len(RANDOM_SEEDS)} seeds)",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve_sample_efficiency.png", dpi=150)
    plt.show()

    # ── Plot 2: Advantage of fine-tune over scratch ───────────────────────────
    advantages = [ft_means[i] - sc_means[i] for i in range(len(SAMPLE_SIZES))]
    colors_adv = ["#4C72B0" if a > 0 else "#DD8452" for a in advantages]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([str(n) for n in SAMPLE_SIZES], advantages,
                  color=colors_adv, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_ylabel("Fine-tune Accuracy − Scratch Accuracy", fontsize=12)
    ax.set_title(
        "Transfer Learning Advantage over Training from Scratch\n"
        "(positive = fine-tune wins, negative = scratch wins)",
        fontsize=12,
    )
    for bar, val in zip(bars, advantages):
        offset = 0.003 if val >= 0 else -0.008
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + offset, f"{val:+.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve_advantage.png", dpi=150)
    plt.show()

    # ── Plot 3: All seeds scatter (transparency shows variance) ───────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        ft_vals = [results["finetune"][n][seed_idx]  for n in SAMPLE_SIZES]
        sc_vals = [results["scratch"][n][seed_idx]   for n in SAMPLE_SIZES]
        ax.plot(SAMPLE_SIZES, ft_vals, "o-", color="#4C72B0",
                alpha=0.35, linewidth=1.2, markersize=5)
        ax.plot(SAMPLE_SIZES, sc_vals, "s--", color="#DD8452",
                alpha=0.35, linewidth=1.2, markersize=5)

    # Bold mean lines on top
    ax.plot(SAMPLE_SIZES, ft_means, "o-", color="#4C72B0", linewidth=3,
            markersize=9, label="Fine-tune mean", zorder=5)
    ax.plot(SAMPLE_SIZES, sc_means, "s--", color="#DD8452", linewidth=3,
            markersize=9, label="Scratch mean", zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(n) for n in SAMPLE_SIZES], fontsize=11)
    ax.set_xlabel("Number of Labeled Chlori Training Samples (log scale)",
                  fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title(
        "Learning Curves — All Seeds (thin=individual, thick=mean)\n"
        f"{len(RANDOM_SEEDS)} seeds × {len(SAMPLE_SIZES)} sample sizes",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Add dummy legend entries for fine-tune vs scratch
    from matplotlib.lines import Line2D
    legend_extra = [
        Line2D([0], [0], color="#4C72B0", linewidth=1, alpha=0.4,
               label="Fine-tune (individual seeds)"),
        Line2D([0], [0], color="#DD8452", linewidth=1, linestyle="--",
               alpha=0.4, label="Scratch (individual seeds)"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_extra,
              labels=ax.get_legend_handles_labels()[1] +
              ["Fine-tune (individual seeds)", "Scratch (individual seeds)"],
              fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("learning_curve_all_seeds.png", dpi=150)
    plt.show()

    # ── Plot 4: Heatmap — accuracy by (method × sample size) ─────────────────
    heatmap_data = pd.DataFrame({
        "Fine-tune":  ft_means,
        "Scratch":    sc_means,
        "Zero-shot":  zs_means,
    }, index=SAMPLE_SIZES).T

    fig, ax = plt.subplots(figsize=(9, 3))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"size": 11})
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_title("Accuracy Heatmap: Method × Sample Size", fontsize=12)
    plt.tight_layout()
    plt.savefig("learning_curve_heatmap.png", dpi=150)
    plt.show()

    print("\nAll plots saved:")
    print("  learning_curve_sample_efficiency.png  ← main figure")
    print("  learning_curve_advantage.png           ← transfer advantage")
    print("  learning_curve_all_seeds.png           ← variance across seeds")
    print("  learning_curve_heatmap.png             ← accuracy overview")
