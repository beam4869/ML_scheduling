"""
LSTM Fine-tune Strategy Comparison: Full vs Head-only vs Scratch
================================================================
Core question:
  When chlori-alkali labeled data is scarce, is it better to:
    (A) Fine-tune ALL layers (LSTM + Dense head) on chlori data?
    (B) Freeze the LSTM, only retrain the Dense head on chlori data?
    (C) Train from scratch on chlori data only?

Hypothesis:
  - At very small N (< ~200): Head-only may outperform full fine-tune
    because full fine-tune with too few samples can overfit / destroy
    the pretrained LSTM's temporal features (catastrophic forgetting).
  - At larger N: full fine-tune should catch up or surpass head-only
    as there is enough data to safely update all layers.

Methods:
  1. Full fine-tune   : unfreeze ALL layers, train on N chlori samples
  2. Head-only        : freeze LSTM layer, only train Dense head on N chlori
  3. Chlori scratch   : random init, train on N chlori samples only
  4. Zero-shot        : ammonia LSTM with no adaptation

Sample sizes: [20, 50, 100, 250, 500, 1000, 2000, 4000]
Seeds: [0, 7, 21, 38, 42]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SAMPLE_SIZES     = [20, 50, 100, 250, 500, 1000, 2000, 4000]
RANDOM_SEEDS     = [0, 7, 21, 38, 42]

LSTM_UNITS       = 64
DENSE_UNITS      = 64
DROPOUT          = 0.2
REC_DROPOUT      = 0.2

LR_PRETRAIN      = 5e-4
LR_FULL_FT       = 1e-4    # lower LR for full fine-tune to prevent forgetting
LR_HEAD_ONLY     = 1e-3    # higher LR OK for head-only (LSTM frozen, safe to go faster)
EPOCHS_PRETRAIN  = 30
EPOCHS_FINETUNE  = 50      # same budget for all methods
BATCH_SIZE       = 64
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
    X_cl_am = sc_am.transform(X_cl.reshape(-1, 96)).reshape(-1, 48, 2)
    return X_am_s, X_cl_s, X_cl_am


# ═══════════════════════════ MODEL BUILDERS ══════════════════════════════════

def build_lstm(lr):
    model = Sequential([
        Input(shape=(48, 2)),
        LSTM(LSTM_UNITS, recurrent_dropout=REC_DROPOUT, name="lstm"),
        Dropout(DROPOUT, name="dropout"),
        Dense(DENSE_UNITS, activation="relu", name="dense_hidden"),
        Dense(1, activation="sigmoid", name="dense_out"),
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def make_full_finetune(source_model):
    """Copy ALL weights, all layers trainable, low LR."""
    m = build_lstm(LR_FULL_FT)
    m.set_weights(source_model.get_weights())
    # All layers already trainable by default
    return m


def make_head_only(source_model):
    """Copy ALL weights, freeze LSTM layer only, higher LR for head."""
    m = build_lstm(LR_HEAD_ONLY)
    m.set_weights(source_model.get_weights())
    # Freeze LSTM (and dropout which has no weights, but freeze anyway)
    m.get_layer("lstm").trainable    = False
    m.get_layer("dropout").trainable = False
    m.compile(optimizer=Adam(LR_HEAD_ONLY),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ═══════════════════════════ SINGLE EXPERIMENT ═══════════════════════════════

def run_one(seed, n_train, source_model, X_cl_am, X_cl_cl, y_cl):
    """
    Evaluate all 4 methods for one (seed, n_train) pair.
    X_cl_am : chlori data scaled with AMMONIA scaler → for transfer methods
    X_cl_cl : chlori data scaled with CHLORI scaler  → for scratch
    """
    idx_all = np.arange(len(y_cl))
    idx_train, idx_test, _, y_te = train_test_split(
        idx_all, y_cl,
        train_size=n_train, random_state=seed, stratify=y_cl,
    )

    eff_batch = min(BATCH_SIZE, n_train)

    # ── 1. Zero-shot ──────────────────────────────────────────────────────────
    y_zero = (source_model.predict(
        X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_zero = accuracy_score(y_te, y_zero)

    # ── 2. Full fine-tune (all layers, low LR) ────────────────────────────────
    ft_full = make_full_finetune(source_model)
    ft_full.fit(X_cl_am[idx_train], y_cl[idx_train],
                epochs=EPOCHS_FINETUNE, batch_size=eff_batch, verbose=0)
    y_full = (ft_full.predict(
        X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_full = accuracy_score(y_te, y_full)

    # ── 3. Head-only (freeze LSTM, higher LR) ────────────────────────────────
    ft_head = make_head_only(source_model)
    ft_head.fit(X_cl_am[idx_train], y_cl[idx_train],
                epochs=EPOCHS_FINETUNE, batch_size=eff_batch, verbose=0)
    y_head = (ft_head.predict(
        X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_head = accuracy_score(y_te, y_head)

    # ── 4. Scratch (random init, chlori scaler) ───────────────────────────────
    scratch = build_lstm(LR_PRETRAIN)
    scratch.fit(X_cl_cl[idx_train], y_cl[idx_train],
                epochs=EPOCHS_FINETUNE, batch_size=eff_batch, verbose=0)
    y_sc = (scratch.predict(
        X_cl_cl[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_sc = accuracy_score(y_te, y_sc)

    return {
        "zero_shot":  acc_zero,
        "full_ft":    acc_full,
        "head_only":  acc_head,
        "scratch":    acc_sc,
    }


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Load & scale ──────────────────────────────────────────────────────────
    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()
    X_am_s, X_cl_cl, X_cl_am = make_scalers(X_am, X_cl)
    print(f"Ammonia: {X_am_s.shape}   Chlori: {X_cl_cl.shape}")

    # ── Pre-train source LSTM once on ALL ammonia ─────────────────────────────
    print(f"\nPre-training LSTM on ALL ammonia ({EPOCHS_PRETRAIN} epochs) …")
    source_model = build_lstm(LR_PRETRAIN)
    source_model.fit(X_am_s, y_am,
                     epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE,
                     validation_split=0.1, verbose=1)
    am_acc = accuracy_score(
        y_am,
        (source_model.predict(X_am_s, verbose=0) > 0.5).astype(int).flatten()
    )
    print(f"Ammonia train accuracy: {am_acc:.4f}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    methods = ["zero_shot", "full_ft", "head_only", "scratch"]
    results = {m: {n: [] for n in SAMPLE_SIZES} for m in methods}

    total = len(SAMPLE_SIZES) * len(RANDOM_SEEDS)
    done  = 0
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"Sample size N = {n_train}")
        for seed in RANDOM_SEEDS:
            res = run_one(seed, n_train, source_model,
                          X_cl_am, X_cl_cl, y_cl)
            for m in methods:
                results[m][n_train].append(res[m])
            done += 1
            print(f"  seed={seed}  "
                  f"full={res['full_ft']:.4f}  "
                  f"head={res['head_only']:.4f}  "
                  f"scratch={res['scratch']:.4f}  "
                  f"zero={res['zero_shot']:.4f}  "
                  f"[{done}/{total}]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    means = {m: [np.mean(results[m][n]) for n in SAMPLE_SIZES] for m in methods}
    stds  = {m: [np.std(results[m][n])  for n in SAMPLE_SIZES] for m in methods}

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'N':>6}  {'Full FT':>8}  {'Head-only':>9}  "
          f"{'Scratch':>8}  {'Zero-shot':>9}  {'Best':>10}")
    print("=" * 75)
    for i, n in enumerate(SAMPLE_SIZES):
        row = {m: means[m][i] for m in methods}
        best_m = max(["full_ft", "head_only", "scratch"], key=lambda m: row[m])
        best_label = {"full_ft": "Full FT", "head_only": "Head-only",
                      "scratch": "Scratch"}[best_m]
        print(f"{n:>6}  {row['full_ft']:>8.4f}  {row['head_only']:>9.4f}  "
              f"{row['scratch']:>8.4f}  {row['zero_shot']:>9.4f}  "
              f"{best_label:>10}")
    print("=" * 75)

    # ── Plot 1: Main learning curve (all 4 methods) ───────────────────────────
    style = {
        "full_ft":   ("o-",  "#4C72B0", "Full Fine-tune (all layers, LR=1e-4)"),
        "head_only": ("D-",  "#2ca02c", "Head-only (freeze LSTM, LR=1e-3)"),
        "scratch":   ("s--", "#DD8452", "Chlori-only Scratch"),
        "zero_shot": ("x:",  "#888888", "Zero-shot (no adaptation)"),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for m, (marker, color, label) in style.items():
        ax.plot(SAMPLE_SIZES, means[m], marker, color=color,
                linewidth=2.2, markersize=8, label=label)
        if m != "zero_shot":
            ax.fill_between(SAMPLE_SIZES,
                            np.array(means[m]) - np.array(stds[m]),
                            np.array(means[m]) + np.array(stds[m]),
                            alpha=0.14, color=color)

    # Annotate full_ft and head_only at each point
    for i, x in enumerate(SAMPLE_SIZES):
        ft_y   = means["full_ft"][i]
        head_y = means["head_only"][i]
        diff   = head_y - ft_y
        # Only annotate the gap when it is notable (> 0.005)
        if abs(diff) > 0.005:
            mid_y = (ft_y + head_y) / 2
            ax.annotate(f"Δ={diff:+.3f}", (x, mid_y),
                        textcoords="offset points",
                        xytext=(14, 0), ha="left", fontsize=8,
                        color="#2ca02c" if diff > 0 else "#4C72B0",
                        fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(n) for n in SAMPLE_SIZES], fontsize=11)
    ax.set_xlabel("Number of Labeled Chlori-Alkali Training Samples (log scale)",
                  fontsize=13)
    ax.set_ylabel("Accuracy on Chlori-Alkali Test Set", fontsize=13)
    ax.set_title(
        "Fine-tune Strategy Comparison: Full Fine-tune vs Head-only vs Scratch\n"
        f"(Ammonia pretrain={EPOCHS_PRETRAIN} ep, "
        f"finetune={EPOCHS_FINETUNE} ep, {len(RANDOM_SEEDS)} seeds)",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("finetune_strategy_comparison.png", dpi=150)
    plt.show()

    # ── Plot 2: Head-only vs Full fine-tune gap ───────────────────────────────
    gap = [means["head_only"][i] - means["full_ft"][i]
           for i in range(len(SAMPLE_SIZES))]
    colors_gap = ["#2ca02c" if g > 0 else "#4C72B0" for g in gap]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(n) for n in SAMPLE_SIZES], gap,
                  color=colors_gap, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_ylabel("Head-only Accuracy − Full Fine-tune Accuracy", fontsize=12)
    ax.set_title(
        "Head-only vs Full Fine-tune Gap\n"
        "(green = head-only wins, blue = full fine-tune wins)",
        fontsize=12,
    )
    for bar, val in zip(bars, gap):
        offset = 0.002 if val >= 0 else -0.007
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + offset, f"{val:+.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("head_only_vs_full_ft_gap.png", dpi=150)
    plt.show()

    # ── Plot 3: Variance comparison (std across seeds) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    x_idx = np.arange(len(SAMPLE_SIZES))
    width = 0.25
    for j, (m, color, label) in enumerate([
        ("full_ft",   "#4C72B0", "Full Fine-tune"),
        ("head_only", "#2ca02c", "Head-only"),
        ("scratch",   "#DD8452", "Scratch"),
    ]):
        ax.bar(x_idx + j * width, stds[m], width,
               color=color, alpha=0.8, edgecolor="black", label=label)
    ax.set_xticks(x_idx + width)
    ax.set_xticklabels([str(n) for n in SAMPLE_SIZES], fontsize=11)
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_ylabel("Std of Accuracy Across Seeds", fontsize=12)
    ax.set_title(
        "Stability Comparison: Std Deviation Across 5 Seeds\n"
        "(lower = more stable / robust to random initialization)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("finetune_strategy_std.png", dpi=150)
    plt.show()

    # ── Plot 4: Heatmap ───────────────────────────────────────────────────────
    heatmap_data = pd.DataFrame({
        "Full Fine-tune": means["full_ft"],
        "Head-only":      means["head_only"],
        "Scratch":        means["scratch"],
        "Zero-shot":      means["zero_shot"],
    }, index=SAMPLE_SIZES).T

    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"size": 11})
    ax.set_xlabel("Number of Chlori Training Samples", fontsize=12)
    ax.set_title("Accuracy Heatmap: Strategy × Sample Size", fontsize=12)
    plt.tight_layout()
    plt.savefig("finetune_strategy_heatmap.png", dpi=150)
    plt.show()

    # ── Final crossover summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY WINNER AT EACH SAMPLE SIZE")
    print("=" * 60)
    for i, n in enumerate(SAMPLE_SIZES):
        row = {m: means[m][i] for m in ["full_ft", "head_only", "scratch"]}
        best = max(row, key=row.get)
        label = {"full_ft": "Full Fine-tune",
                 "head_only": "Head-only", "scratch": "Scratch"}[best]
        ft_vs_head = "head>full" if means["head_only"][i] > means["full_ft"][i] \
                     else "full>head"
        print(f"  N={n:<5}  Best: {label:<18}  "
              f"({ft_vs_head}, "
              f"ft={means['full_ft'][i]:.4f}, "
              f"head={means['head_only'][i]:.4f}, "
              f"sc={means['scratch'][i]:.4f})")
    print("=" * 60)

    print("\nAll plots saved:")
    print("  finetune_strategy_comparison.png   ← main figure")
    print("  head_only_vs_full_ft_gap.png        ← gap analysis")
    print("  finetune_strategy_std.png           ← stability analysis")
    print("  finetune_strategy_heatmap.png       ← accuracy overview")
