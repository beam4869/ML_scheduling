"""
Standard Fine-tune Transfer Learning: Ammonia → Chlori-Alkali
=============================================================
Strategy:
  Phase 1 – Pre-train LSTM on ALL ammonia data (source domain).
  Phase 2 – Fine-tune ALL layers on 2000 labeled chlori samples.

This experiment sweeps over different fine-tune epoch counts to find
the sweet spot and show how accuracy improves with more training.

Comparison baselines included:
  - Chlori-only from scratch (same 2000 samples, same epochs)
  - Ammonia LSTM direct (no adaptation, zero-shot)
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
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
N_CHLORI_TRAIN   = 2000
RANDOM_SEEDS     = [0, 7, 21, 38, 42]

LSTM_UNITS       = 64
DENSE_UNITS      = 64
DROPOUT          = 0.2
REC_DROPOUT      = 0.2

LR_PRETRAIN      = 5e-4       # ammonia pre-training
LR_FINETUNE      = 1e-4       # chlori fine-tuning (smaller to avoid forgetting)
EPOCHS_PRETRAIN  = 30         # pre-train longer on ammonia for better features
BATCH_SIZE       = 64

# Sweep over these fine-tune epoch counts to find the sweet spot
FINETUNE_EPOCH_SWEEP = [5, 10, 20, 30, 50, 80]
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


def scale(X_train, X_test=None):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train.reshape(-1, 96)).reshape(-1, 48, 2)
    if X_test is not None:
        X_te = sc.transform(X_test.reshape(-1, 96)).reshape(-1, 48, 2)
        return X_tr, X_te, sc
    return X_tr, sc


# ═══════════════════════════ MODEL ═══════════════════════════════════════════

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


def clone_pretrained(source_model, lr):
    """Build a fresh model and copy ALL weights from source."""
    new_model = build_lstm(lr)
    new_model.set_weights(source_model.get_weights())
    return new_model


# ═══════════════════════════ EXPERIMENTS ═════════════════════════════════════

def run_seed(seed, source_model, X_am_s, am_scaler,
             X_cl, y_cl, X_cl_am_s, X_cl_cl_s):
    """
    For one seed, evaluate:
      - Zero-shot (ammonia model directly on chlori, no fine-tune)
      - Fine-tune sweep (varying epoch counts)
      - Chlori-only scratch (same epoch counts as fine-tune sweep)
    """
    idx_all = np.arange(len(X_cl))
    idx_train, idx_test, _, y_te = train_test_split(
        idx_all, y_cl,
        train_size=N_CHLORI_TRAIN, random_state=seed, stratify=y_cl,
    )

    results = {"zero_shot": None, "finetune": {}, "scratch": {}}

    # ── Zero-shot baseline ────────────────────────────────────────────────────
    y_pred_zero = (source_model.predict(
        X_cl_am_s[idx_test], verbose=0) > 0.5).astype(int).flatten()
    results["zero_shot"] = accuracy_score(y_te, y_pred_zero)

    # ── Fine-tune sweep ───────────────────────────────────────────────────────
    for n_epochs in FINETUNE_EPOCH_SWEEP:
        ft_model = clone_pretrained(source_model, LR_FINETUNE)
        ft_model.fit(
            X_cl_am_s[idx_train], y_cl[idx_train],
            epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0,
        )
        y_pred = (ft_model.predict(
            X_cl_am_s[idx_test], verbose=0) > 0.5).astype(int).flatten()
        results["finetune"][n_epochs] = accuracy_score(y_te, y_pred)

    # ── Chlori-only scratch sweep (same epoch counts, fair comparison) ────────
    for n_epochs in FINETUNE_EPOCH_SWEEP:
        sc_model = build_lstm(LR_PRETRAIN)
        sc_model.fit(
            X_cl_cl_s[idx_train], y_cl[idx_train],
            epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0,
        )
        y_pred = (sc_model.predict(
            X_cl_cl_s[idx_test], verbose=0) > 0.5).astype(int).flatten()
        results["scratch"][n_epochs] = accuracy_score(y_te, y_pred)

    print(f"    [seed={seed}]  zero-shot={results['zero_shot']:.4f}")
    for n in FINETUNE_EPOCH_SWEEP:
        print(f"      epochs={n:>3}  fine-tune={results['finetune'][n]:.4f}"
              f"  scratch={results['scratch'][n]:.4f}")

    return results, (y_te, idx_test)


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Load & scale data ─────────────────────────────────────────────────────
    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()

    # Ammonia scaler (used to scale chlori for transfer methods)
    X_am_s, am_scaler = scale(X_am)
    # Chlori scaled with its OWN scaler (for scratch baseline)
    X_cl_cl_s, cl_scaler = scale(X_cl)
    # Chlori scaled with AMMONIA scaler (for fine-tune methods)
    X_cl_am_s = am_scaler.transform(
        X_cl.reshape(-1, 96)).reshape(-1, 48, 2)

    print(f"Ammonia: {X_am_s.shape}  Chlori: {X_cl_cl_s.shape}")

    # ── Pre-train source LSTM on ALL ammonia ──────────────────────────────────
    print(f"\nPre-training LSTM on ammonia ({EPOCHS_PRETRAIN} epochs) …")
    source_model = build_lstm(LR_PRETRAIN)
    history = source_model.fit(
        X_am_s, y_am,
        epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE,
        validation_split=0.1, verbose=1,
    )
    am_train_acc = accuracy_score(
        y_am, (source_model.predict(X_am_s, verbose=0) > 0.5).astype(int).flatten())
    print(f"Ammonia train accuracy: {am_train_acc:.4f}")

    # ── Run experiments over seeds ────────────────────────────────────────────
    all_zero   = []
    all_ft     = {n: [] for n in FINETUNE_EPOCH_SWEEP}
    all_sc     = {n: [] for n in FINETUNE_EPOCH_SWEEP}
    last_run   = None

    for seed in RANDOM_SEEDS:
        print(f"\n{'='*50}\nSeed {seed}")
        res, extra = run_seed(
            seed, source_model,
            X_am_s, am_scaler,
            X_cl, y_cl, X_cl_am_s, X_cl_cl_s,
        )
        last_run = (res, extra, seed)
        all_zero.append(res["zero_shot"])
        for n in FINETUNE_EPOCH_SWEEP:
            all_ft[n].append(res["finetune"][n])
            all_sc[n].append(res["scratch"][n])

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Epochs':>8}  {'Fine-tune mean':>15}  {'Fine-tune std':>13}"
          f"  {'Scratch mean':>12}  {'Scratch std':>11}")
    print("=" * 65)
    for n in FINETUNE_EPOCH_SWEEP:
        print(f"{n:>8}  {np.mean(all_ft[n]):>15.4f}  {np.std(all_ft[n]):>13.4f}"
              f"  {np.mean(all_sc[n]):>12.4f}  {np.std(all_sc[n]):>11.4f}")
    print("=" * 65)
    print(f"Zero-shot (no fine-tune): {np.mean(all_zero):.4f} ± {np.std(all_zero):.4f}")

    # ── Plot 1: Accuracy vs epochs curve ─────────────────────────────────────
    ft_means = [np.mean(all_ft[n]) for n in FINETUNE_EPOCH_SWEEP]
    ft_stds  = [np.std(all_ft[n])  for n in FINETUNE_EPOCH_SWEEP]
    sc_means = [np.mean(all_sc[n]) for n in FINETUNE_EPOCH_SWEEP]
    sc_stds  = [np.std(all_sc[n])  for n in FINETUNE_EPOCH_SWEEP]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(FINETUNE_EPOCH_SWEEP, ft_means, "o-", color="#4C72B0",
            label="Standard fine-tune (ammonia pretrained)", linewidth=2, markersize=7)
    ax.fill_between(FINETUNE_EPOCH_SWEEP,
                    np.array(ft_means) - np.array(ft_stds),
                    np.array(ft_means) + np.array(ft_stds),
                    alpha=0.2, color="#4C72B0")
    ax.plot(FINETUNE_EPOCH_SWEEP, sc_means, "s--", color="#DD8452",
            label="Chlori-only (from scratch)", linewidth=2, markersize=7)
    ax.fill_between(FINETUNE_EPOCH_SWEEP,
                    np.array(sc_means) - np.array(sc_stds),
                    np.array(sc_means) + np.array(sc_stds),
                    alpha=0.2, color="#DD8452")
    ax.axhline(np.mean(all_zero), color="gray", linestyle=":",
               linewidth=1.5, label=f"Zero-shot ({np.mean(all_zero):.3f})")
    ax.set_xlabel("Fine-tune Epochs", fontsize=13)
    ax.set_ylabel("Accuracy on Chlori Test Set", fontsize=13)
    ax.set_title(
        f"Standard Fine-tune vs Scratch: Accuracy vs Epochs\n"
        f"(Ammonia pretrain={EPOCHS_PRETRAIN} epochs, "
        f"{N_CHLORI_TRAIN} chlori train samples, {len(RANDOM_SEEDS)} seeds)",
        fontsize=12,
    )
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Annotate final values
    for x, y_ft, y_sc in zip(FINETUNE_EPOCH_SWEEP, ft_means, sc_means):
        ax.annotate(f"{y_ft:.3f}", (x, y_ft), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#4C72B0")
        ax.annotate(f"{y_sc:.3f}", (x, y_sc), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="#DD8452")
    plt.tight_layout()
    plt.savefig("standard_finetune_epoch_curve.png", dpi=150)
    plt.show()

    # ── Plot 2: Bar chart at best epoch ──────────────────────────────────────
    best_n = FINETUNE_EPOCH_SWEEP[int(np.argmax(ft_means))]
    print(f"\nBest fine-tune epoch count: {best_n} "
          f"(acc={np.mean(all_ft[best_n]):.4f})")

    bar_methods = [
        f"Fine-tune\n({best_n} epochs)",
        f"Scratch\n({best_n} epochs)",
        "Zero-shot\n(no adapt)",
    ]
    bar_means = [np.mean(all_ft[best_n]),
                 np.mean(all_sc[best_n]),
                 np.mean(all_zero)]
    bar_stds  = [np.std(all_ft[best_n]),
                 np.std(all_sc[best_n]),
                 np.std(all_zero)]
    colors = ["#4C72B0", "#DD8452", "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bar_methods, bar_means, yerr=bar_stds, capsize=7,
                  color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Accuracy on Chlori Test Set", fontsize=13)
    ax.set_title(
        f"Standard Fine-tune vs Baselines (best epoch={best_n})\n"
        f"Ammonia pretrain={EPOCHS_PRETRAIN} epochs, "
        f"{N_CHLORI_TRAIN} chlori train samples",
        fontsize=12,
    )
    ax.set_ylim(0, 1.1)
    for bar, m, s in zip(bars, bar_means, bar_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.02, f"{m:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("standard_finetune_bar.png", dpi=150)
    plt.show()

    # ── Plot 3: Ammonia pre-training loss curve ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Ammonia Pre-training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title("Ammonia Pre-training Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("ammonia_pretrain_curve.png", dpi=150)
    plt.show()

    # ── Plot 4: Confusion matrices (best epoch, last seed) ───────────────────
    res, (y_te, idx_test), seed = last_run
    best_n_ft  = FINETUNE_EPOCH_SWEEP[int(np.argmax(ft_means))]

    # Re-run final seed to get predictions for confusion matrix
    idx_all = np.arange(len(X_cl))
    idx_tr_final, idx_te_final, _, _ = train_test_split(
        idx_all, y_cl,
        train_size=N_CHLORI_TRAIN, random_state=seed, stratify=y_cl,
    )
    y_true = y_cl[idx_te_final]

    ft_final = clone_pretrained(source_model, LR_FINETUNE)
    ft_final.fit(X_cl_am_s[idx_tr_final], y_cl[idx_tr_final],
                 epochs=best_n_ft, batch_size=BATCH_SIZE, verbose=0)
    y_pred_ft = (ft_final.predict(
        X_cl_am_s[idx_te_final], verbose=0) > 0.5).astype(int).flatten()

    sc_final = build_lstm(LR_PRETRAIN)
    sc_final.fit(X_cl_cl_s[idx_tr_final], y_cl[idx_tr_final],
                 epochs=best_n_ft, batch_size=BATCH_SIZE, verbose=0)
    y_pred_sc = (sc_final.predict(
        X_cl_cl_s[idx_te_final], verbose=0) > 0.5).astype(int).flatten()

    y_pred_zero = (source_model.predict(
        X_cl_am_s[idx_te_final], verbose=0) > 0.5).astype(int).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, y_pred) in zip(axes, [
        (f"Fine-tune ({best_n_ft} ep)", y_pred_ft),
        (f"Scratch ({best_n_ft} ep)",   y_pred_sc),
        ("Zero-shot",                    y_pred_zero),
    ]):
        cm  = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1], yticklabels=[0, 1],
                    annot_kws={"size": 13}, ax=ax)
        ax.set_title(f"{name}\nacc={acc:.4f}", fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle(f"Confusion Matrices — seed={seed}", fontsize=12)
    plt.tight_layout()
    plt.savefig("standard_finetune_confusion.png", dpi=150)
    plt.show()

    print("\nPlots saved:")
    print("  standard_finetune_epoch_curve.png")
    print("  standard_finetune_bar.png")
    print("  ammonia_pretrain_curve.png")
    print("  standard_finetune_confusion.png")
