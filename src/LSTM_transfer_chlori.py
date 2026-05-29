"""
LSTM Transfer Learning: Ammonia → Chlori-Alkali
================================================
Strategy:
  1. Train an LSTM on ALL ammonia data (source domain).
  2. Freeze the LSTM layers (keep the temporal feature extractor).
  3. Replace & retrain only the Dense head on a small chlori-alkali sample
     (2000 samples, same as the RF transfer learning experiments).
  4. Evaluate on the remaining chlori-alkali test set (6000 samples).
  5. Compare against:
       - Baseline : ammonia LSTM tested directly on chlori (no adaptation)
       - Fine-tune : unfreeze all layers, retrain everything on chlori sample
       - Chlori-only: LSTM trained from scratch on the 2000 chlori samples

Results are averaged over multiple random seeds for robustness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
N_CHLORI_TRAIN = 2000          # chlori samples used for transfer / fine-tune
RANDOM_SEEDS   = [0, 7, 21, 38, 42]

# Ammonia source model architecture (keep consistent with Ammonia_LSTM.py)
LSTM_UNITS      = 64
DENSE_UNITS     = 64
DROPOUT         = 0.2
REC_DROPOUT     = 0.2
LR_PRETRAIN     = 1e-3
LR_TRANSFER     = 1e-4          # lower LR for fine-tuning / head-only training
EPOCHS_PRETRAIN = 10            # kept short — 21504 samples, ~few mins
EPOCHS_TRANSFER = 15
BATCH_SIZE      = 64            # larger batch → faster epochs
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════ DATA LOADING ════════════════════════════════════

def load_ammonia():
    df = pd.read_csv(
        "../ammonia_dataset/total dataset for ML with statistic numbers.csv"
    )
    X = df.iloc[:, :96].values.reshape(-1, 48, 2)   # (N, 48, 2)
    y = df.iloc[:, 107].values.astype(int)
    return X, y


def load_chlori():
    df = pd.read_csv(
        "../chlori-alkali_dataset/Tokyo_ML_features_8000.csv"
    )
    X = df.iloc[:, :96].values.reshape(-1, 48, 2)   # (N, 48, 2)
    y = df["label"].values.astype(int)
    return X, y


def scale(X_train, X_test=None):
    """StandardScaler on flattened (N,96), returns scaled (N,48,2)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.reshape(-1, 96)).reshape(-1, 48, 2)
    if X_test is not None:
        X_te = scaler.transform(X_test.reshape(-1, 96)).reshape(-1, 48, 2)
        return X_tr, X_te, scaler
    return X_tr, scaler


# ═══════════════════════════ MODEL BUILDERS ══════════════════════════════════

def build_lstm(lstm_units=LSTM_UNITS, dense_units=DENSE_UNITS,
               dropout=DROPOUT, rec_dropout=REC_DROPOUT, lr=LR_PRETRAIN):
    model = Sequential([
        Input(shape=(48, 2)),
        LSTM(lstm_units, recurrent_dropout=rec_dropout),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def clone_with_new_head(source_model, freeze_lstm=True, lr=LR_TRANSFER):
    """Copy LSTM weights from source; attach a fresh Dense head.
    source_model.layers: [0]=LSTM, [1]=Dropout, [2]=Dense(relu), [3]=Dense(sigmoid)
    """
    new_model = Sequential([
        Input(shape=(48, 2)),
        LSTM(LSTM_UNITS, recurrent_dropout=REC_DROPOUT),
        Dropout(DROPOUT),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    # Copy only LSTM weights (index 0) — Dropout has no weights, Dense head is fresh
    new_model.layers[0].set_weights(source_model.layers[0].get_weights())

    if freeze_lstm:
        new_model.layers[0].trainable = False   # freeze LSTM only

    new_model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                      metrics=["accuracy"])
    return new_model


# ═══════════════════════════ EXPERIMENTS ═════════════════════════════════════

def run_seed(seed, source_model, X_am_scaled, y_am, am_scaler):
    print(f"\n  [seed={seed}]")
    X_cl, y_cl = load_chlori()

    # Scale chlori with its OWN scaler → used for chlori-only scratch model
    cl_scaler = StandardScaler()
    X_cl_scaled = cl_scaler.fit_transform(
        X_cl.reshape(-1, 96)).reshape(-1, 48, 2)

    # Scale chlori with the AMMONIA scaler → used for all transfer methods
    X_cl_am_scaled = am_scaler.transform(
        X_cl.reshape(-1, 96)).reshape(-1, 48, 2)

    # Split chlori: 2000 train / rest test  (same split for all methods)
    X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(
        np.arange(len(X_cl)), y_cl,
        train_size=N_CHLORI_TRAIN, random_state=seed, stratify=y_cl,
    )
    idx_train, idx_test = X_cl_train, X_cl_test   # these are indices

    results = {}

    # ── 1. BASELINE: ammonia LSTM → chlori, no adaptation ────────────────────
    y_pred_base = (source_model.predict(
        X_cl_am_scaled[idx_test], verbose=0) > 0.5).astype(int).flatten()
    results["Baseline (ammonia LSTM)"] = accuracy_score(
        y_cl[idx_test], y_pred_base)

    # ── 2. HEAD-ONLY: freeze LSTM, retrain Dense head on chlori sample ────────
    head_model = clone_with_new_head(source_model, freeze_lstm=True,
                                     lr=LR_TRANSFER)
    head_model.fit(
        X_cl_am_scaled[idx_train], y_cl[idx_train],
        epochs=EPOCHS_TRANSFER, batch_size=BATCH_SIZE, verbose=0,
    )
    y_pred_head = (head_model.predict(
        X_cl_am_scaled[idx_test], verbose=0) > 0.5).astype(int).flatten()
    results["Head-only (freeze LSTM)"] = accuracy_score(
        y_cl[idx_test], y_pred_head)

    # ── 3. FINE-TUNE: unfreeze all, retrain with small lr on chlori sample ────
    ft_model = clone_with_new_head(source_model, freeze_lstm=False,
                                   lr=LR_TRANSFER)
    ft_model.fit(
        X_cl_am_scaled[idx_train], y_cl[idx_train],
        epochs=EPOCHS_TRANSFER, batch_size=BATCH_SIZE, verbose=0,
    )
    y_pred_ft = (ft_model.predict(
        X_cl_am_scaled[idx_test], verbose=0) > 0.5).astype(int).flatten()
    results["Full fine-tune (all layers)"] = accuracy_score(
        y_cl[idx_test], y_pred_ft)

    # ── 4. CHLORI-ONLY: LSTM trained from scratch on 2000 chlori samples ─────
    scratch_model = build_lstm(lr=LR_PRETRAIN)
    scratch_model.fit(
        X_cl_scaled[idx_train], y_cl[idx_train],
        epochs=EPOCHS_TRANSFER, batch_size=BATCH_SIZE, verbose=0,
    )
    y_pred_scratch = (scratch_model.predict(
        X_cl_scaled[idx_test], verbose=0) > 0.5).astype(int).flatten()
    results["Chlori-only (from scratch)"] = accuracy_score(
        y_cl[idx_test], y_pred_scratch)

    for k, v in results.items():
        print(f"    {k}: {v:.4f}")

    return results, (y_cl[idx_test], y_pred_base, y_pred_head,
                     y_pred_ft, y_pred_scratch)


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Pre-train source LSTM on ALL ammonia data ─────────────────────────────
    print("=" * 55)
    print("Pre-training LSTM on ammonia data …")
    X_am, y_am = load_ammonia()
    X_am_scaled, am_scaler = scale(X_am)

    source_model = build_lstm(lr=LR_PRETRAIN)
    source_model.fit(X_am_scaled, y_am,
                     epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE,
                     validation_split=0.1, verbose=1)

    # Quick self-test on ammonia
    am_pred = (source_model.predict(X_am_scaled, verbose=0) > 0.5
               ).astype(int).flatten()
    print(f"Ammonia train accuracy: {accuracy_score(y_am, am_pred):.4f}")
    print("=" * 55)

    # ── Run transfer experiments over multiple seeds ──────────────────────────
    method_names = [
        "Baseline (ammonia LSTM)",
        "Head-only (freeze LSTM)",
        "Full fine-tune (all layers)",
        "Chlori-only (from scratch)",
    ]
    all_results = {m: [] for m in method_names}
    last_run = None

    for seed in RANDOM_SEEDS:
        print(f"\nSeed {seed}")
        res, preds = run_seed(seed, source_model, X_am_scaled, y_am, am_scaler)
        last_run = (res, preds, seed)
        for m in method_names:
            all_results[m].append(res[m])

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<35} {'Mean':>7} {'Std':>7}")
    print("=" * 60)
    for m in method_names:
        print(f"{m:<35} {np.mean(all_results[m]):>7.4f}"
              f" {np.std(all_results[m]):>7.4f}")
    print("=" * 60)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    means  = [np.mean(all_results[m]) for m in method_names]
    stds   = [np.std(all_results[m])  for m in method_names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(method_names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Accuracy on Chlori-Alkali Test Set", fontsize=13)
    ax.set_title(
        f"LSTM Transfer Learning: Ammonia → Chlori-Alkali\n"
        f"({N_CHLORI_TRAIN} chlori training samples, {len(RANDOM_SEEDS)} seeds)",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.01, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("LSTM_transfer_comparison.png", dpi=150)
    plt.show()

    # ── Confusion matrices (last seed) ────────────────────────────────────────
    res, (y_true, y_base, y_head, y_ft, y_scratch), seed = last_run
    pred_dict = {
        "Baseline (ammonia LSTM)":     y_base,
        "Head-only (freeze LSTM)":     y_head,
        "Full fine-tune (all layers)": y_ft,
        "Chlori-only (from scratch)":  y_scratch,
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, y_pred) in zip(axes, pred_dict.items()):
        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1], yticklabels=[0, 1],
                    annot_kws={"size": 13}, ax=ax)
        ax.set_title(f"{name}\nacc={acc:.4f}", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
    plt.suptitle(f"Confusion Matrices — seed={seed}", fontsize=12)
    plt.tight_layout()
    plt.savefig("LSTM_transfer_confusion_matrices.png", dpi=150)
    plt.show()

    print("\nPlots saved: LSTM_transfer_comparison.png, "
          "LSTM_transfer_confusion_matrices.png")
