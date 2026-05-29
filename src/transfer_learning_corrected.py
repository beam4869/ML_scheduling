"""
Transfer Learning with Label-Corrected Chlori-Alkali Dataset
=============================================================
Re-runs ALL transfer learning experiments using the corrected chlori-alkali
dataset where class semantics are aligned with the ammonia source domain:

  class 0 = competing   (high price AND high emission → curtail)
  class 1 = correlating (low price OR low emission   → run)

Experiments:
  Exp A — LSTM: full fine-tune vs scratch       (N sweep)
  Exp B — LSTM: full FT vs head-only vs scratch (N sweep)
  Exp C — RF:   fine-tune (mixed) vs scratch    (N sweep)

Sample sizes: [20, 50, 100, 250, 500, 1000, 2000, 4000]
Seeds       : [0, 7, 21, 38, 42]

Produces paper_fig5_corrected.pdf/.png (same 2×2 layout as original Fig 5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════ CONFIG ══════════════════════════════════════════
SAMPLE_SIZES    = [20, 50, 100, 250, 500, 1000, 2000, 4000]
RANDOM_SEEDS    = [0, 7, 21, 38, 42]

LSTM_UNITS      = 64
DENSE_UNITS     = 64
DROPOUT         = 0.2
REC_DROPOUT     = 0.2
LR_PRETRAIN     = 5e-4
LR_FINETUNE     = 1e-4
EPOCHS_PRETRAIN = 30
EPOCHS_FINETUNE = 50
BATCH_SIZE      = 64

RF_N_ESTIMATORS = 200

AMMONIA_CSV = "../ammonia_dataset/total dataset for ML with statistic numbers.csv"
CHLORI_CSV  = "../chlori-alkali_dataset/Tokyo_ML_features_8000_corrected.csv"

# ═══════════════════════════ DATA ════════════════════════════════════════════

def load_ammonia():
    df = pd.read_csv(AMMONIA_CSV)
    X  = df.iloc[:, :96].values.reshape(-1, 48, 2)
    y  = df.iloc[:, 107].values.astype(int)
    return X, y


def load_chlori():
    df = pd.read_csv(CHLORI_CSV)
    X  = df.iloc[:, :96].values.reshape(-1, 48, 2)
    y  = df["label"].values.astype(int)
    return X, y


def make_scalers(X_am, X_cl):
    sc_am = StandardScaler()
    sc_cl = StandardScaler()
    X_am_s  = sc_am.fit_transform(X_am.reshape(-1, 96)).reshape(-1, 48, 2)
    X_cl_s  = sc_cl.fit_transform(X_cl.reshape(-1, 96)).reshape(-1, 48, 2)
    X_cl_am = sc_am.transform(X_cl.reshape(-1, 96)).reshape(-1, 48, 2)
    return X_am_s, X_cl_s, X_cl_am, sc_am, sc_cl


# ═══════════════════════════ LSTM BUILDERS ═══════════════════════════════════

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


def clone_weights(src, lr, freeze_lstm=False):
    m = build_lstm(lr)
    m.set_weights(src.get_weights())
    if freeze_lstm:
        m.layers[0].trainable = False
        m.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return m


# ═══════════════════════════ SINGLE SEED RUNNER ══════════════════════════════

def run_lstm_seed(seed, n_train, source_model, X_cl_am, X_cl_cl, y_cl):
    """Returns dict: finetune_full, finetune_head, scratch, zero_shot."""
    idx_all = np.arange(len(y_cl))
    idx_tr, idx_te, _, y_te = train_test_split(
        idx_all, y_cl, train_size=n_train,
        random_state=seed, stratify=y_cl,
    )

    # Zero-shot
    acc_zero = accuracy_score(
        y_te, (source_model.predict(X_cl_am[idx_te], verbose=0) > 0.5
               ).astype(int).flatten())

    # Full fine-tune
    ft = clone_weights(source_model, LR_FINETUNE, freeze_lstm=False)
    ft.fit(X_cl_am[idx_tr], y_cl[idx_tr],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train), verbose=0)
    acc_ft = accuracy_score(
        y_te, (ft.predict(X_cl_am[idx_te], verbose=0) > 0.5).astype(int).flatten())

    # Head-only
    ho = clone_weights(source_model, 1e-3, freeze_lstm=True)
    ho.fit(X_cl_am[idx_tr], y_cl[idx_tr],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train), verbose=0)
    acc_ho = accuracy_score(
        y_te, (ho.predict(X_cl_am[idx_te], verbose=0) > 0.5).astype(int).flatten())

    # Scratch
    sc = build_lstm(LR_PRETRAIN)
    sc.fit(X_cl_cl[idx_tr], y_cl[idx_tr],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train), verbose=0)
    acc_sc = accuracy_score(
        y_te, (sc.predict(X_cl_cl[idx_te], verbose=0) > 0.5).astype(int).flatten())

    return dict(ft_full=acc_ft, ft_head=acc_ho, scratch=acc_sc, zero_shot=acc_zero)


def run_rf_seed(seed, n_train, X_am_flat, y_am, X_cl_flat, y_cl):
    """RF: fine-tune = mix ammonia+chlori; scratch = chlori only."""
    idx_all = np.arange(len(y_cl))
    idx_tr, idx_te, _, y_te = train_test_split(
        idx_all, y_cl, train_size=n_train,
        random_state=seed, stratify=y_cl,
    )

    # Fine-tune: ammonia pool + N chlori
    X_mix = np.vstack([X_am_flat, X_cl_flat[idx_tr]])
    y_mix = np.concatenate([y_am, y_cl[idx_tr]])
    rf_ft = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                                   random_state=seed)
    rf_ft.fit(X_mix, y_mix)
    acc_ft = accuracy_score(y_te, rf_ft.predict(X_cl_flat[idx_te]))

    # Scratch: chlori only
    rf_sc = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                                   random_state=seed)
    rf_sc.fit(X_cl_flat[idx_tr], y_cl[idx_tr])
    acc_sc = accuracy_score(y_te, rf_sc.predict(X_cl_flat[idx_te]))

    # Zero-shot RF: train on all ammonia, predict chlori
    rf_z = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                                  random_state=seed)
    rf_z.fit(X_am_flat, y_am)
    acc_zero = accuracy_score(y_te, rf_z.predict(X_cl_flat[idx_te]))

    return dict(ft=acc_ft, scratch=acc_sc, zero_shot=acc_zero)


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()
    X_am_s, X_cl_cl, X_cl_am, sc_am, sc_cl = make_scalers(X_am, X_cl)
    print(f"Ammonia : {X_am_s.shape}  labels 0={( y_am==0).mean():.1%} 1={(y_am==1).mean():.1%}")
    print(f"Chlori  : {X_cl_cl.shape}  labels 0={(y_cl==0).mean():.1%} 1={(y_cl==1).mean():.1%}")

    # RF flat features (scaled with respective scalers)
    X_am_flat = X_am_s.reshape(-1, 96)
    X_cl_flat = X_cl_cl.reshape(-1, 96)

    # ── Pre-train LSTM on ammonia ─────────────────────────────────────────────
    print(f"\nPre-training LSTM on ammonia ({EPOCHS_PRETRAIN} epochs) …")
    source_model = build_lstm(LR_PRETRAIN)
    source_model.fit(X_am_s, y_am, epochs=EPOCHS_PRETRAIN,
                     batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
    am_acc = accuracy_score(
        y_am, (source_model.predict(X_am_s, verbose=0) > 0.5).astype(int).flatten())
    print(f"Ammonia train accuracy: {am_acc:.4f}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    lstm = {k: {n: [] for n in SAMPLE_SIZES}
            for k in ["ft_full", "ft_head", "scratch", "zero_shot"]}
    rf   = {k: {n: [] for n in SAMPLE_SIZES}
            for k in ["ft", "scratch", "zero_shot"]}

    for n in SAMPLE_SIZES:
        print(f"\n{'='*55}\nN = {n}")
        for seed in RANDOM_SEEDS:
            # LSTM
            res_l = run_lstm_seed(seed, n, source_model, X_cl_am, X_cl_cl, y_cl)
            for k in lstm:
                lstm[k][n].append(res_l[k])
            # RF
            res_r = run_rf_seed(seed, n, X_am_flat, y_am, X_cl_flat, y_cl)
            for k in rf:
                rf[k][n].append(res_r[k])
            print(f"  seed={seed}  LSTM ft={res_l['ft_full']:.4f} sc={res_l['scratch']:.4f}"
                  f"  RF ft={res_r['ft']:.4f} sc={res_r['scratch']:.4f}")

    # ── Compute mean / std ────────────────────────────────────────────────────
    def ms(d):  # mean & std arrays across SAMPLE_SIZES
        m = [np.mean(d[n]) for n in SAMPLE_SIZES]
        s = [np.std(d[n])  for n in SAMPLE_SIZES]
        return np.array(m), np.array(s)

    lstm_ft_m,   lstm_ft_s   = ms(lstm["ft_full"])
    lstm_ho_m,   lstm_ho_s   = ms(lstm["ft_head"])
    lstm_sc_m,   lstm_sc_s   = ms(lstm["scratch"])
    lstm_zero                = np.mean([lstm["zero_shot"][n][0] for n in SAMPLE_SIZES])

    rf_ft_m,     rf_ft_s     = ms(rf["ft"])
    rf_sc_m,     rf_sc_s     = ms(rf["scratch"])
    rf_zero                  = np.mean([rf["zero_shot"][n][0] for n in SAMPLE_SIZES])

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY (corrected labels)")
    print("=" * 75)
    header = f"{'N':>6} | {'LSTM FT':>8} {'LSTM SC':>8} {'LSTM Head':>9} | {'RF FT':>7} {'RF SC':>7}"
    print(header)
    print("-" * len(header))
    for i, n in enumerate(SAMPLE_SIZES):
        print(f"{n:>6} | {lstm_ft_m[i]:>8.4f} {lstm_sc_m[i]:>8.4f} {lstm_ho_m[i]:>9.4f}"
              f" | {rf_ft_m[i]:>7.4f} {rf_sc_m[i]:>7.4f}")
    print("=" * 75)
    print(f"Zero-shot: LSTM={lstm_zero:.4f}  RF={rf_zero:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  FIGURE 5 (corrected) — 2×2 multi-panel
    # ══════════════════════════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })

    C_LSTM_FT = "#2166AC"
    C_LSTM_SC = "#6BAED6"
    C_RF_FT   = "#D6604D"
    C_RF_SC   = "#F4A582"
    C_HEAD    = "#1A9850"
    C_ZERO    = "#888888"

    N = SAMPLE_SIZES

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── (a) LSTM vs RF learning curves ───────────────────────────────────────
    ax_a.plot(N, rf_ft_m,   "o-",  color=C_RF_FT,   lw=2, ms=6, label="RF Fine-tune")
    ax_a.plot(N, rf_sc_m,   "o--", color=C_RF_SC,   lw=1.8, ms=6, label="RF Scratch")
    ax_a.plot(N, lstm_ft_m, "s-",  color=C_LSTM_FT, lw=2, ms=6, label="LSTM Fine-tune")
    ax_a.plot(N, lstm_sc_m, "s--", color=C_LSTM_SC, lw=1.8, ms=6, label="LSTM Scratch")
    for m, s, c in [(rf_ft_m, rf_ft_s, C_RF_FT), (rf_sc_m, rf_sc_s, C_RF_SC),
                    (lstm_ft_m, lstm_ft_s, C_LSTM_FT), (lstm_sc_m, lstm_sc_s, C_LSTM_SC)]:
        ax_a.fill_between(N, m - s, m + s, alpha=0.1, color=c)
    ax_a.axhline(rf_zero,   color=C_RF_FT,   ls=":", lw=1.3, alpha=0.6,
                 label=f"RF zero-shot ({rf_zero:.3f})")
    ax_a.axhline(lstm_zero, color=C_LSTM_FT,  ls=":", lw=1.3, alpha=0.6,
                 label=f"LSTM zero-shot ({lstm_zero:.3f})")
    ax_a.set_xscale("log"); ax_a.set_xticks(N)
    ax_a.set_xticklabels([str(n) for n in N], fontsize=8)
    ax_a.set_xlabel("# Labeled Chlori Samples")
    ax_a.set_ylabel("Test Accuracy")
    ax_a.set_title("(a) LSTM vs RF: Fine-tune & Scratch")
    ax_a.set_ylim(0.44, 1.02)
    ax_a.legend(fontsize=8, ncol=2)

    # ── (b) LSTM strategy comparison ─────────────────────────────────────────
    ax_b.plot(N, lstm_ft_m, "s-",  color=C_LSTM_FT, lw=2, ms=6, label="Full fine-tune")
    ax_b.plot(N, lstm_ho_m, "D-",  color=C_HEAD,    lw=2, ms=6, label="Head-only")
    ax_b.plot(N, lstm_sc_m, "o--", color=C_LSTM_SC, lw=1.8, ms=6, label="Scratch")
    ax_b.axhline(lstm_zero, color=C_ZERO, ls=":", lw=1.3, alpha=0.7, label="Zero-shot")
    ax_b.fill_between(N, lstm_ft_m - lstm_ft_s, lstm_ft_m + lstm_ft_s,
                      alpha=0.12, color=C_LSTM_FT)
    ax_b.fill_between(N, lstm_sc_m - lstm_sc_s, lstm_sc_m + lstm_sc_s,
                      alpha=0.12, color=C_LSTM_SC)
    # Find crossover point for annotation
    crossover_N = None
    for i, n in enumerate(N):
        if lstm_ft_m[i] < lstm_sc_m[i]:
            crossover_N = n
            break
    if crossover_N:
        xidx = N.index(crossover_N)
        ax_b.axvspan(N[0]*0.8, N[max(0, xidx-1)]*1.1, alpha=0.06, color=C_LSTM_FT)
        ax_b.text(N[0]*1.1, 0.96, f"FT wins\n(N<{crossover_N})",
                  fontsize=8, color=C_LSTM_FT, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec=C_LSTM_FT, alpha=0.8))
    ax_b.set_xscale("log"); ax_b.set_xticks(N)
    ax_b.set_xticklabels([str(n) for n in N], fontsize=8)
    ax_b.set_xlabel("# Labeled Chlori Samples")
    ax_b.set_ylabel("Test Accuracy")
    ax_b.set_title("(b) LSTM Fine-tune Strategy Comparison")
    ax_b.set_ylim(0.44, 1.02)
    ax_b.legend(fontsize=8)

    # ── (c) Transfer advantage bars ───────────────────────────────────────────
    shared_idx = list(range(len(N)))   # all sizes shared between LSTM and RF
    lstm_adv = lstm_ft_m - lstm_sc_m
    rf_adv   = rf_ft_m   - rf_sc_m

    x = np.arange(len(N))
    w = 0.35
    ax_c.bar(x - w/2, lstm_adv, w, color=[C_LSTM_FT if v >= 0 else "#AAAAAA"
             for v in lstm_adv], alpha=0.85, edgecolor="black", lw=0.6, label="LSTM")
    ax_c.bar(x + w/2, rf_adv,   w, color=[C_RF_FT if v >= 0 else "#AAAAAA"
             for v in rf_adv],   alpha=0.85, edgecolor="black", lw=0.6, label="RF")
    ax_c.axhline(0, color="black", lw=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(n) for n in N], fontsize=9)
    ax_c.set_xlabel("# Labeled Chlori Samples")
    ax_c.set_ylabel("Fine-tune − Scratch Accuracy")
    ax_c.set_title("(c) Transfer Learning Advantage")
    ylim_c = max(abs(np.concatenate([lstm_adv, rf_adv]))) + 0.03
    ax_c.set_ylim(-ylim_c, ylim_c)
    ax_c.legend(fontsize=9)
    for i, (la, ra) in enumerate(zip(lstm_adv, rf_adv)):
        ax_c.text(i - w/2, la + (0.004 if la >= 0 else -0.013),
                  f"{la:+.3f}", ha="center", fontsize=7,
                  color=C_LSTM_FT if la >= 0 else "#555", fontweight="bold")
        ax_c.text(i + w/2, ra + (0.004 if ra >= 0 else -0.013),
                  f"{ra:+.3f}", ha="center", fontsize=7,
                  color=C_RF_FT   if ra >= 0 else "#555", fontweight="bold")

    # ── (d) Stability: std across seeds ───────────────────────────────────────
    x_d = np.arange(len(N))
    w_d = 0.25
    ax_d.bar(x_d - w_d, lstm_ft_s, w_d, color=C_LSTM_FT, alpha=0.85,
             edgecolor="black", lw=0.6, label="LSTM Full fine-tune")
    ax_d.bar(x_d,        lstm_ho_s, w_d, color=C_HEAD,    alpha=0.85,
             edgecolor="black", lw=0.6, label="LSTM Head-only")
    ax_d.bar(x_d + w_d, lstm_sc_s, w_d, color=C_LSTM_SC, alpha=0.85,
             edgecolor="black", lw=0.6, label="LSTM Scratch")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([str(n) for n in N], fontsize=9)
    ax_d.set_xlabel("# Labeled Chlori Samples")
    ax_d.set_ylabel("Std of Accuracy (5 seeds)")
    ax_d.set_title("(d) Stability: Variance Across Random Seeds")
    ax_d.legend(fontsize=9)

    # Panel labels
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    plt.suptitle(
        "Transfer Learning from Ammonia to Chlor-Alkali Energy Market "
        "(Label-Corrected):\n"
        "Sample Efficiency, Strategy Comparison, and Advantage Analysis",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig("paper_fig5_corrected.pdf", bbox_inches="tight")
    plt.savefig("paper_fig5_corrected.png", bbox_inches="tight", dpi=200)
    plt.show()
    print("\nSaved: paper_fig5_corrected.pdf / .png")
