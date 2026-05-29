"""
Run ONLY N=150 and N=200 to find the LSTM transfer learning crossover point.
Uses identical config to sample_size_learning_curve.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─── Same config as sample_size_learning_curve.py ────────────────────────────
SAMPLE_SIZES    = [150, 200]
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
# ─────────────────────────────────────────────────────────────────────────────


def load_ammonia():
    df = pd.read_csv("../ammonia_dataset/total dataset for ML with statistic numbers.csv")
    X = df.iloc[:, :96].values.reshape(-1, 48, 2)
    y = df.iloc[:, 107].values.astype(int)
    return X, y


def load_chlori():
    df = pd.read_csv("../chlori-alkali_dataset/Tokyo_ML_features_8000.csv")
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
    new_model = build_lstm(lr)
    new_model.set_weights(source_model.get_weights())
    return new_model


def run_one(seed, n_train, source_model, X_cl_am, X_cl_cl, y_cl):
    idx_all = np.arange(len(y_cl))
    idx_train, idx_test, _, y_te = train_test_split(
        idx_all, y_cl, train_size=n_train, random_state=seed, stratify=y_cl,
    )

    y_zero = (source_model.predict(X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten()
    acc_zero = accuracy_score(y_te, y_zero)

    ft = clone_all_weights(source_model, LR_FINETUNE)
    ft.fit(X_cl_am[idx_train], y_cl[idx_train],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train), verbose=0)
    acc_ft = accuracy_score(y_te,
        (ft.predict(X_cl_am[idx_test], verbose=0) > 0.5).astype(int).flatten())

    sc = build_lstm(LR_PRETRAIN)
    sc.fit(X_cl_cl[idx_train], y_cl[idx_train],
           epochs=EPOCHS_FINETUNE, batch_size=min(BATCH_SIZE, n_train), verbose=0)
    acc_sc = accuracy_score(y_te,
        (sc.predict(X_cl_cl[idx_test], verbose=0) > 0.5).astype(int).flatten())

    return {"finetune": acc_ft, "scratch": acc_sc, "zero_shot": acc_zero}


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading data …")
    X_am, y_am = load_ammonia()
    X_cl, y_cl = load_chlori()
    X_am_s, X_cl_cl, X_cl_am = make_scalers(X_am, X_cl)

    print(f"\nPre-training LSTM on ammonia ({EPOCHS_PRETRAIN} epochs) …")
    source_model = build_lstm(LR_PRETRAIN)
    source_model.fit(X_am_s, y_am, epochs=EPOCHS_PRETRAIN,
                     batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    results = {
        "finetune":  {n: [] for n in SAMPLE_SIZES},
        "scratch":   {n: [] for n in SAMPLE_SIZES},
        "zero_shot": {n: [] for n in SAMPLE_SIZES},
    }

    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*50}\nN = {n_train}")
        for seed in RANDOM_SEEDS:
            res = run_one(seed, n_train, source_model, X_cl_am, X_cl_cl, y_cl)
            results["finetune"][n_train].append(res["finetune"])
            results["scratch"][n_train].append(res["scratch"])
            results["zero_shot"][n_train].append(res["zero_shot"])
            print(f"  seed={seed}  FT={res['finetune']:.4f}"
                  f"  SC={res['scratch']:.4f}  zero={res['zero_shot']:.4f}")

    print("\n" + "=" * 50)
    print("FINAL RESULTS (mean ± std across 5 seeds):")
    print("=" * 50)
    for n in SAMPLE_SIZES:
        ft_arr = np.array(results["finetune"][n])
        sc_arr = np.array(results["scratch"][n])
        print(f"N={n:4d}  FT={ft_arr.mean():.4f}±{ft_arr.std():.4f}"
              f"   SC={sc_arr.mean():.4f}±{sc_arr.std():.4f}"
              f"   diff={ft_arr.mean()-sc_arr.mean():+.4f}"
              f"   {'FT wins' if ft_arr.mean() > sc_arr.mean() else 'SC wins'}")
