import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

OUT_DIR = (
    '/Users/hongxuan/Library/CloudStorage/'
    'GoogleDrive-hongxuan@umich.edu/My Drive/'
    'AA gourp/transfer learning/PCA_figure_ploting/'
)
CACHE_FILE = os.path.join(OUT_DIR, 'rf_pca_accuracy_cache.json')

# ── RF accuracy: load cache or retrain ───────────────────────────────────────
components = np.arange(3, 51)

if os.path.exists(CACHE_FILE):
    print(f'Loading cached RF results from {CACHE_FILE}')
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    cached_components = np.array(cache['components'])
    cached_accuracies = np.array(cache['accuracies'])
    missing = [int(n) for n in components if n not in cached_components]
    if missing:
        print(f'Cache missing n={missing}, training those...')
        excel_path = (
            '/Users/hongxuan/Library/CloudStorage/'
            'GoogleDrive-hongxuan@umich.edu/My Drive/'
            'AA gourp/machine learning/datasets for ML/'
            'total dataset for ML with statistic numbers.xlsx'
        )
        data = pd.read_excel(excel_path)
        X = data.iloc[:, :96]
        y = data.iloc[:, 106]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        extra_acc = {}
        for n in missing:
            pipeline.set_params(pca__n_components=n)
            pipeline.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            extra_acc[n] = acc
            print(f'  n_components={n:2d}  accuracy={acc:.4f}')
        full = dict(zip(cached_components.tolist(), cached_accuracies.tolist()))
        full.update(extra_acc)
        all_n = sorted(full.keys())
        cached_components = np.array(all_n)
        cached_accuracies = np.array([full[n] for n in all_n])
        with open(CACHE_FILE, 'w') as f:
            json.dump({'components': cached_components.tolist(),
                       'accuracies': cached_accuracies.tolist()}, f, indent=2)
        print(f'Cache updated: {CACHE_FILE}')
    mask = np.isin(cached_components, components)
    components = cached_components[mask]
    accuracies = cached_accuracies[mask]
else:
    print('No cache found — training RF for n=3…50 (this may take a minute)…')
    excel_path = (
        '/Users/hongxuan/Library/CloudStorage/'
        'GoogleDrive-hongxuan@umich.edu/My Drive/'
        'AA gourp/machine learning/datasets for ML/'
        'total dataset for ML with statistic numbers.xlsx'
    )
    data = pd.read_excel(excel_path)
    X = data.iloc[:, :96]
    y = data.iloc[:, 106]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    accuracies = []
    for n in components:
        pipeline.set_params(pca__n_components=n)
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        accuracies.append(acc)
        print(f'  n_components={n:2d}  accuracy={acc:.4f}')
    accuracies = np.array(accuracies)
    with open(CACHE_FILE, 'w') as f:
        json.dump({'components': components.tolist(),
                   'accuracies': accuracies.tolist()}, f, indent=2)
    print(f'Results cached to {CACHE_FILE}')

best_idx = int(np.argmax(accuracies))
best_n   = int(components[best_idx])
best_acc = float(accuracies[best_idx])

# ── Figure: 16×9, explicit axes placement ────────────────────────────────────
# All values in figure-fraction units (0–1).
# Figure: 16 in wide × 9 in tall.
#
# ax1 (PCA scatter) — true square:
#   side = 5.0 in → width_frac = 5/16 = 0.3125, height_frac = 5/9 = 0.5556
#   horizontally: starts at 0.06 (left margin)
#   vertically:   centered between y=0.10 and y=0.93  → center=0.515
#
# ax2 (RF accuracy) — 8.5 × 5 in landscape rectangle:
#   width_frac = 8.5/16 = 0.53125, height_frac = 5/9 = 0.5556
#   same height as ax1 → share the same SQ_Y (perfectly top- and bottom-aligned)

SQ_W = 5.0 / 16          # 0.3125
SQ_H = 5.0 / 9           # 0.5556
SQ_X = 0.06
SQ_Y = 0.515 - SQ_H / 2  # ≈ 0.237  (vertically centred)

RF_X = SQ_X + SQ_W + 0.05   # ≈ 0.4225
RF_W = 8.5 / 16              # 0.53125
RF_H = 5.0 / 9               # 0.5556  — same as SQ_H, so both panels align
RF_Y = SQ_Y                  # identical bottom edge → tops align too

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_axes([SQ_X, SQ_Y, SQ_W, SQ_H])
ax2 = fig.add_axes([RF_X, RF_Y, RF_W, RF_H])

# ── (a) PCA scatter — square axes, equal data ranges ─────────────────────────
csv_path = (
    '/Users/hongxuan/Library/CloudStorage/'
    'GoogleDrive-hongxuan@umich.edu/My Drive/'
    'python/PCA_loop/first two PCA component plot.csv'
)
df = pd.read_csv(csv_path)

group_labels = [
    '[[1,2], [3,4]]',
    '[[4], [1,2,3]]',
    '[[1,4], [2,3]]',
    '[[1], [2,3,4]]',
    '[[2], [1,3,4]]',
]

for i, label in enumerate(group_labels):
    ax1.scatter(
        df[f'x{i}'], df[f'y{i}'],
        color=PALETTE[i], alpha=0.45, s=28,
        label=label, linewidths=0, zorder=3,
    )

ax1.set_xlim(-0.05, 1.15)   # equal range → square data space
ax1.set_ylim(-0.05, 1.15)
ax1.set_xlabel('PCA Component 1', fontsize=13, labelpad=7)
ax1.set_ylabel('PCA Component 2', fontsize=13, labelpad=7)
ax1.set_title('(a)  First Two PCA Components', fontsize=13,
              fontweight='bold', pad=10, loc='left')
ax1.tick_params(labelsize=10.5)
ax1.legend(
    frameon=True, framealpha=0.88, edgecolor='#cccccc',
    fontsize=9, loc='upper left',
    handletextpad=0.4, borderpad=0.6,
    title='Dataset split', title_fontsize=9,
)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.35, zorder=0)

# ── (b) RF accuracy vs PCA dimensions ────────────────────────────────────────
ax2.plot(
    components, accuracies,
    color='#4C72B0', linewidth=2.2,
    marker='o', markersize=5.5,
    markerfacecolor='white', markeredgewidth=1.8,
    markeredgecolor='#4C72B0',
    zorder=3, label='Test accuracy',
)

ax2.scatter([best_n], [best_acc],
            color='#C44E52', s=80, zorder=5,
            label=f'Best  n = {best_n}')
ax2.axvline(best_n, color='#C44E52', linestyle='--', linewidth=1.4,
            alpha=0.6, zorder=2)

# Annotation — shift left if best_n is near right edge
dx = -9 if best_n > 40 else 2.5
dy = -0.035 if best_acc > (accuracies.min() + 0.05) else 0.02
ax2.annotate(
    f'n = {best_n},  Acc = {best_acc:.3f}',
    xy=(best_n, best_acc),
    xytext=(best_n + dx, best_acc + dy),
    fontsize=11, color='#C44E52',
    arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.3),
    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#C44E52',
              alpha=0.9, lw=1.1),
)

ax2.set_xlabel('Number of PCA Components', fontsize=13, labelpad=7)
ax2.set_ylabel('Random Forest Accuracy', fontsize=13, labelpad=7)
ax2.set_title('(b)  RF Accuracy vs. Number of PCA Components', fontsize=13,
              fontweight='bold', pad=10, loc='left')
ax2.set_xticks(np.arange(5, 51, 5))
ax2.set_xlim(2, 51)
ax2.tick_params(labelsize=11)
y_margin = 0.03
ax2.set_ylim(max(0, accuracies.min() - y_margin),
             min(1.01, accuracies.max() + y_margin))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.35, zorder=0)
ax2.legend(frameon=True, framealpha=0.88, edgecolor='#cccccc',
           fontsize=10.5, loc='lower right')

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(os.path.join(OUT_DIR, 'PCA_combined_figure.pdf'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'PCA_combined_figure.png'),
            dpi=300, bbox_inches='tight')
print('\nSaved: PCA_combined_figure.pdf / .png')
plt.show()
