import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
file_path = 'G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/combining last nine months of New England 2024/combined dataset.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, 96:105]  # Adjust the column indices as necessary
y = df.iloc[:, 106]  # Adjust the index for the target column

# Normalize the dataset
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=38)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

rf = RandomForestClassifier(random_state=42)

param_dist = {
    "n_estimators": randint(200, 1500),
    "max_depth": [None] + list(range(5, 31, 5)),
    "min_samples_split": randint(2, 15),
    "min_samples_leaf": randint(1, 15),
    "max_features": ["sqrt", "log2", 0.3, 0.6, 1.0],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=100,               # increase if time permits
    cv=cv,
    scoring="accuracy",       # or f1, roc_auc, ...
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)
print("Best CV score:", search.best_score_)
print("Best params:", search.best_params_)

best_rf = search.best_estimator_
test_acc = best_rf.score(X_test, y_test)
print("Hold‑out accuracy:", test_acc)
# print("Best estimator:", best_rf)
