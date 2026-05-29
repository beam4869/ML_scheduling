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
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, :96]  # Adjust the column indices as necessary
y = df.iloc[:, 106]  # Adjust the index for the target column

# Normalize the dataset
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# Evaluate with the original dataset
accuracy, conf_matrix = train_and_evaluate(X_train, X_test, y_train, y_test)
print("Accuracy with original data:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# PCA and Random Forest
pca_dimensions = range(1, 97)  # Testing 1 to 96 dimensions
accuracies = []

for n in pca_dimensions:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    acc, _ = train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)
    accuracies.append(acc)

# Plotting the accuracies
plt.figure(figsize=(10, 4))
plt.plot(pca_dimensions, accuracies, marker='o')
plt.xlabel('Number of PCA Dimensions',fontsize = 18)
plt.ylabel('Accuracy',fontsize = 18)
plt.title('Accuracy vs. PCA Dimensions',fontsize = 18)
plt.grid(True)
plt.show()
# plt.figure(figsize=(8,6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
# plt.title(f'Confusion Matrix with total dimension, accuracy = {accuracy} )')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
