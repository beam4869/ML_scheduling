import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns

# Load the dataset
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
data = pd.read_excel(file_path)

# Separate features and target
X = data.iloc[:, :96]  # Adjust if necessary
y = data.iloc[:, 106]  # Adjust index based on your description

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline that includes scaling, PCA, and Random Forest
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA()),  # We will adjust the number of components below
    ('rf', RandomForestClassifier(random_state=42))
])

# Testing different numbers of PCA components
components = np.arange(3, 25)  # Test up to the original number of features
best_accuracy = 0
best_n_components = 0
best_confusion_matrix = None

for n in components:
    pipeline.set_params(pca__n_components=n)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy: ",acc)
    if acc > best_accuracy:
        best_accuracy = acc
        best_n_components = n
        best_confusion_matrix = confusion_matrix(y_test, y_pred)
print(f"Best PCA Components: {best_n_components}")
print(f"Best Accuracy: {best_accuracy:.2f}")
# Plot the best confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(best_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.title(f'Confusion Matrix with PCA (dim = {best_n_components} accuracy = {round(best_accuracy,4)} )')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# plt.figure(figsize=(10, 7))
# plt.imshow(best_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title(f'Confusion Matrix with Best PCA Components: {best_n_components}')
# plt.colorbar()
# tick_marks = np.arange(len(np.unique(y)))
# plt.xticks(tick_marks, np.unique(y), rotation=45)
# plt.yticks(tick_marks, np.unique(y))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.grid(False)
# plt.show()


