import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
data = pd.read_excel(file_path)

# Separate features (first 96 columns) and target (107th column)
X = data.iloc[:, :96]  # Features
y = data.iloc[:, 106]  # Classification (107th column)

# Split the data: 70% train, 20% test, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)  # 33% of 30% -> ~10%

# Apply PCA and test different dimensions (you can adjust the number of components)
best_accuracy = 0
best_n_components = 0
best_model = None
all_list = []
# for n_components in range(4, 97, 1):  # Adjust the range for n_components as needed
#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(X_train)
#     X_val_pca = pca.transform(X_val)
    
#     # Train an SVM classifier
#     svm_model = SVC(decision_function_shape='ovo', kernel = 'rbf', random_state=42)
#     svm_model.fit(X_train_pca, y_train)
    
#     # Validate the model on validation set
#     val_accuracy = svm_model.score(X_val_pca, y_val)
#     print("accuracy: ", val_accuracy)
#     all_list.append(val_accuracy)
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_n_components = n_components
#         best_model = svm_model

# Apply PCA with the best number of components found
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the final SVM model on the full training set
svm_model = SVC(decision_function_shape='ovo', kernel = 'rbf' , random_state=42)
svm_model.fit(X_train_pca, y_train)

# Test the model on the test set
y_pred = svm_model.predict(X_test_pca)
accuracy = svm_model.score(X_test_pca, y_test)
# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# df_score = pd.DataFrame({
#     'dim': range(4, 97, 1),
#     'accuracy': all_list
# })
# df_score.to_excel('SVM ovo rbf accuracy with PCA dim.xlsx', index=None)
# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4],annot_kws= {'size': 15})
plt.title(f'Confusion Matrix with PCA (dim = 20 accuracy = {round(accuracy,4)} )')
plt.xlabel('Predicted',fontsize = 14)
plt.ylabel('True',fontsize = 14)
# sns.set(font_scale=8)
plt.show()
