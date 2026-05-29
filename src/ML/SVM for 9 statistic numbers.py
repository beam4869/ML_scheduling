import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
data = pd.read_excel(file_path)

# Separate features (first 96 columns) and target (107th column)
X = data.iloc[:, 96:105]  # Features
print("x:",X)
y = data.iloc[:, 106]  # Classification (107th column)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Split the data: 70% train, 20% test, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)  # 33% of 30% -> ~10%

# Apply PCA and test different dimensions (you can adjust the number of components)
best_accuracy = 0
best_n_components = 0
best_model = None
all_list = []

# Train the final SVM model on the full training set
svm_model = SVC(decision_function_shape='ovo', kernel = 'rbf' , random_state=42)
svm_model.fit(X_train, y_train)

# Test the model on the test set
y_pred = svm_model.predict(X_test)
accuracy = svm_model.score(X_test, y_test)
# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
# print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.title(f'Confusion Matrix 9 statistic numbers accuracy = {accuracy}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
