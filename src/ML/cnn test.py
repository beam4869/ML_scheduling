import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import keras
# from keras.models import Sequential
# from keras.layers import Conv2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, accuracy_score
# Load data
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
data = pd.read_excel(file_path)

# Organize data
X = data.iloc[:, :96]  # First 96 columns as features
y = data.iloc[:, 96]  # The last column as labels

# Reshape features to (num_samples, 48, 2) assuming each sample has 48 points (x, y)
X = X.values.reshape(-1, 48, 2)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)


# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 2, 1)),  # Ensure input shape is correctly formatted
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # Assuming 5 classes
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_labels)
accuracy = accuracy_score(y_test, y_pred_labels)

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
