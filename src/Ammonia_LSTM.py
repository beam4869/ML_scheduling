import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../ammonia_dataset/total dataset for ML with statistic numbers.csv')
X = data.iloc[:, :96].values  # First 96 columns are features
y = data.iloc[:, 107].values   # two groupiings (0 and 1) in the last column

# Reshape X to (samples, 48, 2) for LSTM input
X = X.reshape(-1, 48, 2)

# Scale the features (price and emission) across all samples and time steps
scaler = StandardScaler()
X_reshaped = X.reshape(-1, 96)  # Flatten to (samples * timesteps, features)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 48, 2)  # Reshape back to (samples, 48, 2)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
n_classes = len(np.unique(y_encoded))

# One-hot encode if multi-class (>2 classes), keep as is for binary
# n_classes = 5
if n_classes > 2:
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_encoded = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
else:
    y_encoded = y_encoded.reshape(-1, 1)  # For binary classification
print("y_encoded:", y_encoded)

# Define hyperparameter space
hyperparameters = {
    'units': [32, 64, 128, 256],              # LSTM units
    'dropout': [0.2, 0.3, 0.5],               # Dropout rate
    'dense_units': [32, 64, 128],             # Units in dense layer
    'learning_rate': [0.001, 0.0001, 0.00001],# Learning rate for Adam
    'num_layers': [1, 2, 3],                  # Number of LSTM layers
    'recurrent_dropout': [0.2, 0.3, 0.5]      # Recurrent dropout rate
}

# Number of random search iterations
n_iter = 20
k = 4  # Number of folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Store results
results = []

for _ in range(n_iter):
    # Randomly sample hyperparameters and cast to native Python types for Keras
    params = {key: np.random.choice(values) for key, values in hyperparameters.items()}
    params['units']             = int(params['units'])
    params['dense_units']       = int(params['dense_units'])
    params['num_layers']        = int(params['num_layers'])
    params['dropout']           = float(params['dropout'])
    params['recurrent_dropout'] = float(params['recurrent_dropout'])
    params['learning_rate']     = float(params['learning_rate'])
    fold_accuracies = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]
        
        # Build the LSTM model
        model = Sequential()
        for i in range(params['num_layers']):
            units = params['units']
            print("units:", units)
            if i == 0:
                # model.add(LSTM(params['units'], input_shape=(48, 2),
                model.add(LSTM(units, input_shape=(48, 2),
                               return_sequences=True if params['num_layers'] > 1 else False,
                               recurrent_dropout=params['recurrent_dropout']))
            else:
                model.add(LSTM(units,
                               return_sequences=True if i < params['num_layers'] - 1 else False,
                               recurrent_dropout=params['recurrent_dropout']))
            model.add(Dropout(params['dropout']))
        model.add(Dense(params['dense_units'], activation='relu'))
        if n_classes > 2:
            model.add(Dense(n_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        optimizer = Adam(learning_rate=params['learning_rate'])
        if n_classes > 2:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val, verbose=0)
        if n_classes > 2:
            val_pred = np.argmax(val_pred, axis=1)
            y_val_true = np.argmax(y_val, axis=1)
        else:
            val_pred = (val_pred > 0.5).astype(int).flatten()
            y_val_true = y_val.flatten()
        acc = accuracy_score(y_val_true, val_pred)
        print(f"Validation accuracy: {acc:.4f}")
        fold_accuracies.append(acc)
    
    # Average accuracy across folds
    avg_acc = np.mean(fold_accuracies)
    results.append((params, avg_acc))

# Find the best hyperparameters
best_params, best_acc = max(results, key=lambda x: x[1])
print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validation accuracy: {best_acc:.4f}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Build the model with best hyperparameters
model = Sequential()
for i in range(int(best_params['num_layers'])):
    if i == 0:
        model.add(LSTM(int(best_params['units']), input_shape=(48, 2),
                       return_sequences=True if best_params['num_layers'] > 1 else False,
                       recurrent_dropout=float(best_params['recurrent_dropout'])))
    else:
        model.add(LSTM(int(best_params['units']),
                       return_sequences=True if i < best_params['num_layers'] - 1 else False,
                       recurrent_dropout=float(best_params['recurrent_dropout'])))
    model.add(Dropout(float(best_params['dropout'])))
model.add(Dense(int(best_params['dense_units']), activation='relu'))
if n_classes > 2:
    model.add(Dense(n_classes, activation='softmax'))
else:
    model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=best_params['learning_rate'])
if n_classes > 2:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation split
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate on test set
test_pred = model.predict(X_test, verbose=0)
if n_classes > 2:
    test_pred_classes = np.argmax(test_pred, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
else:
    test_pred_classes = (test_pred > 0.5).astype(int).flatten()
    y_test_true = y_test.flatten()
test_acc = accuracy_score(y_test_true, test_pred_classes)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test_true, test_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()