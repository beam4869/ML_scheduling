import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_process_data(path):
    data = pd.read_excel(path)
    X = data.iloc[:, :96].values  # First 96 columns are features
    y = data.iloc[:, 107].values   # Last column is the label
    X = X.reshape(-1, 48, 2)
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, 96)  # Flatten to (samples * timesteps, features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(-1, 48, 2)  # Reshape back to (samples, 48, 2)
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
    return X, y_encoded, n_classes,scaler, label_encoder

def plot_results(history, y_pred_classes, y_test_classes):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], linestyle='dashed', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], linestyle='dashed', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def build_model(X, y_encoded, n_classes, best_params):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    print("y_test:", y_test)

    # Build the model with best hyperparameters
    model = Sequential()
    for i in range(best_params['num_layers']):
        units = int(best_params['units'])
        if i == 0:
            model.add(LSTM(best_params['units'], input_shape=(48, 2),
                        return_sequences=True if best_params['num_layers'] > 1 else False,
                        recurrent_dropout=best_params['recurrent_dropout']))
        else:
            model.add(LSTM(best_params['units'],
                        return_sequences=True if i < best_params['num_layers'] - 1 else False,
                        recurrent_dropout=best_params['recurrent_dropout']))
        model.add(Dropout(best_params['dropout']))
    model.add(Dense(best_params['dense_units'], activation='relu'))
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
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate on test set
    test_pred = model.predict(X_test, verbose=0)
    if n_classes > 2:
        test_pred_classes = np.argmax(test_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
    else:
        test_pred_classes = (test_pred > 0.5).astype(int).flatten()
        y_test_true = y_test.flatten()
        y_test_classes = y_test_true
    test_acc = accuracy_score(y_test_true, test_pred_classes)
    print(f"Test accuracy: {test_acc:.4f}")
    print("y predict:", test_pred)
    print("y predict class:", test_pred_classes)
    return history, X_test, y_test, test_pred_classes,y_test_classes,model

path = 'G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/combining last nine months of New England 2024/combined dataset.xlsx'
X, y_encoded, n_classes, scaler, label_encoder = load_process_data(path)
print("num_classes:", n_classes)
best_params = {'units': 256, 'dropout': 0.2, 'dense_units': 64, 'learning_rate': 0.001, 'num_layers': 3, 'recurrent_dropout': 0.2}
history, X_test, y_test, test_pred,y_test_classes,model = build_model(X, y_encoded, n_classes, best_params)
plot_results(history, test_pred, y_test)


# plot_results(history, test_pred, y_test_classes)
model.save('two_classification_trained_lstm_model_200epochs.keras')
joblib.dump(scaler, 'two_classification_lstm_scaler.pkl')
joblib.dump(label_encoder, 'two_classification_lstm_label_encoder.pkl')