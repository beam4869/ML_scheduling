import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import keras_tuner as kt

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :96].values  # First 96 columns are features
    y = df.iloc[:, 106].values  # 107th column is the classification
    return X, y

# Preprocess dataset
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape into (samples, timesteps, features) for LSTM
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 48, 2)  # 48 time steps, 2 features per time step
    return X_reshaped, y, scaler

# Build LSTM Model
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), return_sequences=False, input_shape=(48, 2)))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Perform time series cross-validation
def time_series_cv(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    histories = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        tuner = kt.Hyperband(build_lstm_model, objective='val_accuracy', max_epochs=10, factor=3)
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
        histories.append(history)
    return histories

# Run the pipeline
filepath = 'total dataset for ML with statistic numbers.csv'
X, y = load_data(filepath)
X, y, scaler = preprocess_data(X, y)
histories = time_series_cv(X, y)
print("histories: ", histories)