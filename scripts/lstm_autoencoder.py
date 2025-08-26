import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Import your modular loader and preprocessing functions
from utils.loader import load_ecg5000
from utils.preprocessing import normalize_sequences, create_windows
from utils.general import plot_signal_with_anomalies

# --------------------- Load & Preprocess ---------------------
X, y = load_ecg5000()            # uses loader module
X_norm = normalize_sequences(X)  # row-wise normalization

# Separate normal vs anomaly sequences
normal_X = X_norm[y == 1]
anomaly_X = X_norm[y != 1]

# Create windows for Autoencoder
window_size = 30
normal_windows = create_windows(normal_X, window_size)
anomaly_windows = create_windows(anomaly_X, window_size)

# Add channel dimension
normal_windows = normal_windows[..., np.newaxis]
anomaly_windows = anomaly_windows[..., np.newaxis]

# Split into training and validation sets
X_train, X_val = train_test_split(normal_windows, test_size=0.2, random_state=42)

# --------------------- LSTM Autoencoder ---------------------
inputs = Input(shape=(window_size,1))
encoded = LSTM(32, activation='relu', return_sequences=False)(inputs)
encoded = RepeatVector(window_size)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Early stopping callback
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the autoencoder
history_ae = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=[es]
)

# --------------------- Reconstruction & Threshold ---------------------
def reconstruction_errors(model, data):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=(1,2))
    return mse

normal_errors = reconstruction_errors(autoencoder, normal_windows)
threshold = normal_errors.mean() + 3 * normal_errors.std()

# --------------------- Detect anomalies ---------------------
def detect_seq_anomalies(seq):
    windows = create_windows(seq[np.newaxis, :], window_size)
    windows = windows[..., np.newaxis]
    errors = reconstruction_errors(autoencoder, windows)
    anomaly_indices = np.where(errors > threshold)[0]
    return anomaly_indices

anomalies = detect_seq_anomalies(normal_X[0])
plot_signal_with_anomalies(normal_X[0], anomalies, "Normal ECG LSTM-AE Anomalies")

