import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Import your modular loader and preprocessing functions
from utils.loader import load_ecg5000
from utils.preprocessing import normalize_sequences, create_sequences
from utils.general import detect_anomalies, plot_signal_with_anomalies

# --------------------- Load & Preprocess ---------------------
X, y = load_ecg5000()            # uses loader module
X_norm = normalize_sequences(X)  # row-wise normalization

# Separate normal vs anomaly sequences
normal_X = X_norm[y == 1]
anomaly_X = X_norm[y != 1]

# Create sequences for CNN+GRU
window_size = 30
X_seq, y_seq = create_sequences(normal_X, window_size)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# Add channel dimension for Conv1D
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# --------------------- CNN+GRU Model ---------------------
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(window_size,1)),
    Dropout(0.2),
    Conv1D(32, 3, activation='relu'),
    GRU(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Early stopping callback
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[es]
)

# --------------------- Forecast & Anomalies ---------------------
# Test on a single normal sequence
test_signal = normal_X[0]
anomalies, errors, preds, threshold = detect_anomalies(test_signal, model, window_size)
plot_signal_with_anomalies(test_signal, anomalies, "Normal ECG CNN+GRU Anomalies")
