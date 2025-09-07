import numpy as np
import matplotlib.pyplot as plt


def create_windows(data, window_size):
    windows = []
    for seq in data:
        for i in range(len(seq) - window_size + 1):
            windows.append(seq[i:i + window_size])
    return np.array(windows)

# --------------------- Prediction for Sliding Window Models ---------------------
def predict_sequence(model, sequence, window_size):
    preds = []
    for i in range(len(sequence) - window_size):
        input_seq = sequence[i:i + window_size].reshape(1, window_size, 1)
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0][0])
    return np.array(preds)

# --------------------- Anomaly Detection ---------------------
def detect_anomalies(signal, model, window_size, threshold=None):
    preds = predict_sequence(model, signal, window_size)
    actual = signal[window_size:]
    errors = np.abs(actual - preds)
    if threshold is None:
        threshold = np.mean(errors) + 3 * np.std(errors)
    anomalies = np.where(errors > threshold)[0] + window_size
    return anomalies, errors, preds, threshold

def plot_signal_with_anomalies(signal, anomalies, title="ECG Signal"):
    plt.figure(figsize=(12,4))
    plt.plot(signal, label='ECG Signal')
    plt.scatter(anomalies, signal[anomalies], color='red', label='Anomalies')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
