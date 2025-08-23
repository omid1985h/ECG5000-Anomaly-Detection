import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_sequences(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def encode_labels(y):
    return LabelEncoder().fit_transform(y)

def create_sequences(data, window_size):
    X_seq, y_seq = [], []
    for seq in data:
        for i in range(len(seq) - window_size):
            X_seq.append(seq[i:i + window_size])
            y_seq.append(seq[i + window_size])
    return np.array(X_seq), np.array(y_seq)
