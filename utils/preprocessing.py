import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_sequences(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def encode_labels(y):
    return LabelEncoder().fit_transform(y)
