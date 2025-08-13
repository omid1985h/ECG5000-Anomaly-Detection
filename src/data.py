import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_ecg5000_data(base_path=None):
    """
    Load ECG5000 dataset (train + test) from base_path or default Desktop location.
    
    Returns:
        X: np.ndarray, ECG signals
        y: np.ndarray, class labels (encoded)
        label_encoder: sklearn LabelEncoder fitted on y
    """
    if base_path is None:
        base_path = os.path.expanduser("~/Desktop/ECG5000")
    
    train_file = os.path.join(base_path, "ECG5000_TRAIN.txt")
    test_file = os.path.join(base_path, "ECG5000_TEST.txt")
    
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)
    
    data = np.concatenate([train_data, test_data], axis=0)
    X = data[:, 1:]
    y = data[:, 0]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize signals per sample (row-wise)
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    
    return X_norm, y_encoded, label_encoder
