import os
import numpy as np

def load_ecg5000(desktop_path=None):
    if desktop_path is None:
        desktop_path = os.path.expanduser("~/Desktop/ECG5000")
    train_file = os.path.join(desktop_path, "ECG5000_TRAIN.txt")
    test_file = os.path.join(desktop_path, "ECG5000_TEST.txt")
    
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)
    data = np.concatenate([train_data, test_data], axis=0)
    
    X = data[:, 1:]
    y = data[:, 0]
    return X, y
