import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load(seed=0, test_size=0.25):
    """sklearn Digits (8x8, 10 classes), scaled to [0,1], stratified split."""
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
