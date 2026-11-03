"""Data: scikit-learn Digits (8x8) with per-seed stratified splits and test-time distribution shifts."""
import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def shift_images(X, dx=1):
    """Shift every 8x8 image `dx` pixels to the right (zero fill)."""
    I = X.reshape(-1, 8, 8)
    out = np.zeros_like(I)
    out[:, :, dx:] = I[:, :, :8 - dx]
    return out.reshape(len(X), 64)


def rotate_images(X, deg=15.0):
    I = X.reshape(-1, 8, 8)
    out = np.stack([rotate(im, deg, reshape=False, order=1, mode="constant", cval=0.0) for im in I])
    return np.clip(out, 0, 1).reshape(len(X), 64)


def noisy_images(X, sigma=0.15, seed=0):
    rng = np.random.default_rng(10_000 + seed)
    return np.clip(X + rng.normal(0, sigma, X.shape), 0, 1)


def load_split(seed=0, test_size=0.25):
    """Stratified 75/25 split that depends on `seed`; also returns shifted / rotated / noisy test copies."""
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                shifted=shift_images(Xte), rotated=rotate_images(Xte), noisy=noisy_images(Xte, seed=seed))


def split_validation(d, seed=0, val_frac=0.2):
    """Carve a validation set out of the *training* data (used for hyper-parameter selection)."""
    Xf, Xv, yf, yv = train_test_split(d["Xtr"], d["ytr"], test_size=val_frac, random_state=seed, stratify=d["ytr"])
    return Xf, yf, Xv, yv
