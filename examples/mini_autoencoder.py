"""Minimal autoencoder in pure NumPy: forward pass, hand-written backprop, plain gradient descent.
A readable 60-line version of what asym_ae/ does with reusable classes, Adam and mini-batches.
Run: python examples/mini_autoencoder.py   (needs numpy and scikit-learn)
"""
import numpy as np
from sklearn.datasets import load_digits

X = load_digits().data / 16.0                 # 1797 images, 64 pixels each, scaled to [0, 1]
rng = np.random.default_rng(0)
k = 8                                         # bottleneck size (the "code")

# weights: encoder 64 -> 32 -> k, decoder k -> 32 -> 64
W1 = rng.normal(0, np.sqrt(2/64), (64, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(1/32), (32, k));  b2 = np.zeros(k)
W3 = rng.normal(0, np.sqrt(2/k),  (k, 32));  b3 = np.zeros(32)
W4 = rng.normal(0, np.sqrt(1/32), (32, 64)); b4 = np.zeros(64)

relu    = lambda z: np.maximum(z, 0)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

lr = 0.1
for epoch in range(301):
    # ---------- forward pass ----------
    h1 = relu(X @ W1 + b1)                    # encoder hidden layer
    z  = h1 @ W2 + b2                         # THE CODE: k numbers per image
    h2 = relu(z @ W3 + b3)                    # decoder hidden layer
    out = sigmoid(h2 @ W4 + b4)               # reconstructed image
    loss = ((out - X) ** 2).mean()

    # ---------- backward pass (chain rule, from the end to the start) ----------
    d_out = 2 * (out - X) / X.size
    d_a4  = d_out * out * (1 - out)           # through the sigmoid
    dW4, db4 = h2.T @ d_a4, d_a4.sum(0)
    d_h2  = d_a4 @ W4.T
    d_a3  = d_h2 * (h2 > 0)                   # through the ReLU
    dW3, db3 = z.T @ d_a3, d_a3.sum(0)
    d_z   = d_a3 @ W3.T
    dW2, db2 = h1.T @ d_z, d_z.sum(0)
    d_h1  = d_z @ W2.T
    d_a1  = d_h1 * (h1 > 0)
    dW1, db1 = X.T @ d_a1, d_a1.sum(0)

    # ---------- update: move each weight a little against its gradient ----------
    for p, g in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2),(W3,dW3),(b3,db3),(W4,dW4),(b4,db4)]:
        p -= lr * g

    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  reconstruction error {loss:.4f}")

print("code shape:", z.shape, "-> each image is now just", k, "numbers")
