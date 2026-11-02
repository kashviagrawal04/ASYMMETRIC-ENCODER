"""Minimal NumPy neural-network toolkit: dense layers, manual backprop, Adam."""
import numpy as np


def _act(name, z):
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if name == "linear":
        return z
    raise ValueError(name)


def _act_grad(name, a, z):
    if name == "relu":
        return (z > 0).astype(a.dtype)
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "linear":
        return np.ones_like(a)
    raise ValueError(name)


class MLP:
    """Fully-connected net. `sizes` e.g. [64,128,64,8]; `acts` one per layer."""

    def __init__(self, sizes, acts, rng):
        assert len(acts) == len(sizes) - 1
        self.acts = acts
        self.W, self.b = [], []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / m) if acts[i] == "relu" else np.sqrt(1.0 / m)  # He / LeCun
            self.W.append(rng.normal(0, scale, (m, n)))
            self.b.append(np.zeros(n))
        self._cache = []

    @property
    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def forward(self, x, train=False):
        self._cache = []
        a = x
        for W, b, act in zip(self.W, self.b, self.acts):
            z = a @ W + b
            out = _act(act, z)
            if train:
                self._cache.append((a, z, out))
            a = out
        return a

    def backward(self, dout):
        """Return grads (dW list, db list) and gradient w.r.t. the input."""
        dWs, dbs = [None] * len(self.W), [None] * len(self.W)
        g = dout
        for i in reversed(range(len(self.W))):
            a_in, z, out = self._cache[i]
            dz = g * _act_grad(self.acts[i], out, z)
            dWs[i] = a_in.T @ dz
            dbs[i] = dz.sum(0)
            g = dz @ self.W[i].T
        return dWs, dbs, g

    def params(self):
        return self.W + self.b


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
