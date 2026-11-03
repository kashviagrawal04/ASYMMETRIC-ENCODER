"""Auto-encoder with independently configurable encoder / decoder depth and an optional
variance-covariance latent regulariser (VICReg-style) to counter dimensional collapse."""
import numpy as np
from .nn import MLP, Adam
from .evaluate import effective_rank

D_IN = 64
ENC_H = {0: [], 1: [128], 2: [128, 64]}
DEC_H = {0: [], 1: [128], 2: [64, 128]}
NAMED = {"symmetric": (2, 2), "heavy_enc": (2, 0), "heavy_dec": (0, 2), "linear_sym": (0, 0)}


def latent_reg(z, eps=1e-4):
    """Variance hinge + off-diagonal covariance penalty on a batch of latents.
    Returns (loss, dloss/dz). L = mean_j relu(1-std_j)^2 + (1/k) sum_{a!=b} C_ab^2."""
    B, k = z.shape
    zc = z - z.mean(0)
    s = np.sqrt((zc ** 2).sum(0) / (B - 1) + eps)
    hinge = np.maximum(0.0, 1.0 - s)
    dz = (-2.0 * hinge / k) * zc / ((B - 1) * s)
    C = zc.T @ zc / (B - 1)
    off = C - np.diag(np.diag(C))
    G = 2.0 * off / k
    dz = dz + 2.0 * zc @ G / (B - 1)
    return (hinge ** 2).mean() + (off ** 2).sum() / k, dz


class AutoEncoder:
    """arch: a name in NAMED, or a tuple (encoder_depth, decoder_depth) with depths in {0,1,2}."""

    def __init__(self, arch, k, seed=0):
        ed, dd = NAMED[arch] if isinstance(arch, str) else arch
        linear = arch == "linear_sym"
        rng = np.random.default_rng(seed)
        self.arch, self.k, self.ed, self.dd = arch, k, ed, dd
        hid = "linear" if linear else "relu"
        es = [D_IN] + ENC_H[ed] + [k]
        self.enc = MLP(es, [hid] * len(ENC_H[ed]) + ["linear"], rng)
        ds = [k] + DEC_H[dd] + [D_IN]
        self.dec = MLP(ds, [hid] * len(DEC_H[dd]) + ["linear" if linear else "sigmoid"], rng)
        self.rng = rng
        self.trace = []

    n_enc = property(lambda self: self.enc.n_params)
    n_dec = property(lambda self: self.dec.n_params)

    def encode(self, X): return self.enc.forward(X)
    def reconstruct(self, X): return self.dec.forward(self.enc.forward(X))
    def decode(self, Z): return self.dec.forward(Z)

    def fit(self, X, epochs=300, bs=64, lr=1e-3, corrupt=0.0, reg=0.0, trace_every=0, log=None):
        """MSE training. corrupt>0: masking-noise denoising. reg>0: latent variance/covariance penalty.
        trace_every>0 records (epoch, train mse, latent effective rank) along the way."""
        opt = Adam(self.enc.params() + self.dec.params(), lr=lr)
        n, hist = len(X), []
        self.trace = []
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, bs):
                xb = X[idx[s:s + bs]]
                xin = xb * (self.rng.random(xb.shape) >= corrupt) if corrupt > 0 else xb
                z = self.enc.forward(xin, train=True)
                yhat = self.dec.forward(z, train=True)
                diff = yhat - xb
                tot += (diff ** 2).sum()
                dWd, dbd, dz = self.dec.backward(2.0 * diff / diff.size)
                if reg > 0 and len(xb) >= 4:
                    _, dzr = latent_reg(z)
                    dz = dz + reg * dzr
                dWe, dbe, _ = self.enc.backward(dz)
                opt.step(dWe + dbe + dWd + dbd)
            hist.append(tot / X.size)
            if trace_every and ((ep + 1) % trace_every == 0 or ep == 0):
                self.trace.append((ep + 1, hist[-1], effective_rank(self.encode(X))))
            if log and (ep + 1) % 50 == 0:
                log(f"  ep {ep + 1:4d}  train mse {hist[-1]:.5f}")
        return hist

    def to_dict(self, nd=4):
        """Decoder + encoder weights (rounded) for the JS demo."""
        f = lambda m: dict(W=[np.round(w, nd).tolist() for w in m.W], b=[np.round(b, nd).tolist() for b in m.b], acts=m.acts)
        return dict(arch=str(self.arch), k=self.k, enc=f(self.enc), dec=f(self.dec))
