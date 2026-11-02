"""Autoencoder with independently configurable encoder / decoder capacity."""
import numpy as np
from .nn import MLP, Adam

D_IN = 64

# (encoder hidden widths, decoder hidden widths). Bottleneck is linear; output is sigmoid.
ARCHS = {
    "symmetric":        dict(enc=[128, 64], dec=[64, 128], desc="deep encoder / deep decoder"),
    "heavy_enc":        dict(enc=[128, 64], dec=[],        desc="deep encoder / single-layer decoder"),
    "heavy_dec":        dict(enc=[],        dec=[64, 128], desc="single-layer encoder / deep decoder"),
    "linear_sym":       dict(enc=[],        dec=[],        desc="linear encoder / linear decoder", linear=True),
}


class AutoEncoder:
    def __init__(self, arch, k, seed=0):
        cfg = ARCHS[arch]
        rng = np.random.default_rng(seed)
        self.arch, self.k = arch, k
        hid_act = "linear" if cfg.get("linear") else "relu"
        es = [D_IN] + cfg["enc"] + [k]
        self.enc = MLP(es, [hid_act] * len(cfg["enc"]) + ["linear"], rng)
        ds = [k] + cfg["dec"] + [D_IN]
        out_act = "linear" if cfg.get("linear") else "sigmoid"
        self.dec = MLP(ds, [hid_act] * len(cfg["dec"]) + [out_act], rng)
        self.rng = rng

    @property
    def n_enc(self): return self.enc.n_params
    @property
    def n_dec(self): return self.dec.n_params

    def encode(self, X): return self.enc.forward(X)
    def reconstruct(self, X): return self.dec.forward(self.enc.forward(X))

    def fit(self, X, epochs=300, bs=64, lr=1e-3, corrupt=0.0, log=None):
        """Train with MSE. corrupt>0 -> masking-noise denoising objective (input masked, target clean)."""
        opt = Adam(self.enc.params() + self.dec.params(), lr=lr)
        n, hist = len(X), []
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
                dy = 2.0 * diff / diff.size
                dWd, dbd, dz = self.dec.backward(dy)
                dWe, dbe, _ = self.enc.backward(dz)
                opt.step(dWe + dbe + dWd + dbd)
            hist.append(tot / X.size)
            if log and (ep + 1) % 50 == 0:
                log(f"  ep {ep+1:4d}  train mse {hist[-1]:.5f}")
        return hist
