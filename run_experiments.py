"""Main sweep: architecture x bottleneck width x seed (+ denoising ablation, + PCA/pixel baselines)."""
import json, sys, time
import numpy as np
from sklearn.decomposition import PCA
from asym_ae.data import load
from asym_ae.models import AutoEncoder, ARCHS
from asym_ae import evaluate as E

SEEDS = [0, 1, 2, 3, 4]
KS = [2, 4, 8, 16, 32]
EPOCHS = 300
Xtr, Xte, ytr, yte = load(seed=0)
rows = []


def log(m): print(m, flush=True)


def evaluate(Ztr, Zte, Xte_rec, seed):
    return dict(
        mse=float(((Xte_rec - Xte) ** 2).mean()),
        probe=E.probe(Ztr, ytr, Zte, yte),
        probe10=E.probe(Ztr, ytr, Zte, yte, frac=0.10, seed=seed),
        knn=E.knn(Ztr, ytr, Zte, yte),
        erank=E.effective_rank(Ztr),
    )


t0 = time.time()
# --- baselines
rows.append(dict(model="pixels", k=64, seed=0, corrupt=0.0, **dict(
    mse=0.0, probe=E.probe(Xtr, ytr, Xte, yte), probe10=E.probe(Xtr, ytr, Xte, yte, 0.10, 0),
    knn=E.knn(Xtr, ytr, Xte, yte), erank=E.effective_rank(Xtr)), n_enc=0, n_dec=0))
for k in KS:
    p = PCA(k).fit(Xtr)
    Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    rec = p.inverse_transform(Zte)
    rows.append(dict(model="pca", k=k, seed=0, corrupt=0.0, **evaluate(Ztr, Zte, rec, 0), n_enc=64 * k, n_dec=64 * k))
log("baselines done")

# --- autoencoders
for corrupt in [0.0, 0.3]:
    ks = KS if corrupt == 0.0 else [8, 16]
    for arch in ARCHS:
        for k in ks:
            for seed in SEEDS:
                ae = AutoEncoder(arch, k, seed=seed)
                ae.fit(Xtr, epochs=EPOCHS, corrupt=corrupt)
                Ztr, Zte = ae.encode(Xtr), ae.encode(Xte)
                m = evaluate(Ztr, Zte, ae.reconstruct(Xte), seed)
                rows.append(dict(model=arch, k=k, seed=seed, corrupt=corrupt, n_enc=ae.n_enc, n_dec=ae.n_dec, **m))
            sub = [r for r in rows if r["model"] == arch and r["k"] == k and r["corrupt"] == corrupt]
            log(f"[{time.time()-t0:6.0f}s] corrupt={corrupt} {arch:10s} k={k:2d} "
                f"mse={np.mean([r['mse'] for r in sub]):.4f} probe={np.mean([r['probe'] for r in sub]):.3f} "
                f"probe10={np.mean([r['probe10'] for r in sub]):.3f} knn={np.mean([r['knn'] for r in sub]):.3f}")
    json.dump(rows, open("results/results.json", "w"), indent=1)

# --- 2-D latents for visualisation (seed 0, k=2, clean)
lat = {}
for arch in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]:
    ae = AutoEncoder(arch, 2, seed=0); ae.fit(Xtr, epochs=EPOCHS)
    lat[arch] = ae.encode(Xte).tolist()
lat["labels"] = yte.tolist()
json.dump(lat, open("results/latents2d.json", "w"))
json.dump(rows, open("results/results.json", "w"), indent=1)
log(f"done in {time.time()-t0:.0f}s")
