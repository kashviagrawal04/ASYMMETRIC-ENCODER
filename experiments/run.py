"""Experiment runner.  Usage: python experiments/run.py STAGE [STAGE ...]
stages: main | denoise | reg | grid | diagnose | control
Every stage writes results/<stage>.json incrementally. Splits depend on the seed, so seed-to-seed
variance includes data-split variance, and comparisons between models are paired by seed."""
import json, os, sys, time
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asym_ae.data import load_split, split_validation
from asym_ae.models import AutoEncoder, NAMED
from asym_ae import evaluate as E

SEEDS = [0, 1, 2, 3, 4]
KS = [2, 4, 8, 16, 32]
EPOCHS = 300
T0 = time.time()
os.makedirs("results", exist_ok=True)


def log(m): print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
def save(stage, rows): json.dump(rows, open(f"results/{stage}.json", "w"), indent=1)


def train_eval(arch, k, seed, d, **fit_kw):
    ae = AutoEncoder(arch, k, seed=seed)
    ae.fit(d["Xtr"], epochs=fit_kw.pop("epochs", EPOCHS), **fit_kw)
    m = E.full_eval(ae.encode, ae.reconstruct, d, seed)
    return dict(n_enc=ae.n_enc, n_dec=ae.n_dec, **m)


def stage_main():
    rows = []
    for seed in SEEDS:
        d = load_split(seed)
        rows.append(dict(model="pixels", k=64, seed=seed, n_enc=0, n_dec=0,
                         **E.full_eval(lambda X: X, lambda X: X, d, seed)))
        for k in KS:
            p = PCA(k, random_state=seed).fit(d["Xtr"])
            rows.append(dict(model="pca", k=k, seed=seed, n_enc=64 * k, n_dec=64 * k,
                             **E.full_eval(p.transform, lambda X, p=p: p.inverse_transform(p.transform(X)), d, seed)))
    log("baselines done"); save("main", rows)
    for arch in NAMED:
        for k in KS:
            for seed in SEEDS:
                rows.append(dict(model=arch, k=k, seed=seed, **train_eval(arch, k, seed, load_split(seed))))
            sub = [r for r in rows if r["model"] == arch and r["k"] == k]
            log(f"main {arch:10s} k={k:2d} probe={np.mean([r['probe'] for r in sub]):.3f} erank={np.mean([r['erank'] for r in sub]):.2f}")
            save("main", rows)


def stage_denoise():
    rows = []
    for arch in NAMED:
        for k in [8, 16]:
            for seed in SEEDS:
                rows.append(dict(model=arch, k=k, seed=seed, corrupt=0.3,
                                 **train_eval(arch, k, seed, load_split(seed), corrupt=0.3)))
            log(f"denoise {arch} k={k}"); save("denoise", rows)


def stage_reg():
    lam = json.load(open("results/diagnose_selected.json"))["lambda"] if os.path.exists("results/diagnose_selected.json") else 0.1
    log(f"regularised sweep with lambda={lam}")
    rows = []
    for arch in NAMED:
        for k in [2, 4, 8, 16]:
            for seed in SEEDS:
                rows.append(dict(model=arch + "+reg", k=k, seed=seed, reg=lam,
                                 **train_eval(arch, k, seed, load_split(seed), reg=lam)))
            log(f"reg {arch} k={k}"); save("reg", rows)


def stage_grid():
    rows = []
    for ed in [0, 1, 2]:
        for dd in [0, 1, 2]:
            for k in [2, 4, 8]:
                for seed in SEEDS[:3]:
                    rows.append(dict(model=f"e{ed}d{dd}", ed=ed, dd=dd, k=k, seed=seed,
                                     **train_eval((ed, dd), k, seed, load_split(seed))))
                log(f"grid e{ed}d{dd} k={k}"); save("grid", rows)


def stage_control():
    """Untrained (random-initialisation) encoders: how much of the probe accuracy comes from training?"""
    rows = []
    for arch in NAMED:
        for k in KS:
            for seed in SEEDS:
                d = load_split(seed)
                ae = AutoEncoder(arch, k, seed=seed)          # no fit()
                rows.append(dict(model=arch + " (random init)", k=k, seed=seed, **E.full_eval(ae.encode, ae.reconstruct, d, seed)))
            log(f"control {arch} k={k}"); save("control", rows)


def stage_diagnose():
    """Why does a deep encoder + light decoder collapse? Test remedies. Regulariser strength is chosen on a
    validation split carved from TRAINING data; the test set is only used for the final report."""
    variants = {  # name: (arch, kwargs)
        "base": ("heavy_enc", {}),
        "1000 epochs": ("heavy_enc", dict(epochs=1000)),
        "lr 3e-3": ("heavy_enc", dict(lr=3e-3)),
        "reg 0.01": ("heavy_enc", dict(reg=0.01)),
        "reg 0.1": ("heavy_enc", dict(reg=0.1)),
        "reg 1.0": ("heavy_enc", dict(reg=1.0)),
        "1-layer decoder": ((2, 1), {}),
        "2-layer decoder (symmetric)": ("symmetric", {}),
    }
    rows = []
    for k in [2, 4]:
        for name, (arch, kw) in variants.items():
            for seed in SEEDS[:3]:
                d = load_split(seed)
                Xf, yf, Xv, yv = split_validation(d, seed)
                kw2 = dict(kw); ep = kw2.pop("epochs", EPOCHS)
                ae = AutoEncoder(arch, k, seed=seed)
                ae.fit(Xf, epochs=ep, trace_every=25, **kw2)
                v = E.quick_eval(ae.encode, ae.reconstruct, Xf, yf, Xv, yv)
                # test metrics from a model trained on the full train split (same hyper-parameters)
                ae2 = AutoEncoder(arch, k, seed=seed)
                ae2.fit(d["Xtr"], epochs=ep, trace_every=25, **kw2)
                t = E.quick_eval(ae2.encode, ae2.reconstruct, d["Xtr"], d["ytr"], d["Xte"], d["yte"])
                rows.append(dict(variant=name, k=k, seed=seed, val_probe=v["probe"], val_erank=v["erank"],
                                 test_probe=t["probe"], test_knn=t["knn"], test_mse=t["mse"], test_erank=t["erank"],
                                 trace=[(int(a), float(b), float(c)) for a, b, c in ae2.trace]))
            sub = [r for r in rows if r["variant"] == name and r["k"] == k]
            log(f"diagnose k={k} {name:28s} val_probe={np.mean([r['val_probe'] for r in sub]):.3f} erank={np.mean([r['test_erank'] for r in sub]):.2f}")
            save("diagnose", rows)
    # select lambda on validation probe (mean over k and seeds)
    best, bv = None, -1
    for lam in [0.01, 0.1, 1.0]:
        vals = [r["val_probe"] for r in rows if r["variant"] == f"reg {lam}"]
        if np.mean(vals) > bv: best, bv = lam, float(np.mean(vals))
    json.dump(dict(**{"lambda": best}, val_probe=bv), open("results/diagnose_selected.json", "w"))
    log(f"selected lambda={best} (val probe {bv:.3f})")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        globals()[f"stage_{s}"]()
    log("done")
