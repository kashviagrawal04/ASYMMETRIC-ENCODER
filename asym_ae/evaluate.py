"""Representation-quality metrics. Labels are used ONLY here, never in training."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

LABEL_FRACS = {"lab1": 0.01, "lab5": 0.05, "lab10": 0.10, "lab25": 0.25}


def effective_rank(Z):
    """Participation ratio of latent covariance eigenvalues, (sum l)^2 / sum l^2. Equals k for an
    isotropic code and tends to 1 when the code collapses onto a line."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    s2 = (lam ** 2).sum()
    return float(lam.sum() ** 2 / s2) if s2 > 0 else 1.0


class LinearProbe:
    """Standardise + logistic regression, fit once on frozen train latents, scored on any latent set."""

    def __init__(self, Ztr, ytr):
        self.sc = StandardScaler().fit(Ztr)
        self.clf = LogisticRegression(max_iter=3000).fit(self.sc.transform(Ztr), ytr)

    def score(self, Z, y):
        return float(self.clf.score(self.sc.transform(Z), y))


def subsample_per_class(y, frac, seed):
    rng = np.random.default_rng(seed)
    sel = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        sel.extend(rng.choice(ci, max(1, int(round(frac * len(ci)))), replace=False))
    return np.array(sel)


def probe_with_fraction(Ztr, ytr, Zte, yte, frac, seed=0):
    sel = subsample_per_class(ytr, frac, seed)
    return LinearProbe(Ztr[sel], ytr[sel]).score(Zte, yte)


def knn_acc(Ztr, ytr, Zte, yte, k=5):
    sc = StandardScaler().fit(Ztr)
    return float(KNeighborsClassifier(k).fit(sc.transform(Ztr), ytr).score(sc.transform(Zte), yte))


def cluster_nmi(Z, y, seed=0):
    """K-means (10 clusters) on standardised latents vs true classes: label-free clustering quality."""
    Zs = StandardScaler().fit_transform(Z)
    return float(normalized_mutual_info_score(y, KMeans(10, n_init=5, random_state=seed).fit_predict(Zs)))


def quick_eval(encode, reconstruct, Xtr, ytr, Xte, yte):
    Ztr, Zte = encode(Xtr), encode(Xte)
    return dict(mse=float(((reconstruct(Xte) - Xte) ** 2).mean()),
                probe=LinearProbe(Ztr, ytr).score(Zte, yte),
                knn=knn_acc(Ztr, ytr, Zte, yte), erank=effective_rank(Ztr))


def full_eval(encode, reconstruct, d, seed=0):
    """All metrics for one representation. `d` comes from data.load_split."""
    Ztr, Zte = encode(d["Xtr"]), encode(d["Xte"])
    pr = LinearProbe(Ztr, d["ytr"])
    out = dict(
        mse=float(((reconstruct(d["Xte"]) - d["Xte"]) ** 2).mean()),
        probe=pr.score(Zte, d["yte"]),
        knn=knn_acc(Ztr, d["ytr"], Zte, d["yte"]),
        nmi=cluster_nmi(Zte, d["yte"], seed),
        trust=float(trustworthiness(d["Xte"], Zte, n_neighbors=10)),
        erank=effective_rank(Ztr),
        rob_shift=pr.score(encode(d["shifted"]), d["yte"]),
        rob_rot=pr.score(encode(d["rotated"]), d["yte"]),
        rob_noise=pr.score(encode(d["noisy"]), d["yte"]),
    )
    for name, frac in LABEL_FRACS.items():
        out[name] = probe_with_fraction(Ztr, d["ytr"], Zte, d["yte"], frac, seed)
    return out
