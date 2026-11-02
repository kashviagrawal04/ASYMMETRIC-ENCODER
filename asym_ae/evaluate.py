"""Representation-quality metrics (labels are used ONLY for evaluation)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def probe(Ztr, ytr, Zte, yte, frac=1.0, seed=0):
    """Linear probe (logistic regression) on frozen latents, optionally with a fraction of labels."""
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        sel = []
        for c in np.unique(ytr):
            ci = np.flatnonzero(ytr == c)
            sel.extend(rng.choice(ci, max(1, int(round(frac * len(ci)))), replace=False))
        sel = np.array(sel)
        Ztr, ytr = Ztr[sel], ytr[sel]
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Ztr), ytr)
    return clf.score(sc.transform(Zte), yte)


def knn(Ztr, ytr, Zte, yte, k=5):
    sc = StandardScaler().fit(Ztr)
    return KNeighborsClassifier(k).fit(sc.transform(Ztr), ytr).score(sc.transform(Zte), yte)


def effective_rank(Z):
    """Participation ratio of latent covariance eigenvalues: (sum l)^2 / sum l^2."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())
