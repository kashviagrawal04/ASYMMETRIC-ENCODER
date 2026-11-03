"""Paired seed-level comparisons (same seed => same data split for both models)."""
import numpy as np
import pandas as pd
from scipy import stats


def paired_compare(df, metric, a, b, k, key=("model",)):
    """Mean difference a-b, 95% bootstrap CI over seeds, and paired t-test p-value."""
    xa = df[(df.model == a) & (df.k == k)].sort_values("seed")[metric].to_numpy()
    xb = df[(df.model == b) & (df.k == k)].sort_values("seed")[metric].to_numpy()
    n = min(len(xa), len(xb)); xa, xb = xa[:n], xb[:n]
    diff = xa - xb
    rng = np.random.default_rng(0)
    boots = rng.choice(diff, (5000, n), replace=True).mean(1)
    p = stats.ttest_rel(xa, xb).pvalue if n > 1 and diff.std() > 0 else 1.0
    return dict(a=a, b=b, k=k, metric=metric, diff=float(diff.mean()),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), p=float(p), n=n)


def table(df, metric, models, ref, ks):
    rows = [paired_compare(df, metric, m, ref, k) for m in models for k in ks if m != ref]
    return pd.DataFrame(rows)
