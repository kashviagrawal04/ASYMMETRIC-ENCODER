"""Figures and summary tables from results/*.json."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asym_ae.style import apply, COL, NAME
from asym_ae import stats

apply(); os.makedirs("figures", exist_ok=True)
load = lambda s: pd.DataFrame(json.load(open(f"results/{s}.json"))) if os.path.exists(f"results/{s}.json") else None
M, DN, RG, GR, DG = load("main"), load("denoise"), load("reg"), load("grid"), load("diagnose")
ORDER = ["pca", "linear_sym", "symmetric", "heavy_dec", "heavy_enc"]
KS = [2, 4, 8, 16, 32]
SUB = {"probe": "Linear-probe accuracy", "lab10": "Probe accuracy, 10% labels", "knn": "5-NN accuracy", "nmi": "K-means NMI (label-free clustering)",
       "trust": "Trustworthiness (10-NN)", "erank": "Latent effective rank", "mse": "Test reconstruction MSE",
       "rob_shift": "Probe acc., 1-px shifted test", "rob_rot": "Probe acc., 15° rotated test", "rob_noise": "Probe acc., noisy test (σ=0.15)"}


def agg(df, by=("model", "k")):
    return df.groupby(list(by)).agg(**{f"{m}_{s}": (m, f) for m in SUB for s, f in [("m", "mean"), ("s", "std")] if m in df}).reset_index()


def curve(ax, A, metric, models=ORDER, log=False):
    for m in models:
        d = A[A.model == m].sort_values("k")
        if d.empty: continue
        ax.errorbar(d.k, d[f"{metric}_m"], yerr=d[f"{metric}_s"], color=COL[m], lw=2, marker="o", ms=4.5, capsize=2,
                    ls="--" if m == "pca" else "-", label=NAME[m])
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
    if log: ax.set_yscale("log")
    ax.set_xlabel("Bottleneck width k"); ax.set_title(SUB[metric], loc="left", fontsize=10)


if M is not None:
    A = agg(M); A.to_csv("results/summary_main.csv", index=False)
    pix = M[M.model == "pixels"][["probe", "lab10", "knn", "nmi", "trust", "rob_shift", "rob_rot", "rob_noise"]].mean()
    for fname, metrics, shape, size in [("fig_main_metrics.png", ["probe", "lab10", "knn", "nmi", "trust", "erank"], (2, 3), (13, 7)),
                                          ("fig_robustness.png", ["rob_shift", "rob_rot", "rob_noise"], (1, 3), (13, 3.8)),
                                          ("fig_mse.png", ["mse"], (1, 1), (6.2, 4))]:
        fig, axs = plt.subplots(*shape, figsize=size); axs = np.atleast_1d(axs).ravel()
        for ax, m in zip(axs, metrics):
            curve(ax, A, m, log=(m == "mse"))
            if m in pix: ax.axhline(pix[m], color=COL["pixels"], lw=1.2, ls=":", label=NAME["pixels"])
        axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
        fig.tight_layout(); fig.savefig(f"figures/{fname}", dpi=200); plt.close(fig)
    # label efficiency
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    fr = [("lab1", .01), ("lab5", .05), ("lab10", .10), ("lab25", .25), ("probe", 1.0)]
    for ax, k in zip(axs, [2, 4, 8]):
        for m in ORDER:
            d = M[(M.model == m) & (M.k == k)]
            ax.plot([f for _, f in fr], [d[c].mean() for c, _ in fr], color=COL[m], lw=2, marker="o", ms=4, ls="--" if m == "pca" else "-", label=NAME[m])
        ax.set_xscale("log"); ax.set_xticks([.01, .05, .1, .25, 1]); ax.set_xticklabels(["1%", "5%", "10%", "25%", "100%"])
        ax.set_xlabel("Fraction of training labels used by the probe"); ax.set_title(f"k = {k}", loc="left", fontsize=10)
    axs[0].set_ylabel("Probe test accuracy"); axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig("figures/fig_label_efficiency.png", dpi=200); plt.close(fig)
    # paired statistics vs symmetric and vs PCA
    rows = [stats.table(M, mt, ["heavy_enc", "heavy_dec", "linear_sym", "pca"], "symmetric", KS).assign(ref="symmetric") for mt in ["probe", "lab10", "nmi", "rob_shift", "erank"]]
    st = pd.concat(rows); st.to_csv("results/stats_vs_symmetric.csv", index=False)
    print(st[st.metric == "probe"].round(4).to_string(index=False))
    print(A.pivot(index="k", columns="model", values="probe_m").round(3))

if GR is not None:
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for j, k in enumerate([2, 4, 8]):
        for i, (mt, cm, fmt) in enumerate([("probe", "Blues", "{:.2f}"), ("erank", "Greens", "{:.1f}")]):
            G = GR[GR.k == k].groupby(["ed", "dd"])[mt].mean().unstack()
            ax = axs[i, j]; ax.grid(False)
            im = ax.imshow(G.values, cmap=cm, vmin=G.values.min() * .98, vmax=G.values.max() * 1.01, origin="upper")
            for a in range(3):
                for b in range(3):
                    v = G.values[a, b]; ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=10, color="white" if (v - G.values.min()) / (np.ptp(G.values) + 1e-9) > .6 else "#0b0b0b")
            ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["linear", "1 hidden", "2 hidden"]); ax.set_yticklabels(["linear", "1 hidden", "2 hidden"])
            ax.set_xlabel("Decoder depth"); ax.set_ylabel("Encoder depth")
            ax.set_title(("Probe accuracy" if mt == "probe" else "Effective rank") + f", k={k}", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig_depth_grid.png", dpi=200); plt.close(fig)
    g = GR.groupby(["model", "k"])[["probe", "erank", "mse", "n_enc", "n_dec"]].mean().round(4); g.to_csv("results/summary_grid.csv")

if DG is not None:
    VC = {"base": "#eb6834", "1000 epochs": "#eda100", "lr 3e-3": "#e87ba4", "reg 0.01": "#9085e9", "reg 0.1": "#6250d6", "reg 1.0": "#4a3aa7",
          "1-layer decoder": "#1baf7a", "2-layer decoder (symmetric)": "#2a78d6"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k in zip(axs, [2, 4]):
        for v, c in VC.items():
            rs = DG[(DG.variant == v) & (DG.k == k)]
            tr = np.array([[t[2] for t in r] for r in rs.trace]); ep = [t[0] for t in rs.trace.iloc[0]]
            ax.plot(ep, tr.mean(0), color=c, lw=3.2 if v == "base" else 2, ls=":" if v == "base" else "-", zorder=5 if v == "base" else 2, label=v)
        ax.set_xscale("log"); ax.set_xlabel("Training epoch"); ax.set_ylabel("Latent effective rank"); ax.axhline(k, color="#c3c2b7", ls=":", lw=1)
        ax.set_title(f"Collapse diagnosis, k = {k} (mean of 3 seeds)", loc="left", fontsize=10)
    axs[1].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout(); fig.savefig("figures/fig_collapse_trace.png", dpi=200); plt.close(fig)
    t = DG.groupby(["k", "variant"])[["val_probe", "test_probe", "test_knn", "test_mse", "test_erank"]].mean().round(4)
    t.to_csv("results/summary_diagnose.csv"); print(t.to_string())

if RG is not None and M is not None:
    both = pd.concat([M[M.model.isin(["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]) & M.k.isin([2, 4, 8, 16])], RG])
    A2 = agg(both)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, mt in zip(axs, ["probe", "erank", "mse"]):
        for m in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]:
            for suffix, ls in [("", "-"), ("+reg", "--")]:
                d = A2[A2.model == m + suffix].sort_values("k")
                if d.empty: continue
                ax.errorbar(d.k, d[f"{mt}_m"], yerr=d[f"{mt}_s"], color=COL[m], ls=ls, lw=2, marker="o", ms=4, capsize=2, label=NAME[m] + (" + reg" if suffix else ""))
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels([2, 4, 8, 16]); ax.set_xlabel("Bottleneck width k")
        if mt == "mse": ax.set_yscale("log")
        ax.set_title(SUB[mt], loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout(); fig.savefig("figures/fig_regularised.png", dpi=200); plt.close(fig)
    A2.to_csv("results/summary_reg.csv", index=False)

if DN is not None and M is not None:
    base = M[M.model.isin(["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]) & M.k.isin([8, 16])]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    archs = ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]
    for ax, k in zip(axs, [8, 16]):
        x = np.arange(4); w = .38
        for j, (df, alpha) in enumerate([(base, 1.0), (DN, .55)]):
            v = [df[(df.model == m) & (df.k == k)].probe.mean() for m in archs]; s = [df[(df.model == m) & (df.k == k)].probe.std() for m in archs]
            ax.bar(x + (j - .5) * w, v, w - .04, yerr=s, capsize=2, alpha=alpha, color=[COL[m] for m in archs])
        ax.set_xticks(x); ax.set_xticklabels(["Symm.", "Heavy enc", "Heavy dec", "Linear"], fontsize=9); ax.set_ylim(.7, 1); ax.set_title(f"k = {k}", loc="left", fontsize=10)
    axs[0].set_ylabel("Linear-probe accuracy")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#52514e", label="clean input"), Patch(fc="#52514e", alpha=.55, label="30% masking noise")], frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig("figures/fig_denoise.png", dpi=200); plt.close(fig)
print("figures done")
