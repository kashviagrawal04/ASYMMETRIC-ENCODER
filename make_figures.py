import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pd.DataFrame(json.load(open("results/results.json")))
L = json.load(open("results/latents2d.json"))
COL = {"symmetric": "#2a78d6", "heavy_enc": "#eb6834", "heavy_dec": "#1baf7a", "linear_sym": "#6250d6", "pca": "#52514e"}
NAME = {"symmetric": "Symmetric (deep/deep)", "heavy_enc": "Heavy encoder / light decoder",
        "heavy_dec": "Light encoder / heavy decoder", "linear_sym": "Linear AE", "pca": "PCA"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
clean = R[R.corrupt == 0.0]
agg = clean.groupby(["model", "k"]).agg(**{f"{m}_{s}": (m, f) for m in ["mse", "probe", "probe10", "knn", "erank"]
                                              for s, f in [("m", "mean"), ("s", "std")]}).reset_index()
agg.to_csv("results/summary_clean.csv", index=False)

def curve(metric, ylabel, fname, log=False, title=None):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for m in ["pca", "linear_sym", "symmetric", "heavy_dec", "heavy_enc"]:
        d = agg[agg.model == m].sort_values("k")
        ax.errorbar(d.k, d[f"{metric}_m"], yerr=d[f"{metric}_s"], color=COL[m], lw=2, marker="o", ms=5,
                    capsize=2, label=NAME[m], ls="--" if m == "pca" else "-")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
    if log: ax.set_yscale("log")
    ax.set_xlabel("Bottleneck width k"); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"figures/{fname}", dpi=200); plt.close(fig)

curve("probe", "Linear-probe test accuracy", "fig_probe.png", title="Linear probe on frozen latents (mean ± sd, 5 seeds)")
curve("probe10", "Linear-probe accuracy, 10% labels", "fig_probe10.png", title="Low-label probe (10% of training labels)")
curve("knn", "5-NN test accuracy", "fig_knn.png", title="k-NN on frozen latents")
curve("mse", "Test reconstruction MSE (log)", "fig_mse.png", log=True, title="Reconstruction error")
curve("erank", "Effective rank of latents", "fig_erank.png", title="Latent effective rank (participation ratio)")

# 2-D latent scatter
y = np.array(L["labels"])
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
cm = plt.get_cmap("tab10")
for ax, m in zip(axs, ["linear_sym", "symmetric", "heavy_enc", "heavy_dec"]):
    Z = np.array(L[m]); ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=7, alpha=.85, linewidths=0)
    ax.set_title(NAME[m], fontsize=9.5, loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.tight_layout(); fig.savefig("figures/fig_latent2d.png", dpi=200); plt.close(fig)

# denoising ablation k=8,16
dn = R[(R.model.isin(["symmetric", "heavy_enc", "heavy_dec", "linear_sym"])) & R.k.isin([8, 16])]
g = dn.groupby(["model", "k", "corrupt"]).agg(probe=("probe", "mean"), sd=("probe", "std"), mse=("mse", "mean"),
                                               knn=("knn", "mean"), probe10=("probe10", "mean")).reset_index()
g.to_csv("results/summary_denoise.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, k in zip(axs, [8, 16]):
    x = np.arange(4); w = .38
    for j, (c, lab, a) in enumerate([(0.0, "clean input", 1.0), (0.3, "30% masking noise", 0.55)]):
        vals = [g[(g.model == m) & (g.k == k) & (g.corrupt == c)].probe.values[0] for m in COL if m != "pca"]
        sds = [g[(g.model == m) & (g.k == k) & (g.corrupt == c)].sd.values[0] for m in COL if m != "pca"]
        ax.bar(x + (j - .5) * w, vals, w - .04, yerr=sds, capsize=2, label=lab, alpha=a,
               color=[COL[m] for m in COL if m != "pca"])
    ax.set_xticks(x); ax.set_xticklabels(["Symm.", "Heavy enc", "Heavy dec", "Linear"], fontsize=9)
    ax.set_title(f"k = {k}", loc="left", fontsize=10); ax.set_ylim(0.7, 1.0)
axs[0].set_ylabel("Linear-probe accuracy")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc="#52514e", alpha=1, label="clean input"), Patch(fc="#52514e", alpha=.55, label="30% masking noise")], frameon=False, fontsize=8.5, loc="upper right", ncol=2)
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("figures/fig_denoise.png", dpi=200); plt.close(fig)

# parameter table
p = clean.groupby("model")[["n_enc", "n_dec"]].first()
print(p)
pd.set_option("display.width", 200)
t = agg.pivot(index="k", columns="model", values="probe_m").round(3); print(t)
print(agg.pivot(index="k", columns="model", values="probe10_m").round(3))
print(agg.pivot(index="k", columns="model", values="knn_m").round(3))
print(agg.pivot(index="k", columns="model", values="mse_m").round(4))
print(agg.pivot(index="k", columns="model", values="erank_m").round(2))
print(g.round(3))
