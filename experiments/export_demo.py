"""Train k=2 models on the seed-0 split, save latents + weights for the HTML demo, and draw
latent scatters, decoded manifolds and a reconstruction gallery."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asym_ae.data import load_split
from asym_ae.models import AutoEncoder
from asym_ae.style import apply, NAME

apply()
lam = json.load(open("results/diagnose_selected.json"))["lambda"] if os.path.exists("results/diagnose_selected.json") else 0.1
d = load_split(0)
os.makedirs("figures", exist_ok=True); os.makedirs("demo", exist_ok=True)

SPECS = [("symmetric", {}), ("heavy_dec", {}), ("heavy_enc", {}), ("heavy_enc", dict(reg=lam))]
def label(arch, kw): return NAME[arch] + (f" + VICReg-style reg (λ={lam})" if kw else "")

models, demo = [], {"models": [], "labels": d["yte"].tolist(), "images": np.round(d["Xte"], 3).tolist()}
for arch, kw in SPECS:
    ae = AutoEncoder(arch, 2, seed=0); ae.fit(d["Xtr"], **kw)
    Z = ae.encode(d["Xte"])
    models.append((label(arch, kw), ae, Z))
    entry = ae.to_dict(); entry.pop("enc"); entry["name"] = label(arch, kw); entry["latents"] = np.round(Z, 3).tolist()
    demo["models"].append(entry)
json.dump(demo, open("demo/demo_data.json", "w"), separators=(",", ":"))
json.dump({n: Z.tolist() for n, _, Z in models} | {"labels": d["yte"].tolist()}, open("results/latents2d.json", "w"))

# latent scatters
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, (n, ae, Z) in zip(axs, models):
    ax.scatter(Z[:, 0], Z[:, 1], c=d["yte"], cmap="tab10", s=7, alpha=.85, linewidths=0)
    ax.set_title(n.replace(" + VICReg-style reg", "\n+ latent reg"), fontsize=9, loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.tight_layout(); fig.savefig("figures/fig_latent2d.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# decoded manifolds (what the decoder paints over the latent plane)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.7))
n = 14
for ax, (nm, ae, Z) in zip(axs, models):
    lo, hi = np.percentile(Z, 3, 0), np.percentile(Z, 97, 0)
    g0, g1 = np.linspace(lo[0], hi[0], n), np.linspace(hi[1], lo[1], n)
    img = np.zeros((8 * n, 8 * n))
    for r, y in enumerate(g1):
        for c, x in enumerate(g0):
            img[8 * r:8 * r + 8, 8 * c:8 * c + 8] = ae.decode(np.array([[x, y]])).reshape(8, 8)
    ax.imshow(img, cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(nm.replace(" + VICReg-style reg", "\n+ latent reg"), fontsize=9, loc="left")
fig.tight_layout(); fig.savefig("figures/fig_manifold.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# reconstruction gallery at k=4
rows = [("Input", None)]
rec = {}
for arch in ["symmetric", "heavy_dec", "heavy_enc"]:
    ae = AutoEncoder(arch, 4, seed=0); ae.fit(d["Xtr"]); rec[NAME[arch]] = ae.reconstruct(d["Xte"])
p = PCA(4).fit(d["Xtr"]); rec["PCA"] = p.inverse_transform(p.transform(d["Xte"]))
idx = [int(np.flatnonzero(d["yte"] == c)[0]) for c in range(10)]
names = ["Input"] + list(rec)
fig, axs = plt.subplots(len(names), 10, figsize=(10, 1.05 * len(names)))
for r, nm in enumerate(names):
    for c, i in enumerate(idx):
        im = d["Xte"][i] if nm == "Input" else rec[nm][i]
        axs[r, c].imshow(im.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1); axs[r, c].axis("off")
    axs[r, 0].text(-0.15, 0.5, nm, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=8)
fig.subplots_adjust(left=0.2, wspace=0.05, hspace=0.05); fig.savefig("figures/fig_reconstructions.png", dpi=200); plt.close(fig)
print("demo + figures written")

# self-contained demo page: template + data inlined (works from file:// and GitHub Pages)
tpl = open("demo/template.html").read()
open("demo/index.html", "w").write(tpl.replace("__DATA__", json.dumps(demo, separators=(",", ":"))))
print("demo/index.html written")
