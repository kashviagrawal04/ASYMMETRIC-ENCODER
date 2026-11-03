"""Streamlit app: Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning.
Run locally:  streamlit run streamlit_app.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from asym_ae import evaluate as E
from asym_ae.data import load_split
from asym_ae.models import AutoEncoder

ROOT = Path(__file__).parent
st.set_page_config(page_title="Asymmetric Auto-Encoder Explorer", page_icon="🧠", layout="wide")
DEPTH = {"none (single layer)": 0, "1 hidden layer": 1, "2 hidden layers": 2}
NAME = {"symmetric": "Symmetric (deep/deep)", "heavy_enc": "Heavy encoder / light decoder", "heavy_dec": "Light encoder / heavy decoder",
        "linear_sym": "Linear AE", "pca": "PCA", "pixels": "Raw pixels"}


# ----------------------------------------------------------------------------- data
@st.cache_data
def demo_data():
    return json.load(open(ROOT / "demo" / "demo_data.json"))


@st.cache_data
def results_main():
    return pd.read_csv(ROOT / "results" / "summary_main.csv")


@st.cache_data
def results_stats():
    return pd.read_csv(ROOT / "results" / "stats_vs_symmetric.csv")


def decode(model, z):
    a = np.atleast_2d(np.asarray(z, dtype=float))
    for W, b, act in zip(model["dec"]["W"], model["dec"]["b"], model["dec"]["acts"]):
        a = a @ np.array(W) + np.array(b)
        a = np.maximum(a, 0) if act == "relu" else 1 / (1 + np.exp(-np.clip(a, -30, 30))) if act == "sigmoid" else a
    return a


def show_digit(ax, v):
    ax.imshow(np.asarray(v).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off")


# ----------------------------------------------------------------------------- header
st.title("Asymmetric Auto-Encoder Bottlenecks")
st.caption("Self-supervised latent representation learning · BCSE332L Deep Learning · Kashvi Agrawal (23BDS0336), Sudeep Sharma (23BDS0315)")
st.write("An auto-encoder squeezes an image into a few numbers (the *code*) and rebuilds it. This app lets you explore what happens when the "
         "**encoder and decoder have different sizes**: look at trained models, move through their latent space, or train your own.")
tab_over, tab_lat, tab_train, tab_res = st.tabs(["Overview", "Latent explorer", "Train your own", "Results"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    M = results_main()
    g = lambda m, k, c: float(M[(M.model == m) & (M.k == k)][f"{c}_m"].iloc[0])
    st.subheader("Headline result")
    st.write("With a **narrow bottleneck**, a balanced (symmetric) deep auto-encoder gives much better codes than a deep encoder with a "
             "single-layer decoder, which collapses onto almost a single line. With a wide bottleneck (16+ numbers) on this easy dataset, nothing matters.")
    cols = st.columns(3)
    for col, k in zip(cols, [2, 4, 8]):
        col.metric(f"Probe accuracy, k = {k}", f"{g('symmetric', k, 'probe'):.3f}",
                   f"{g('heavy_enc', k, 'probe') - g('symmetric', k, 'probe'):+.3f} for heavy encoder / light decoder", delta_color="off")
    c1, c2 = st.columns(2)
    c1.metric("Effective rank of the 2-D code: symmetric", f"{g('symmetric', 2, 'erank'):.2f} of 2")
    c2.metric("Effective rank of the 2-D code: heavy encoder / light decoder", f"{g('heavy_enc', 2, 'erank'):.2f} of 2", "collapsed", delta_color="inverse")
    st.image(str(ROOT / "figures" / "fig_main_metrics.png"), caption="Six representation-quality metrics versus bottleneck width (mean ± sd over 5 seeds)")
    with st.expander("How to read the numbers"):
        st.markdown("- **Probe accuracy**: a very simple classifier trained on the frozen code (labels are never used to train the auto-encoder).\n"
                    "- **Effective rank**: how many code dimensions carry information. Equal to *k* is healthy; near 1 means the code collapsed onto a line.\n"
                    "- **k**: the size of the bottleneck, i.e. how many numbers describe each image.")

# ----------------------------------------------------------------------------- latent explorer
with tab_lat:
    D = demo_data()
    models = D["models"]
    st.write("Each model compresses an 8×8 digit into **2 numbers**. Pick a model, move the sliders, and its decoder draws the digit for that point.")
    name = st.selectbox("Model", [m["name"] for m in models])
    mdl = next(m for m in models if m["name"] == name)
    Z = np.array(mdl["latents"]); labels = np.array(D["labels"])
    lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    pad = (hi - lo) * 0.15
    c_ctrl, c_plane, c_img = st.columns([1, 2, 1])
    z1 = c_ctrl.slider("z1", float(lo[0] - pad[0]), float(hi[0] + pad[0]), float(Z[:, 0].mean()), key=f"z1_{name}")
    z2 = c_ctrl.slider("z2", float(lo[1] - pad[1]), float(hi[1] + pad[1]), float(Z[:, 1].mean()), key=f"z2_{name}")
    show_man = c_ctrl.checkbox("Show decoded manifold behind the points", value=False)
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    erank = float(lam.sum() ** 2 / (lam ** 2).sum())
    c_ctrl.metric("Effective rank of this code", f"{erank:.2f} of 2")
    if erank < 1.4:
        c_ctrl.warning("The code has collapsed onto a line: most of this plane is unused.")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ext = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    if show_man:
        n = 14; gx = np.linspace(ext[0], ext[1], n); gy = np.linspace(ext[3], ext[2], n)
        img = np.zeros((8 * n, 8 * n))
        for r, y in enumerate(gy):
            for c, x in enumerate(gx):
                img[8 * r:8 * r + 8, 8 * c:8 * c + 8] = decode(mdl, [x, y]).reshape(8, 8)
        ax.imshow(img, cmap="gray_r", extent=ext, aspect="auto", alpha=0.85)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=12, alpha=0.9, linewidths=0)
    ax.plot([z1], [z2], marker="+", color="black", ms=18, mew=2.5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.legend(*sc.legend_elements(), title="digit", fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    c_plane.pyplot(fig, clear_figure=True)

    fig2, axs = plt.subplots(2, 1, figsize=(2.6, 5.2))
    show_digit(axs[0], decode(mdl, [z1, z2])); axs[0].set_title("decoded from (z1, z2)", fontsize=9)
    j = int(np.argmin(((Z - [z1, z2]) ** 2).sum(1)))
    show_digit(axs[1], D["images"][j]); axs[1].set_title(f"nearest test digit: {labels[j]}", fontsize=9)
    c_img.pyplot(fig2, clear_figure=True)
    st.image(str(ROOT / "figures" / "fig_manifold.png"), caption="What each decoder paints over the 2-D latent plane (from the report)")

# ----------------------------------------------------------------------------- train your own
@st.cache_data(show_spinner=False)
def train(enc_d, dec_d, k, epochs, reg, seed):
    d = load_split(seed)
    ae = AutoEncoder((enc_d, dec_d), k, seed=seed)
    ae.fit(d["Xtr"], epochs=epochs, reg=reg, trace_every=max(1, epochs // 20))
    Ztr, Zte = ae.encode(d["Xtr"]), ae.encode(d["Xte"])
    rec = ae.reconstruct(d["Xte"])
    out = dict(mse=float(((rec - d["Xte"]) ** 2).mean()), probe=E.LinearProbe(Ztr, d["ytr"]).score(Zte, d["yte"]),
               knn=E.knn_acc(Ztr, d["ytr"], Zte, d["yte"]), erank=E.effective_rank(Ztr),
               n_enc=int(ae.n_enc), n_dec=int(ae.n_dec), trace=[(int(a), float(b), float(c)) for a, b, c in ae.trace])
    idx = [int(np.flatnonzero(d["yte"] == c)[0]) for c in range(10)]
    return out, d["Xte"][idx], rec[idx], (Zte if k == 2 else None), d["yte"]


with tab_train:
    st.write("Choose how big the encoder and decoder are and train a model right here (it takes a few seconds). "
             "Try **encoder: 2 hidden layers, decoder: linear, k = 2** and watch the effective rank collapse, then give the decoder one hidden layer.")
    a, b, c = st.columns(3)
    enc_name = a.selectbox("Encoder", list(DEPTH), index=2)
    dec_name = a.selectbox("Decoder", list(DEPTH), index=0)
    k = b.select_slider("Bottleneck size k (numbers per image)", options=[2, 4, 8, 16], value=2)
    epochs = b.slider("Training epochs", 20, 200, 100, step=10)
    reg = c.select_slider("Latent regulariser weight (0 = off)", options=[0.0, 0.01, 0.1, 1.0], value=0.0)
    seed = c.number_input("Random seed", 0, 99, 0)
    if st.button("Train model", type="primary"):
        with st.spinner("Training the auto-encoder (pure NumPy)..."):
            res, X10, R10, Z2, yte = train(DEPTH[enc_name], DEPTH[dec_name], k, epochs, reg, int(seed))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probe accuracy", f"{res['probe']:.3f}"); m2.metric("5-NN accuracy", f"{res['knn']:.3f}")
        m3.metric("Effective rank", f"{res['erank']:.2f} of {k}"); m4.metric("Reconstruction MSE", f"{res['mse']:.4f}")
        st.caption(f"Parameters: encoder {res['n_enc']:,}, decoder {res['n_dec']:,}.  Compare with the report: symmetric k=2 probe ≈ 0.74, heavy-encoder/light-decoder ≈ 0.55.")
        l, r = st.columns(2)
        tr = pd.DataFrame(res["trace"], columns=["epoch", "train MSE", "effective rank"]).set_index("epoch")
        l.write("Effective rank of the code during training"); l.line_chart(tr[["effective rank"]])
        r.write("Training reconstruction error"); r.line_chart(tr[["train MSE"]])
        fig, axs = plt.subplots(2, 10, figsize=(10, 2.2))
        for i in range(10):
            show_digit(axs[0, i], X10[i]); show_digit(axs[1, i], R10[i])
        axs[0, 0].set_title("input", fontsize=8, loc="left"); axs[1, 0].set_title("reconstruction", fontsize=8, loc="left")
        st.pyplot(fig, clear_figure=True)
        if Z2 is not None:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(Z2[:, 0], Z2[:, 1], c=yte, cmap="tab10", s=10, linewidths=0); ax.set_title("2-D code of the test digits", fontsize=9)
            ax.set_xlabel("z1"); ax.set_ylabel("z2"); st.pyplot(fig, clear_figure=True)
    else:
        st.info("Set the options and press **Train model**.")

# ----------------------------------------------------------------------------- results
with tab_res:
    M = results_main()
    st.subheader("Linear-probe accuracy by bottleneck width (5 seeds)")
    order = ["symmetric", "heavy_enc", "heavy_dec", "linear_sym", "pca", "pixels"]
    tbl = pd.DataFrame({NAME[m]: M[M.model == m].set_index("k")["probe_m"] for m in order}).round(3)
    st.dataframe(tbl, width="stretch")
    for f, cap in [("fig_depth_grid.png", "Encoder depth × decoder depth grid: probe accuracy (top) and effective rank (bottom)"),
                   ("fig_collapse_trace.png", "Collapse diagnosis: effective rank during training"),
                   ("fig_label_efficiency.png", "Label efficiency"), ("fig_robustness.png", "Robustness to shifts, rotations and noise"),
                   ("fig_regularised.png", "Effect of the latent regulariser")]:
        st.image(str(ROOT / "figures" / f), caption=cap)
    st.subheader("Paired comparison with the symmetric model")
    S = results_stats()
    S = S[S.metric == "probe"][["a", "k", "diff", "lo", "hi", "p"]].rename(columns={"a": "model", "diff": "mean difference", "lo": "95% CI low", "hi": "95% CI high", "p": "paired p-value"})
    st.dataframe(S.round(4), width="stretch", hide_index=True)
    st.caption("Source code, full report and raw results: github.com/kashviagrawal04/ASYMMETRIC-ENCODER")
