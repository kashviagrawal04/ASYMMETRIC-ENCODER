"""Builds report/Asymmetric_AE_Case_Study.docx from results/*.json (tables are generated, not typed)."""
import json
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from asym_ae.models import AutoEncoder
from asym_ae.stats import paired_compare
REPO = "https://github.com/kashviagrawal04/ASYMMETRIC-ENCODER"
DEMO = open("results/_demo_url.txt").read().strip() if os.path.exists("results/_demo_url.txt") else None
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

M = pd.read_json("results/main.json"); R = pd.read_json("results/reg.json"); D = pd.read_json("results/denoise.json")
C = pd.read_json("results/control.json"); G = pd.read_json("results/grid.json"); X = pd.read_json("results/diagnose.json")
ST = pd.read_csv("results/stats_vs_symmetric.csv"); SR = pd.read_csv("results/stats_reg_vs_plain.csv")
lam = json.load(open("results/diagnose_selected.json"))["lambda"]
NM = {"symmetric": "Symmetric", "heavy_enc": "Heavy enc. / light dec.", "heavy_dec": "Light enc. / heavy dec.", "linear_sym": "Linear AE", "pca": "PCA", "pixels": "Raw pixels"}
ORD = ["symmetric", "heavy_enc", "heavy_dec", "linear_sym", "pca"]

def mean(df, model, k, m): return df[(df.model == model) & (df.k == k)][m].mean()
def sd(df, model, k, m): return df[(df.model == model) & (df.k == k)][m].std()

d = Document()
sec = d.sections[0]; sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)
st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11); st.paragraph_format.space_after = Pt(6)
for n, sz in [("Heading 1", 16), ("Heading 2", 13)]:
    h = d.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.color.rgb = RGBColor(0, 0, 0)

def P(text, bold=False, italic=False, align=None, size=None):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align is not None: p.alignment = align
    return p
def B(text):
    p = d.add_paragraph(style="List Bullet")
    if "||" in text:
        a, b = text.split("||", 1); p.add_run(a).bold = True; p.add_run(b)
    else: p.add_run(text)
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), color); tcPr.append(s)
def link_par(label, url, size=12):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    p = d.add_paragraph(); r0 = p.add_run(label + " "); r0.font.size = Pt(size)
    rid = d.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
    run = OxmlElement("w:r"); rp = OxmlElement("w:rPr")
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rp.append(u)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); rp.append(sz)
    run.append(rp); tx = OxmlElement("w:t"); tx.text = url; run.append(tx); h.append(run); p._p.append(h)
    return p
HEADINGS = []
def H(text, level):
    HEADINGS.append([text, level]); return d.add_heading(text, level)

import os
TOC = json.load(open("results/_toc.json")) if os.path.exists("results/_toc.json") else []
def add_field_run(p, instr=None, kind=None, text=None):
    r = p.add_run()
    if kind:
        f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), kind); r._r.append(f)
    if instr:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r._r.append(it)
    if text is not None: r.text = text
    return r
def toc_block():
    p = P("Table of Contents", bold=True, size=16)
    entries = TOC or [["(updating)", 1, 0]]
    for i, (text, level, page) in enumerate(entries):
        p = d.add_paragraph(); pf = p.paragraph_format
        pf.space_after = Pt(2); pf.space_before = Pt(5 if level == 1 else 0); pf.left_indent = Inches(0 if level == 1 else 0.3)
        pf.tab_stops.add_tab_stop(Inches(6.5), alignment=2, leader=1)      # right tab, dotted leader
        if i == 0:
            add_field_run(p, kind="begin"); add_field_run(p, instr=' TOC \\o "1-2" \\u '); add_field_run(p, kind="separate")
        r = p.add_run(f"{text}\t{page if page else ''}"); r.bold = (level == 1); r.font.size = Pt(11)
        if i == len(entries) - 1: add_field_run(p, kind="end")
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
def footer_page_numbers():
    fp = d.sections[0].footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field_run(fp, kind="begin"); add_field_run(fp, instr=" PAGE "); add_field_run(fp, kind="separate"); add_field_run(fp, text="1"); add_field_run(fp, kind="end")
def table(header, rows, cap=None, hl=None):
    t = d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0, 0, 0); shade(c, "D9D9D9")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, x in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(x)); r.font.size = Pt(9)
            if i > 0: cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if hl and (ri, i) in hl: r.bold = True; shade(cells[i], "EDEDED")
    for row in t.rows[:-1]:                      # keep each table on one page
        for c in row.cells:
            for p in c.paragraphs: p.paragraph_format.keep_with_next = True
    if cap: P(cap, italic=True, size=9)
    else: d.add_paragraph()
def FIG(path, cap, w=6.2):
    d.add_picture(path, width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(cap, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
def best_hl(rows, ncol, hi=True):
    hl = set()
    for ri, r in enumerate(rows):
        vals = [float(str(c).split(" ")[0]) for c in r[1:]]
        hl.add((ri, 1 + (int(np.argmax(vals)) if hi else int(np.argmin(vals)))))
    return hl

REFS = [
 'G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," Science, vol. 313, no. 5786, pp. 504-507, 2006.',
 'P. Vincent, H. Larochelle, Y. Bengio and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in Proc. ICML, 2008, pp. 1096-1103.',
 'K. He, X. Chen, S. Xie, Y. Li, P. Dollar and R. Girshick, "Masked autoencoders are scalable vision learners," in Proc. CVPR, 2022, pp. 16000-16009.',
 'D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors," Nature, vol. 323, pp. 533-536, 1986.',
 'A. Bardes, J. Ponce and Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," in Proc. ICLR, 2022.',
 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.',
 'O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," in Proc. EUSIPCO, 2007, pp. 606-610.',
 'J. Venna and S. Kaski, "Neighborhood preservation in nonlinear projection methods: An experimental study," in Proc. ICANN, 2001, pp. 485-491.',
 'G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," arXiv:1610.01644, 2016.',
 'L. Jing, P. Vincent, Y. LeCun and Y. Tian, "Understanding dimensional collapse in contrastive self-supervised learning," in Proc. ICLR, 2022.',
 'K. He, X. Zhang, S. Ren and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in Proc. ICCV, 2015, pp. 1026-1034.',
 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',
 'C. R. Harris et al., "Array programming with NumPy," Nature, vol. 585, pp. 357-362, 2020.',
 'E. Alpaydin and C. Kaynak, "Optical Recognition of Handwritten Digits," UCI Machine Learning Repository, 1998.',
]
# ------------------------------------------------------------------ cover page
from docx.enum.text import WD_BREAK
for _ in range(5): d.add_paragraph()
t = P("Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=26)
t.runs[0].font.color.rgb = RGBColor(0, 0, 0)
d.add_paragraph()
P("Course Code - BCSE332L", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
P("Course Title - Deep Learning", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
P("Instructor - Prof. VARALAKSHMI M", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
for _ in range(5): d.add_paragraph()
P("Team Members", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=14)
P("Kashvi Agrawal    23BDS0336", align=WD_ALIGN_PARAGRAPH.LEFT, size=13)
P("Sudeep Sharma    23BDS0315", align=WD_ALIGN_PARAGRAPH.LEFT, size=13)
for _ in range(2): d.add_paragraph()
link_par("GitHub repository:", REPO)
if DEMO: link_par("Live demo:", DEMO)
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

toc_block()
H("Abstract", 1)
P("Auto-encoders learn compact codes without labels, and recent self-supervised methods often give the decoder less capacity than the encoder. This project asks how the capacity asymmetry between encoder and decoder shapes the code that is learned. "
  "We built an auto-encoder framework from scratch in NumPy with independently configurable encoder and decoder depth, and judged frozen latents on the 8×8 Digits dataset with a linear probe, low-label probes, clustering, neighbourhood preservation, effective rank, robustness and computational cost, "
  "over 5 seeds with a fresh data split per seed and paired statistics. "
  f"A symmetric deep model was best at narrow bottlenecks (probe accuracy {mean(M,'symmetric',2,'probe'):.3f} against {mean(M,'heavy_enc',2,'probe'):.3f} for a deep encoder with a single-layer decoder at 2 dimensions), and the light-decoder model suffered dimensional collapse (effective rank {mean(M,'heavy_enc',2,'erank'):.2f} of 2). "
  "A random-encoder control shows that training matters most when the bottleneck is narrow. A latent regulariser removes the collapse and repairs clustering but not linear separability, whereas one hidden decoder layer fixes both. "
  "From 16 dimensions upwards all models, PCA and raw pixels are indistinguishable on this easy dataset.")
P("Keywords: auto-encoder, asymmetric architecture, latent representation, bottleneck, dimensional collapse, linear probe, effective rank, NumPy.", italic=True, size=10.5)
H("Summary of Findings", 1)
P("Auto-encoders learn representations without labels by squeezing an input through a bottleneck and reconstructing it [1]. Most designs mirror the encoder and decoder; "
  "recent self-supervised work such as masked auto-encoders [3] often breaks that symmetry, frequently with a light decoder, on the argument that a weak decoder forces the encoder to do the semantic work. "
  "This project tests that idea in a controlled, fully reproducible setting. I built an auto-encoder framework from scratch in NumPy in which encoder and decoder depth are set independently, "
  "and evaluated it on frozen latents with a linear probe, low-label probes, k-NN, label-free clustering, neighbourhood preservation, latent effective rank and robustness to shifts, rotations and noise, "
  "with every comparison paired by seed.")
P("Headline results", bold=True)
B(f"A light decoder did not help.|| Symmetric deep auto-encoders were best whenever the bottleneck was narrow: probe accuracy {mean(M,'symmetric',2,'probe'):.3f} / {mean(M,'symmetric',4,'probe'):.3f} / {mean(M,'symmetric',8,'probe'):.3f} at k = 2 / 4 / 8, "
  f"versus {mean(M,'heavy_enc',2,'probe'):.3f} / {mean(M,'heavy_enc',4,'probe'):.3f} / {mean(M,'heavy_enc',8,'probe'):.3f} for a deep encoder with a single-layer decoder.")
B(f"Deep encoder + single-layer decoder suffers dimensional collapse.|| Effective rank {mean(M,'heavy_enc',2,'erank'):.2f} of 2 at k = 2 and {mean(M,'heavy_enc',4,'erank'):.2f} of 4 at k = 4, and clustering quality (NMI) falls from {mean(M,'symmetric',2,'nmi'):.2f} to {mean(M,'heavy_enc',2,'nmi'):.2f} at k = 2.")
B("Diagnosis.|| The collapse is an early training plateau (rank drops to about 1 within the first 25 to 100 epochs) that longer training or a higher learning rate only partly escapes. A variance-covariance regulariser removes the collapse "
  "and repairs clustering, but does not restore linear separability. Giving the decoder one hidden layer does both.")
B("Limits of the evidence.|| On this small, easy dataset (raw pixels already give 0.969 probe accuracy) every model, PCA included, is indistinguishable at k ≥ 16, so the conclusions concern narrow bottlenecks only.")

# ------------------------------------------------------------------ 1
H("1. Problem Statement", 1)
P("Given unlabeled data x ∈ ℝ⁶⁴, learn an encoder f: ℝ⁶⁴ → ℝᵏ and decoder g: ℝᵏ → ℝ⁶⁴ by minimising reconstruction error ‖x − g(f(x))‖². The objective is not reconstruction itself but a code z = f(x) that is useful downstream without label supervision.")
P("Research question. ", bold=True).add_run("At a fixed bottleneck width k, how does the capacity asymmetry between encoder and decoder affect the quality and geometry of the learned latent representation, and why?").bold = False
P("Hypotheses", bold=True)
B("H1 (light decoder helps).|| A weak decoder cannot do much work, so the encoder must produce a more informative, linearly separable code.")
B("H2 (heavy decoder compensates).|| A strong decoder can reconstruct from a simple encoder's code, giving good reconstruction but a less structured latent space.")
B("H3 (noise helps).|| A masking-noise denoising objective regularises the latent space, most where the decoder is weak.")
B("H4 (collapse is the mechanism).|| If a light decoder hurts, it does so by letting the latent space collapse to fewer effective dimensions; remedies that restore effective rank should restore accuracy.")

# ------------------------------------------------------------------ 2
H("2. Work Done", 1)
P("Version 1 of this project compared four architectures on one metric family and left the main finding unexplained. Version 2 extends it in five directions:")
B("Broader architecture space.|| A full 3 × 3 encoder-depth × decoder-depth grid (9 architectures, 3 bottleneck widths) instead of four hand-picked models.")
B("Mechanism, not just ranking.|| A dedicated diagnosis of the light-decoder collapse: per-epoch effective-rank traces, longer training, a higher learning rate, a one-layer decoder, and a new variance-covariance latent regulariser with strength chosen on a validation split.")
B("Richer evaluation.|| Added 1 / 5 / 10 / 25 %-label probes, K-means NMI, trustworthiness, and robustness to 1-px shifts, 15° rotations and Gaussian noise.")
B("More honest statistics.|| Every seed now draws its own stratified train/test split (so variance includes data-split variance) and models are compared with paired differences, bootstrap confidence intervals and paired t-tests.")
B("Engineering.|| A package layout with 18 automated tests (finite-difference gradient checks for every architecture and the regulariser), CI, a Makefile, a license, and a self-contained interactive latent-space explorer whose decoder runs in JavaScript.")

# ------------------------------------------------------------------ 3
H("3. Implementation", 1)
H("3.1 Data", 2)
P("scikit-learn Digits [12, 14]: 1,797 grayscale 8×8 images (64 features, 10 classes), scaled to [0, 1], split 75/25 (1,347 train / 450 test) with class stratification. The split depends on the seed. "
  "Three transformed copies of each test set probe invariance: a 1-pixel right shift, a 15° rotation and additive Gaussian noise (σ = 0.15). MNIST was the original plan, but dataset downloads were blocked in my environment, so Digits (bundled with scikit-learn) was used throughout.")
H("3.2 Models", 2)
P("A from-scratch NumPy [13] library (dense layers, manual backpropagation [4], Adam [6]). Encoder and decoder depth are independent: depth 0 is a single linear/sigmoid layer, depth 1 adds a 128-unit ReLU layer, depth 2 adds 128-64 (encoder) or 64-128 (decoder). The bottleneck is linear with width k ∈ {2, 4, 8, 16, 32} and the output is a sigmoid. "
  "Named variants: symmetric (2/2), heavy encoder / light decoder (2/0), light encoder / heavy decoder (0/2) and a fully linear AE. Parameter counts at k = 2:")
g2 = G[G.k == 2].groupby("model")[["n_enc", "n_dec"]].first()
table(["Architecture (enc/dec depth)", "Encoder params", "Decoder params"], [[f"{m}  ({m[1]}/{m[3]})", f"{int(r.n_enc):,}", f"{int(r.n_dec):,}"] for m, r in g2.iterrows()])
H("3.3 Training, regulariser and evaluation", 2)
P("Mean-squared-error loss, Adam [6] (learning rate 1e-3), batch size 64, 300 epochs, He initialisation [11]; identical hyper-parameters for every architecture. "
  "The denoising ablation [2] zeroes each input pixel with probability 0.3 and reconstructs the clean image. "
  "The latent regulariser is VICReg-style [5]: L_reg = mean_j max(0, 1 − std_j)² + (1/k) Σ_{a≠b} C_ab², where C is the batch covariance of the latents; it is added to the loss with weight λ and its gradient is derived by hand and verified against finite differences. "
  f"λ was selected from {{0.01, 0.1, 1.0}} on a validation split carved out of the training data (λ = {lam}); the test set was never used for selection.")
P("Metrics (labels are used only here, never in training):", bold=True)
B("Linear probe|| (standardise + logistic regression on frozen latents), also with 1, 5, 10 and 25 % of training labels [9].")
B("5-NN accuracy and K-means NMI.|| NMI clusters the latents into 10 groups without labels and compares them with the true classes.")
B("Trustworthiness.|| With 10 neighbours: how well local neighbourhoods of the input are preserved in the latent space [8].")
B("Effective rank|| (Σλ)² / Σλ² of the latent covariance eigenvalues: equal to k for an isotropic code, close to 1 for a code collapsed onto a line [7].")
B("Robustness|| (probe trained on clean latents, scored on shifted / rotated / noisy test latents) and reconstruction MSE.")
P("Baselines are PCA at every k and the raw 64 pixels. Experiments run in five stages (main sweep, denoising ablation, depth grid, collapse diagnosis, regularised sweep): 5 seeds for the main, denoising and regularised stages, 3 seeds for the depth grid and the diagnosis.")

# ------------------------------------------------------------------ 4
H("4. Results", 1)
KS = [2, 4, 8, 16, 32]
H("4.1 Main sweep: linear-probe accuracy", 2)
rows = [[k] + [f"{mean(M,m,k,'probe'):.3f} ± {sd(M,m,k,'probe'):.3f}" if m != "pca" else f"{mean(M,m,k,'probe'):.3f} ± {sd(M,m,k,'probe'):.3f}" for m in ORD] for k in KS]
table(["k"] + [NM[m] for m in ORD], rows, f"Test accuracy of a linear probe on frozen latents, mean ± sd over 5 seeds (each seed has its own split). Raw pixels: {M[M.model=='pixels'].probe.mean():.3f}. Best per row in bold.", hl=best_hl(rows, 5))
FIG("figures/fig_main_metrics.png", "Figure 1. Six representation-quality metrics versus bottleneck width (mean ± sd, 5 seeds). Dotted line: raw pixels.", 6.5)
H("4.2 Is the difference real? Paired comparison with the symmetric model", 2)
rows = []
for mt, lab in [("probe", "probe acc."), ("nmi", "NMI")]:
    for m in ["heavy_enc", "heavy_dec", "pca"]:
        row = [f"{NM[m]} − Symm., {lab}"]
        for k in [2, 4, 8, 16]:
            r = ST[(ST.a == m) & (ST.metric == mt) & (ST.k == k)].iloc[0]
            row.append(f"{r['diff']:+.3f} (p={r['p']:.3f})" + ("*" if r["p"] < 0.05 else ""))
        rows.append(row)
table(["Comparison", "k=2", "k=4", "k=8", "k=16"], rows, "Mean paired difference over 5 seeds (same seed = same split) and paired t-test p-value; * = p < 0.05. With n = 5 these tests are weak; the confidence intervals are in results/stats_vs_symmetric.csv.")
H("4.3 Several metrics at a glance", 2)
for k in [2, 8]:
    rows = [[NM[m]] + [f"{mean(M,m,k,c):.3f}" if c != "erank" else f"{mean(M,m,k,c):.2f}" for c in ["probe", "lab1", "lab10", "knn", "nmi", "trust", "erank", "mse"]] for m in ORD]
    table(["k = %d" % k, "Probe", "1% labels", "10% labels", "5-NN", "NMI", "Trust.", "Eff. rank", "Rec. MSE"], rows)
P(f"Raw pixels: probe {M[M.model=='pixels'].probe.mean():.3f}, 1% labels {M[M.model=='pixels'].lab1.mean():.3f}, 10% labels {M[M.model=='pixels'].lab10.mean():.3f}, NMI {M[M.model=='pixels'].nmi.mean():.3f}.", italic=True, size=9)
FIG("figures/fig_label_efficiency.png", "Figure 2. Probe accuracy as a function of how many labels the probe sees.", 6.5)
H("4.4 Robustness", 2)
rows = [[NM[m]] + [f"{mean(M,m,64 if m == 'pixels' else 8,c):.3f}" for c in ["probe", "rob_shift", "rob_rot", "rob_noise"]] for m in ORD + ["pixels"]]
table(["k = 8", "Clean", "1-px shift", "15° rotation", "Noise σ=0.15"], rows, "Probe accuracy on transformed test images (probe trained on clean data only).")
H("4.5 Encoder-depth × decoder-depth grid", 2)
FIG("figures/fig_depth_grid.png", "Figure 3. Probe accuracy (top) and effective rank (bottom) for all nine depth combinations (mean of 3 seeds).", 6.0)
rows = []
for m in sorted(G.model.unique()):
    rows.append([f"{m}  (enc {m[1]} / dec {m[3]})"] + [f"{mean(G,m,k,'probe'):.3f} ± {sd(G,m,k,'probe'):.3f}" for k in [2, 4, 8]])
table(["Architecture", "k = 2", "k = 4", "k = 8"], rows, "Probe accuracy, mean ± sd over 3 seeds.")
H("4.6 Diagnosing the collapse", 2)
FIG("figures/fig_collapse_trace.png", "Figure 4. Latent effective rank during training for the heavy-encoder / light-decoder model and its remedies (mean of 3 seeds).", 6.4)
rows = []
for v in ["base", "1000 epochs", "lr 3e-3", "reg 0.01", "reg 0.1", "reg 1.0", "1-layer decoder", "2-layer decoder (symmetric)"]:
    row = [v]
    for k in [2, 4]:
        s = X[(X.variant == v) & (X.k == k)]
        row += [f"{s.val_probe.mean():.3f}", f"{s.test_probe.mean():.3f}", f"{s.test_erank.mean():.2f}"]
    rows.append(row)
table(["Variant (deep encoder)", "k=2 val probe", "k=2 test probe", "k=2 eff. rank", "k=4 val probe", "k=4 test probe", "k=4 eff. rank"], rows,
      f"Heavy-encoder remedies, mean of 3 seeds. Validation probe (from a held-out part of the training data) was used to select λ = {lam}; test columns are reported for completeness.")
H("4.7 Latent regulariser across all architectures", 2)
FIG("figures/fig_regularised.png", f"Figure 5. Plain (solid) vs regularised (dashed, λ = {lam}) models, 5 seeds.", 6.5)
rows = []
for m in ["heavy_enc", "symmetric"]:
    for k in [2, 4, 8, 16]:
        a, b = mean(M, m, k, "probe"), mean(R, m + "+reg", k, "probe")
        n1, n2 = mean(M, m, k, "nmi"), mean(R, m + "+reg", k, "nmi")
        e1, e2 = mean(M, m, k, "erank"), mean(R, m + "+reg", k, "erank")
        rows.append([NM[m], k, f"{a:.3f} → {b:.3f}", f"{n1:.3f} → {n2:.3f}", f"{e1:.2f} → {e2:.2f}"])
table(["Model", "k", "Probe", "NMI", "Effective rank"], rows, "Plain → regularised (mean of 5 seeds).")
H("4.8 Reconstruction, latent geometry and denoising", 2)
FIG("figures/fig_manifold.png", "Figure 6. What each decoder paints over the 2-D latent plane (k = 2). The collapsed heavy-encoder model uses a thin line of it.", 6.5)
FIG("figures/fig_reconstructions.png", "Figure 7. Reconstructions of one test digit per class at k = 4.", 5.5)
rows = []
for m in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]:
    rows.append([NM[m]] + [f"{mean(M,m,k,'probe'):.3f} → {mean(D.assign(),m,k,'probe'):.3f}" for k in [8, 16]])
table(["Model", "k = 8", "k = 16"], rows, "Probe accuracy, clean training → 30% masking-noise training (5 seeds).")


H("4.9 Computational cost", 2)
cost = []
for m in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]:
    ae = AutoEncoder(m, 8); me, md = sum(w.size for w in ae.enc.W), sum(w.size for w in ae.dec.W)
    cost.append([NM[m], f"{ae.n_enc:,}", f"{ae.n_dec:,}", f"{me + md:,}", f"{mean(M,m,8,'mse'):.4f}", f"{mean(M,m,8,'probe'):.3f}"])
cost.append(["PCA", "512", "512", "1,024", f"{mean(M,'pca',8,'mse'):.4f}", f"{mean(M,'pca',8,'probe'):.3f}"])
table(["k = 8", "Encoder params", "Decoder params", "MACs per image", "Rec. MSE", "Probe acc."], cost,
      "Cost of encoding and decoding one image (multiply-accumulates = number of weights in the fully connected layers) against what it buys.")
P("For fully connected layers the number of multiply-accumulate operations equals the number of weights, so parameter count is a faithful cost proxy here (this is not true for convolutional networks). "
  "The symmetric model costs about twice as much as the light-encoder / heavy-decoder model (33,792 vs 17,408 operations) for a similar reconstruction error and 0.026 more probe accuracy. "
  "The heavy-encoder / light-decoder model costs the same as the light-encoder / heavy-decoder model but reconstructs much worse (MSE 0.0208 vs 0.0129) and clusters worse (NMI 0.569 vs 0.700), with no reliable probe advantage (0.912 vs 0.909), so it is the less efficient of the two asymmetric designs. The linear models and PCA are about 30 times cheaper than the symmetric model.")
H("4.10 Does training matter? Random-encoder control", 2)
base = pd.concat([M[M.model.isin(["symmetric", "heavy_enc", "heavy_dec", "linear_sym"])], C])
rows = []
for k in KS:
    rows.append([k, f"{mean(base,'symmetric (random init)',k,'probe'):.3f}", f"{mean(base,'heavy_dec (random init)',k,'probe'):.3f}"] + [f"{mean(base,m,k,'probe'):.3f}" for m in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]])
table(["k", "Random deep encoder", "Random linear encoder", "Symmetric", "Heavy enc.", "Heavy dec.", "Linear AE"], rows,
      "Linear-probe accuracy of untrained (randomly initialised) encoders against trained models, mean of 5 seeds. Models that share an encoder architecture share the same random encoder.")
P("An untrained encoder is a useful floor: it shows how much of a probe score comes from training rather than from the architecture and the data alone. "
  "Training raises probe accuracy at every width. The paired gain of the symmetric model over a random deep encoder is +0.408 at k = 2, +0.349 at k = 4, +0.252 at k = 8 and +0.124 at k = 16, and all gains up to k = 16 are significant (paired p < 0.01 for every architecture). "
  f"At k = 32 the gain shrinks to +0.067 for the deep encoder and +0.020 for the linear one: a random linear projection to 32 dimensions already scores {mean(base,'heavy_dec (random init)',32,'probe'):.3f}, close to the {M[M.model=='pixels'].probe.mean():.3f} of raw pixels.")

# ------------------------------------------------------------------ 5
H("5. Discussion", 1)
H("5.1 Hypotheses, one by one", 2)
B(f"H1 (light decoder helps): not supported.|| At k = 2, 4, 8 the deep-encoder / single-layer-decoder model scored {mean(M,'heavy_enc',2,'probe'):.3f}, {mean(M,'heavy_enc',4,'probe'):.3f}, {mean(M,'heavy_enc',8,'probe'):.3f} against {mean(M,'symmetric',2,'probe'):.3f}, {mean(M,'symmetric',4,'probe'):.3f}, {mean(M,'symmetric',8,'probe'):.3f} for the symmetric model. "
  "The gap is significant at k = 2 (−0.194, p = 0.0004) and k = 8 (−0.023, p = 0.012); at k = 4 it is of the same size as the seed spread (p = 0.14). It is far larger and clearer on clustering: NMI is lower by 0.29, 0.25 and 0.15 at k = 2, 4, 8 (all p < 0.002).")
B("H2 (heavy decoder compensates): partly supported.|| The light-encoder / heavy-decoder model reconstructs about as well as the symmetric one (MSE 0.013 vs 0.012 at k = 8, against 0.025 for the linear AE), so decoder depth, not encoder depth, drives reconstruction quality. "
  f"Its probe accuracy ({mean(M,'heavy_dec',8,'probe'):.3f} at k = 8) is only slightly above PCA ({mean(M,'pca',8,'probe'):.3f}): good reconstruction from a linear encoder does not yield a better-structured code.")
dn = {m: (mean(M, m, 8, "probe"), mean(D, m, 8, "probe")) for m in ["symmetric", "heavy_enc", "heavy_dec", "linear_sym"]}
B(f"H3 (noise helps): not supported.|| Masking noise changed probe accuracy by at most 0.012 in either direction (k = 8 and 16, all four architectures), with no consistent sign, which is within seed noise.")
B("H4 (collapse is the mechanism): partly supported, and the more interesting result.|| See 5.2.")
H("5.2 What the collapse is, and is not", 2)
P(f"With a single-layer decoder, a deep encoder maps the data onto an almost one-dimensional curve: effective rank {mean(M,'heavy_enc',2,'erank'):.2f} of 2 at k = 2, {mean(M,'heavy_enc',4,'erank'):.2f} of 4 at k = 4 and {mean(M,'heavy_enc',8,'erank'):.2f} of 8 at k = 8 (symmetric: 1.93 / 3.61 / 5.69). This is a form of dimensional collapse, a failure mode studied in contrastive self-supervised learning [10]. The decoded-manifold figure shows the consequence: the digit classes sit along one line and most of the latent plane decodes to meaningless blobs.")
B("It is an optimisation plateau, not a hard capacity limit.|| The rank trace falls to about 1 within the first 25 to 100 epochs. At k = 4 it recovers slowly, reaching about 2.1 after 1000 epochs, or about 2.5 within 300 epochs at a three times higher learning rate. At k = 2 it had not recovered after 1000 epochs.")
B("Fixing the rank is not enough.|| The regulariser lifts the effective rank to 3.9 of 4 and 1.9 of 2, yet the validation probe accuracy barely moves (k = 4: 0.780 vs 0.785 for the base model) and can fall with a stronger weight. The code becomes decorrelated, not more informative for classification.")
B("Decoder expressiveness is what matters.|| One hidden layer in the decoder lifts validation probe accuracy to 0.869 at k = 4 and 0.667 at k = 2, matching the symmetric model at k = 4 (0.869) and closing most of the gap at k = 2. This is the only remedy that restores both rank and accuracy. In the depth grid, every single-layer-decoder architecture is weaker than its one-hidden-layer-decoder counterpart at k = 2 and 4 (for the deepest encoder, 0.556 vs 0.660 at k = 2 and 0.805 vs 0.901 at k = 4), and at k = 4 the 2/1 architecture (0.901) was at least as good as the full 2/2 model (0.841, sd 0.059; that difference is within seed noise).")
B("Regularisation does repair clustering.|| For the collapsed model the regulariser improves NMI by 0.146 (k = 2, p = 0.003), 0.169 (k = 4, p = 0.0006) and 0.114 (k = 8, p = 0.02) and improves probe accuracy at k = 2 (+0.034, p = 0.03), but not at k ≥ 4. On healthy models it forces near-isotropic codes (effective rank 7.98 of 8) and slightly hurts clustering (symmetric, k = 8: NMI −0.038, p = 0.03), so it is a remedy for collapse rather than a general improvement.")
P("My interpretation, which the experiments support but do not prove, is that a single-layer decoder can only read out a (sigmoid-squashed) linear function of z, so the encoder is pushed towards a nearly linear, low-dimensional solution early in training, and the plateau it then sits on is hard to leave. This differs from masked auto-encoders at scale [3], where the decoder is light relative to a huge encoder but still multi-layer and the task (reconstructing heavily masked inputs) is hard. A one-layer decoder on 8×8 digits is probably too weak, not just lighter, so these results say little about that regime.")
H("5.3 Other observations", 2)
B("Depth on both sides matters most at narrow bottlenecks.|| At k = 2 the symmetric model leads the grid (0.741; the linear/linear model is at 0.527) and symmetric also leads on 5-NN accuracy (0.83 vs 0.57 for the heavy-encoder model) and trustworthiness (0.91 vs 0.79). By k = 8 any architecture with a non-linear layer on both sides is at 0.93 to 0.94, and by k = 16 all models converge.")
B("Label efficiency.|| With one labelled example per class (1% of labels), probes on the k = 8 latents reach 0.673 (symmetric) and 0.58 to 0.65 (others) versus 0.562 on raw pixels, with 8 times fewer features. With 10% of labels raw pixels (0.918) already catch up with the best latents (0.900), and at 100% of labels pixels win (0.969 vs 0.935 at k = 8).")
B("Robustness.|| At k = 8 the bottleneck gives a large noise-robustness gain: 0.84 to 0.88 on noisy inputs versus 0.64 for pixels. No representation is shift-invariant: a one-pixel shift drops every model below 0.43, because nothing in training asks for invariance. Rotation robustness varies (heavy-encoder 0.714, symmetric 0.651, pixels 0.306) and was not predicted by clean accuracy.")
B("Wide bottlenecks.|| At k ≥ 16 all models, PCA and raw pixels reach 0.95 to 0.97. The heavy-encoder model is nominally ahead of the symmetric model at k = 16 (+0.008, p = 0.006), a gap too small to interpret on this dataset.")
B("Training matters most where compression is severe.|| The random-encoder control (4.10) shows that at k = 2 training lifts probe accuracy from about 0.33 (untrained) to between 0.55 and 0.74 (trained), while at k = 32 a random linear projection already gets 0.948. Differences between architectures are therefore only meaningful at narrow widths, which is exactly where we found them.")
B("Cost-efficiency.|| For fully connected networks the symmetric model costs about twice the operations of the light-encoder / heavy-decoder model for 0.026 more probe accuracy at k = 8, and the heavy-encoder / light-decoder design costs the same as the light-encoder / heavy-decoder design while reconstructing and clustering worse.")
H("5.4 Limitations", 2)
B("Small and easy data.|| 1,797 images at 8×8, 450 test images (one accuracy point ≈ 0.2%). Findings may not transfer to MNIST, CIFAR or natural images, or to convolutional / transformer models.")
B("Few seeds.|| 5 seeds for the main sweep and 3 for the depth grid and the diagnosis. Paired tests with n = 5 have little power, and no multiple-comparison correction was applied; treat p-values as guides.")
B("Capacity is not matched.|| Variants differ by up to about 130 times in encoder or decoder parameters, so depth and parameter count are confounded (the depth grid partly separates them).")
B("Shared hyper-parameters.|| Only λ was tuned. A longer schedule or per-architecture tuning might shrink the collapse for the heavy-encoder model, as the 1000-epoch and higher-learning-rate runs hint.")
B("One regulariser, one noise type.|| The study covers a single latent regulariser and one corruption (masking).")
H("5.5 Next steps", 2)
B("Repeat on MNIST / Fashion-MNIST / CIFAR-10 with convolutional encoders and masked-patch objectives, where the masked auto-encoder premise applies.")
B("Test the optimisation-plateau explanation directly (learning-rate warm-up, normalisation layers, initialisation scale) and match parameter counts.")
B("Add invariance-inducing training (augmentation or contrastive terms) to see whether asymmetry changes robustness once the objective asks for it.")

H("Reproducibility", 1)
P("Code, raw per-run results (results/*.json), summary and statistics tables, figures, tests and the interactive latent-space explorer are in the project repository (github.com/kashviagrawal04/ASYMMETRIC-ENCODER). "
  "`make experiments` regenerates every number in about 15 minutes on two CPU cores; `python build_report.py` rebuilds this document from the result files. "
  "An interactive Streamlit app (streamlit_app.py) lets readers explore the trained models and train their own" + (f"; live demo: {DEMO}" if DEMO else "") + ".")
H("References", 1)
for i, ref in enumerate(REFS, 1):
    P(f"[{i}] {ref}", size=10)
footer_page_numbers()
json.dump(HEADINGS, open("results/_headings.json", "w"))
d.save("report/Asymmetric_AE_Case_Study.docx")
