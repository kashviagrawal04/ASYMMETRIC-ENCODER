from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd

A = pd.read_csv("results/summary_clean.csv")
def v(model, k, m, nd=3): return f"{A[(A.model==model)&(A.k==k)][m+'_m'].values[0]:.{nd}f}"
def vs(model, k, m, nd=3): return f"{v(model,k,m,nd)} ± {A[(A.model==model)&(A.k==k)][m+'_s'].values[0]:.{nd}f}"

d = Document()
sec = d.sections[0]; sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)
st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
for n, sz in [("Heading 1", 16), ("Heading 2", 13)]:
    h = d.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.color.rgb = RGBColor(0x1c, 0x3a, 0x6b)

def P(text, bold=False, italic=False, align=None):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if align: p.alignment = align
    return p
def B(text):
    p = d.add_paragraph(style="List Bullet"); 
    if "||" in text:
        a, b = text.split("||", 1); p.add_run(a).bold = True; p.add_run(b)
    else: p.add_run(text)
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), color); tcPr.append(s)
def table(header, rows, widths=None, bold_best=None):
    t = d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255); shade(c, "1C3A6B")
    for row in rows:
        cells = t.add_row().cells
        for i, x in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(x)); r.font.size = Pt(9.5)
            if i > 0: cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_paragraph()
def FIG(path, cap, w=6.0):
    d.add_picture(path, width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = P(cap, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER); p.runs[0].font.size = Pt(9.5)

# ---- Title
t = P("Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
t.runs[0].font.size = Pt(20); t.runs[0].font.color.rgb = RGBColor(0x1c, 0x3a, 0x6b)
P("Project Case Study", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Kashvi Agrawal", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Problem statement  •  Work done  •  Implementation  •  Results and discussion", align=WD_ALIGN_PARAGRAPH.CENTER)

# ---- Summary
d.add_heading("Summary", 1)
P("Auto-encoders learn representations without labels by compressing an input through a bottleneck and reconstructing it. "
  "Most designs mirror the encoder and decoder. Recent self-supervised work (for example masked auto-encoders) deliberately breaks this "
  "symmetry, often with a light decoder, on the argument that a weak decoder pushes the encoder to carry the semantic work. "
  "This project tests that idea in a controlled setting: I built an auto-encoder framework from scratch in NumPy in which encoder and decoder "
  "capacity are set independently, and compared four architectures across five bottleneck widths and five seeds, scoring the frozen latents with "
  "a linear probe, k-NN, and a latent effective-rank measure.")
P("Headline finding: in this setting, a light decoder did not improve the latent space, and a symmetric deep auto-encoder was best whenever the "
  "bottleneck was narrow (k ≤ 8). A heavy encoder paired with a light decoder reached a much worse optimum at narrow bottlenecks, with latents that "
  "nearly collapsed to one dimension at k = 2. At k ≥ 16, all models, including PCA and raw pixels, landed within seed noise of each other because "
  "the dataset is easy.")

# ---- 1 Problem
d.add_heading("1. Problem Statement", 1)
P("Given unlabeled data x ∈ ℝ⁶⁴, learn an encoder f: ℝ⁶⁴ → ℝᵏ and decoder g: ℝᵏ → ℝ⁶⁴ by minimizing reconstruction error ‖x − g(f(x))‖². "
  "The goal is not reconstruction itself but a latent code z = f(x) that is useful for downstream tasks without any label supervision.")
P("Research question. ", bold=True).add_run("How does the capacity asymmetry between encoder and decoder, at a fixed bottleneck width k, affect the quality of the learned latent representation?").bold = False
P("Hypotheses tested:", bold=True)
B("H1 (light decoder helps).|| A weak decoder cannot do much work, so the encoder must produce a more linearly separable, informative code. Heavy encoder / light decoder should give better probe accuracy than a symmetric model.")
B("H2 (heavy decoder compensates).|| A strong decoder can unfold a simple (even linear) encoder's code, so a light encoder / heavy decoder model should reconstruct well but give a less structured latent.")
B("H3 (noise helps).|| A denoising objective should regularize the latent space and help most where the decoder is weak.")

# ---- 2 Work done
d.add_heading("2. Work Done", 1)
B("Built a NumPy neural-network library (dense layers, manual backpropagation, Adam), and verified the analytic gradients against finite differences (relative agreement to about 8 significant digits).")
B("Defined four auto-encoder architectures that differ only in encoder and decoder depth, with a linear bottleneck of width k ∈ {2, 4, 8, 16, 32}.")
B("Trained each configuration for 300 epochs with 5 random seeds (100 clean runs), plus a masking-noise denoising ablation (40 runs).")
B("Evaluated frozen latents with label-free training and label-based evaluation only: full-label linear probe, 10%-label linear probe, 5-NN, reconstruction MSE, and latent effective rank. Added PCA at every k and raw pixels as reference baselines.")
B("Produced figures and summary tables directly from the saved results files, and wrote this case study from those numbers.")

# ---- 3 Implementation
d.add_heading("3. Implementation", 1)
d.add_heading("3.1 Data", 2)
P("scikit-learn Digits: 1,797 grayscale 8×8 digit images (64 features, 10 classes), scaled to [0, 1] and split 75/25 into 1,347 training and 450 test images, stratified by class. "
  "I had intended to use MNIST, but dataset downloads were blocked in my environment, so Digits (bundled with scikit-learn) was used. This is a much smaller and easier problem than MNIST, which shapes how the results should be read.")
d.add_heading("3.2 Architectures", 2)
P("All models use ReLU hidden layers, a linear bottleneck, and a sigmoid output; the exception is the Linear AE, which is linear throughout. "
  "Parameter counts below are for k = 2 (they grow slightly with k).")
table(["Model", "Encoder", "Decoder", "Enc. params", "Dec. params"], [
    ["Symmetric", "64-128-64-k", "k-64-128-64", "16,706", "16,768"],
    ["Heavy encoder / light decoder", "64-128-64-k", "k-64", "16,706", "192"],
    ["Light encoder / heavy decoder", "64-k", "k-64-128-64", "130", "16,768"],
    ["Linear AE", "64-k", "k-64", "130", "192"],
])
d.add_heading("3.3 Training", 2)
P("Mean-squared-error loss, Adam (learning rate 1e-3), batch size 64, 300 epochs, He initialization for ReLU layers. "
  "The same hyperparameters were used for every architecture; none were tuned on the test set. "
  "For the denoising ablation, each input pixel was independently zeroed with probability 0.3 while the target remained the clean image.")
d.add_heading("3.4 Evaluation metrics", 2)
B("Linear probe.|| Standardize frozen latents, fit logistic regression on training latents, report test accuracy. Repeated with a stratified 10% subset of training labels to mimic a low-label regime.")
B("5-NN accuracy.|| Non-parametric check that class structure is local in latent space.")
B("Reconstruction MSE.|| Test-set mean squared error.")
B("Effective rank.|| Participation ratio of the latent covariance eigenvalues, (Σλ)² / Σλ². A value near k means all latent dimensions carry variance; a value near 1 signals dimensional collapse.")
P("Code is organized as a small package (nn.py, models.py, evaluate.py) with a single run_experiments.py that regenerates every number in this report.")

# ---- 4 Results
d.add_heading("4. Results", 1)
d.add_heading("4.1 Linear-probe accuracy", 2)
ks = [2, 4, 8, 16, 32]
order = [("symmetric", "Symmetric"), ("heavy_enc", "Heavy enc / light dec"), ("heavy_dec", "Light enc / heavy dec"), ("linear_sym", "Linear AE"), ("pca", "PCA")]
table(["k"] + [n for _, n in order], [[k] + [vs(m, k, "probe") if m != "pca" else v(m, k, "probe") for m, _ in order] for k in ks])
P("Test accuracy of a logistic-regression probe on frozen latents (mean ± sd over 5 seeds; PCA is deterministic). Raw 64-pixel input scores 0.969 under the same probe.", italic=True).runs[0].font.size = Pt(9.5)
FIG("figures/fig_probe.png", "Figure 1. Linear-probe accuracy versus bottleneck width.", 5.6)
d.add_heading("4.2 Low-label probe and k-NN", 2)
table(["k"] + [f"{n}" for _, n in order], [[k] + [v(m, k, "probe10") for m, _ in order] for k in ks])
P("Linear-probe accuracy using 10% of training labels (raw pixels: 0.898).", italic=True).runs[0].font.size = Pt(9.5)
table(["k"] + [f"{n}" for _, n in order], [[k] + [v(m, k, "knn") for m, _ in order] for k in ks])
P("5-NN test accuracy (raw pixels: 0.973).", italic=True).runs[0].font.size = Pt(9.5)
d.add_heading("4.3 Reconstruction and latent geometry", 2)
table(["k"] + [f"{n}" for _, n in order], [[k] + [v(m, k, "mse", 4) for m, _ in order] for k in ks])
P("Test reconstruction MSE (lower is better).", italic=True).runs[0].font.size = Pt(9.5)
table(["k"] + [f"{n}" for _, n in order], [[k] + [v(m, k, "erank", 2) for m, _ in order] for k in ks])
P("Effective rank of the latent codes (maximum possible equals k).", italic=True).runs[0].font.size = Pt(9.5)
FIG("figures/fig_mse.png", "Figure 2. Reconstruction error versus bottleneck width (log scale).", 5.3)
FIG("figures/fig_latent2d.png", "Figure 3. Two-dimensional latent codes of test digits (k = 2, seed 0), colored by digit class. Labels were never used in training.", 6.5)
d.add_heading("4.4 Denoising ablation", 2)
FIG("figures/fig_denoise.png", "Figure 4. Linear-probe accuracy with and without 30% masking noise during training (k = 8 and 16).", 6.2)

# ---- 5 Discussion
d.add_heading("5. Discussion", 1)
d.add_heading("5.1 What the results say about the hypotheses", 2)
B("H1 (light decoder helps) is not supported.|| At k = 2, 4 and 8 the heavy-encoder / light-decoder model scored 0.539, 0.812 and 0.916, against 0.703, 0.884 and 0.939 for the symmetric model. Its 10%-label probe at k = 2 (0.426) was the worst of all models. At k = 8 it was only marginally above PCA (0.916 vs 0.900) and not distinguishable from it given the seed spread.")
B("H2 (heavy decoder, light encoder) is partly supported.|| This model reconstructed far better than the linear models (k = 8 MSE 0.0129 vs 0.0251 for the Linear AE, almost matching the symmetric 0.0125), but its probe accuracy stayed close to PCA (0.906 vs 0.900 at k = 8). A heavy decoder can reconstruct from a code that is not especially class-structured; reconstruction quality and representation quality are different things.")
B("H3 (noise helps) is not supported here.|| Masking noise changed probe accuracy by at most about 0.014 in either direction. At k = 8 it lowered accuracy for all four models, by 0.002–0.014 (seed sd is about 0.005–0.014). At k = 16 the symmetric model gained 0.006 while the other three lost 0.002–0.011 (largest for heavy encoder, 0.954 to 0.943); these shifts are of the same size as seed noise. Masking also raised the clean-test reconstruction error, as expected.")
d.add_heading("5.2 Why the light decoder underperformed (and what is a hypothesis)", 2)
P("The effective-rank table shows the clearest pattern: with a light decoder and a deep encoder, the latents were nearly one-dimensional at k = 2 (effective rank 1.01) and strongly under-used at k = 4 (1.24 of 4). "
  "Figure 3 shows the same thing visually, with all ten digit classes lying along a single line. That is a failure to use the bottleneck rather than a clever compression. "
  "A plausible explanation, which I did not test directly, is that a single-layer decoder can only read out a linear (plus sigmoid) function of the code, so the encoder must arrange z so that this simple readout works. "
  "Early in training the strongest reconstruction gradient lies along one dominant direction, and a deep encoder can follow it and then fail to open up additional dimensions within 300 epochs. "
  "Testing this would need longer training, a learning-rate or initialization sweep for this architecture, and a per-epoch effective-rank trace.")
P("This differs from the masked auto-encoder setting that motivates H1. There, the decoder is light relative to a very large encoder, the input is heavily masked (about 75%), and the task is hard. "
  "Here the input is a tiny 8×8 image reconstructed in full, and the decoder is a single layer, which is probably too weak, not merely lighter. The comparison therefore says little about MAE-style design at scale.")
d.add_heading("5.3 Why the symmetric model won at narrow bottlenecks", 2)
P("When k is small, information is scarce and the code must be an efficient nonlinear summary of a curved data manifold. "
  "Only the symmetric model has nonlinear capacity on both sides: the encoder can bend the manifold into a compact code and the decoder can unbend it. "
  "Its advantage shrinks as k grows (0.884 → 0.939 → 0.950 at k = 4, 8, 16), and at k ≥ 16 it is no better than PCA. "
  "The light-encoder and linear models are limited by their linear encoders to PCA-like codes, which is consistent with their probe accuracies sitting at or just above PCA.")
d.add_heading("5.4 Wide bottlenecks: everything converges", 2)
P("At k = 16 and 32, all five representations reach 0.95–0.97 probe accuracy, which equals the raw pixel baseline (0.969). On Digits, a linear classifier on the pixels is already near ceiling, so there is nothing for a learned latent to add. "
  "Differences at these widths (under 0.012) are within seed-to-seed spread and should not be interpreted.")
d.add_heading("5.5 Limitations", 2)
B("Small and easy data.|| 1,797 images at 8×8; the test set has 450 samples, so one accuracy point is about 0.2%. Results may not transfer to MNIST, CIFAR or natural images.")
B("Capacity is not matched.|| Asymmetric variants have very different parameter counts (about 17k vs 0.2k per side), so effects of asymmetry and total capacity are partly confounded.")
B("No per-architecture tuning.|| Shared hyperparameters may favor the symmetric model; the light-decoder collapse could shrink with tuning.")
B("One data split.|| Seeds vary initialization and shuffling but not the train/test split.")
B("Only linear and k-NN evaluation,|| no downstream fine-tuning, and a single noise type and level.")
d.add_heading("5.6 Next steps", 2)
B("Repeat on MNIST/Fashion-MNIST or CIFAR-10 with a convolutional encoder and masked-patch objective, where the MAE-style premise actually applies.")
B("Add a middle-ground decoder (one hidden layer) to locate where the heavy-encoder collapse starts, and trace effective rank during training.")
B("Match total parameter counts, and add a regularized or variational bottleneck for comparison.")

d.add_heading("Reproducibility", 1)
P("All code, raw results (results.json), summary tables and figures are in the project folder asym-ae. "
  "Running run_experiments.py followed by make_figures.py regenerates every number and figure in this report in about 10 minutes on two CPU cores.")
d.save("report/Asymmetric_AE_Case_Study.docx")
