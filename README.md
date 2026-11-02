# ASYMMETRIC-ENCODER

**Author:** Kashvi Agrawal

Pure-NumPy study of how encoder/decoder capacity asymmetry shapes the latent space of an auto-encoder.
Labels are used only to *evaluate* frozen latents (linear probe, k-NN, effective rank).

    asym_ae/nn.py        dense layers, manual backprop, Adam
    asym_ae/models.py    AutoEncoder with independent encoder/decoder depth (4 architectures)
    asym_ae/evaluate.py  linear probe, k-NN, latent effective rank
    run_experiments.py   full sweep (arch x k x 5 seeds, + denoising ablation, + PCA/pixel baselines)
    make_figures.py      figures + summary tables
    quicktest.py         gradient check + smoke test

Run: `pip install numpy scikit-learn matplotlib pandas && python run_experiments.py && python make_figures.py`
(~10 min on 2 CPU cores). Dataset: scikit-learn Digits (bundled; no download).

The full case study is in `report/` (Word and PDF).
