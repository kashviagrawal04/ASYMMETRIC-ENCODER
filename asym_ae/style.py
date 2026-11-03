"""Shared plotting style and names (validated categorical palette, fixed assignment per model)."""
import matplotlib.pyplot as plt

COL = {"symmetric": "#2a78d6", "heavy_enc": "#eb6834", "heavy_dec": "#1baf7a", "linear_sym": "#6250d6", "pca": "#52514e", "pixels": "#e87ba4"}
NAME = {"symmetric": "Symmetric (deep/deep)", "heavy_enc": "Heavy encoder / light decoder",
        "heavy_dec": "Light encoder / heavy decoder", "linear_sym": "Linear AE", "pca": "PCA", "pixels": "Raw pixels"}


def apply():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
