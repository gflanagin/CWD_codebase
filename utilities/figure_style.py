"""Shared figure style for the paper (Bulletin of Mathematical Biology).

Figures are drawn at the text width they are printed at, so lettering is
8 pt at final size, and saved at 600 dpi (combination art).
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

WIDTH = 6.5              # inches; the manuscript text width, under BMB's 174 mm
FONT = 8                 # pt at final size; BMB asks for 8-12 pt

plt.rcParams.update({
    # BMB prefers Arial or Helvetica; math stays in the matching sans set,
    # since mathtext misplaces sub- and superscripts in a custom Arial set
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
    "legend.fontsize": FONT, "legend.title_fontsize": FONT,
    "xtick.labelsize": FONT, "ytick.labelsize": FONT,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})


def panel_label(ax, k, inside=False, color="black"):
    """Letter (a), (b), ... for the k-th panel.

    Outside the axes at the top-left corner by default; inside, with a dark
    outline, for map panels that fill their axes.
    """
    text = f"({chr(ord('a') + k)})"
    if inside:
        ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])
    else:
        ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right",
                va="bottom", fontweight="bold")
