"""Where a sight-based cull can push the local reproduction index below one.

Filled contours give the smallest constant cull rate that brings R_c(v) below
one, v > alpha (R0 - 1) / (1 - R0_pre), over R0 and the deposition ratio
varrho = rho_R/rho_I at the adopted route ratio m. The hatched region has
R0_pre >= 1, where no cull rate suffices, and its edge is also drawn at the two
m arms of the ratio probe. Closed form only; no solver output is read.

    python utilities/make_cull_reach_figure.py

Needs only numpy and matplotlib.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import WIDTH  # noqa: E402
from sweep_setup import r0_parts  # noqa: E402

OUT = Path("figures/cull_reach.png")
M_RANGE = (2.6, 18.2)        # the m arms: Garlick et al. (2013) and free-ranging Miller et al. (2006)
TICKS = [1, 3, 10, 30, 100]  # cull rate, yr^-1; the log scale is clipped to this range


def preclinical_share(rho, m, tau):
    """R0_pre / R0: the direct route splits by duration, the environmental by deposit."""
    return (1.0 / (1.0 + tau) + m / (1.0 + rho * tau)) / (1.0 + m)


def main():
    p = json.loads(Path("parameters.json").read_text(encoding="utf-8"))
    rx = p["reaction"]
    alpha = float(rx["clinical_removal_rate"])
    tau = float(rx["preclinical_to_clinical_rate"]) / alpha
    rho_ref = (float(rx["environmental_shedding_rate_clinical"])
               / float(rx["environmental_shedding_rate_preclinical"]))
    R0_dir, R0_env, _ = r0_parts(rx, float(p["land_cover"]["carrying_capacity_base"]))
    m_ref = R0_env / R0_dir
    v_max = float(p["optimal_control"]["control_maximum"])
    print(f"alpha = {alpha:.3f}, tau = {tau:.4f}, m = {m_ref:.3f}, "
          f"rho = {rho_ref:.3f}, v_max = {v_max:g}")

    R0 = np.linspace(1.0, 6.0, 601)
    rho = np.geomspace(1.0, 10.0, 481)
    RR, PP = np.meshgrid(R0, rho)
    pre = RR * preclinical_share(PP, m_ref, tau)
    with np.errstate(divide="ignore"):
        need = np.where(pre < 1.0, alpha * (RR - 1.0) / (1.0 - pre), np.inf)

    fig, ax = plt.subplots(figsize=(0.65 * WIDTH, 3.0), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "blues", plt.get_cmap("Blues")(np.linspace(0.08, 0.95, 256)))
    norm = LogNorm(vmin=TICKS[0], vmax=TICKS[-1])
    shown = np.clip(np.where(np.isfinite(need), need, np.nan), TICKS[0], TICKS[-1])
    fill = ax.pcolormesh(RR, PP, shown, cmap=cmap, norm=norm, shading="auto",
                         rasterized=True)

    plt.rcParams["hatch.color"] = "0.75"
    plt.rcParams["hatch.linewidth"] = 0.5
    ax.contourf(RR, PP, pre, levels=[1.0, np.inf], colors=["0.95"],
                hatches=["////"])
    cap = ax.contour(RR, PP, need, levels=[v_max], colors="black",
                     linewidths=1.2)
    ax.clabel(cap, fmt={v_max: rf"$v_{{\max}} = {v_max:g}$"}, fontsize=7,
              manual=[(1.55, 5.5)])

    # Edge of the hatched region at the two m arms of the ratio probe, drawn
    # with a white halo and labeled on the curve so it reads over both fills.
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    # (style, label height, horizontal offset from the curve, alignment)
    for m, style, at, dx, ha in ((M_RANGE[0], (0, (4, 2)), 8.0, -0.1, "right"),
                                 (m_ref, "-", None, 0, ""),
                                 (M_RANGE[1], (0, (1, 1.2)), 6.0, 0.1, "left")):
        edge = 1.0 / preclinical_share(rho, m, tau)
        ax.plot(edge, rho, color="black", ls=style, lw=1.4, path_effects=halo)
        if at is not None:
            x = 1.0 / preclinical_share(at, m, tau) + dx
            ax.text(x, at, f"$m = {m:g}$", ha=ha, va="center", fontsize=7,
                    bbox=dict(fc="white", ec="black", lw=0.5, pad=1.2))
    ax.text(5.0, 1.45, "no cull rate\nsuffices", ha="center", va="center",
            fontsize=7, bbox=dict(fc="white", ec="none", pad=1.5))

    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(1.0, 10.0)
    ax.set_xlabel(r"$R_0$")
    ax.set_ylabel(r"Deposition ratio $\varrho = \rho_R/\rho_I$")

    bar = fig.colorbar(fill, ax=ax, ticks=TICKS)
    bar.ax.minorticks_off()
    bar.ax.set_yticklabels([f"{v:g} ({100 * v / (v + alpha):.0f}%)" for v in TICKS])
    bar.set_label("Cull rate needed, yr$^{-1}$ (share of clinical deer culled)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
