"""Figures for the habitat ablation, one set per region.

Four arms at one cell (delta = 0.25, landscape-average R0 = 3.5, so
R0-tilde = 3.5 / c-bar; c4 = 0.5), crossing the two habitat mechanisms:

    slope anisotropy   eta(theta) up the slope, 2 - eta along it  on / eta = 1
    land cover         p(x) and c(x) at their measured values / flattened

"maps" shows the measured-land-cover arms: one row of optimal v per slope
treatment, with the uncontrolled and controlled 0.1 deer/km^2 fronts on
every panel. "region" shows terrain, kappa*p(x) and S_inf in one row.

    python utilities/make_ablation_figure.py [--run NAME] [maps|region] [region ...]

Runs are read from cluster_runs/ablation; the region panel reads
meshes100/<region>_mesh/bundle.h5, written by export_mesh_to_xdmf.py.
"""
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import WIDTH, panel_label  # noqa: E402

OUT = "figures"
REGIONS = ["upper_arkansas", "estes_park"]
# R0-tilde = 3.5 / c-bar, as it appears in point names; bundle.h5 supplies ell
RUNS = {
    # c-bar = 0.8260, 0.7782 and 0.8374
    "ablation": {"root": "cluster_runs/ablation", "suffix": "_100km",
                 "R0T": {"upper_arkansas": "4p237", "estes_park": "4p498",
                         "carbondale": "4p18"},
                 "bundle": "meshes100/{region}_mesh/bundle.h5"},
}
RUN = RUNS["ablation"]
KAPPA_KM2 = 4.0          # parameters.json kappa = 4.0e6 m^2/yr

GROUPS = [("aniso", r"slope-anisotropic $D$"),
          ("iso", r"isotropic $D$")]
# baselines are written every 2 yr. The frame at t holds the state at t and the
# control of block [t, t+1), so the last panel is t = 18: no block starts at 20.
YEARS = [2, 8, 14, 18]

V_MAX = 3.0              # control_upper[0]; the box is [0, v_max]
FRONT = 0.1              # E+I+R front level (deer/km^2)
FRONT_LW = 1.6           # front contour width (pt)
BASE_INK = "white"       # uncontrolled front, dashed
CTRL_INK = "#39d0ff"     # optimal front, solid


def cell(region):
    return f"delta_0p25_R0_{RUN['R0T'][region]}"


def run_dir(region, group):
    return f"{RUN['root']}/{region}_{group}_lc/{cell(region)}__c4_0p5"


def baseline_dir(region, group="aniso"):
    return f"{RUN['root']}/{region}_{group}_lc/baseline_{cell(region)}"


def mesh_of(d):
    with h5py.File(f"{d}/CWD_control_v.h5", "r") as f:
        return f["Mesh/mesh/geometry"][:], f["Mesh/mesh/topology"][:]


def fld(d, fn, nm, fr):
    with h5py.File(f"{d}/CWD_{fn}.h5", "r") as f:
        return f["Function"][nm][str(fr)][:, 0]


def infected(d, fr):
    return (fld(d, "E", "Exposed", fr) + fld(d, "I", "Preclinical", fr)
            + fld(d, "R", "Clinical", fr))


def fig_region_map(region):
    """Terrain, kappa*p(x) and S_inf.

    Mirrors make_results_figures.fig_region so the second region is shown on
    the same terms as the study region. ell(x) is read from the bundle's own
    HDF5, whose geometry is checked to match the run's node for node.
    """
    d = baseline_dir(region)
    g, t = mesh_of(d)
    x, y, z = g[:, 0] / 1000, g[:, 1] / 1000, g[:, 2]
    tri = Triangulation(x, y, t)
    S0 = fld(d, "S_infinity", "Disease_Free_Equilibrium", 0)
    focus = np.argmax(infected(d, 0))   # the seeded focus, marked on every panel
    with h5py.File(RUN["bundle"].format(region=region), "r") as f:
        if not np.array_equal(f["Mesh/mesh/geometry"][:], g):
            sys.exit(f"{region}: bundle and run node orders differ")
        ell = f["Function/Permeability_ell/0"][:, 0]

    # One row: the handover state is omitted, since t = 2 of the "maps"
    # figure shows the same front.
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 0.42 * WIDTH), layout="compressed")
    panels = [
        (z, "elevation anomaly (m)", "terrain"),
        (KAPPA_KM2 * ell, r"$\kappa\,p(\mathbf{x})$ (km$^2$/yr)", "viridis"),
        (S0, r"$S_\infty$ (deer/km$^2$)", "YlGn"),
    ]
    for k, (a, (v, lab, cm)) in enumerate(zip(ax.flat, panels)):
        h = a.tripcolor(tri, v, cmap=cm, shading="gouraud")
        cb = fig.colorbar(h, ax=a, orientation="horizontal", shrink=0.9, pad=0.03)
        cb.ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.set_title(lab)
        panel_label(a, k, inside=True, color="white")
        a.set_aspect("equal")
        a.plot(x[focus], y[focus], marker="x", ms=5, mew=1.2, color="white",
               path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
        # pin the limits so the marker cannot widen them
        a.set_xlim(x.min(), x.max()); a.set_ylim(y.min(), y.max())
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    a = ax[0]
    a.plot([x.min() + 2, x.min() + 12], [y.min() + 2.5] * 2, "k-", lw=1.5)
    a.text(x.min() + 7, y.min() + 3.8, "10 km", ha="center")
    os.makedirs(OUT, exist_ok=True)
    name = f"ablation_region_{region}{RUN['suffix']}.png"
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print(name)


def fig_region(region):
    """Optimal control under each slope treatment, measured land cover only.

    Each row is one slope treatment's optimal v, with its uncontrolled and
    controlled fronts drawn over it. The flattened-land-cover arms enter the
    results table, not this figure.
    """
    rows = [(g, baseline_dir(region, g), run_dir(region, g)) for g, _ in GROUPS]
    missing = [d for g, b, d in rows
               if not os.path.isfile(f"{d}/optimization_history.json")]
    if missing:
        sys.exit(f"{region}: unfinished {missing}")

    g0, t = mesh_of(rows[0][2])
    x, y = g0[:, 0] / 1000, g0[:, 1] / 1000
    tri = Triangulation(x, y, t)
    vnorm = Normalize(0, V_MAX)

    # margins in inches, so the colorbar and legend keep their room
    nr, nc = len(rows), len(YEARS)
    L, R_, WS, HS = 0.07, 0.98, 0.02, 0.03
    TOP, BOT = 0.22, 0.62
    W = WIDTH
    pw = W * (R_ - L) / (nc + (nc - 1) * WS)
    ph = pw * (y.max() - y.min()) / (x.max() - x.min())
    H = ph * (nr + (nr - 1) * HS) + TOP + BOT
    fig, ax = plt.subplots(nr, nc, figsize=(W, H), squeeze=False)
    fig.subplots_adjust(left=L, right=R_, bottom=BOT / H, top=1 - TOP / H,
                        wspace=WS, hspace=HS)

    def front(a, I, style, color):
        cs = a.tricontour(tri, I, levels=[FRONT], linewidths=FRONT_LW,
                          linestyles=style, colors=[color])
        cs.set(path_effects=[pe.withStroke(linewidth=FRONT_LW + 1.4,
                                           foreground="black", alpha=0.6)])

    for i, (g, b, d) in enumerate(rows):
        for j, yr in enumerate(YEARS):
            a = ax[i, j]
            hv = a.tripcolor(tri, fld(d, "control_v", "Targeted_Control", yr),
                             cmap="inferno", norm=vnorm, shading="gouraud")
            front(a, infected(b, yr), "dashed", BASE_INK)
            front(a, infected(d, yr), "solid", CTRL_INK)
            a.set_aspect("equal")
            a.set_xlim(x.min(), x.max()); a.set_ylim(y.min(), y.max())
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            panel_label(a, i * nc + j, inside=True, color="white")
            if i == 0:
                a.set_title(rf"$t = {yr}$")
        ax[i, 0].text(-0.08, 0.5, dict(GROUPS)[g], rotation=90, va="center",
                      ha="center", transform=ax[i, 0].transAxes)

    yc = 0.45 * BOT / H
    cax = fig.add_axes([L, yc, 0.3, 0.08 / H])
    cb = fig.colorbar(hv, cax=cax, orientation="horizontal")
    cb.set_label(r"$v$ (yr$^{-1}$)", labelpad=1)
    cb.ax.tick_params(pad=1)

    # dark key so the white dashed front reads as it does on the panels
    handles = [plt.Line2D([], [], color=BASE_INK, ls="--", lw=FRONT_LW),
               plt.Line2D([], [], color=CTRL_INK, ls="-", lw=FRONT_LW)]
    leg = fig.legend(handles, ["uncontrolled", "optimal"],
                     title=rf"$E{{+}}I{{+}}R = {FRONT}$ deer/km$^2$",
                     loc="center left", bbox_to_anchor=(0.5, yc + 0.04 / H),
                     facecolor="0.15", edgecolor="none", framealpha=1,
                     labelcolor="white", ncol=2, handlelength=2.2, columnspacing=1.0)
    leg.get_title().set_color("white")

    # scale bar on the first panel, matching study_region.png
    a = ax[0, 0]
    a.plot([x.min() + 2, x.min() + 12], [y.min() + 2.5] * 2, "-",
           color="white", lw=1.5,
           path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])
    a.text(x.min() + 2, y.min() + 4.5, "10 km", ha="left", color="white",
           path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])

    os.makedirs(OUT, exist_ok=True)
    name = f"ablation_v_maps_{region}{RUN['suffix']}.png"
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print(name)


if __name__ == "__main__":
    # "maps" renders only the control panels, "region" only the region panel
    args = sys.argv[1:]
    if "--run" in args:
        i = args.index("--run")
        RUN = RUNS[args[i + 1]]
        del args[i:i + 2]
    args = args or REGIONS
    what = {"maps": [fig_region], "region": [fig_region_map]}
    todo = [a for a in args if a not in what]
    fns = next((what[a] for a in args if a in what), [fig_region])
    for r in (todo or REGIONS):
        for fn in fns:
            fn(r)
