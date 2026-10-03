"""Figures for the habitat ablation, one set per region.

Four arms at one cell (delta = 0.25, landscape-average R0 = 3.5, so
R0-tilde = 3.5 / c-bar; c4 = 0.5), crossing the two habitat mechanisms:

    slope anisotropy   eta(theta) up the slope, 2 - eta along it  on / eta = 1
    land cover         p(x) and c(x) at their measured values / flattened

"maps" shows the measured-land-cover arms: for each slope treatment, the
uncontrolled E+I+R above the optimal v, with both 0.1 deer/km^2 fronts on
every panel. "region" shows terrain, kappa(x), S_inf and the handover state.

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
# baselines are written every 2 yr; the control horizon ends at year 20
YEARS = [2, 8, 14, 20]

V_MAX = 3.0              # control_upper[0]; the box is [0, v_max]
FRONT = 0.1              # E+I+R front level (deer/km^2)
ICMAP = "viridis"        # uncontrolled E+I+R; kept clear of v's inferno
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
    """Terrain, kappa(x), S_inf and the handover infected field.

    Mirrors make_results_figures.fig_region so the second region is shown on
    the same terms as the study region. ell(x) is read from the bundle's own
    HDF5, whose geometry is checked to match the run's node for node.
    """
    d = baseline_dir(region)
    g, t = mesh_of(d)
    x, y, z = g[:, 0] / 1000, g[:, 1] / 1000, g[:, 2]
    tri = Triangulation(x, y, t)
    S0 = fld(d, "S_infinity", "Disease_Free_Equilibrium", 0)
    I0 = infected(d, 0)
    with h5py.File(RUN["bundle"].format(region=region), "r") as f:
        if not np.array_equal(f["Mesh/mesh/geometry"][:], g):
            sys.exit(f"{region}: bundle and run node orders differ")
        ell = f["Function/Permeability_ell/0"][:, 0]

    fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 0.92 * WIDTH), layout="compressed")
    panels = [
        (z, "elevation anomaly (m)", "terrain"),
        (KAPPA_KM2 * ell, r"$\kappa(\mathbf{x})$ (km$^2$/yr)", "viridis"),
        (S0, r"$S_\infty$ (deer/km$^2$)", "YlGn"),
        (I0, r"$E{+}I{+}R$ at $t=0$ (deer/km$^2$)", "magma_r"),
    ]
    for k, (a, (v, lab, cm)) in enumerate(zip(ax.flat, panels)):
        h = a.tripcolor(tri, v, cmap=cm, shading="gouraud")
        cb = fig.colorbar(h, ax=a, shrink=0.88, pad=0.02)
        if lab.startswith("$E"):
            a.tricontour(tri, v, levels=[FRONT], colors=CTRL_INK, linewidths=0.8)
            cb.ax.axhline(FRONT, color=CTRL_INK, lw=1.0)
        a.set_title(lab)
        panel_label(a, k, inside=True, color="white")
        a.set_aspect("equal")
        # tricontour drops tripcolor's tight limits; pin them so panels match
        a.set_xlim(x.min(), x.max()); a.set_ylim(y.min(), y.max())
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    a = ax[0, 0]
    a.plot([x.min() + 2, x.min() + 12], [y.min() + 2.5] * 2, "k-", lw=1.5)
    a.text(x.min() + 7, y.min() + 3.8, "10 km", ha="center")
    os.makedirs(OUT, exist_ok=True)
    name = f"ablation_region_{region}{RUN['suffix']}.png"
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print(name)


def fig_region(region):
    """Uncontrolled burden and optimal control, measured land cover only.

    Rows pair each slope treatment's uncontrolled E+I+R with its optimal v.
    The flattened-land-cover arms enter the results table, not this figure.
    """
    rows = []
    for g, _ in GROUPS:
        rows += [(g, "base", baseline_dir(region, g)), (g, "opt", run_dir(region, g))]
    missing = [d for g, k, d in rows
               if k == "opt" and not os.path.isfile(f"{d}/optimization_history.json")]
    if missing:
        sys.exit(f"{region}: unfinished {missing}")

    g0, t = mesh_of(rows[0][2])
    x, y = g0[:, 0] / 1000, g0[:, 1] / 1000
    tri = Triangulation(x, y, t)

    # the baseline's E+I+R also supplies the dashed front on its control row
    base = {g: {yr: infected(d, yr) for yr in YEARS}
            for g, k, d in rows if k == "base"}
    inorm = Normalize(0, max(float(q.max()) for f in base.values()
                             for q in f.values()))
    vnorm = Normalize(0, V_MAX)

    nr, nc = len(rows), len(YEARS)
    L, R_, B, T, WS, HS = 0.15, 0.85, 0.11, 0.962, 0.02, 0.06
    W = WIDTH
    pw = W * (R_ - L) / (nc + (nc - 1) * WS)
    ph = pw * (y.max() - y.min()) / (x.max() - x.min())
    H = ph * (nr + (nr - 1) * HS) / (T - B)
    fig, ax = plt.subplots(nr, nc, figsize=(W, H))
    fig.subplots_adjust(left=L, right=R_, bottom=B, top=T, wspace=WS, hspace=HS)

    def front(a, I, style, color):
        cs = a.tricontour(tri, I, levels=[FRONT], linewidths=0.9,
                          linestyles=style, colors=[color])
        cs.set(path_effects=[pe.withStroke(linewidth=2.0, foreground="black",
                                           alpha=0.6)])

    for i, (g, kind, d) in enumerate(rows):
        for j, yr in enumerate(YEARS):
            a = ax[i, j]
            if kind == "base":
                hi = a.tripcolor(tri, base[g][yr], cmap=ICMAP, norm=inorm,
                                 shading="gouraud")
            else:
                hv = a.tripcolor(tri, fld(d, "control_v", "Targeted_Control", yr - 1),
                                 cmap="inferno", norm=vnorm, shading="gouraud")
                front(a, infected(d, yr), "solid", CTRL_INK)
            front(a, base[g][yr], "dashed", BASE_INK)
            a.set_aspect("equal")
            a.set_xlim(x.min(), x.max()); a.set_ylim(y.min(), y.max())
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            panel_label(a, i * nc + j, inside=True, color="white")
            if i == 0:
                a.set_title(f"year {yr}")
            if j == 0:
                lab = r"uncontrolled" "\n" r"$E{+}I{+}R$" if kind == "base" \
                    else r"optimal" "\n" r"$v$"
                a.text(-0.12, 0.5, lab, rotation=90, va="center", ha="center",
                       multialignment="center", transform=a.transAxes)

    # one bracketed label per pair of rows sharing a slope treatment
    for key, label in GROUPS:
        rs = [i for i, r in enumerate(rows) if r[0] == key]
        top = ax[rs[0], 0].get_position().y1
        bot = ax[rs[-1], 0].get_position().y0
        xb = ax[rs[0], 0].get_position().x0 - 0.085
        fig.text(xb - 0.02, 0.5 * (top + bot), label, rotation=90,
                 va="center", ha="center")
        fig.add_artist(plt.Line2D([xb, xb], [bot, top], color="0.3", lw=0.8))

    y0, y1 = ax[-1, 0].get_position().y0, ax[0, 0].get_position().y1
    cc = fig.add_axes([0.865, y0 + 0.25 * (y1 - y0), 0.015, 0.5 * (y1 - y0)])
    cbi = fig.colorbar(hi, cax=cc)
    cbi.set_label(r"$E{+}I{+}R$ (deer/km$^2$)", labelpad=2)
    cbi.ax.tick_params(pad=1.5)
    cbi.outline.set_linewidth(0.4)

    cax = fig.add_axes([0.15, 0.05, 0.3, 0.012])
    cb = fig.colorbar(hv, cax=cax, orientation="horizontal")
    cb.set_label(r"$v$ (yr$^{-1}$)", labelpad=1)
    cb.ax.tick_params(pad=1)

    # dark key so the white dashed front reads as it does on the panels
    handles = [plt.Line2D([], [], color=BASE_INK, ls="--", lw=0.9),
               plt.Line2D([], [], color=CTRL_INK, ls="-", lw=0.9)]
    leg = fig.legend(handles, ["uncontrolled", "optimal"],
                     title=rf"$E{{+}}I{{+}}R = {FRONT}$ deer/km$^2$",
                     loc="center left", bbox_to_anchor=(0.5, 0.05),
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
