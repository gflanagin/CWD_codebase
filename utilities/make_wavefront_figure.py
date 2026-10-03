"""The uncontrolled wave on the flat disk, and how far the control holds it back.

Each row is one R0. The first two columns are radial profiles of the
uncontrolled state at the slowest and fastest reservoir, at the times in
PROFILE_YEARS. The third column is the leading edge against time for every delta,
uncontrolled and optimal. The leading edge is the outermost radius at which the
radially binned E+I+R is at least 0.1 deer/km^2. Runs are read from
cluster_runs/flat_grid.

    python utilities/make_wavefront_figure.py            # figure
    python utilities/make_wavefront_figure.py --table    # numbers for the text

Needs only h5py and matplotlib.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flat_transect import read_series, node_areas  # noqa: E402
from figure_style import WIDTH, panel_label  # noqa: E402

ROOT = Path("cluster_runs/flat_grid")
OUT = Path("figures/flat_grid/wavefronts.png")
DELTAS = ["0p125", "0p25", "0p5", "1"]
R0S = ["2", "3p5", "5"]
# profile year per delta: the slow reservoir's wave needs twice as long to develop
PROFILE_YEARS = {"0p125": 12.0, "1": 6.0}
EDGE = 0.1               # leading-edge level (deer/km^2)
BIN_KM = 0.25
RIM_KM = 40.0
INTERIOR_KM = 38.0       # edges past this are treated as at the rim
T_C = 20.0
LABELS = {"S": "Susceptible", "E": "Exposed", "I": "Preclinical",
          "R": "Clinical", "W": "Environment"}
COLOURS = {"E": "#8c6bb1", "I": "#2b8cbe", "R": "#d7301f", "W": "#8c510a"}
DCOLOURS = dict(zip(DELTAS, plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 4))))


def value(tag):
    return float(tag.replace("p", "."))


class Disk:
    """Radial bins shared by every run on the disk."""

    def __init__(self, run):
        g, t, _ = read_series(run, "S_infinity", "Disease_Free_Equilibrium")
        mass = node_areas(g, t)
        radius = np.linalg.norm(g[:, :2], axis=1) / 1000.0
        edges = np.arange(0.0, radius.max() + BIN_KM, BIN_KM)
        self.r = 0.5 * (edges[1:] + edges[:-1])
        self.index = np.clip(np.digitize(radius, edges) - 1, 0, len(self.r) - 1)
        self.mass = mass
        self.weight = np.bincount(self.index, mass, len(self.r))

    def profile(self, values):
        return (np.bincount(self.index, self.mass * values, len(self.r))
                / self.weight)

    def series(self, run, key):
        return {t: self.profile(v)
                for t, v in read_series(run, key, LABELS[key])[2].items()}

    def edges(self, run):
        """{time: leading-edge radius}; 0 where no bin reaches EDGE."""
        E, I, R = (self.series(run, k) for k in "EIR")
        out = {}
        for t in sorted(E):
            ok = np.nonzero(E[t] + I[t] + R[t] >= EDGE)[0]
            out[t] = self.r[ok.max()] if len(ok) else 0.0
        return out


def baseline(d, R0):
    return ROOT / f"baseline_delta_{d}_R0_{R0}"


def optimal(d, R0):
    return ROOT / f"delta_{d}_R0_{R0}__c4_0p5"


def first_crossing(edges, radius):
    """Interpolated time at which the edge first reaches radius."""
    ts = sorted(edges)
    for a, b in zip(ts, ts[1:]):
        if edges[a] < radius <= edges[b]:
            return a + (b - a) * (radius - edges[a]) / (edges[b] - edges[a])
    return np.nan


def speed(edges):
    """Least-squares edge speed over the horizon, while the edge is interior."""
    ts = [t for t in sorted(edges) if t <= T_C and 0 < edges[t] < INTERIOR_KM]
    if len(ts) < 3:
        return np.nan
    return np.polyfit(ts, [edges[t] for t in ts], 1)[0]


def table(disk):
    print(f"{'R0':>4} {'delta':>6} {'speed base':>10} {'speed opt':>9} {'ratio':>6}"
          f" {'t20 base':>8} {'t20 opt':>7} {'delay':>6} {'edge@20 base':>12} {'opt':>6}")
    for R0 in R0S:
        for d in DELTAS:
            b, o = disk.edges(baseline(d, R0)), disk.edges(optimal(d, R0))
            sb, so = speed(b), speed(o)
            tb, to = first_crossing(b, 20.0), first_crossing(o, 20.0)
            print(f"{value(R0):4g} {value(d):6g} {sb:10.2f} {so:9.2f} {so / sb:6.2f}"
                  f" {tb:8.1f} {to:7.1f} {to - tb:6.1f} {b[T_C]:12.1f} {o[T_C]:6.1f}")


def figure(disk):
    state = {(d, R0): {k: disk.series(baseline(d, R0), k) for k in "SEIRW"}
             for d in PROFILE_YEARS for R0 in R0S}
    s_inf = disk.profile(next(iter(read_series(
        baseline(DELTAS[0], R0S[0]), "S_infinity",
        "Disease_Free_Equilibrium")[2].values())))
    # each class scaled by its largest value over all profile panels, so the
    # panels compare magnitudes as well as shapes
    top = {k: max(float(f[k][PROFILE_YEARS[d]].max()) for (d, _), f in state.items())
           for k in "EIRW"}

    fig, ax = plt.subplots(len(R0S), 3, figsize=(WIDTH, 6.2),
                           constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for i, R0 in enumerate(R0S):
        for j, (d, yr) in enumerate(PROFILE_YEARS.items()):
            a, f = ax[i, j], state[(d, R0)]
            a.plot(disk.r, f["S"][yr] / s_inf, color="0.45", lw=1.2,
                   ls="--", label=r"$S/S_\infty$")
            for k in "EIRW":
                a.plot(disk.r, f[k][yr] / top[k], color=COLOURS[k],
                       lw=1.2, label=k)
            a.set_xlim(0, RIM_KM)
            a.set_ylim(0, 1.05)
            if i == 0:
                a.set_title(rf"$\delta = {value(d):g}$, year ${yr:g}$")
            if i == len(R0S) - 1:
                a.set_xlabel("radius (km)")
            if j == 0:
                a.set_ylabel(rf"$R_0 = {value(R0):g}$" "\n" "scaled density")
            else:
                a.tick_params(labelleft=False)

        a = ax[i, 2]
        for d in DELTAS:
            for run, ls in ((baseline(d, R0), "-"), (optimal(d, R0), "--")):
                ts, rs = interior(disk.edges(run))
                a.plot(ts, rs, ls=ls, color=DCOLOURS[d], lw=1.2,
                       label=rf"$\delta = {value(d):g}$" if ls == "-" else None)
        a.axvline(T_C, color="0.6", lw=0.7)
        a.set_xlim(0, 30)
        a.set_ylim(0, RIM_KM)
        a.set_ylabel("leading edge (km)")
        if i == 0:
            a.set_title("leading edge")
        if i == len(R0S) - 1:
            a.set_xlabel("years after handover")

    for k, a in enumerate(ax.flat):
        panel_label(a, k)
    ax[0, 0].legend(frameon=False, loc="upper right", handlelength=1.6)
    handles, labels = ax[0, 2].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", ls="-", lw=1.0),
                plt.Line2D([], [], color="0.3", ls="--", lw=1.0)]
    labels += ["uncontrolled", "optimal"]
    # the R0 = 5 edges leave the panel before year 20, so its right half is free
    ax[-1, 2].legend(handles, labels, frameon=False, loc="lower right",
                     handlelength=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


def interior(edges):
    """Edge curve up to the first frame on the rim or after burn-out."""
    ts = []
    for t in sorted(edges):
        if t > 0 and not 0 < edges[t] < INTERIOR_KM:
            break
        ts.append(t)
    return ts, [edges[t] for t in ts]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", action="store_true")
    disk = Disk(baseline(DELTAS[0], R0S[0]))
    if p.parse_args().table:
        table(disk)
    else:
        figure(disk)
