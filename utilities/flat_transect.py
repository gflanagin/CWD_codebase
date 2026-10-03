"""Optimal control on a flat, uniform disk, and its radial transect.

The disk has the index focus at its centre and a single land-cover class, so
the solution is radially symmetric. Four stages, each skipped once its output
exists:

  build   flat disk mesh (gmsh) and a uniform land-cover bundle
  params  one point file via sweep_setup.py
  run     the optimization, through CWD_optimal_control.py
  plot    radial profiles of S, E, I, R, W and v

    python utilities/flat_transect.py --out runs/flat_transect \\
        --delta 1 --R0 5 --figure figures/flat_transect.png

Build only the bundle used by the flat-disk grid (condor/README.md):

    python utilities/flat_transect.py --out runs/flat_disk --radius-km 40 \\
        --permeability 1 --build-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ build
def build_bundle(folder, radius_km, h_m, land_cover_class, permeability=None):
    """Flat disk mesh with uniform land-cover arrays and a centred focus.

    ``permeability`` overrides the class's own l; the flat grid sets it to 1 so
    that D = kappa I exactly.
    """
    import gmsh
    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.io.gmsh import read_from_msh

    folder.mkdir(parents=True, exist_ok=True)
    mesh_path = folder / "terrain.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("flat_disk")
    radius = 1000.0 * radius_km
    disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [disk], 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_m)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_m)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(mesh_path))
    gmsh.finalize()

    domain = read_from_msh(str(mesh_path), MPI.COMM_SELF, gdim=3).mesh
    nodes = fem.functionspace(domain, ("CG", 1)).dofmap.index_map.size_local

    parameters = json.loads((REPO / "parameters.json").read_text(
        encoding="utf-8"))
    code = str(land_cover_class)
    if permeability is None:
        permeability = parameters["land_cover"]["diffusivity"][code]
    np.save(folder / "land_cover_classes.npy",
            np.full(nodes, land_cover_class, dtype=np.int32))
    np.save(folder / "land_cover_diffusivity.npy",
            np.full(nodes, permeability, dtype=np.float64))

    seed = parameters["initial_conditions"]["infected_gaussians"][0]
    parameters["initial_conditions"]["infected_gaussians"] = [
        dict(seed, center=[0.0, 0.0])]
    (folder / "effective_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8")
    capacity = parameters["land_cover"]["carrying_capacity_fractions"][code]
    print(f"flat disk: radius {radius_km:g} km, h = {h_m:g} m, {nodes:,} nodes;"
          f" class {code}, c = {capacity}, l = {permeability}")


# ----------------------------------------------------------------- params
def make_params(bundle, params, delta, R0, c3):
    subprocess.run(
        [sys.executable, str(REPO / "utilities" / "sweep_setup.py"),
         "--base", str(bundle / "effective_parameters.json"),
         "--overlay", str(REPO / "parameters.json"),
         "--out", str(params), "--delta", str(delta), "--R0", str(R0),
         "--c4", "0", "--c3", str(c3), "--time-step-years", "0.05"],
        check=True)
    names = (params / "manifest.txt").read_text(encoding="utf-8").split()
    return params / f"{names[0]}.json"


# -------------------------------------------------------------------- run
def run_optimization(bundle, point, run):
    run.mkdir(parents=True, exist_ok=True)
    frames_per_year = round(1.0 / json.loads(point.read_text(
        encoding="utf-8"))["time"]["time_step_years"])
    with (run / "solve.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(REPO / "CWD_optimal_control.py"),
             "--mesh-folder", str(bundle), "--parameters", str(point),
             "--output-folder", str(run),
             "--disease-state", str(run / "disease_state.npy"),
             "--output-every", str(frames_per_year), "--no-spinup-series",
             "--checkpoint-every", "5"],
            check=True, stdout=log, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------- plot
def read_series(run, name, label):
    """{time: nodal values} for one XDMF/HDF5 series, and the geometry."""
    import h5py
    with h5py.File(run / f"CWD_{name}.h5", "r") as f:
        geometry = f["Mesh/mesh/geometry"][:]
        topology = f["Mesh/mesh/topology"][:]
        frames = {float(k): f["Function"][label][k][:, 0]
                  for k in f["Function"][label]}
    return geometry, topology, frames


def node_areas(geometry, topology):
    p = geometry[:, :3]
    area = 0.5 * np.linalg.norm(np.cross(p[topology[:, 1]] - p[topology[:, 0]],
                                         p[topology[:, 2]] - p[topology[:, 0]]),
                                axis=1)
    mass = np.zeros(len(p))
    for k in range(3):
        np.add.at(mass, topology[:, k], area / 3.0)
    return mass


class Transect:
    """Radially binned, area-weighted profiles of one run's saved series."""

    def __init__(self, run, bin_km):
        self.series = {key: read_series(run, key, label)[2] for key, label in (
            ("S", "Susceptible"), ("E", "Exposed"), ("I", "Preclinical"),
            ("R", "Clinical"), ("W", "Environment"),
            ("control_v", "Targeted_Control"))}
        geometry, topology, s_inf = read_series(run, "S_infinity",
                                                "Disease_Free_Equilibrium")
        mass = node_areas(geometry, topology)
        radius = np.linalg.norm(geometry[:, :2], axis=1) / 1000.0
        edges = np.arange(0.0, radius.max() + bin_km, bin_km)
        self.r_max = edges[-1]
        self.centres = 0.5 * (edges[1:] + edges[:-1])
        self.index = np.clip(np.digitize(radius, edges) - 1, 0,
                             len(self.centres) - 1)
        self.mass = mass
        self.weight = np.bincount(self.index, mass, len(self.centres))
        self.s_inf = self.profile(next(iter(s_inf.values())))
        self.bin_km = bin_km
        horizon = json.loads((run / "optimization_history.json").read_text(
            encoding="utf-8"))["time"]["control_horizon_years"]
        self.control_years = [t for t in sorted(self.series["control_v"])
                              if t < horizon]

    def profile(self, values):
        return (np.bincount(self.index, self.mass * values, len(self.centres))
                / self.weight)

    def at(self, key, t):
        frames = self.series[key]
        return self.profile(frames[min(frames, key=lambda s: abs(s - t))])

    def front(self, t):
        """r_50: where E+I+R falls to half its interior plateau."""
        infected = self.at("E", t) + self.at("I", t) + self.at("R", t)
        plateau = infected[: max(1, int(2.0 / self.bin_km))].max()
        below = np.nonzero(infected < 0.5 * plateau)[0]
        return self.centres[below[0]] if plateau > 0 and len(below) else np.nan


def plot(runs, figure, years, bin_km, labels=None):
    """One row of panels per run, one column per year, sharing the v axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_style import WIDTH, panel_label

    runs = [runs] if isinstance(runs, (str, Path)) else list(runs)
    transects = [Transect(Path(run), bin_km) for run in runs]
    first = transects[0]
    if not years:
        # three years spread over the span in which the front is interior
        interior = [t for t in first.control_years
                    if np.isfinite(first.front(t))
                    and first.front(t) < 0.7 * first.r_max]
        picks = np.linspace(0, len(interior) - 1, 3).round().astype(int)
        years = [interior[k] for k in sorted(set(picks))]

    # one v axis across all panels, set by the largest effort in the window
    v_top = max(tr.series["control_v"][t].max()
                for tr in transects for t in tr.control_years)
    colours = {"E": "#8c6bb1", "I": "#2b8cbe", "R": "#d7301f", "W": "#8c510a"}
    fig, axes = plt.subplots(len(runs), len(years),
                             figsize=(WIDTH, 1.45 * len(runs) + 0.6),
                             sharey=True, squeeze=False,
                             constrained_layout=True)
    for i, tr in enumerate(transects):
        for j, t in enumerate(years):
            ax = axes[i, j]
            ax.plot(tr.centres, tr.at("S", t) / tr.s_inf, color="0.45",
                    lw=1.2, ls="--", label=r"$S/S_\infty$")
            for key in ("E", "I", "R", "W"):
                p = tr.at(key, t)
                ax.plot(tr.centres, p / np.nanmax(p), color=colours[key],
                        lw=1.2, label=key)
            clinical = tr.at("R", t) * tr.at("S", t)
            ax.plot(tr.centres, clinical / np.nanmax(clinical), color="0.2",
                    lw=0.8, ls=":", label=r"$RS$")
            r50 = tr.front(t)
            if np.isfinite(r50):
                ax.axvline(r50, color="0.6", lw=0.7)
            twin = ax.twinx()
            twin.plot(tr.centres, tr.at("control_v", t), color="k", lw=2.0,
                      label=r"$v$")
            twin.set_ylim(0.0, 1.05 * v_top)
            if j < len(years) - 1:
                twin.tick_params(right=False, labelright=False)
            else:
                twin.set_ylabel(r"$v$ (yr$^{-1}$)")
            if i == 0:
                ax.set_title(f"year {t:g}")
            if i == len(runs) - 1:
                ax.set_xlabel("radius (km)")
            ax.set_xlim(0.0, tr.r_max)
            ax.set_ylim(0.0, 1.05)
        label = labels[i] + "\n" if labels else ""
        axes[i, 0].set_ylabel(label + "scaled density")
    for k, ax in enumerate(axes.flat):
        panel_label(ax, k)
    handles = axes[0, 0].get_legend_handles_labels()
    twin_handles = twin.get_legend_handles_labels()
    fig.legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1],
               frameon=False, loc="outside lower center", ncol=7)
    figure = Path(figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    print(f"wrote {figure} (years {', '.join(f'{t:g}' for t in years)})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True,
                        help="Working folder: bundle/, params/, run/.")
    parser.add_argument("--figure", default=str(REPO / "figures" /
                                                 "flat_transect.png"))
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--R0", type=float, default=5.0,
                        help="R0-tilde; on a uniform c = 1 domain this is "
                             "also the local R0 everywhere.")
    parser.add_argument("--c3", type=float, default=0.5)
    parser.add_argument("--radius-km", type=float, default=25.0)
    parser.add_argument("--h", type=float, default=250.0,
                        help="Mesh size in metres (default: %(default)s).")
    parser.add_argument("--land-cover-class", type=int, default=52,
                        help="NLCD class used everywhere (default: 52, shrub "
                             "or scrub, c = 1.0: the study region's dominant "
                             "class).")
    parser.add_argument("--permeability", type=float, default=None,
                        help="Uniform l overriding the class's own (class 52 "
                             "has l = 0.81); 1 makes D = kappa I.")
    parser.add_argument("--build-only", action="store_true",
                        help="Build the bundle and stop: the flat grid's "
                             "sweep uses it as its mesh folder.")
    parser.add_argument("--years", type=float, nargs="*", default=None,
                        help="Years to plot (default: three spread over the "
                             "span in which the front is interior).")
    parser.add_argument("--bin-km", type=float, default=0.25)
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot an existing run.")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    bundle, params, run = out / "bundle", out / "params", out / "run"
    if not args.plot_only:
        if not (bundle / "terrain.msh").is_file():
            build_bundle(bundle, args.radius_km, args.h, args.land_cover_class,
                         args.permeability)
        if args.build_only:
            return
        point = make_params(bundle, params, args.delta, args.R0, args.c3)
        if not (run / "optimization_history.json").is_file():
            print(f"optimizing {point.stem}; log: {run / 'solve.log'}")
            run_optimization(bundle, point, run)
    plot(run, Path(args.figure).expanduser(), args.years, args.bin_km)


if __name__ == "__main__":
    main()
