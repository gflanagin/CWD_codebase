#!/usr/bin/env python
"""Mass-weighted area statistics for a saved ``optimal_control.npy``.

A saved control is in solver DOF order, which is not ``terrain.msh`` node
order, so the lumped mass is assembled through dolfinx exactly as
``CWDControlProblem`` does. Reports the space-time mean, the space-time
fraction at the lower bound, and the never-treated area (at the lower bound in
every block). The first two are checked against optimization_history.json.

    python utilities/control_area_stats.py --mesh-folder BUNDLE RUN_DIR...
    python utilities/control_area_stats.py --mesh-folder BUNDLE \\
        --csv areas.csv cluster_runs/flat_grid/delta_*__c4_*
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

try:
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector
    from dolfinx.io.gmsh import read_from_msh
    from petsc4py import PETSc
    import ufl
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        f"needs dolfinx ({error}). Activate the solver environment first."
    )

SQUARE_METRES_PER_SQUARE_KM = 1.0e6


def lumped_mass_for(mesh_path, comm):
    """Assemble int(phi_i) on the scalar CG1 space, in solver dof order.

    Mirrors CWDControlProblem: same reader, same gdim, same element family and
    degree, so the dof numbering is the one a saved control is indexed by.
    """
    mesh_data = read_from_msh(str(mesh_path), comm, gdim=3)
    domain = mesh_data.mesh
    V = fem.functionspace(domain, ("CG", 1))
    dq = ufl.TestFunction(V)
    vector = assemble_vector(fem.form(1.0 * dq * ufl.dx))
    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    mass = vector.array_r[:n_owned].copy()
    vector.destroy()
    return mass


def statistics(control, mass, lower=0.0, tolerance=1e-12):
    """Space-time and union statistics for one control, all mass weighted."""
    n = mass.size
    slab = control[:, :n]
    blocks = slab.shape[0]
    total_area = float(mass.sum())

    weights = mass / total_area
    at_zero = slab <= lower + tolerance

    # Equal to the solver's space-time statistic when all blocks have equal length.
    space_time_mean = float((slab * weights[None, :]).sum() / blocks)
    space_time_zero = float((at_zero * weights[None, :]).sum() / blocks)

    never = np.all(at_zero, axis=0)
    never_area = float(mass[never].sum())

    return {
        "blocks": blocks,
        "nodes": n,
        "domain_area_km2": total_area / SQUARE_METRES_PER_SQUARE_KM,
        "space_time_mean": space_time_mean,
        "space_time_zero_fraction": space_time_zero,
        "never_treated_area_fraction": never_area / total_area,
        "never_treated_area_km2": never_area / SQUARE_METRES_PER_SQUARE_KM,
        "never_treated_nodes": int(never.sum()),
    }


def recorded(run_dir):
    """The solver's own numbers, for the ordering checksum."""
    path = Path(run_dir) / "optimization_history.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        targeted = json.load(stream).get("controls", {}).get("targeted", {})
    return {
        "space_time_mean": targeted.get("space_time_mean"),
        "space_time_zero_fraction": targeted.get("fraction_at_lower_bound"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run folders.")
    parser.add_argument("--mesh-folder", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--tolerance", type=float, default=2e-3,
                        help="Relative tolerance for the ordering checksum.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise SystemExit("run serially; a saved control is a serial artifact.")

    meshes = sorted(Path(args.mesh_folder).glob("*.msh"))
    if len(meshes) != 1:
        raise SystemExit(
            f"expected exactly one .msh in {args.mesh_folder}, found {len(meshes)}"
        )
    mass = lumped_mass_for(meshes[0], comm)
    print(f"lumped mass from {meshes[0].name}: {mass.size} dofs, "
          f"{mass.sum() / SQUARE_METRES_PER_SQUARE_KM:.4f} km^2 total")
    print(f"node area span: {mass.max() / mass.min():.3g}\n")

    header = (f"{'run':42s} {'mean v':>8s} {'z_st':>8s} {'never':>8s} "
              f"{'never km2':>10s} {'check':>6s}")
    print(header)
    print("-" * len(header))

    rows, failures = [], []
    for run in args.runs:
        path = Path(run) / "optimal_control.npy"
        if not path.exists():
            print(f"{Path(run).name:42s} no optimal_control.npy")
            continue
        control = np.load(path)
        if control.ndim != 2:
            raise SystemExit(f"{path}: expected (blocks, nodes)")
        if control.shape[1] != mass.size:
            raise SystemExit(
                f"{path}: {control.shape[1]} nodes against the mesh's "
                f"{mass.size} -- wrong mesh folder for this run."
            )
        stats = statistics(control, mass)

        # Check against the solver's recorded statistics.
        mark, ref = "n/a", recorded(run)
        if ref and ref["space_time_mean"]:
            errors = [
                abs(stats[key] - ref[key]) / max(abs(ref[key]), 1e-30)
                for key in ("space_time_mean", "space_time_zero_fraction")
                if ref.get(key) is not None
            ]
            worst = max(errors) if errors else 0.0
            mark = "ok" if worst <= args.tolerance else f"OFF {worst:.1%}"
            if worst > args.tolerance:
                failures.append((Path(run).name, worst))

        stats["run"] = Path(run).name
        rows.append(stats)
        print(f"{stats['run']:42s} {stats['space_time_mean']:8.4f} "
              f"{100 * stats['space_time_zero_fraction']:7.2f}% "
              f"{100 * stats['never_treated_area_fraction']:7.2f}% "
              f"{stats['never_treated_area_km2']:10.1f} {mark:>6s}")

    if args.csv and rows:
        import csv
        fields = ["run", "blocks", "nodes", "domain_area_km2",
                  "space_time_mean", "space_time_zero_fraction",
                  "never_treated_area_fraction", "never_treated_area_km2",
                  "never_treated_nodes"]
        with open(args.csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
        print(f"\nwrote {args.csv}")

    if failures:
        print(f"\n{len(failures)} run(s) failed the ordering checksum:")
        for name, worst in failures:
            print(f"  {name}: worst relative error {worst:.2%}")
        print("The union areas above are NOT trustworthy for those runs.")
        return 1
    print("\nAll recomputed statistics match the solver's own to within "
          f"{args.tolerance:.1%}; the dof ordering is confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
