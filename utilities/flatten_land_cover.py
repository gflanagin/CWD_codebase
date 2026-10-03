#!/usr/bin/env python
"""Area-weighted means of the two land-cover fields, and a flattened diffusivity.

Flattening replaces a field by its area-weighted mean, which removes the
spatial structure without changing the landscape mean. c(x) is flattened by
setting every land_cover.carrying_capacity_fractions entry to c-bar in the
run's parameter file; p(x) is read from land_cover_diffusivity.npy, so a flat
array is written instead. Weights are cell areas, with nodal values averaged
over each cell's vertices.

    python utilities/flatten_land_cover.py --mesh-folder BUNDLE [--write]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

try:
    from dolfinx.io.gmsh import read_from_msh
except ImportError:  # pragma: no cover - environment guard
    sys.exit("dolfinx is required; run inside the FEniCS environment.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_parameters import land_cover_to_carrying_capacity  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Area-weighted land-cover means for a mesh bundle.")
    parser.add_argument("--mesh-folder", required=True)
    parser.add_argument(
        "--parameters", default="parameters.json",
        help="Supplies land_cover.carrying_capacity_fractions. Defaults to the "
             "repository file, not the bundle snapshot.")
    parser.add_argument(
        "--write", action="store_true",
        help="Write land_cover_diffusivity_flat.npy beside the bundle.")
    return parser.parse_args()


def cell_areas_and_map(domain):
    coords = domain.geometry.x
    geom = domain.geometry
    for accessor in (lambda: geom.dofmaps(0), lambda: geom.dofmaps[0],
                     lambda: geom.dofmap):
        try:
            dofmap = accessor()
        except (AttributeError, TypeError):
            continue
        break
    else:  # pragma: no cover
        raise AttributeError("no usable geometry dofmap on this dolfinx build")
    cells = np.asarray(dofmap).reshape(-1, 3)
    p0, p1, p2 = coords[cells[:, 0]], coords[cells[:, 1]], coords[cells[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    return areas, cells


def area_weighted_mean(nodal, cells, areas):
    """Landscape mean of a nodal field: cell means, weighted by cell area."""
    return float(np.average(nodal[cells].mean(axis=1), weights=areas))


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        sys.exit("Run this diagnostic on a single rank.")

    folder = Path(args.mesh_folder)
    meshes = sorted(folder.glob("*.msh"))
    if len(meshes) != 1:
        sys.exit(f"Expected exactly one .msh in {folder}, found {len(meshes)}.")

    params = json.loads(Path(args.parameters).read_text())
    mesh_data = read_from_msh(str(meshes[0]), comm, gdim=3)
    domain = getattr(mesh_data, "mesh", mesh_data)
    areas, cells = cell_areas_and_map(domain)

    classes = np.load(folder / "land_cover_classes.npy").astype(np.int32)
    lcd = np.load(folder / "land_cover_diffusivity.npy")
    for name, arr in (("land_cover_classes.npy", classes),
                      ("land_cover_diffusivity.npy", lcd)):
        if arr.size != domain.geometry.x.shape[0]:
            sys.exit(f"{name} has {arr.size} entries against "
                     f"{domain.geometry.x.shape[0]} mesh nodes; not a bundle.")

    K = land_cover_to_carrying_capacity(classes, params)
    K0 = float(params["land_cover"]["carrying_capacity_base"])
    c_nodal = K / K0

    c_bar = area_weighted_mean(c_nodal, cells, areas)
    l_bar = area_weighted_mean(lcd, cells, areas)

    print(f"mesh            {meshes[0]}")
    print(f"surface area    {areas.sum() / 1e6:,.1f} km^2")
    print(f"nodes           {classes.size:,}")
    print()
    print(f"c-bar  (area-weighted mean carrying-capacity fraction) {c_bar:.4f}")
    print(f"l-bar  (area-weighted mean land-cover diffusivity)     {l_bar:.4f}")
    print()
    print("For the no-land-cover arm on this mesh:")
    print(f"  set every carrying_capacity_fractions entry and "
          f"carrying_capacity_fraction_default to {c_bar:.4f}")
    print(f"  pass --land-cover-diffusivity <flat array at {l_bar:.4f}>")

    if args.write:
        out = folder / "land_cover_diffusivity_flat.npy"
        np.save(out, np.full_like(lcd, l_bar))
        print(f"\nwrote {out}  (constant {l_bar:.6f}, {lcd.size:,} nodes)")


if __name__ == "__main__":
    main()
