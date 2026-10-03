"""Build the four habitat-ablation bundles for one mesh.

The arms cross slope anisotropy (on, or isotropy = 1) with land cover
(measured, or flattened to its area-weighted mean; see flatten_land_cover.py).
Each arm is its own bundle folder, since each has its own cached disease-free
equilibrium: the mesh, class array and offsets are symlinked, and the arm owns
its diffusivity array and effective_parameters.json.

    python utilities/ablation_bundles.py --mesh BUNDLE --out ablation --focus X Y

writes <out>/<region>_{aniso,iso}_{lc,nolc}/ and prints c-bar; the matching
best-habitat R0 is R0-tilde = R0 / c-bar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.io.gmsh import read_from_msh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_parameters import land_cover_to_carrying_capacity  # noqa: E402
from flatten_land_cover import area_weighted_mean, cell_areas_and_map  # noqa: E402

SHARED = ("terrain.msh", "land_cover_classes.npy", "coord_offsets.npy")


def link(target, name):
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, help="Source mesh bundle.")
    parser.add_argument("--out", required=True, help="Parent of the arm folders.")
    parser.add_argument("--focus", type=float, nargs=2, metavar=("X", "Y"),
                        default=None,
                        help="Move the index focus to these mesh coordinates "
                             "(m, offsets from the centroid); default keeps "
                             "the bundle's.")
    args = parser.parse_args()

    src = Path(args.mesh).resolve()
    region = src.name.removesuffix("_mesh")
    base = json.loads((src / "effective_parameters.json").read_text())
    if args.focus:
        base["initial_conditions"]["infected_gaussians"][0]["center"] = args.focus

    domain = read_from_msh(str(src / "terrain.msh"), MPI.COMM_SELF, gdim=3).mesh
    areas, cells = cell_areas_and_map(domain)
    classes = np.load(src / "land_cover_classes.npy").astype(np.int32)
    ell = np.load(src / "land_cover_diffusivity.npy")
    K0 = float(base["land_cover"]["carrying_capacity_base"])
    c_bar = area_weighted_mean(
        land_cover_to_carrying_capacity(classes, base) / K0, cells, areas)
    l_bar = area_weighted_mean(ell, cells, areas)

    for slope in ("aniso", "iso"):
        for cover in ("lc", "nolc"):
            arm = Path(args.out) / f"{region}_{slope}_{cover}"
            arm.mkdir(parents=True, exist_ok=True)
            for name in SHARED:
                if (src / name).exists():
                    link(src / name, arm / name)
            params = json.loads(json.dumps(base))
            if slope == "iso":
                params["diffusion_tensor"]["isotropy"] = 1.0
            if cover == "lc":
                link(src / "land_cover_diffusivity.npy",
                     arm / "land_cover_diffusivity.npy")
            else:
                np.save(arm / "land_cover_diffusivity.npy",
                        np.full_like(ell, l_bar))
                table = params["land_cover"]
                table["carrying_capacity_fractions"] = {
                    code: c_bar for code in table["carrying_capacity_fractions"]}
                table["carrying_capacity_fraction_default"] = c_bar
            (arm / "effective_parameters.json").write_text(
                json.dumps(params, indent=2), encoding="utf-8")
            print(f"wrote {arm}")

    focus = base["initial_conditions"]["infected_gaussians"][0]["center"]
    print(f"\n{region}: c-bar = {c_bar:.4f}, l-bar = {l_bar:.4f}, "
          f"focus = {focus}")


if __name__ == "__main__":
    main()
