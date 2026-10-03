"""Write an XDMF/HDF5 copy of a mesh bundle, for inspection in ParaView.

The solver does not read this file. Fields written on the mesh:

    Land_Cover_Class        the sampled NLCD code
    Permeability_ell        p(x), the land-cover factor multiplying kappa
    Carrying_Capacity       K(x) = K0 c(x), in deer/km^2
    Elevation               z, in metres, mean-centred as the mesh is
    Susceptible_Equilibrium S_infinity, if cached beside the mesh

A ParaView colormap, <output>_colormap.json, covering the classes present is
written alongside. Inputs that are absent are skipped.

    python utilities/export_mesh_to_xdmf.py --mesh-folder region_mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import read_from_msh

UTILITIES_DIR = Path(__file__).resolve().parent
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (  # noqa: E402
    land_cover_to_carrying_capacity,
    load_parameters,
)
from lc_colorbar import save_paraview_colormap  # noqa: E402
from susceptible_spinup import EQUILIBRIUM_ARRAY_NAME  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-folder", default=None,
                        help="Bundle folder holding one .msh, the land-cover "
                             "arrays and effective_parameters.json. Anything "
                             "missing is skipped.")
    parser.add_argument("--mesh", default=None,
                        help="Explicit mesh path (default: terrain.msh, or the "
                             "single .msh in --mesh-folder).")
    parser.add_argument("--land-cover-classes", default=None,
                        help="Explicit class-array path.")
    parser.add_argument("--land-cover-diffusivity", default=None,
                        help="Explicit permeability-array path.")
    parser.add_argument("--susceptible-equilibrium", default=None,
                        help="Explicit cached S_infinity path.")
    parser.add_argument("--parameters", default=None,
                        help="Parameter file, needed for the habitat column.")
    parser.add_argument("--output", default=None,
                        help="XDMF file to write (default: terrain.xdmf, or "
                             "bundle.xdmf inside --mesh-folder).")
    return parser


def resolve(args):
    """Locate whatever the bundle actually contains; absent is not an error."""
    folder = None
    if args.mesh_folder is not None:
        folder = Path(args.mesh_folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Mesh folder does not exist: {folder}")

    if args.mesh is not None:
        mesh_path = Path(args.mesh).expanduser().resolve()
    elif folder is not None:
        candidates = sorted(folder.glob("*.msh"))
        if len(candidates) != 1:
            names = ", ".join(p.name for p in candidates) or "none"
            raise ValueError(
                f"Expected exactly one .msh in {folder}; found: {names}. "
                "Use --mesh to select one."
            )
        mesh_path = candidates[0]
    else:
        mesh_path = Path("terrain.msh").resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    # Other inputs are looked for beside the mesh.
    search = folder if folder is not None else mesh_path.parent

    def optional(explicit, name):
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Not found: {path}")
            return path
        path = search / name
        return path if path.is_file() else None

    parameters_path = args.parameters
    if parameters_path is None:
        for name in ("effective_parameters.json", "parameters.json"):
            if (search / name).is_file():
                parameters_path = search / name
                break

    output = args.output
    if output is None:
        output = ((folder / "bundle.xdmf") if folder
                  else mesh_path.with_suffix(".xdmf"))

    return (
        mesh_path,
        optional(args.land_cover_classes, "land_cover_classes.npy"),
        optional(args.land_cover_diffusivity, "land_cover_diffusivity.npy"),
        optional(args.susceptible_equilibrium, EQUILIBRIUM_ARRAY_NAME),
        Path(parameters_path).expanduser().resolve() if parameters_path else None,
        Path(output).expanduser().resolve(),
    )


def main():
    args = build_parser().parse_args()
    (mesh_path, classes_path, diffusivity_path, equilibrium_path,
     parameters_path, output_path) = resolve(args)

    comm = MPI.COMM_WORLD
    root = comm.rank == 0
    domain = read_from_msh(str(mesh_path), comm, gdim=3).mesh
    scalar_space = fem.functionspace(domain, ("CG", 1))
    expected = scalar_space.dofmap.index_map.size_local \
        + scalar_space.dofmap.index_map.num_ghosts

    if root:
        print(f"Mesh: {mesh_path}  ({expected:,} local CG1 nodes)")

    written = []
    skipped = []

    def field(name, values):
        """Attach one nodal array to the mesh under a given name."""
        values = np.asarray(values, dtype=np.float64)
        if values.size != expected:
            skipped.append(
                f"{name}: array has {values.size:,} values, mesh expects "
                f"{expected:,} -- not a matching bundle"
            )
            return
        function = fem.Function(scalar_space, name=name)
        function.x.array[:] = values
        function.x.scatter_forward()
        written.append((name, function))

    # Geometry-node order, the same order the land-cover arrays were sampled in.
    field("Elevation", domain.geometry.x[:, 2])

    classes = None
    parameters = None
    if parameters_path is not None:
        parameters = load_parameters(str(parameters_path))

    if classes_path is not None:
        classes = np.load(classes_path).astype(np.int32)
        field("Land_Cover_Class", classes)
    else:
        skipped.append("Land_Cover_Class: no land_cover_classes.npy")

    if diffusivity_path is not None:
        field("Permeability_ell", np.load(diffusivity_path))
    else:
        skipped.append("Permeability_ell: no land_cover_diffusivity.npy")

    if classes is not None and parameters is not None:
        field("Carrying_Capacity",
              land_cover_to_carrying_capacity(classes, parameters))
    else:
        skipped.append("Carrying_Capacity: needs the class array and a "
                       "parameter file")

    if equilibrium_path is not None:
        field("Susceptible_Equilibrium", np.load(equilibrium_path))
    else:
        skipped.append(f"Susceptible_Equilibrium: no {EQUILIBRIUM_ARRAY_NAME} "
                       "cached yet")

    with XDMFFile(comm, str(output_path), "w") as handle:
        handle.write_mesh(domain)
        for _name, function in written:
            handle.write_function(function, 0.0)

    # Colormap for the classes present, ordered by node share.
    colormap_path = None
    colormap_name = None
    if classes is not None and root:
        values, counts = np.unique(classes, return_counts=True)
        colormap_path = output_path.with_name(output_path.stem + "_colormap.json")
        # Preset named for the bundle so repeated imports stay distinct.
        colormap_name = f"NLCD {mesh_path.parent.name}"
        save_paraview_colormap(
            colormap_path,
            codes=[int(v) for v in values[np.argsort(-counts)]],
            class_names=(parameters or {}).get("land_cover", {}).get("class_names"),
            name=colormap_name,
            verbose=False,
        )

    if root:
        print(f"Wrote {output_path}")
        for name, function in written:
            data = function.x.array
            print(f"  {name:24s} min {data.min():10.4g}  "
                  f"max {data.max():10.4g}  mean {data.mean():10.4g}")
        for note in skipped:
            print(f"  skipped -- {note}")
        if colormap_path is not None:
            print(f"Wrote {colormap_path}")
            print(f"  In ParaView: colour by Land_Cover_Class, then in the "
                  f"Color Map Editor use\n  'Import a colour map' on that JSON "
                  f"and pick '{colormap_name}'. Any class with\n  no palette "
                  f"entry -- NLCD GeoTIFF NoData is 250 -- is drawn magenta.")
        print("\nEach field is a separate named block sharing this mesh, so in "
              "ParaView open\nthe file and switch arrays to compare them.")


if __name__ == "__main__":
    main()
