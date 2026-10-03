"""Locate a solver input bundle (mesh, nodal arrays, parameter file) and the
command-line flags that select it."""

from __future__ import annotations

from pathlib import Path

from shared_parameters import DEFAULT_PARAMETERS


def explain_nodal_array_mismatch(name, size, expected, domain, comm):
    """Error message for a bundle array that does not fit this function space.

    Bundle arrays are in serial DOLFINx node order, so the usual cause is a
    run under MPI.
    """
    if comm.size > 1 and size == domain.geometry.index_map().size_global:
        return (
            f"\n{name} has {size:,} values -- the whole mesh -- but this rank's "
            f"function space\nexpects {expected:,}, its share of a "
            f"{comm.size}-way partition.\n\n"
            "Mesh bundles are serial. The generator writes the arrays in the "
            "node order\nDOLFINx produces on a single rank, which is neither "
            "the .msh order nor any\nparallel partition's, so they cannot be "
            "read back under mpirun. The spin-up\ncaches beside them have the "
            "same limitation.\n\nRun this on one rank:\n\n"
            "    python CWD_optimal_control.py --mesh-folder ...\n"
        )
    return (
        f"\n{name} has {size:,} values, but this mesh and function space "
        f"expect {expected:,}.\nThe mesh and arrays are not a matching "
        "bundle -- regenerate the arrays against\nthis mesh with "
        "utilities/resample_land_cover_on_existing_mesh.py, or point at the "
        "bundle\nthe mesh belongs to.\n"
    )


def add_bundle_arguments(parser):
    """Attach the standard mesh-bundle flags to an argument parser."""
    parser.add_argument(
        "--mesh-folder",
        default=None,
        help=(
            "Folder containing one .msh file, land_cover_classes.npy, "
            "land_cover_diffusivity.npy, and effective_parameters.json."
        ),
    )
    parser.add_argument("--parameters", default=None,
                        help="Combined model parameter JSON; overrides bundle discovery.")
    parser.add_argument("--mesh", default=None,
                        help="Explicit mesh path; overrides --mesh-folder discovery.")
    parser.add_argument("--land-cover-classes", default=None,
                        help="Explicit class-array path; overrides --mesh-folder discovery.")
    parser.add_argument("--land-cover-diffusivity", default=None,
                        help="Explicit diffusivity-array path; overrides --mesh-folder discovery.")
    parser.add_argument(
        "--susceptible-equilibrium",
        default=None,
        help=(
            "Path to the cached disease-free susceptible field. Defaults to "
            "susceptible_equilibrium.npy beside the mesh; written there if it "
            "has to be computed."
        ),
    )
    parser.add_argument(
        "--recompute-equilibrium",
        action="store_true",
        help=(
            "Re-run the disease-free spin-up even if a matching cached field "
            "exists."
        ),
    )
    parser.add_argument(
        "--disease-state",
        default=None,
        help=(
            "Path to the cached spun-up epizootic used as the initial "
            "condition. Defaults to disease_initial_state.npy beside the "
            "mesh; written there if it has to be computed."
        ),
    )
    parser.add_argument(
        "--recompute-disease-state",
        action="store_true",
        help=(
            "Re-run the disease spin-up even if a matching cached state "
            "exists."
        ),
    )
    return parser


def require_file(path, description):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def resolve_input_files(parsed_args):
    """Resolve a mesh bundle, while allowing individual path overrides."""
    mesh_folder = None
    if parsed_args.mesh_folder is not None:
        mesh_folder = Path(parsed_args.mesh_folder).expanduser().resolve()
        if not mesh_folder.is_dir():
            raise NotADirectoryError(f"Mesh folder does not exist: {mesh_folder}")

    if parsed_args.mesh is not None:
        mesh_path = require_file(parsed_args.mesh, "mesh")
    elif mesh_folder is not None:
        mesh_candidates = sorted(mesh_folder.glob("*.msh"))
        if len(mesh_candidates) != 1:
            names = ", ".join(path.name for path in mesh_candidates) or "none"
            raise ValueError(
                f"Expected exactly one .msh file in {mesh_folder}; found: {names}. "
                "Use --mesh to select one explicitly."
            )
        mesh_path = mesh_candidates[0]
    else:
        mesh_path = require_file("terrain.msh", "mesh")

    classes_path = require_file(
        parsed_args.land_cover_classes
        or (mesh_folder / "land_cover_classes.npy" if mesh_folder else "land_cover_classes.npy"),
        "land-cover class array",
    )
    diffusivity_path = require_file(
        parsed_args.land_cover_diffusivity
        or (
            mesh_folder / "land_cover_diffusivity.npy"
            if mesh_folder
            else "land_cover_diffusivity.npy"
        ),
        "land-cover diffusivity array",
    )

    if parsed_args.parameters is not None:
        spatial_path = require_file(parsed_args.parameters, "model parameter file")
    elif mesh_folder is not None:
        effective_path = mesh_folder / "effective_parameters.json"
        fallback_path = mesh_folder / "parameters.json"
        if effective_path.is_file():
            spatial_path = effective_path
        elif fallback_path.is_file():
            spatial_path = fallback_path
        else:
            raise FileNotFoundError(
                "The mesh folder contains no effective_parameters.json "
                "or parameters.json."
            )
    else:
        spatial_path = require_file(DEFAULT_PARAMETERS, "parameter file")

    return mesh_path, classes_path, diffusivity_path, spatial_path
