from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
import numpy as np
import argparse
from pathlib import Path
import sys

UTILITIES_DIR = Path(__file__).resolve().parent
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (  # noqa: E402
    DEFAULT_PARAMETERS,
    land_cover_to_diffusivity,
    load_parameters,
)
from land_cover_sampling import (  # noqa: E402
    lumped_node_areas,
    mesh_cell_vertices,
    read_crs,
    report_node_class_coverage,
    sample_land_cover_at_nodes,
)

def build_parser():
    parser = argparse.ArgumentParser(
        description="Regenerate nodal land-cover arrays for an existing mesh."
    )
    parser.add_argument("--mesh", default="terrain.msh")
    parser.add_argument("--land-cover", default="land_cover.tif",
                        help="Categorical land-cover raster, in any CRS.")
    parser.add_argument("--dem", default="dem.tif",
                        help="DEM the mesh was built from. Only its CRS is "
                             "read: mesh coordinates are in it, and the "
                             "sampling points are warped from it into the "
                             "land-cover raster's CRS.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Proceed even if most nodes receive class 0.")
    parser.add_argument("--coord-offsets", default="coord_offsets.npy")
    parser.add_argument("--classes-output", default="land_cover_classes.npy")
    parser.add_argument("--diffusivity-output", default="land_cover_diffusivity.npy")
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS))
    return parser


def main():
    args = build_parser().parse_args()
    spatial_parameters = load_parameters(args.parameters)

    mesh_data = read_from_msh(args.mesh, MPI.COMM_WORLD, gdim=3)
    domain = mesh_data.mesh
    coords = domain.geometry.x
    x_offset, y_offset = np.load(args.coord_offsets)

    land_cover = sample_land_cover_at_nodes(
        args.land_cover, coords, x_offset, y_offset, read_crs(args.dem)
    )
    report_node_class_coverage(
        land_cover, args.allow_partial,
        class_names=spatial_parameters['land_cover'].get('class_names'),
        parameters=spatial_parameters,
        node_areas=lumped_node_areas(coords, mesh_cell_vertices(domain)),
    )
    diffusivity = land_cover_to_diffusivity(land_cover, spatial_parameters)

    np.save(args.diffusivity_output, diffusivity)
    np.save(args.classes_output, land_cover.astype(np.int32))

    water_value = float(spatial_parameters["land_cover"]["diffusivity"].get("11", 0.0))
    print(f"Saved! min={diffusivity.min():.4g}, max={diffusivity.max():.4g}")
    print(f"Water nodes (barrier): {(diffusivity == water_value).sum()}")


if __name__ == "__main__":
    main()
