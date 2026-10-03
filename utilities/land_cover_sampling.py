"""Sample a categorical land-cover raster at mesh nodes, with coverage checks.

The query points are warped into the raster's CRS, so the tile may be in any
CRS at any resolution, and each node gets one nearest-neighbour lookup.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_points, transform_bounds

# NLCD class 0 (nodata) is a mapped class in parameters.json, so a tile that
# misses the mesh would otherwise pass silently.
COVERAGE_ERROR_FRACTION = 0.995     # below this, refuse without allow_partial
UNKNOWN_CLASS_WARN_FRACTION = 0.01  # above this, say so
UNKNOWN_CLASS_ERROR_FRACTION = 0.50 # above this, refuse without allow_partial

# Codes absent from parameters.json (usually an NLCD NoData fill such as 250)
# take the default values.
UNMAPPED_CLASS_ERROR_FRACTION = 0.001


def read_crs(raster_path):
    """CRS of a raster, as a rasterio CRS object."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise SystemExit(f"{raster_path} has no CRS.")
        return src.crs


def sample_land_cover_at_nodes(land_cover_path, mesh_verts, x_offset, y_offset,
                               mesh_crs):
    """Nearest-neighbour land-cover class at each mesh vertex.

    ``mesh_verts`` are in ``mesh_crs`` -- the DEM's CRS -- de-centred by the
    offsets recorded when the mesh was built. Nodes falling outside the raster
    are returned as class 0, matching NLCD's own nodata convention, so that
    :func:`report_node_class_coverage` can see them.
    """
    xs = np.asarray(mesh_verts[:, 0], dtype=np.float64) + x_offset
    ys = np.asarray(mesh_verts[:, 1], dtype=np.float64) + y_offset

    with rasterio.open(land_cover_path) as src:
        if src.crs is None:
            raise SystemExit(
                f"{land_cover_path} has no CRS; cannot sample it at mesh nodes."
            )
        if src.crs != mesh_crs:
            # Warp the query points into the raster's CRS.
            xs, ys = warp_points(mesh_crs, src.crs, xs.tolist(), ys.tolist())
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
        data = src.read(1)
        transform = src.transform
        declared_nodata = src.nodata

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) & (rows < data.shape[0])
        & (cols >= 0) & (cols < data.shape[1])
    )
    land_cover = np.zeros(rows.shape, dtype=np.int32)
    land_cover[inside] = data[rows[inside], cols[inside]]

    outside = int((~inside).sum())
    if outside:
        print(f"  {outside} of {land_cover.size} mesh nodes fall outside "
              f"{land_cover_path}; recorded as class 0.")
    if declared_nodata is not None:
        hits = int((land_cover == int(declared_nodata)).sum())
        if hits:
            print(f"  {hits} of {land_cover.size} mesh nodes sample the "
                  f"raster's declared NoData value ({int(declared_nodata)}). "
                  "The tile\n  reaches the mesh but its valid data does not; "
                  "this is what a clipped or\n  reprojected tile looks like "
                  "at the corners.")
    return land_cover


def mesh_cell_vertices(domain):
    """Triangle connectivity into ``domain.geometry.x``, as (n_cells, 3).

    Accepts either form of ``domain.geometry.dofmap`` (2-D array or
    AdjacencyList) without importing dolfinx. Raises for non-triangular meshes.
    """
    dofmap = domain.geometry.dofmap
    if hasattr(dofmap, "array") and hasattr(dofmap, "offsets"):
        offsets = np.asarray(dofmap.offsets)
        widths = np.unique(np.diff(offsets))
        if widths.size != 1:
            raise ValueError(
                f"mixed cell types in geometry dofmap (widths {widths}); "
                "only all-triangle surface meshes are supported."
            )
        cells = np.asarray(dofmap.array).reshape(-1, int(widths[0]))
    else:
        cells = np.asarray(dofmap)
        if cells.ndim == 1:
            raise ValueError(
                "geometry dofmap is flat with no offsets; cannot infer the "
                "cell size."
            )
    if cells.shape[1] < 3:
        raise ValueError(
            f"cells have {cells.shape[1]} vertices; expected triangles."
        )
    # First three columns are the vertices for both P1 and higher-order
    # geometries; the rest, if any, are edge/interior nodes.
    return cells[:, :3]


def lumped_node_areas(points, cells):
    """Surface area attributed to each node: 1/3 of each incident triangle.

    ``points`` is (n_nodes, 3) and ``cells`` (n_cells, 3). Areas are on the
    embedded surface, so the result sums to the mesh's surface area.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"points must be (n, 2 or 3); got {points.shape}")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"cells must be (n, 3) triangles; got {cells.shape}")
    if points.shape[1] == 2:                      # planar mesh: pad z = 0
        points = np.column_stack([points, np.zeros(len(points))])

    edge1 = points[cells[:, 1]] - points[cells[:, 0]]
    edge2 = points[cells[:, 2]] - points[cells[:, 0]]
    triangle_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    areas = np.zeros(len(points), dtype=np.float64)
    for corner in range(3):
        np.add.at(areas, cells[:, corner], triangle_area / 3.0)
    return areas


def report_node_class_coverage(land_cover, allow_partial=False,
                               class_names=None, parameters=None,
                               node_areas=None):
    """Audit the sampled classes for the tile-missed-the-mesh failure.

    Prints the class histogram over the mesh nodes and the mean carrying
    capacity and permeability, area-weighted when ``node_areas`` (from
    :func:`lumped_node_areas`) is given. Pass it for any graded mesh.
    """
    if node_areas is not None:
        node_areas = np.asarray(node_areas, dtype=np.float64)
        if node_areas.shape != land_cover.shape:
            raise ValueError(
                f"node_areas has shape {node_areas.shape} but land_cover has "
                f"{land_cover.shape}; they must be one value per node."
            )
        if not np.all(np.isfinite(node_areas)) or np.any(node_areas < 0.0):
            raise ValueError("node_areas must be finite and non-negative.")
        if node_areas.sum() <= 0.0:
            raise ValueError("node_areas sum to zero; the mesh has no area.")
    values, counts = np.unique(land_cover, return_counts=True)
    unknown = int(counts[values == 0].sum()) if (values == 0).any() else 0
    fraction = unknown / land_cover.size if land_cover.size else 0.0

    names = {int(k): v for k, v in (class_names or {}).items()}
    cap = dif = None
    mapped = None
    if parameters is not None:
        lc = parameters["land_cover"]
        cap = (lc["carrying_capacity_fractions"],
               float(lc["carrying_capacity_fraction_default"]))
        dif = (lc["diffusivity"], float(lc["diffusivity_default"]))
        # A code is mapped only if both tables carry it.
        mapped = set(int(k) for k in cap[0]) & set(int(k) for k in dif[0])

    print(f"Land-cover classes at {land_cover.size:,} mesh nodes:")
    head = f"  {'code':>4} {'class':30}{'nodes':>10}{'share':>8}"
    if cap: head += f"{'c':>6}{'ell':>6}"
    print(head)
    order = sorted(zip(values.tolist(), counts.tolist()), key=lambda t: -t[1])
    for code, count in order:
        unmapped_here = mapped is not None and code not in mapped
        label = names.get(code, "?")
        row = (f"  {code:>4} {label[:30]:30}{count:10,}"
               f"{count / land_cover.size * 100:7.2f}%")
        if cap:
            row += (f"{float(cap[0].get(str(code), cap[1])):6.2f}"
                    f"{float(dif[0].get(str(code), dif[1])):6.2f}")
        if unmapped_here:
            row += "   <- UNMAPPED, taking defaults"
        print(row)
    if cap:
        capacity = np.array([float(cap[0].get(str(int(v)), cap[1]))
                             for v in land_cover])
        permeability = np.array([float(dif[0].get(str(int(v)), dif[1]))
                                 for v in land_cover])
        if node_areas is None:
            weighting = "per-node (UNWEIGHTED -- pass node_areas for a mesh)"
            mc = capacity.mean()
            ml = permeability.mean()
        else:
            weighting = "area-weighted"
            total_area = node_areas.sum()
            mc = float(np.dot(capacity, node_areas) / total_area)
            ml = float(np.dot(permeability, node_areas) / total_area)
            print(f"  mesh surface area = {total_area / 1e6:,.1f} km^2")
            print(f"  (unweighted per-node mean c would be "
                  f"{capacity.mean():.3f})")
        print(f"  {weighting} mean c = {mc:.3f}   mean ell = {ml:.3f}")
        print(f"  -> landscape-mean R0 is {mc:.3f} x the best-habitat R0; "
              f"calibrate R0_nom = R0_obs / {mc:.3f}")
    if fraction:
        print(f"  Class 0 (unknown/nodata): {fraction * 100:.2f}% of nodes")

    if mapped is not None:
        stray = [(int(v), int(c)) for v, c in order if int(v) not in mapped]
        stray_fraction = sum(c for _v, c in stray) / land_cover.size
        if stray:
            codes = ", ".join(str(v) for v, _c in stray)
            print(f"  Unmapped classes ({codes}): "
                  f"{stray_fraction * 100:.2f}% of nodes")
        if stray_fraction >= UNMAPPED_CLASS_ERROR_FRACTION and not allow_partial:
            raise SystemExit(
                f"\nLand-cover class(es) {codes} cover "
                f"{stray_fraction * 100:.2f}% of the mesh and have no entry in "
                "parameters.json.\n\nThose nodes silently take the mid-range "
                f"defaults -- c = {cap[1]}, ell = {dif[1]} -- so the mesh "
                "builds, the model runs,\nand that fraction of the landscape "
                "carries no land-cover structure at all.\n\n"
                "Codes above 95 are not NLCD land-cover classes. The usual "
                "cause is a NoData\nfill: NLCD GeoTIFFs from MRLC use 250, "
                "and it appears wherever the tile was\nclipped or reprojected "
                "and its valid data does not reach the mesh. Re-download\n"
                "covering a larger area, preferably in the raster's native "
                "CRS so that no\nreprojection wedges are introduced, then run "
                "utilities/resample_land_cover_on_existing_mesh.py -- the mesh "
                "itself is fine and\ndoes not need rebuilding.\n\n"
                "If the code is real and simply missing, add it to "
                "land_cover.diffusivity and\nland_cover.carrying_capacity_"
                "fractions.\n\nPass --allow-partial to accept the defaults "
                "and proceed."
            )

    if fraction >= UNKNOWN_CLASS_ERROR_FRACTION and not allow_partial:
        raise SystemExit(
            f"\n{fraction * 100:.1f}% of mesh nodes received class 0 "
            "(unknown/nodata).\n\n"
            + ("No land-cover classes were transferred at all. "
               if not [v for v in values if v != 0] else
               "Most of the mesh received no land-cover data. ")
            + "The land-cover raster's\nfootprint may reach the DEM while its "
            "valid data does not -- check for a large\nnodata margin, or a "
            "tile clipped to a neighbouring area.\n\n"
            "Pass --allow-partial to proceed anyway."
        )
    if fraction >= UNKNOWN_CLASS_WARN_FRACTION:
        print("  WARNING: class 0 is mapped in parameters.json, so these "
              "nodes will\n  silently become ordinary habitat in the model.")
    return fraction


def check_coverage(land_cover_path, dem_path, allow_partial=False):
    """Verify the land-cover raster actually reaches the DEM.

    The land cover must contain the DEM extent; it may be larger, in another
    CRS, at another resolution.
    """
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise SystemExit(f"{dem_path} has no CRS; cannot check coverage.")
        dem_box = transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)
    with rasterio.open(land_cover_path) as lc:
        if lc.crs is None:
            raise SystemExit(f"{land_cover_path} has no CRS; cannot check coverage.")
        lc_box = transform_bounds(lc.crs, "EPSG:4326", *lc.bounds)

    overlap_x = max(0.0, min(dem_box[2], lc_box[2]) - max(dem_box[0], lc_box[0]))
    overlap_y = max(0.0, min(dem_box[3], lc_box[3]) - max(dem_box[1], lc_box[1]))
    dem_area = (dem_box[2] - dem_box[0]) * (dem_box[3] - dem_box[1])
    covered = (overlap_x * overlap_y) / dem_area if dem_area > 0 else 0.0

    def box(b):
        return f"{b[0]:10.4f} {b[1]:9.4f} {b[2]:10.4f} {b[3]:9.4f}"

    print("Footprints in lon/lat (W S E N):")
    print(f"  DEM        {box(dem_box)}")
    print(f"  Land cover {box(lc_box)}")
    print(f"  DEM covered by land cover: {covered * 100:.2f}%")

    if covered >= COVERAGE_ERROR_FRACTION:
        if covered < 1.0:
            print("  NOTE: coverage is not quite complete; expect a thin "
                  "border of class 0 along one edge.")
        return covered

    margin_x = 0.05 * (dem_box[2] - dem_box[0])
    margin_y = 0.05 * (dem_box[3] - dem_box[1])
    wanted = (dem_box[0] - margin_x, dem_box[1] - margin_y,
              dem_box[2] + margin_x, dem_box[3] + margin_y)
    problem = ("does not overlap the DEM at all" if covered == 0
               else f"covers only {covered * 100:.1f}% of the DEM")
    message = (
        f"\n{land_cover_path} {problem}.\n\n"
        f"  DEM        {box(dem_box)}\n"
        f"  Land cover {box(lc_box)}\n\n"
        "Mesh nodes outside the land-cover raster become class 0 (nodata), "
        "and class 0\nis a mapped class in parameters.json -- so the mesh "
        "would build and the model\nwould run, on a landscape with no "
        "land-cover structure at all.\n\n"
        "Download a land-cover tile from https://www.mrlc.gov/viewer/ covering "
        "at least:\n"
        f"    {wanted[0]:.4f} W to {wanted[2]:.4f} W, "
        f"{wanted[1]:.4f} N to {wanted[3]:.4f} N\n\n"
        "It may be much larger than that, and in any CRS or resolution; it is "
        "sampled\ndirectly at the mesh nodes and never resampled onto the DEM "
        "grid. It just has\nto contain the DEM.\n\n"
        "Pass --allow-partial to proceed anyway and accept class-0 regions."
    )
    if allow_partial:
        print(message.replace("\n", "\n  "))
        print("\n  --allow-partial given; continuing.")
        return covered
    raise SystemExit(message)
