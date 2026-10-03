"""Reproject (and optionally crop) a DEM from geographic degrees to projected metres.

    python utilities/reproject_dem.py --dem dem.tif --output region_utm.tif \\
        --bounds -107.70 38.43 -107.22 38.91

With no --target-crs the UTM zone of the cropped region's centroid is used;
with no --bounds the whole raster is reprojected. Resampling is bilinear and
the output resolution defaults to 30 m. Refuses an already-projected raster
unless --force is given.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    Resampling, calculate_default_transform, reproject, transform_bounds,
)


# Thresholds on the triangulation size handed to mmg, after mesh.downsample.
MESH_VERTICES_WARN = 200_000
MESH_VERTICES_REFUSE = 1_000_000

RESAMPLING_CHOICES = {
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
    "nearest": Resampling.nearest,
}


def utm_epsg_for(longitude, latitude):
    """EPSG code of the WGS 84 UTM zone containing a point."""
    zone = int(math.floor((longitude + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    return (32600 if latitude >= 0 else 32700) + zone, zone


def describe(path):
    with rasterio.open(path) as src:
        xres, yres = abs(src.transform.a), abs(src.transform.e)
        return {
            "crs": src.crs, "width": src.width, "height": src.height,
            "xres": xres, "yres": yres,
            "projected": bool(src.crs and src.crs.is_projected),
            "bounds": src.bounds,
        }


def report_mesh_size(width, height, downsample, hmax, resolution):
    """Warn when the output would produce a triangulation mmg cannot chew."""
    vertices = (width // max(downsample, 1)) * (height // max(downsample, 1))
    print(f"  Input triangulation: ~{vertices:,} vertices "
          f"({width}x{height} px at mesh.downsample = {downsample})")

    if resolution > hmax:
        print(f"  NOTE: {resolution:g} m pixels are coarser than mesh.hmax = "
              f"{hmax:g} m, so the mesh cannot resolve terrain at its own "
              f"target edge length. Consider a finer --resolution.")

    if vertices >= MESH_VERTICES_REFUSE:
        suggestion = resolution * math.sqrt(vertices / (MESH_VERTICES_REFUSE / 4))
        print(f"  WARNING: that is very large. Remeshing is unlikely to finish. "
              f"Try --resolution {int(round(suggestion / 10) * 10)} or a "
              f"smaller --bounds.")
    elif vertices >= MESH_VERTICES_WARN:
        print("  WARNING: that is large; remeshing will be slow. Consider a "
              "coarser --resolution if the run does not complete.")


def reproject_dem(dem_path, output_path, target_crs=None, resolution=30.0,
                  bounds=None, bounds_crs="EPSG:4326", resampling="bilinear",
                  downsample=3, hmax=150.0, force=False):
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{dem_path} has no CRS, so it cannot be reprojected. Set one "
                "in QGIS (Layer Properties > Source > Assign CRS) first."
            )

        src_xres, src_yres = abs(src.transform.a), abs(src.transform.e)
        print(f"Source: {dem_path}")
        print(f"  CRS:        {src.crs.to_string()} "
              f"({'projected' if src.crs.is_projected else 'geographic'})")
        print(f"  Pixel size: {src_xres:g} x {src_yres:g} "
              f"{'m' if src.crs.is_projected else 'deg'}")
        print(f"  Size:       {src.width} x {src.height} px")

        if src.crs.is_projected and not force:
            raise SystemExit(
                f"\n{dem_path} is already in a projected CRS "
                f"({src.crs.to_string()}), so its units are already linear and "
                "this script has nothing to convert. This usually means the "
                "file has been reprojected once already.\n"
                "Pass --force to warp it anyway (to change zone or resolution)."
            )

        # Crop first, so the UTM zone follows the cropped extent.
        extent = tuple(src.bounds)
        if bounds is not None:
            requested = transform_bounds(bounds_crs, src.crs, *bounds)
            extent = (
                max(requested[0], src.bounds.left),
                max(requested[1], src.bounds.bottom),
                min(requested[2], src.bounds.right),
                min(requested[3], src.bounds.top),
            )
            if extent[0] >= extent[2] or extent[1] >= extent[3]:
                full = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                raise SystemExit(
                    "\n--bounds does not overlap the DEM.\n"
                    f"  requested (in {bounds_crs}): {bounds}\n"
                    f"  DEM covers (EPSG:4326):      "
                    f"{full[0]:.5f} {full[1]:.5f} {full[2]:.5f} {full[3]:.5f}"
                )
            print(f"  Cropping to {bounds} ({bounds_crs})")

        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *extent)
        if target_crs is None:
            epsg, zone = utm_epsg_for((west + east) / 2, (south + north) / 2)
            target_crs = f"EPSG:{epsg}"
            print(f"  Auto-selected target CRS: {target_crs} (UTM zone {zone}N)")
            mid = (south + north) / 2
            span = utm_epsg_for(east, mid)[1] - utm_epsg_for(west, mid)[1]
            if span:
                print(f"  NOTE: the selected region spans "
                      f"{span + 1} UTM zones. A single zone still works, but "
                      "distortion grows away from the chosen one.")

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *extent,
            resolution=resolution,
        )

        if resampling == "nearest":
            print("\n  WARNING: nearest-neighbour resampling on elevation "
                  "leaves stair-steps that become spurious slope in the "
                  "diffusion tensor. Prefer bilinear or cubic.")

        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile.copy()
        profile.update({
            "crs": target_crs, "transform": transform,
            "width": width, "height": height, "nodata": nodata,
            "count": 1, "compress": "deflate",
        })
        profile.pop("tiled", None)

        print(f"\nTarget: {output_path}")
        print(f"  CRS:        {target_crs}")
        print(f"  Pixel size: {resolution:g} x {resolution:g} m")
        print(f"  Size:       {width} x {height} px "
              f"({width * resolution / 1000:.1f} x "
              f"{height * resolution / 1000:.1f} km)")
        report_mesh_size(width, height, downsample, hmax, resolution)

        Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform, dst_crs=target_crs,
                dst_nodata=nodata,
                resampling=RESAMPLING_CHOICES[resampling],
            )

    with rasterio.open(output_path) as dst:
        data = dst.read(1, masked=True)
        blank = float(np.ma.getmaskarray(data).mean())
        print(f"\n  Elevation range: {data.min():.1f} to {data.max():.1f} m")
        if blank > 0.001:
            print(f"  Nodata: {blank * 100:.1f}% of pixels")
            if blank > 0.02:
                print("  WARNING: a nodata border becomes artificial flat "
                      "terrain in the mesh. Tighten --bounds, or see "
                      "crop_dem_valid_window.py.")
        print(f"\nWrote {output_path}")
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reproject a DEM from geographic degrees to projected metres, "
            "optionally cropping in the same pass. Step 1 of the meshing "
            "workflow."
        ),
        epilog=(
            "Example: reproject_dem.py --dem dem.tif --output region_utm.tif "
            "--bounds -107.70 38.43 -107.22 38.91"
        ),
    )
    parser.add_argument("--dem", default="dem.tif",
                        help="Source DEM, normally in geographic degrees "
                             "(default: dem.tif).")
    parser.add_argument("--output", default="dem_utm.tif",
                        help="Reprojected output (default: dem_utm.tif).")
    parser.add_argument("--target-crs", default=None,
                        help="Target CRS. Default: the UTM zone containing the "
                             "raster's centroid.")
    parser.add_argument("--resolution", type=float, default=30.0,
                        help="Output pixel size in metres (default: 30, "
                             "matching NLCD).")
    parser.add_argument("--bounds", type=float, nargs=4, default=None,
                        metavar=("W", "S", "E", "N"),
                        help="Optional crop extent, in --bounds-crs.")
    parser.add_argument("--bounds-crs", default="EPSG:4326",
                        help="CRS of --bounds (default: EPSG:4326 lon/lat).")
    parser.add_argument("--resampling", default="bilinear",
                        choices=sorted(RESAMPLING_CHOICES),
                        help="Resampling method (default: bilinear; do not use "
                             "nearest on elevation).")
    parser.add_argument("--mesh-downsample", type=int, default=3,
                        help="mesh.downsample, used only to size the "
                             "triangulation warning (default: 3).")
    parser.add_argument("--mesh-hmax", type=float, default=150.0,
                        help="mesh.hmax, used only for the resolution advisory "
                             "(default: 150).")
    parser.add_argument("--force", action="store_true",
                        help="Warp even if the source is already projected.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resolution <= 0:
        raise SystemExit("--resolution must be positive")
    reproject_dem(
        args.dem, args.output, target_crs=args.target_crs,
        resolution=args.resolution, bounds=args.bounds,
        bounds_crs=args.bounds_crs, resampling=args.resampling,
        downsample=args.mesh_downsample, hmax=args.mesh_hmax, force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
