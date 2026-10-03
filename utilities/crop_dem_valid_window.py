"""Crop a DEM to its valid-data window, with an inset margin.

Use when a DEM carries a nodata border. The margin insets further, past the
ragged edge a reprojection leaves.
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.windows import Window


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dem", default="dem.tif",
                        help="DEM to crop (default: dem.tif).")
    parser.add_argument("--output", default="dem_cropped.tif",
                        help="Cropped result (default: dem_cropped.tif).")
    parser.add_argument(
        "--margin", type=int, nargs=4, default=(40, 40, 30, 30),
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        help="Extra pixels to inset past the valid-data window on each side "
             "(default: 40 40 30 30).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    left, right, top, bottom = args.margin
    if min(args.margin) < 0:
        raise SystemExit("--margin values cannot be negative")

    with rasterio.open(args.dem) as source:
        data = source.read(1)
        nodata = source.nodata
        valid = np.ones(data.shape, dtype=bool) if nodata is None \
            else ~np.isclose(data, nodata)
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            raise SystemExit(f"{args.dem} contains no valid pixels")

        row_min, row_max = rows[0] + top, rows[-1] - bottom
        col_min, col_max = cols[0] + left, cols[-1] - right
        if row_max <= row_min or col_max <= col_min:
            raise SystemExit(
                "The margins remove the whole valid window; reduce --margin."
            )

        window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        cropped = source.read(1, window=window)
        profile = source.profile.copy()
        profile.update({
            "width": col_max - col_min,
            "height": row_max - row_min,
            "transform": source.window_transform(window),
        })

    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(cropped, 1)

    remaining = 0 if nodata is None else int(np.isclose(cropped, nodata).sum())
    print(f"Wrote {args.output}: {col_max - col_min} x {row_max - row_min}")
    print(f"Remaining nodata pixels: {remaining}")


if __name__ == "__main__":
    main()
