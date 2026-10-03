"""A ParaView categorical colormap for the NLCD land-cover classes.

Class names come from parameters.json when available. Codes with no palette
entry (for example an NLCD NoData fill of 250) are drawn magenta. Usually
called through export_mesh_to_xdmf.py, which restricts the legend to the
classes present:

    python utilities/export_mesh_to_xdmf.py --mesh-folder region_mesh
"""

from __future__ import annotations

import json

# Approximate NLCD legend colours, RGB in 0-1.
LC_COLORS = {
    0:  {"name": "Unknown / nodata",              "color": [0.35, 0.35, 0.35]},
    11: {"name": "Open Water",                    "color": [0, 0, 1]},
    12: {"name": "Perennial Ice/Snow",            "color": [1, 1, 1]},
    21: {"name": "Developed Open Space",          "color": [192/255, 192/255, 192/255]},
    22: {"name": "Developed Low Intensity",       "color": [160/255, 160/255, 160/255]},
    23: {"name": "Developed Medium Intensity",    "color": [128/255, 128/255, 128/255]},
    24: {"name": "Developed High Intensity",      "color": [96/255, 96/255, 96/255]},
    31: {"name": "Barren Rock",                   "color": [1, 102/255, 102/255]},
    41: {"name": "Deciduous Forest",              "color": [0, 204/255, 0]},
    42: {"name": "Evergreen Forest",              "color": [0, 153/255, 0]},
    43: {"name": "Mixed Forest",                  "color": [0, 102/255, 0]},
    52: {"name": "Shrub/Scrub",                   "color": [1, 1, 102/255]},
    71: {"name": "Grassland/Herbaceous",          "color": [178/255, 1, 102/255]},
    81: {"name": "Pasture/Hay",                   "color": [1, 1, 153/255]},
    82: {"name": "Cultivated Crops",              "color": [1, 102/255, 1]},
    90: {"name": "Woody Wetlands",                "color": [153/255, 1, 204/255]},
    95: {"name": "Emergent Herbaceous Wetlands",  "color": [153/255, 1, 1]},
}

UNMAPPED_COLOR = [1.0, 0.0, 1.0]
DEFAULT_NAME = "NLCD Land Cover"


def save_paraview_colormap(output_json, codes=None, class_names=None,
                           opacity=None, name=DEFAULT_NAME, verbose=True):
    """Write a ParaView categorical colormap for `codes`.

    codes       the classes to include, in the order they should be listed;
                None means the whole NLCD palette.
    class_names optional {code: name} from parameters.json, preferred over the
                built-in names so that the legend and the parameter file agree.
    name        the preset name ParaView will list; use one specific to the
                bundle, since ParaView adds rather than replaces presets.
    opacity     if given, emit IndexedOpacities at this value.
    """
    if codes is None:
        codes = sorted(LC_COLORS)
    codes = [int(code) for code in codes]
    names = {int(k): v for k, v in (class_names or {}).items()}

    annotations = []
    colors = []
    rows = []
    for code in codes:
        entry = LC_COLORS.get(code)
        color = entry["color"] if entry else UNMAPPED_COLOR
        label = names.get(code) or (entry["name"] if entry else None)
        if label is None:
            label = f"UNMAPPED {code}"
        # ParaView matches annotations on the string of the float value.
        annotations.append(str(float(code)))
        annotations.append(label)
        colors.extend(color)
        rows.append((code, label, entry is not None))

    preset = {
        "Name": str(name),
        "ColorSpace": "RGB",
        "NanColor": [0.5, 0.5, 0.5],
        "Annotations": annotations,
        "IndexedColors": colors,
        "InterpretValuesAsCategories": True,
    }
    if opacity is not None:
        preset["IndexedOpacities"] = [float(opacity)] * len(codes)
    colormap = [preset]

    with open(output_json, "w", encoding="utf-8") as stream:
        json.dump(colormap, stream, indent=2)

    if verbose:
        print(f"Wrote {output_json}  ({len(codes)} classes, preset '{name}')")
        unmapped = [code for code, _label, known in rows if not known]
        if unmapped:
            print(f"  {len(unmapped)} class(es) have no palette entry and are "
                  f"drawn magenta: {', '.join(str(c) for c in unmapped)}")
    return output_json


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="nlcd_colormap.json",
                        help="Colormap JSON to write.")
    parser.add_argument("--land-cover-classes", default=None,
                        help="Class array; restricts the legend to the "
                             "classes actually present.")
    parser.add_argument("--parameters", default=None,
                        help="Parameter file supplying land_cover.class_names.")
    parser.add_argument("--opacity", type=float, default=None,
                        help="Emit an opacity map at this value. Omitted by "
                             "default: it makes the surface translucent, "
                             "which tends to wash the render out to grey.")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="Preset name shown in ParaView. Make it specific: "
                             "importing the same name repeatedly eventually "
                             "fails outright.")
    return parser


def main():
    import sys
    from pathlib import Path

    utilities = Path(__file__).resolve().parent
    if str(utilities) not in sys.path:
        sys.path.insert(0, str(utilities))

    args = build_parser().parse_args()

    codes = None
    if args.land_cover_classes is not None:
        import numpy as np
        present = np.load(args.land_cover_classes).astype(np.int32)
        values, counts = np.unique(present, return_counts=True)
        codes = [int(v) for v in values[np.argsort(-counts)]]

    class_names = None
    if args.parameters is not None:
        from shared_parameters import load_parameters
        class_names = load_parameters(args.parameters)["land_cover"].get(
            "class_names"
        )

    save_paraview_colormap(args.output, codes=codes, class_names=class_names,
                           opacity=args.opacity, name=args.name)


if __name__ == "__main__":
    main()
