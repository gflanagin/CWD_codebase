# Mesh generation: DEM and land cover to a mesh bundle

This document covers building a **mesh bundle**, the terrain surface and
per-node land-cover arrays the solver reads. Running the solver on a bundle is
covered in [`README_solver.md`](README_solver.md). The bundles used in the paper
are in the data archive, so this pipeline is needed only for a new region.

## Environment

```bash
conda env create --file environment.yml
conda activate cwd-fem
python utilities/verify_environment.py     # package versions and the PyVista .mmg accessor
python utilities/smoke_test_fenics.py      # minimal DOLFINx/PETSc assembly
```

Keep DOLFINx, PETSc, HDF5, MPI and their Python bindings in this one
environment so their compiled versions agree. The workflow is serial.

## What a mesh bundle is

| File | Contents | Used as |
| --- | --- | --- |
| `terrain.msh` | triangulated terrain surface embedded in 3-D | DOLFINx mesh (`gdim = 3`) |
| `land_cover_classes.npy` | NLCD class code per CG1 node | input to the carrying capacity `c(x)` |
| `land_cover_diffusivity.npy` | permeability `p(x)` per CG1 node | multiplier in the diffusion tensor |
| `effective_parameters.json` | `parameters.json` with command-line overrides folded in | every model constant |

The four files are one bundle: the arrays follow the final mesh's node order, so
never mix files from different runs. Two more files appear beside them:
`coord_offsets.npy`, the X/Y centre subtracted during meshing, and the cached
spin-ups the solver writes on its first run.

Carrying capacity is not stored; the solver computes it from the classes at
start-up.

## Parameters that affect the mesh

`parameters.json` is the single configuration for the mesher and the solver.
The mesh metric uses the same slope factor as the diffusion tensor,

```
eta(theta) = iota + (1 - iota) exp(-gamma tan(theta))
```

with `iota = diffusion_tensor.isotropy`, `gamma = diffusion_tensor.slope_penalty`
and `theta` the local slope. Edges have length `hmax` along contours and
`sqrt(eta) hmax` up the slope; with `trace_preserving` the contour edges
stretch to `sqrt(2 - eta) hmax`. The finest edge is `sqrt(iota) hmax`. Changing
`isotropy`, `slope_penalty`, `trace_preserving`, `hmax` or the
`land_cover.diffusivity` table therefore needs a new bundle; changing a
carrying-capacity value does not.

| Key | Meaning |
| --- | --- |
| `mesh.downsample` | elevation grid stride before the initial triangulation |
| `mesh.smoothing_sigma` | Gaussian elevation smoothing (`--no-smooth` disables it) |
| `mesh.hmax` | coarsest target edge length, in metres (250 for the paper's meshes) |

## The pipeline

Every script takes `--help`.

### 1. Prepare the DEM

Download a 3DEP elevation tile from the
[National Map Downloader](https://apps.nationalmap.gov/downloader/). 3DEP tiles
are in geographic degrees and the mesher needs metres, so crop and reproject in
one pass:

```bash
python utilities/reproject_dem.py --dem dem.tif --output region_utm.tif \
    --bounds -107.70 38.43 -107.22 38.91
```

With no `--target-crs` it picks the UTM zone of the region's centroid. It
resamples bilinearly, since nearest-neighbour stair-steps become spurious slope,
and defaults to 30 m, the native NLCD resolution. If a nodata border remains:

```bash
python utilities/crop_dem_valid_window.py --dem region_utm.tif \
    --output region_cropped.tif --margin 40 40 30 30
```

Download an NLCD land-cover tile from the [MRLC viewer](https://www.mrlc.gov/viewer/).
It need not share the DEM's grid or CRS; it only has to contain the DEM's
extent, because it is sampled directly at the mesh nodes.

### 2. Check land-cover coverage (optional)

```bash
python check_land_cover_coverage.py --land-cover land_cover.tif --dem region_cropped.tif
```

Nodes outside the NLCD tile receive class 0, which maps to real parameter
values, so a tile that misses the DEM would otherwise give a valid-looking
bundle with no land-cover structure. This exits nonzero in that case. Step 3
runs the same check and also audits the sampled classes afterwards.

### 3. Generate the bundle

```bash
python generate_mesh_and_attributes.py \
    --dem region_cropped.tif --land-cover land_cover.tif \
    --parameters parameters.json --output-folder region_mesh --hmax 250
```

The script triangulates the (optionally smoothed and downsampled) DEM, centres
the coordinates, builds the slope metric, remeshes with MMG, writes
`terrain.msh`, then reloads the final mesh through DOLFINx and samples the NLCD
classes in final node order. Initial-condition coordinates in
`parameters.json` are in the centred mesh coordinates; `coord_offsets.npy`
converts.

The solver picks up `effective_parameters.json` from `--mesh-folder`. When
sweeping, use `utilities/sweep_setup.py --overlay parameters.json` so the
reaction and control settings come from the current `parameters.json` while
the mesh-specific sections stay with the bundle.

### 4. Inspect

```bash
python utilities/export_mesh_to_xdmf.py --mesh-folder region_mesh
```

writes `bundle.xdmf` with `Land_Cover_Class`, `Permeability_ell`,
`Carrying_Capacity`, `Elevation` and, once cached, `Susceptible_Equilibrium`,
plus a ParaView colormap for the classes present (`utilities/lc_colorbar.py`).
Classes above 95 are not NLCD land cover and are drawn magenta; they usually
mean an NLCD nodata fill reached the mesh.

### Re-deriving the arrays without remeshing

The mesh geometry does not depend on land cover, so a changed land-cover table
needs only new arrays:

```bash
python utilities/resample_land_cover_on_existing_mesh.py \
    --mesh region_mesh/terrain.msh --land-cover land_cover.tif \
    --dem region_cropped.tif --coord-offsets region_mesh/coord_offsets.npy \
    --classes-output region_mesh/land_cover_classes.npy \
    --diffusivity-output region_mesh/land_cover_diffusivity.npy
```

## Land-cover mappings

Both tables live in `land_cover` in `parameters.json` and are applied by
`utilities/shared_parameters.py`. They measure different things: carrying
capacity `c(x)` is habitat quality and permeability `p(x)` is ease of
movement, and for several classes the two run opposite ways. Open water has
`c = 0` and `p = 0.01`. Class 0 (nodata) takes 0.5 in both. Every entry is
expert judgement and should be calibrated against regional density estimates.

## Mesh-generation files

| File | Role |
| --- | --- |
| `generate_mesh_and_attributes.py` | step 3 |
| `check_land_cover_coverage.py` | step 2 |
| `land_cover_to_capacity.py` | carrying-capacity mapping wrapper |
| `utilities/shared_parameters.py` | parameter loader and both land-cover conversions |
| `utilities/reproject_dem.py`, `utilities/crop_dem_valid_window.py` | step 1 |
| `utilities/land_cover_sampling.py` | NLCD sampling at nodes and the coverage checks |
| `utilities/resample_land_cover_on_existing_mesh.py` | new arrays on an existing mesh |
| `utilities/export_mesh_to_xdmf.py`, `utilities/lc_colorbar.py` | ParaView export |
| `utilities/flat_transect.py` | the flat-disk bundle (`--build-only`) |
| `utilities/ablation_bundles.py`, `utilities/flatten_land_cover.py` | the four ablation arms of a bundle |
