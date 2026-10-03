# CWD_code

Finite-element code for a spatially explicit S-E-I-R-W reaction-diffusion model
of chronic wasting disease (CWD) in mule deer on a terrain surface built from
elevation and land-cover data, and for the optimal control of a sight-based
cull acting on the clinical class.

It accompanies

> G. Flanagin, A. Escoto, Z. Chao. *A Reaction-Diffusion Model for Chronic
> Wasting Disease Spread in Deer on a Topographically Realistic Landscape, with
> Optimal Culling Control.* Submitted to the Bulletin of Mathematical Biology.

The mesh bundles and simulation output behind the paper are archived
separately at [Zenodo DOI to be added].

## Installation

The code runs on Linux (tested on Ubuntu 24.04 under WSL2 and on an HTCondor
pool) with the Conda environment in `environment.yml`: Python 3.11,
DOLFINx 0.11, Gmsh 4.15, mmgpy 0.16, Rasterio 1.4 and SciPy 1.16.

```bash
conda env create --file environment.yml
conda activate cwd-fem
python utilities/verify_environment.py
```

Everything is serial. The nodal arrays of a mesh bundle are indexed by local
degree of freedom, so the solver refuses to run under more than one MPI rank.

## Repository layout

| Path | Contents |
| --- | --- |
| `cwd_control_problem.py` | the discrete state, adjoint and gradient: mesh, spaces, diffusion tensor, the factorized operator, and every variational form |
| `CWD_optimal_control.py` | the driver: forward-only runs, the optimizer, the Taylor test, output |
| `generate_mesh_and_attributes.py` | DEM and NLCD rasters to a mesh bundle |
| `check_land_cover_coverage.py`, `land_cover_to_capacity.py` | mesh-generation helpers |
| `parameters.json` | every model, mesh and optimizer constant |
| `utilities/` | spin-ups, bundle handling, sweep generation, the mesh-free adjoint check, and the figure scripts |
| `condor/` | the HTCondor job wrapper, cluster bootstrap, and the commands that reproduce the paper's runs |

Documentation is split three ways:

- [`README_mesh_generation.md`](README_mesh_generation.md): building a mesh
  bundle from a DEM and an NLCD tile.
- [`README_solver.md`](README_solver.md): the model, the control problem, the
  driver and its outputs, and a symbol-to-code table for the paper.
- [`condor/README.md`](condor/README.md): reproducing every simulation and
  figure in the paper.

## Quick start

Build the flat 40 km disk used for the paper's main grid (about 94,000 nodes):

```bash
python utilities/flat_transect.py --out runs/flat_disk --radius-km 40 \
    --permeability 1 --build-only
```

Check the reduced gradient, then run the uncontrolled baseline and one
optimization:

```bash
python utilities/check_reaction_adjoint.py
python CWD_optimal_control.py --mesh-folder runs/flat_disk/bundle --gradient-check
python CWD_optimal_control.py --mesh-folder runs/flat_disk/bundle --forward-only \
    --output-folder runs/baseline
python CWD_optimal_control.py --mesh-folder runs/flat_disk/bundle \
    --output-folder runs/optimal
```

The first run on a bundle computes and caches the disease-free equilibrium and
the spun-up epizootic beside the mesh; later runs reuse them.

## Source data

The landscape meshes are built from USGS 3DEP 1/3 arc-second elevation tiles
([National Map Downloader](https://apps.nationalmap.gov/downloader/)) and the
2025 Annual NLCD land-cover product ([MRLC](https://www.mrlc.gov/viewer/)).
Neither is redistributed here; the data archive contains the finished mesh
bundles.

## Citation and license

Cite the paper above and this archive; see `CITATION.cff`. The code is released
under the MIT License (`LICENSE`).
