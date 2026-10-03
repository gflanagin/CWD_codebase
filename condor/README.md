# Reproducing the paper's simulations

Every simulation in the paper is one optimization (or one uncontrolled
baseline) of `CWD_optimal_control.py` on a mesh bundle, with a per-point
parameter file written by `utilities/sweep_setup.py`. The runs were made on an
HTCondor pool, one core per point; this folder holds the job wrapper and the
commands. Any point can also be run directly, as shown at the end.

The finished outputs are in the data archive. The figure scripts read them from
`cluster_runs/<experiment>/` relative to the repository root.

## Settings shared by every run

All of these are the values in `parameters.json`, so `--overlay
parameters.json` carries them into every point file.

| Setting | Value | Key |
| --- | --- | --- |
| horizon `T`, control horizon `T_c` | 30 yr, 20 yr | `time.final_time_years`, `optimal_control.control_horizon_years` |
| time step | 0.05 yr | `time.time_step_years` (passed as `--time-step-years 0.05`) |
| `c1`, `c2`, `c3`, `c4` | 10, 0, 0.5, 0.5 | `optimal_control.cost_*` |
| `v_max` | 3 /yr | `optimal_control.control_maximum` |
| detection threshold `R_det` | 60.8356 clinical deer-years | `disease_spinup.detection_threshold` |

`R_det` was measured once with `utilities/anchor_detection_threshold.py`
(`prepare` writes a probe parameter file, `read --apply` writes the
accumulated burden back into `parameters.json`) and is held fixed.

`--R0` in `sweep_setup.py` is the best-habitat reproduction number `R0-tilde`.
On a landscape the average is `R0-tilde * c-bar`, where `c-bar` is the
area-weighted mean carrying-capacity fraction, so a landscape-average target
`R0` is reached with `--R0 R0/c-bar`. On the flat disk `c-bar = 1`.

## Environment on an HTCondor pool

The generated submit files assume a shared filesystem visible from every
execute node. Paths must not contain whitespace: HTCondor's submit language
cannot quote them.

```bash
export CWD_CLUSTER_ROOT=/path/to/shared/cwd     # holds repo/, conda/, runs/
bash $CWD_CLUSTER_ROOT/repo/condor/bootstrap_cluster.sh
```

`bootstrap_cluster.sh` installs Miniforge and the `cwd-fem` environment under
the root, checks that the PETSc/DOLFINx stack imports, and compiles one form to
warm the FFCx cache. It is idempotent. For interactive commands afterwards:

```bash
CWD=$CWD_CLUSTER_ROOT
source $CWD/conda/miniforge/etc/profile.d/conda.sh
conda activate $CWD/conda/cwd-fem
export PY=$CWD/conda/cwd-fem/bin/python XDG_CACHE_HOME=$CWD/jit_cache
cd $CWD/repo
```

FFCx compiles forms at run time with the environment's `gcc`, so the
environment's `bin/` must be on `PATH`. `run_solve.sh` arranges this for batch
jobs.

## How a sweep runs

`utilities/sweep_setup.py` writes one parameter file per point. A point's name
is `delta_<d>_R0_<r>__c4_<c>`; everything before `__` is its **spin-up group**,
since the disease spin-up depends only on the reaction rates.

`utilities/make_condor_sweep.py` turns that folder into a DAG with:

- one uncontrolled baseline per spin-up group, run before the group's points;
  it computes and caches the spun-up initial state the points share;
- one node run alone first, so concurrent jobs do not race on a cold FFCx
  cache;
- idempotent nodes, so a rescue DAG reruns only what failed, and resumption of
  an evicted point from its last `control_checkpoint.npy`.

`condor/run_solve.sh` is the job both node types run. It pins every threading
library to one thread and memory-maps the forward trajectory on the execute
node's scratch. Summarize a finished sweep with
`utilities/sweep_report.py <outroot>`.

## 1. Flat-disk grid (12 points)

A flat disk of radius 40 km with uniform land cover (`c = p = 1`), so `K = K0`
and `D = kappa I`, with the index case at the centre. `delta` in {0.125, 0.25,
0.5, 1} by `R0` in {2, 3.5, 5}.

```bash
$PY utilities/flat_transect.py --out $CWD/flat_disk --radius-km 40 \
    --permeability 1 --build-only

$PY utilities/sweep_setup.py --base $CWD/flat_disk/bundle/effective_parameters.json \
    --overlay parameters.json --out sweep_params_flat \
    --delta 0.125 0.25 0.5 1.0 --R0 2 3.5 5 --c4 0.5 --c3 0.5 --v-max 3 \
    --time-step-years 0.05
```

The DAG needs `susceptible_equilibrium.npy` in the bundle, which the first
forward run writes:

```bash
$PY CWD_optimal_control.py --mesh-folder $CWD/flat_disk/bundle \
    --parameters sweep_params_flat/delta_0p25_R0_3p5__c4_0p5.json \
    --output-folder $CWD/runs/flat_probe --forward-only \
    --disease-state $CWD/runs/flat_probe/state.npy

$PY utilities/make_condor_sweep.py --mesh $CWD/flat_disk/bundle \
    --params $CWD/repo/sweep_params_flat --outroot $CWD/runs/flat_grid \
    --submit-dir $CWD/repo/condor/flat_grid --python $PY --request-memory 4G \
    --extra-args "--checkpoint-every 5 --output-every 20 --no-spinup-series"
condor_submit_dag $CWD/repo/condor/flat_grid/sweep.dag
```

## 2. Objective comparison on the flat disk

The cell `delta = 0.25`, `R0 = 3.5` under two further objectives that move
weight from the susceptible deficit onto the reservoir, with `c2` scaled so the
uncontrolled `J` is unchanged. In the uncontrolled run,
`int(S_inf - S) / int(W) = 2.428`, so `c2 = 2.428 (10 - c1)`:

| `c1` | `c2` | `W` share of the uncontrolled `J` |
| --- | --- | --- |
| 5 | 12.14 | 50% |
| 0 | 24.28 | 100% |

`sweep_setup.py` has no `c1`/`c2` axis, so the two point files are patched from
the generated one. The spin-up and baseline are copied from the flat grid.

```bash
RUNS=$CWD/runs; P=sweep_params_flat_c2; O=$RUNS/flat_c2; g=delta_0p25_R0_3p5
mkdir -p $P
$PY utilities/sweep_setup.py --base $CWD/flat_disk/bundle/effective_parameters.json \
    --overlay parameters.json --out _tmp_c2 --delta 0.25 --R0 3.5 --c4 0.5 \
    --c3 0.5 --v-max 3 --time-step-years 0.05
n=$(cat _tmp_c2/manifest.txt)
for arm in "5 12.14 c1_5_c2_12p14" "0 24.28 c1_0_c2_24p28"; do set -- $arm
  $PY - _tmp_c2/$n.json $P/${n}__$3.json $1 $2 <<'EOF'
import json, sys
src, dst, c1, c2 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
p = json.load(open(src))
p["optimal_control"]["cost_susceptible_deficit"] = c1
p["optimal_control"]["cost_environment"] = c2
json.dump(p, open(dst, "w"), indent=2, ensure_ascii=False)
EOF
  echo ${n}__$3 >> $P/manifest.txt
done
rm -rf _tmp_c2

mkdir -p $O/spinup_cache
cp $RUNS/flat_grid/spinup_cache/$g.* $O/spinup_cache/
cp -r $RUNS/flat_grid/baseline_$g $O/

$PY utilities/make_condor_sweep.py --mesh $CWD/flat_disk/bundle \
    --params $CWD/repo/$P --outroot $O --submit-dir $CWD/repo/condor/flat_c2 \
    --python $PY --request-memory 4G \
    --extra-args "--checkpoint-every 5 --output-every 20 --no-spinup-series"
condor_submit_dag $CWD/repo/condor/flat_c2/sweep.dag
```

## 3. Landscape ablation (three regions, four arms each)

Three 100 km by 100 km meshes at `hmax = 250 m`, built with
`generate_mesh_and_attributes.py` and archived in the data record. Each region
is run at one cell, `delta = 0.25` and landscape-average `R0 = 3.5`, under four
arms that cross slope anisotropy (on, or `isotropy = 1`) with land cover
(measured, or flattened to its area-weighted mean):

| Region | Nodes | `c-bar` | `R0-tilde = 3.5 / c-bar` | Focus (mesh m) |
| --- | --- | --- | --- | --- |
| `upper_arkansas` | 350,315 | 0.8260 | 4.237 | (39, -44) |
| `estes_park` | 323,798 | 0.7782 | 4.498 | (7045, -7045) |
| `carbondale` | 370,354 | 0.8374 | 4.18 | (45, -10045) |

`utilities/ablation_bundles.py` writes the four arm bundles for a region and
prints `c-bar`; check it against the table.

```bash
RUNS=$CWD/runs; AB=$CWD/ablation; M=$CWD/meshes100
declare -A R0T=([upper_arkansas]=4.237 [estes_park]=4.498 [carbondale]=4.18)
declare -A FOCUS=([upper_arkansas]="39 -44" [estes_park]="7045 -7045" [carbondale]="45 -10045")
for region in upper_arkansas estes_park carbondale; do
  $PY utilities/ablation_bundles.py --mesh $M/${region}_mesh --out $AB \
      --focus ${FOCUS[$region]}
  for arm in aniso_lc aniso_nolc iso_lc iso_nolc; do A=${region}_$arm
    $PY utilities/sweep_setup.py --base $AB/$A/effective_parameters.json \
        --overlay parameters.json --out sweep_params_ablation/$A \
        --delta 0.25 --R0 ${R0T[$region]} --c4 0.5 --c3 0.5 --v-max 3 \
        --time-step-years 0.05
    $PY utilities/make_condor_sweep.py --mesh $AB/$A \
        --params $CWD/repo/sweep_params_ablation/$A --outroot $RUNS/ablation/$A \
        --submit-dir $CWD/repo/condor/ablation/$A --python $PY \
        --request-memory 12G --request-disk 24G \
        --extra-args "--checkpoint-every 5 --output-every 20 --no-spinup-series"
    condor_submit_dag $CWD/repo/condor/ablation/$A/sweep.dag
  done
done
```

## Figures and reported numbers

With the outputs under `cluster_runs/` and, for the ablation region panels, the
three bundles under `meshes100/<region>_mesh/` with their `bundle.h5` written
by `utilities/export_mesh_to_xdmf.py`:

```bash
python utilities/make_wavefront_figure.py            # flat grid wavefronts
python utilities/make_wavefront_figure.py --table    # front positions quoted in the text
python utilities/make_transect_figure.py             # objective comparison
python utilities/ablation_report.py cluster_runs/ablation upper_arkansas --focus 39 -44
python utilities/ablation_report.py cluster_runs/ablation estes_park carbondale
python utilities/make_ablation_figure.py --run ablation upper_arkansas estes_park carbondale
```

`utilities/control_area_stats.py` computes treated and never-treated areas from
saved controls; it builds the lumped mass through DOLFINx, because saved
controls are in solver degree-of-freedom order, not `terrain.msh` node order.

## Cost

| Experiment | Nodes | Time per point (one core) | Memory | Scratch |
| --- | --- | --- | --- | --- |
| flat disk | 93,701 | 1 h (up to 1.6) | 4 GB | about 2.3 GB |
| landscape ablation | 324k to 370k | 4.3 h (up to 8.5) | 12 GB | about 8.5 GB |

The trajectory (`601 x 5 x nodes x 8` bytes) is memory-mapped on scratch.
Request memory above the optimizer's resident workspace if the pool charges a
job for the page cache of its own mapped files; the symptom is jobs going on
hold for memory part-way through a healthy solve.

## Running a point without HTCondor

```bash
G=delta_0p25_R0_3p5; N=${G}__c4_0p5; B=runs/flat_disk/bundle; P=sweep_params_flat
python CWD_optimal_control.py --mesh-folder $B --parameters $P/$N.json \
    --output-folder runs/flat_grid/baseline_$G --forward-only \
    --disease-state runs/flat_grid/spinup_cache/$G.npy
python CWD_optimal_control.py --mesh-folder $B --parameters $P/$N.json \
    --output-folder runs/flat_grid/$N \
    --disease-state runs/flat_grid/spinup_cache/$G.npy \
    --trajectory-file /tmp/$N.npy --output-every 20
```
