# The solver and the control problem

This document covers running `CWD_optimal_control.py` on a mesh bundle. Building
a bundle is covered in [`README_mesh_generation.md`](README_mesh_generation.md).
The model, the adjoint derivation and the discretization are in the paper; the
[symbol-to-code table](#symbol-to-code-correspondence) at the end maps one onto
the other.

## Compartments

| | | sheds | transmits directly | visible in the field | mobility |
| --- | --- | --- | --- | --- | --- |
| `S` | susceptible | no | no | n/a | 1 |
| `E` | exposed, latent | no | no | no | 1 |
| `I` | pre-clinical | `rho_I` | `beta_1` | no | 1 |
| `R` | clinical | `rho_R = 4 rho_I` | `beta_1` | yes | 0.376 |
| `W` | environmental prion load | | | | does not diffuse |

The model has no background mortality: `sigma_1`, `sigma_2` and `alpha` are the
only exits from `E`, `I` and `R`, and `alpha` absorbs disease mortality together
with any removal that targets symptomatic animals. `v = 0` is therefore an
unmanaged herd.

## The control problem

One control, the per-capita removal rate `v(x, t)` on the clinical class,
entering the `R` equation as `-v R`. It is sight-based: `R` is the class a field
crew can identify, and `v` cannot reach the pre-clinical class `I`.

```
J(v) = int_0^T int_Gamma [ c1 (S_inf - S) + c2 W ] dGamma dt
     + int_0^T_c int_Gamma [ c3 v^2 + c4 v ] dGamma dt
```

`S_inf` is the disease-free equilibrium, written to `CWD_S_infinity.xdmf`. The
control is a variable on `[0, T_c]` and is zero on the continuation tail
`(T_c, T]`, whose running cost prices the state left at `T_c`. The paper uses
`T = 30`, `T_c = 20` and `c2 = 0`. `terminal_cost_scale` multiplies an optional
closed-form terminal cost and is 0 throughout.

`v` is piecewise constant in time over control blocks of
`control_block_years` (1 yr), one CG1 field per block. Stationarity is

```
v_b* = P_[0, v_max]( (mean_{n in b} R^n lambda_R^{n+1} - c4) / (2 c3) )
```

so `c4` sets a threshold below which the optimum is exactly zero and `c3` sets
the slope above it. Since `v >= 0`, the `c4 v` term is smooth on the admissible
set and needs no proximal method.

### One iteration

1. **Forward.** Advance the controlled state with the IMEX scheme
   `(M_L + dt A) y^{n+1} = M_L y^n + dt F(y^n, v^n)`, recording the trajectory
   and accumulating `J`.
2. **Backward.** Sweep the discrete adjoint from `lambda^N` down, accumulating
   the reduced gradient
   `dJ/dv_b = 2 c3 v_b + c4 - mean_{n in b} R^n lambda_R^{n+1}`.

The operator `A` is symmetric and block diagonal by compartment, so the
factorizations built once at start-up serve both sweeps
(`utilities/block_solver.py`). The adjoint is the exact adjoint of the discrete
scheme, so the gradient is the exact derivative of the discrete objective.

### Optimizer

`lbfgs` (default) is SciPy's L-BFGS-B on the reduced problem. It optimizes the
scaled variable `z = M_b^{1/2} v`, where `M_b` combines the lumped mass with the
block length, so that the Euclidean product SciPy uses is the space-time `L2`
product. Without this the line-search slope would be wrong by the spread of the
lumped mass, about four orders of magnitude on a graded landscape mesh.
`projected-gradient` is a projected steepest descent with an Armijo line
search, kept as a reference.

The reduced problem is not convex. Both methods find a stationary point.

## Running the driver

```bash
python CWD_optimal_control.py --mesh-folder BUNDLE --output-folder OUT
```

`--mesh-folder` finds the folder's single `*.msh`, `land_cover_classes.npy`,
`land_cover_diffusivity.npy` and `effective_parameters.json`, and stops if any
is missing or if an array length does not match the mesh. `--parameters`,
`--mesh`, `--land-cover-classes` and `--land-cover-diffusivity` override
discovery individually.

**Uncontrolled baseline.** `--forward-only` runs one forward solve and writes
`forward_summary.json` with `J` and its breakdown. With no `--initial-control`
the control is zero. No trajectory is stored, so it needs no extra memory.

**Gradient check.** Two checks, cheapest first:

```bash
python utilities/check_reaction_adjoint.py
python CWD_optimal_control.py --mesh-folder BUNDLE --gradient-check
```

The first needs no mesh and no DOLFINx and runs in seconds. It checks the
adjoint rows and the control gradient against finite differences at random
states, and the closed-form terminal weights. The second is a Taylor test of the
full discrete gradient along a random direction and along one supported on the
last control block, which isolates the continuation tail:

```
R0 = |J(v + eps d) - J(v)|                      rate 1
R1 = |J(v + eps d) - J(v) - eps <grad J, d>|    rate 2
```

Directions are scaled to unit RMS rather than unit `L2` norm, because the
space-time measure `|Gamma| T` is of order `1e10` and a unit-`L2` direction
would perturb the control below round-off. The driver reports how far `R1` sits
above the round-off floor and exits nonzero when the test is inconclusive.

### Flags

| Flag | Effect |
| --- | --- |
| `--forward-only` | one forward solve, no optimization |
| `--gradient-check` | Taylor test, then exit |
| `--initial-control FILE.npy` | start from, or with `--forward-only` replay, a saved control |
| `--max-iterations N` | override `optimal_control.max_iterations` |
| `--optimizer lbfgs\|projected-gradient` | override `optimal_control.optimizer` |
| `--control-block-years Y` | block length; `0` gives one value per time step |
| `--control-horizon-years Y` | `T_c`; `0` controls the whole horizon |
| `--trajectory-file FILE.npy` | memory-map the forward trajectory instead of holding it in RAM |
| `--checkpoint-every N`, `--no-checkpoint` | how often to write `control_checkpoint.npy` |
| `--output-every N` | write every Nth step to XDMF |
| `--write-adjoint` | also write the adjoint fields |
| `--susceptible-equilibrium FILE`, `--recompute-equilibrium` | location of the cached `S_inf`, or force its recomputation |
| `--disease-state FILE`, `--recompute-disease-state` | location of the cached initial epizootic, or force its recomputation |
| `--spinup-series DIR`, `--no-spinup-series` | XDMF series of the disease spin-up |
| `--budget RATE` | optional per-block budget on mean effort (below) |
| `--heuristic EFFORT` | score the clinical-density rule `v = min(v_max, k R)` matched to a mean effort, instead of optimizing |

### Memory

The backward sweep needs the whole forward trajectory, `(N + 1) x 5 x nodes x 8`
bytes: 2.3 GB on the flat disk and about 8.5 GB on a 100 km landscape mesh at
`dt = 0.05` over 30 years. The driver prints the figure at start-up. Pass
`--trajectory-file` to back it with a memory map.

## Initial condition

Two spin-ups run before the control clock starts. Each is cached beside the mesh
with a signature of everything it depends on, and is recomputed if any of those
change.

1. **Disease-free equilibrium** (`utilities/susceptible_spinup.py`). The `S`
   equation alone, from `S = K`, until `max |dS/dt|` falls below
   `drift_tolerance` times the peak of `K` (60-year cap). `K` is a
   piecewise-constant land-cover lookup and not a steady state: diffusion
   smooths it across patch boundaries and water draws it down. The result
   `S_inf` is the initial susceptible field and the reference in the objective.
2. **Epizootic** (`utilities/disease_spinup.py`). The uncontrolled system from a
   Gaussian index case, run until the accumulated clinical burden
   `int int R dGamma dt` reaches `disease_spinup.detection_threshold`. Every
   parameter combination is therefore handed over at the same stage of
   detection rather than the same age.

## Cost weights

The `optimal_control` block of `parameters.json`:

| Key | Symbol | Value |
| --- | --- | --- |
| `cost_susceptible_deficit` | `c1` | 10 |
| `cost_environment` | `c2` | 0 |
| `cost_control` | `c3` | 0.5 |
| `cost_control_l1` | `c4` | 0.5 |
| `control_maximum` | `v_max` | 3 /yr |
| `control_block_years` | | 1 |
| `control_horizon_years` | `T_c` | 20 |
| `terminal_cost_scale` | | 0 |
| `lbfgs_memory` | `m` | 10 |
| `max_iterations` | | 40 |

Only the ratios of the weights matter. At least one of `c3` and `c4` must be
positive.

### Optional effort budget

`control_budget` (default `null`) caps the mass-weighted spatial mean of `v` in
each block, enforced by an augmented Lagrangian around the L-BFGS-B solve. The
per-block multipliers are the shadow prices of the budget and are written to
`optimization_history.json`. It cannot be combined with a nonzero `c4`. It is
not used in the paper.

## Parameters

The sources of the values in `parameters.json` are given in the paper. Units:
`S`, `E`, `I`, `R` in deer/km², `W` in prion units/km², rates per year. Mesh
coordinates are in metres, so `kappa` is in m²/yr.

The transmission rates are back-solved from a target `R0` and depend on
`carrying_capacity_base`, `alpha`, `delta` and the `rho`s. Change any of those
and rescale `beta_1` and `beta_2` together; `utilities/sweep_setup.py` does
this for `--delta` and `--R0`. `diffusion_tensor.isotropy`, `slope_penalty`,
`trace_preserving` and the `land_cover.diffusivity` table also enter the mesh,
so changing them needs a new bundle.

## Mass lumping

Mass terms use a vertex quadrature rule, which gives `M_L = diag(M 1)`. The
consistent mass matrix has positive off-diagonal entries that break the
M-matrix structure of `M + dt A` and let the implicit step undershoot on sharp
data; lumping preserves positivity and total mass at no loss of order for P1.
The adjoint load vector is lumped alongside, so each term is differentiated on
the rule it was assembled with. Reaction terms stay on a Gauss rule.

## Outputs

| File | Contents |
| --- | --- |
| `CWD_S`, `CWD_E`, `CWD_I`, `CWD_R`, `CWD_W` (`.xdmf`/`.h5`) | state series |
| `CWD_control_v` | the control |
| `CWD_S_infinity` | disease-free equilibrium |
| `CWD_adjoint_*` | adjoint fields, with `--write-adjoint` |
| `optimal_control.npy` | the control, shape `(blocks, dofs)`, in solver dof order |
| `optimization_history.json` | per-iteration cost and gradient norm, cost breakdown, diagnostics |
| `control_checkpoint.npy`, `.json` | latest iterate, for resuming with `--initial-control` |
| `forward_summary.json` | `J` and its breakdown, from `--forward-only` |

`optimal_control.npy` is in DOLFINx degree-of-freedom order, which is not
`terrain.msh` node order. Weight it only through a mass built by DOLFINx, as
`utilities/control_area_stats.py` does.

The end-of-run report gives two different zero statistics. The fraction of
space-time at `v = 0` counts a node once per block, so it is not an area. The
never-treated area is the area at `v = 0` in every block, the ground a programme
never visits. Both are mass weighted and are recorded under `controls.targeted`
in `optimization_history.json`.

## Symbol-to-code correspondence

`cwd_control_problem.py` builds the mixed space and land-cover fields,
assembles and factorizes the operator, and holds every form below; its forward
and backward methods advance the discrete state and adjoint. Symbols not listed
have the same name in the source as in the paper.

| Math | Code | Role |
| --- | --- | --- |
| `beta_1` | `direct_transmission_rate_preclinical`, `direct_transmission_rate_clinical` | direct transmission; the two keys must be equal |
| `beta_2` | `environmental_transmission_rate` | environmental transmission |
| `L(y^n, v^n)` | `state_rhs_form` | forward load vector |
| `[dL/dy]^T lambda + dt g` | `adjoint_rhs_form` | adjoint load vector |
| `dt g` | `terminal_rhs_form` | terminal adjoint, at `lambda(T) = 0` |
| `T_c` | `control_horizon_years` | end of the control; tail steps carry block index `-1` |
| `dJ_h/dv^n` | `gradient_form` | reduced gradient |
| `S_inf` | `S_infinity` | disease-free reference |
| `M_L` | `lumped_mass` | lumped mass, inner product |
| `beta(n), N^(b), dt_b` | `block_of_step`, `block_step_counts`, `block_weights` | block averaging |
| `c1, c2` | `cost_susceptible_deficit`, `cost_environment` | state weights |
| `c3, c4` | `cost_control`, `cost_control_l1` | control weights |
| `v_min, v_max` | `control_minimum`, `control_maximum` | admissible box |
| `N_b` | `control_block_years` | control block length |
| `m` | `lbfgs_memory` | L-BFGS-B history length |
| `M_b^{1/2}` | `block_weights`, `lumped_mass` | change of variables for L-BFGS-B |
| `R_det` | `disease_spinup.detection_threshold` | clinical burden that ends the spin-up |
