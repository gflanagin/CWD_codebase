"""Disease-free spin-up: relax S to its equilibrium before any infection.

Runs the S equation alone (I = R = W = 0) from S = K until max|dS/dt| falls
below the drift tolerance, with the same lumped-mass semi-implicit step as the
state solver. The result is cached beside the mesh bundle under a signature of
the mesh, land-cover arrays and parameters it depends on, and is recomputed
when any of them change.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import TestFunction, TrialFunction, dot, grad

from block_solver import configure_direct_solver


# Cached field and its signature, stored in the mesh bundle folder.
EQUILIBRIUM_ARRAY_NAME = "susceptible_equilibrium.npy"
EQUILIBRIUM_SIGNATURE_NAME = "susceptible_equilibrium.json"

# Quadrature degree for the reaction term; matches QUADRATURE_DEGREE in
# cwd_control_problem.py.
REACTION_QUADRATURE_DEGREE = 4


def default_equilibrium_paths(mesh_path):
    """Where the cached field and its signature live for a given mesh."""
    folder = Path(mesh_path).expanduser().resolve().parent
    return folder / EQUILIBRIUM_ARRAY_NAME, folder / EQUILIBRIUM_SIGNATURE_NAME


def _digest(array):
    """Short content hash of a numeric array, for cache invalidation."""
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def spinup_time_step(parameters):
    """Spin-up step, defaulting to the main simulation's step when unset."""
    configured = parameters["susceptible_spinup"]["time_step_years"]
    if configured is None:
        return float(parameters["time"]["time_step_years"])
    return float(configured)


def equilibrium_signature(domain, carrying_capacity, diffusivity, parameters):
    """The inputs the disease-free equilibrium depends on, for cache validation."""
    tensor = parameters["diffusion_tensor"]
    spinup = parameters["susceptible_spinup"]
    signature = {
        "geometry_digest": _digest(domain.geometry.x),
        "carrying_capacity_digest": _digest(carrying_capacity),
        "diffusivity_digest": _digest(diffusivity),
        "node_count": int(np.asarray(carrying_capacity).size),
        "intrinsic_growth_rate": float(
            parameters["reaction"]["intrinsic_growth_rate"]
        ),
        "kappa": float(tensor["kappa"]),
        "isotropy": float(tensor["isotropy"]),
        "slope_penalty": float(tensor["slope_penalty"]),
        "susceptible_diffusion_scale": float(
            tensor["compartment_scales"]["susceptible"]
        ),
        "time_step_years": float(spinup_time_step(parameters)),
        "drift_tolerance": float(spinup["drift_tolerance"]),
    }
    # Added only when on, so existing caches stay valid.
    if bool(tensor.get("trace_preserving", False)):
        signature["trace_preserving"] = True
    return signature


def _print(comm, message):
    if comm.rank == 0:
        print(message)


def _progress(comm, completed_steps, total_steps, drift, width=40):
    if comm.rank != 0:
        return
    fraction = completed_steps / total_steps if total_steps else 1.0
    filled = min(width, int(width * fraction))
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\rSpin-up progress:    [{bar}] {fraction:6.1%} "
        f"({completed_steps}/{total_steps})  drift {drift:.2e}/yr"
    )
    sys.stdout.flush()


def run_susceptible_spinup(domain, V_scal, DTens, K_func, parameters,
                           comm=MPI.COMM_WORLD, verbose=True):
    """Relax S alone, from S = K, until it stops moving.

    Returns ``(values, info)`` where ``values`` is the nodal array of the
    relaxed field (same layout as ``Function(V_scal).x.array``) and ``info``
    records how the solve terminated.
    """
    spinup_parameters = parameters["susceptible_spinup"]
    growth_rate = float(parameters["reaction"]["intrinsic_growth_rate"])
    susceptible_scale = float(
        parameters["diffusion_tensor"]["compartment_scales"]["susceptible"]
    )
    dt = spinup_time_step(parameters)
    max_duration = float(spinup_parameters["max_duration_years"])
    drift_tolerance = float(spinup_parameters["drift_tolerance"])
    total_steps = max(1, int(round(max_duration / dt)))

    index_map = V_scal.dofmap.index_map
    n_owned = index_map.size_local * V_scal.dofmap.index_map_bs

    u = TrialFunction(V_scal)
    v = TestFunction(V_scal)

    S0 = Function(V_scal)
    S1 = Function(V_scal)
    S0.x.array[:] = K_func.x.array
    S0.x.scatter_forward()

    # Mass on the vertex rule (lumped), everything else on the Gauss rule.
    dl = ufl.Measure(
        "dx", domain=domain,
        metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
    )
    dq = ufl.Measure(
        "dx", domain=domain,
        metadata={"quadrature_degree": REACTION_QUADRATURE_DEGREE},
    )

    a = u * v * dl + dt * susceptible_scale * dot(DTens * grad(u), grad(v)) * dq

    # The S row of the state system with I = R = W = 0. The conditional
    # compiles to a ternary, so S0/K is never evaluated where K = 0.
    l = S0 * v * dl + dt * v * ufl.conditional(
        ufl.lt(K_func, 0.01),
        -growth_rate * S0,
        growth_rate * S0 * (1 - S0 / K_func),
    ) * dq

    a_form = fem.form(a)
    l_form = fem.form(l)

    A = assemble_matrix(a_form)
    A.assemble()
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    configure_direct_solver(solver, prefix="spinup_")

    # Refilled in place each step.
    b = assemble_vector(l_form)

    # Drift is relative to the peak carrying capacity, per year.
    capacity_scale = comm.allreduce(
        float(np.max(np.abs(K_func.x.array[:n_owned]))) if n_owned else 0.0,
        op=MPI.MAX,
    )
    if capacity_scale <= 0.0:
        capacity_scale = 1.0

    if verbose:
        _print(
            comm,
            f"Susceptible spin-up: up to {max_duration:g} yr "
            f"({total_steps} steps of {dt:g} yr), stopping once the drift "
            f"falls below {drift_tolerance:g}/yr.",
        )

    converged = False
    drift = float("inf")
    completed_steps = 0
    for step in range(total_steps):
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, l_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver.solve(b, S1.x.petsc_vec)
        S1.x.scatter_forward()

        local_change = (
            float(np.max(np.abs(S1.x.array[:n_owned] - S0.x.array[:n_owned])))
            if n_owned
            else 0.0
        )
        drift = comm.allreduce(local_change, op=MPI.MAX) / (dt * capacity_scale)

        S0.x.array[:] = S1.x.array
        S0.x.scatter_forward()
        completed_steps = step + 1

        if verbose and completed_steps % 25 == 0:
            _progress(comm, completed_steps, total_steps, drift)

        if drift < drift_tolerance:
            converged = True
            break

    if verbose:
        _progress(comm, completed_steps, total_steps, drift)
        if comm.rank == 0:
            print()
        if converged:
            _print(
                comm,
                f"Susceptible spin-up converged after "
                f"{completed_steps * dt:g} yr (drift {drift:.3g}/yr).",
            )
        else:
            _print(
                comm,
                f"WARNING: susceptible spin-up hit its {max_duration:g} yr cap "
                f"with drift {drift:.3g}/yr, still above the "
                f"{drift_tolerance:g}/yr tolerance. The saved field is not at "
                "equilibrium; raise susceptible_spinup.max_duration_years.",
            )

    b.destroy()
    A.destroy()
    solver.destroy()

    info = {
        "converged": bool(converged),
        "years_run": float(completed_steps * dt),
        "steps_run": int(completed_steps),
        "final_drift_per_year": float(drift),
    }
    return S0.x.array.copy(), info


def _cache_mismatch(stored, expected):
    """Names of the signature entries that disagree, if any."""
    if not isinstance(stored, dict):
        return ["<unreadable signature>"]
    changed = []
    for key, value in expected.items():
        if key not in stored:
            changed.append(key)
        elif isinstance(value, float):
            if not np.isclose(float(stored[key]), value, rtol=1e-12, atol=0.0):
                changed.append(key)
        elif stored[key] != value:
            changed.append(key)
    return changed


def susceptible_initial_condition(domain, V_scal, DTens, K_func,
                                  diffusivity_values, parameters,
                                  array_path, signature_path,
                                  comm=MPI.COMM_WORLD, recompute=False,
                                  verbose=True):
    """The initial S field: the cached disease-free equilibrium, or a fresh one.

    Returns ``S = K`` when ``susceptible_spinup.enabled`` is false.
    """
    if not bool(parameters["susceptible_spinup"]["enabled"]):
        _print(
            comm,
            "Susceptible spin-up disabled; starting S at the carrying "
            "capacity K.",
        )
        return K_func.x.array.copy()

    array_path = Path(array_path)
    signature_path = Path(signature_path)
    expected = equilibrium_signature(
        domain, K_func.x.array, diffusivity_values, parameters
    )

    reason = None
    if recompute:
        reason = "recompute requested"
    elif not array_path.is_file():
        reason = f"no cached field at {array_path}"
    elif not signature_path.is_file():
        reason = f"cached field has no signature file at {signature_path}"
    else:
        try:
            with signature_path.open("r", encoding="utf-8") as stream:
                stored = json.load(stream)
        except (OSError, json.JSONDecodeError):
            stored = None
        changed = _cache_mismatch((stored or {}).get("inputs"), expected)
        if changed:
            reason = "cached field is stale (changed: " + ", ".join(changed) + ")"
        else:
            values = np.load(array_path)
            if values.size != K_func.x.array.size:
                reason = (
                    f"cached field has {values.size} values but this mesh and "
                    f"function space expect {K_func.x.array.size}"
                )
            else:
                _print(comm, f"Susceptible initial condition: {array_path}")
                result = stored.get("result") or {}
                if result:
                    years = float(result.get("years_run", float("nan")))
                    final_drift = float(
                        result.get("final_drift_per_year", float("nan"))
                    )
                    _print(
                        comm,
                        f"  (spun up for {years:g} yr, final drift "
                        f"{final_drift:.3g}/yr)",
                    )
                return values

    _print(comm, f"Computing susceptible equilibrium: {reason}.")
    values, info = run_susceptible_spinup(
        domain, V_scal, DTens, K_func, parameters, comm=comm, verbose=verbose
    )

    # Nodal arrays are partition-dependent, so only serial runs write a cache.
    if comm.size > 1:
        _print(
            comm,
            "Running on more than one rank, so the equilibrium is not cached "
            "(nodal arrays are partition-dependent). Run once in serial to "
            "write the cache.",
        )
        return values

    np.save(array_path, values)
    with signature_path.open("w", encoding="utf-8") as stream:
        json.dump({"inputs": expected, "result": info}, stream, indent=2)
    _print(comm, f"Saved susceptible equilibrium to {array_path}")
    return values
