"""Disease spin-up: advance the uncontrolled epizootic to the handover state.

Runs the uncontrolled system (v = 0) from the seeded foci until the stopping
rule in parameters["disease_spinup"] is met (a fixed duration, a domain-wide
prevalence, or an accumulated clinical burden), and caches the resulting mixed
state beside the mesh bundle under a signature of every input it depends on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

DISEASE_ARRAY_NAME = "disease_initial_state.npy"
DISEASE_SIGNATURE_NAME = "disease_initial_state.json"


def default_disease_paths(mesh_path):
    """Where the cached state and its signature live for a given mesh."""
    folder = Path(mesh_path).expanduser().resolve().parent
    return folder / DISEASE_ARRAY_NAME, folder / DISEASE_SIGNATURE_NAME


def _digest(array):
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def spinup_time_step(parameters):
    """Spin-up step; must equal time.time_step_years (or be null).

    The caller's stepper is built at time.time_step_years, so any other value
    would only change the bookkeeping here.
    """
    configured = parameters["disease_spinup"]["time_step_years"]
    solve_step = float(parameters["time"]["time_step_years"])
    if configured is None:
        return solve_step
    if abs(float(configured) - solve_step) > 1.0e-12:
        raise ValueError(
            "disease_spinup.time_step_years "
            f"({float(configured)}) must equal time.time_step_years "
            f"({solve_step}), or be null. To share one handover state across "
            "runs with different time steps, pass it with --disease-state."
        )
    return solve_step


def disease_signature(domain, carrying_capacity, diffusivity,
                      susceptible_equilibrium, parameters):
    """The inputs the spun-up state depends on, for cache validation."""
    tensor = parameters["diffusion_tensor"]
    reaction = parameters["reaction"]
    spinup = parameters["disease_spinup"]
    scales = tensor["compartment_scales"]
    foci = parameters["initial_conditions"]["infected_gaussians"]
    signature = {
        "geometry_digest": _digest(domain.geometry.x),
        "carrying_capacity_digest": _digest(carrying_capacity),
        "diffusivity_digest": _digest(diffusivity),
        "susceptible_equilibrium_digest": _digest(susceptible_equilibrium),
        "node_count": int(np.asarray(carrying_capacity).size),
        "reaction": {key: float(value) for key, value in reaction.items()
                     if key != "_notes"},
        "kappa": float(tensor["kappa"]),
        "isotropy": float(tensor["isotropy"]),
        "slope_penalty": float(tensor["slope_penalty"]),
        "compartment_scales": {key: float(value)
                               for key, value in scales.items()},
        "carrying_capacity_base": float(
            parameters["land_cover"]["carrying_capacity_base"]
        ),
        "infected_gaussians": [
            {
                "center": [float(value) for value in focus["center"]],
                "mean": float(focus["mean"]),
                "standard_deviation": float(focus["standard_deviation"]),
            }
            for focus in foci
        ],
        "seed_mode": str(
            parameters["initial_conditions"]["seed_mode"]
        ),
        # Only the value used by the active stopping rule.
        "mode": str(spinup["mode"]),
        "stopping_value": {
            "duration": lambda: float(spinup["duration_years"]),
            "prevalence": lambda: float(spinup["target_prevalence"]),
            "detection": lambda: float(spinup["detection_threshold"]),
        }[str(spinup["mode"])](),
        "time_step_years": float(spinup_time_step(parameters)),
    }
    # Optional keys are added only when they apply, so existing caches stay
    # valid.
    if str(spinup["mode"]) == "detection":
        signature["detection_exponent"] = float(spinup["detection_exponent"])
    if bool(tensor.get("trace_preserving", False)):
        signature["trace_preserving"] = True
    return signature


def _print(comm, message):
    if comm.rank == 0:
        print(message)


def _progress(comm, years, max_years, prevalence, target, width=40):
    if comm.rank != 0:
        return
    # Bar driven by log prevalence, the stopping quantity.
    if prevalence <= 0.0:
        fraction = 0.0
    else:
        fraction = min(1.0, np.log(prevalence / 1e-6) / np.log(target / 1e-6))
    filled = min(width, int(width * max(0.0, fraction)))
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\rDisease spin-up:     [{bar}] prevalence {prevalence:8.4%} "
        f"of {target:.2%} target   t = {years:7.1f} yr"
    )
    sys.stdout.flush()


def _progress_burden(comm, years, accumulated, target, prevalence, width=40):
    if comm.rank != 0:
        return
    # Bar driven by log(burden / target) over six decades.
    if accumulated <= 0.0 or target <= 0.0:
        fraction = 0.0
    else:
        fraction = 1.0 + np.log10(accumulated / target) / 6.0
    filled = min(width, int(width * min(1.0, max(0.0, fraction))))
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\rDisease spin-up:     [{bar}] burden {accumulated:10.4g} "
        f"of {target:g}   t = {years:7.1f} yr   prevalence {prevalence:7.3%}"
    )
    sys.stdout.flush()


def _progress_time(comm, years, total_years, prevalence, width=40):
    if comm.rank != 0:
        return
    fraction = min(1.0, years / total_years) if total_years > 0 else 1.0
    filled = min(width, int(width * max(0.0, fraction)))
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\rDisease spin-up:     [{bar}] t = {years:7.1f} of {total_years:g} yr"
        f"   prevalence {prevalence:8.4%}"
    )
    sys.stdout.flush()


def run_disease_spinup(advance, prevalence, read_state, parameters,
                       comm=MPI.COMM_WORLD, verbose=True, observer=None,
                       clinical_burden=None):
    """Advance the uncontrolled system until the stopping rule is met.

    The caller supplies the stepping machinery:

      advance()         -- advance the state one time step, in place, with v = 0
      prevalence()      -- domain-wide prevalence, collective
      read_state()      -- the current mixed nodal array, as a copy
      clinical_burden() -- int R^p dx over the domain (detection mode only)

    observer (optional) is called as observer(step, years) after each step.

    Returns ``(values, info)``.
    """
    spinup = parameters["disease_spinup"]
    mode = str(spinup["mode"])
    dt = spinup_time_step(parameters)

    if mode == "duration":
        return _run_for_duration(advance, prevalence, read_state, spinup, dt,
                                 comm, verbose, observer)
    if mode == "detection":
        if clinical_burden is None:
            raise ValueError(
                "disease_spinup.mode is 'detection', which stops on the "
                "accumulated clinical burden, but the caller supplied no "
                "clinical_burden() callable."
            )
        return _run_to_detection(advance, prevalence, clinical_burden,
                                 read_state, spinup, dt, comm, verbose,
                                 observer)
    return _run_to_prevalence(advance, prevalence, read_state, spinup, dt,
                              comm, verbose, observer)


def _run_for_duration(advance, prevalence, read_state, spinup, dt, comm,
                      verbose, observer=None):
    """Advance a fixed number of years, independent of the size of the mesh."""
    years = float(spinup["duration_years"])
    total_steps = max(1, int(round(years / dt)))
    start = prevalence()
    if verbose:
        _print(
            comm,
            f"Disease spin-up: advancing the uncontrolled system for "
            f"{years:g} yr ({total_steps} steps of {dt:g} yr), from "
            f"{start:.4%} prevalence.",
        )

    if observer is not None:
        observer(0, 0.0)
    for step in range(total_steps):
        advance()
        if observer is not None:
            observer(step + 1, (step + 1) * dt)
        # Prevalence is evaluated only for the progress line.
        if verbose and (step + 1) % 200 == 0:
            _progress_time(comm, (step + 1) * dt, years, prevalence())

    current = prevalence()
    if verbose:
        _progress_time(comm, total_steps * dt, years, current)
        if comm.rank == 0:
            print()
        _print(
            comm,
            f"Disease spin-up finished {years:g} yr at {current:.3%} "
            "domain-wide prevalence.",
        )
        if current <= start:
            _print(
                comm,
                "  WARNING: prevalence did not grow over the spin-up. Check "
                "that the landscape-mean R0 exceeds one.",
            )

    return read_state(), {
        "converged": True,
        "mode": "duration",
        "years_run": float(total_steps * dt),
        "steps_run": int(total_steps),
        "initial_prevalence": float(start),
        "final_prevalence": float(current),
        "duration_years": float(years),
    }


def _run_to_prevalence(advance, prevalence, read_state, spinup, dt, comm,
                       verbose, observer=None):
    """Advance until domain-wide prevalence reaches the target."""
    target = float(spinup["target_prevalence"])
    max_years = float(spinup["max_duration_years"])
    total_steps = max(1, int(round(max_years / dt)))

    start = prevalence()
    if verbose:
        _print(
            comm,
            f"Disease spin-up: advancing the uncontrolled system to "
            f"{target:.2%} prevalence, from {start:.4%}, up to {max_years:g} yr "
            f"({total_steps} steps of {dt:g} yr).",
        )
    if start >= target:
        # The foci already exceed the target.
        if verbose:
            _print(comm, "  Initial prevalence already at or above target.")
        return read_state(), {
            "converged": True, "mode": "prevalence",
            "years_run": 0.0, "steps_run": 0,
            "initial_prevalence": float(start),
            "final_prevalence": float(start),
            "target_prevalence": float(target),
        }

    reached = False
    current = start
    completed = 0
    if observer is not None:
        observer(0, 0.0)
    for step in range(total_steps):
        advance()
        completed = step + 1
        if observer is not None:
            observer(completed, completed * dt)
        current = prevalence()
        if verbose and completed % 200 == 0:
            _progress(comm, completed * dt, max_years, current, target)
        if current >= target:
            reached = True
            break

    if verbose:
        _progress(comm, completed * dt, max_years, current, target)
        if comm.rank == 0:
            print()
        if reached:
            _print(
                comm,
                f"Disease spin-up reached {current:.3%} prevalence after "
                f"{completed * dt:g} yr.",
            )
        else:
            _print(
                comm,
                f"WARNING: disease spin-up hit its {max_years:g} yr cap at "
                f"{current:.3%} prevalence, short of the {target:.2%} target. "
                "The saved state is not the intended initial condition; raise "
                "disease_spinup.max_duration_years, lower the target, or check "
                "that R0 exceeds one for these parameters.",
            )

    info = {
        "converged": bool(reached),
        "mode": "prevalence",
        "years_run": float(completed * dt),
        "steps_run": int(completed),
        "initial_prevalence": float(start),
        "final_prevalence": float(current),
        "target_prevalence": float(target),
    }
    return read_state(), info


def _run_to_detection(advance, prevalence, clinical_burden, read_state, spinup,
                      dt, comm, verbose, observer=None):
    """Advance until the accumulated clinical burden reaches the threshold.

    The stopping rule is

        int_0^t int_Gamma R(x,s)^p dx ds  >=  B,

    accumulated with the trapezoid rule in time.

    The returned state is the one after the step on which the threshold was
    crossed (``years_run``); ``crossing_years`` interpolates within that step
    and is a diagnostic only.
    """
    target = float(spinup["detection_threshold"])
    exponent = float(spinup["detection_exponent"])
    max_years = float(spinup["max_duration_years"])
    total_steps = max(1, int(round(max_years / dt)))

    start = prevalence()
    rate = clinical_burden()
    accumulated = 0.0
    if verbose:
        _print(
            comm,
            f"Disease spin-up: advancing the uncontrolled system until "
            f"int int R^{exponent:g} dx dt reaches {target:g}, from "
            f"{start:.4%} prevalence, up to {max_years:g} yr "
            f"({total_steps} steps of {dt:g} yr).",
        )

    reached = False
    completed = 0
    before = 0.0
    current = start
    if observer is not None:
        observer(0, 0.0)
    for step in range(total_steps):
        advance()
        completed = step + 1
        if observer is not None:
            observer(completed, completed * dt)
        # Trapezoid rule in time.
        following = clinical_burden()
        before = accumulated
        accumulated += 0.5 * (rate + following) * dt
        rate = following
        if verbose and completed % 200 == 0:
            current = prevalence()
            _progress_burden(comm, completed * dt, accumulated, target, current)
        if accumulated >= target:
            reached = True
            break

    current = prevalence()
    crossing = float(completed * dt)
    if reached and accumulated > before:
        crossing = (completed - 1 + (target - before)
                    / (accumulated - before)) * dt

    if verbose:
        _progress_burden(comm, completed * dt, accumulated, target, current)
        if comm.rank == 0:
            print()
        if reached:
            _print(
                comm,
                f"Disease spin-up reached an accumulated clinical burden of "
                f"{accumulated:.4g} (target {target:g}) after "
                f"{completed * dt:g} yr, at {current:.3%} prevalence.",
            )
        else:
            _print(
                comm,
                f"WARNING: disease spin-up hit its {max_years:g} yr cap at an "
                f"accumulated clinical burden of {accumulated:.4g}, short of "
                f"the {target:g} target. Raise "
                "disease_spinup.max_duration_years, lower the threshold, or "
                "check that R0 exceeds one for these parameters.",
            )

    return read_state(), {
        "converged": bool(reached),
        "mode": "detection",
        "years_run": float(completed * dt),
        "steps_run": int(completed),
        "initial_prevalence": float(start),
        "final_prevalence": float(current),
        "detection_threshold": float(target),
        "detection_exponent": float(exponent),
        "accumulated_burden": float(accumulated),
        "crossing_years": float(crossing),
    }


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


def disease_initial_state(expected, expected_size, compute,
                          array_path, signature_path,
                          comm=MPI.COMM_WORLD, recompute=False, verbose=True):
    """The initial mixed state: the cached spun-up epizootic, or a fresh one.

    ``compute()`` runs the spin-up and returns ``(values, info)``; it is only
    called when the cache is absent or stale. ``expected`` is the signature from
    :func:`disease_signature`.
    """
    array_path = Path(array_path)
    signature_path = Path(signature_path)

    reason = None
    if recompute:
        reason = "recompute requested"
    elif not array_path.is_file():
        reason = f"no cached state at {array_path}"
    elif not signature_path.is_file():
        reason = f"cached state has no signature file at {signature_path}"
    else:
        try:
            with signature_path.open("r", encoding="utf-8") as stream:
                stored = json.load(stream)
        except (OSError, json.JSONDecodeError):
            stored = None
        changed = _cache_mismatch((stored or {}).get("inputs"), expected)
        if changed:
            reason = "cached state is stale (changed: " + ", ".join(changed) + ")"
        else:
            values = np.load(array_path)
            if values.size != expected_size:
                reason = (
                    f"cached state has {values.size} values but this mesh and "
                    f"function space expect {expected_size}"
                )
            else:
                _print(comm, f"Disease initial condition: {array_path}")
                result = stored.get("result") or {}
                if result:
                    _print(
                        comm,
                        f"  (spun up {float(result.get('years_run', float('nan'))):g} yr "
                        f"to {float(result.get('final_prevalence', float('nan'))):.3%} "
                        "prevalence)",
                    )
                    if not result.get("converged", True):
                        _print(
                            comm,
                            "  WARNING: that run did not reach its target "
                            "prevalence.",
                        )
                return values

    _print(comm, f"Computing disease initial state: {reason}.")
    values, info = compute()

    # Nodal arrays are partition-dependent, so only serial runs write a cache.
    if comm.size > 1:
        _print(
            comm,
            "Running on more than one rank, so the spun-up state is not cached "
            "(nodal arrays are partition-dependent). Run once in serial to "
            "write the cache.",
        )
        return values

    np.save(array_path, values)
    with signature_path.open("w", encoding="utf-8") as stream:
        json.dump({"inputs": expected, "result": info}, stream, indent=2)
    _print(comm, f"Saved disease initial state to {array_path}")
    return values
