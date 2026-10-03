"""Optimal culling control for the CWD reaction-diffusion model.

Each L-BFGS-B evaluation (or projected-gradient iteration) is one forward solve,
recording the trajectory and J(v), followed by one backward adjoint sweep that
accumulates the reduced gradient per control block. See README_solver.md.

Examples
--------
    python CWD_optimal_control.py --mesh-folder BUNDLE --output-folder OUT
    python CWD_optimal_control.py --mesh-folder BUNDLE --gradient-check
    python CWD_optimal_control.py --mesh-folder BUNDLE --forward-only --output-folder BASE
    python CWD_optimal_control.py --mesh-folder BUNDLE --output-folder OUT \
        --initial-control OUT/control_checkpoint.npy      # resume a run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from dolfinx import io
from dolfinx.fem import Function

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from mesh_bundle import add_bundle_arguments, resolve_input_files  # noqa: E402
from shared_parameters import load_parameters  # noqa: E402

from cwd_control_problem import (  # noqa: E402
    CONTROL_NAME,
    CONTROL_SYMBOL,
    COST_SCALE,
    COST_UNITS,
    SQUARE_METRES_PER_SQUARE_KM,
    CWDControlProblem,
)
from effort_budget import EffortBudget  # noqa: E402


############################
# command line
############################
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Solve the culling optimal-control problem for the CWD "
            "reaction-diffusion model by forward/backward sweeps."
        )
    )
    add_bundle_arguments(parser)
    parser.add_argument(
        "--output-folder",
        default="control_outputs",
        help="Folder for optimal-control results (default: control_outputs).",
    )
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override optimal_control.max_iterations.")
    parser.add_argument(
        "--budget-multipliers", default=None, metavar="HISTORY_JSON",
        help=(
            "Warm-start the budget multipliers from a previous run's "
            "optimization_history.json, or from a .npy of per-block values."
        ),
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        metavar="RATE",
        help=(
            "Override optimal_control.control_budget: cap the mass-weighted "
            "spatial mean of v at RATE (1/yr) in every control block, enforced "
            "by an augmented Lagrangian. Requires --optimizer lbfgs. Pass 0 for "
            "no budget."
        ),
    )
    parser.add_argument(
        "--optimizer", choices=("lbfgs", "projected-gradient"), default=None,
        help=(
            "Override optimal_control.optimizer. 'lbfgs' is SciPy's L-BFGS-B "
            "on the reduced problem; 'projected-gradient' is projected "
            "steepest descent with an Armijo line search."
        ),
    )
    parser.add_argument(
        "--control-block-years", type=float, default=None,
        help=(
            "Override optimal_control.control_block_years, the span over which "
            "v is held constant. Pass 0 for one value per time step."
        ),
    )
    parser.add_argument(
        "--control-horizon-years", type=float, default=None,
        help=(
            "Override optimal_control.control_horizon_years, the span over "
            "which v is a variable; the rest of the horizon runs at v = 0. "
            "Pass 0 to control the whole horizon."
        ),
    )
    parser.add_argument("--initial-step-size", type=float, default=None,
                        help="Override optimal_control.initial_step_size.")
    parser.add_argument(
        "--auto-initial-step", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Scale the first Armijo step so the most sensitive node moves "
            "across the full admissible control range (default: on). Disable "
            "to use optimal_control.initial_step_size verbatim."
        ),
    )
    parser.add_argument("--initial-control", default=None,
                        help="Warm start from a saved optimal_control.npy or "
                             "control_checkpoint.npy.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=1,
        help=(
            "Save the current control to <output-folder>/"
            "control_checkpoint.npy every Nth optimizer iteration "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Do not write per-iteration control checkpoints.",
    )
    parser.add_argument(
        "--trajectory-file",
        default=None,
        help=(
            "Memory-map the recorded forward trajectory to this .npy path "
            "instead of holding it in RAM."
        ),
    )
    parser.add_argument("--output-every", type=int, default=3,
                        help="Write every Nth time step to XDMF (default: 3).")
    parser.add_argument("--spinup-series", default=None,
                        help="Folder for the disease spin-up XDMF series "
                             "(default: <output>/spinup). Written only when "
                             "the spin-up runs rather than loading its cache.")
    parser.add_argument("--no-spinup-series", action="store_true",
                        help="Do not write the spin-up series.")
    parser.add_argument("--spinup-log-every", type=int, default=10,
                        help="Write every Nth spin-up step (default: 10).")
    parser.add_argument("--write-adjoint", action="store_true",
                        help="Also write the adjoint fields of the final sweep.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gradient-check", action="store_true",
        help=(
            "Run a Taylor test of the reduced gradient against finite "
            "differences and exit without optimizing."
        ),
    )
    parser.add_argument(
        "--gradient-check-directions", choices=("all", "core"), default="all",
        help=(
            "Which Taylor directions to test: a full-support direction and one "
            "supported on the last control block. 'all' and 'core' currently "
            "select the same two."
        ),
    )
    mode.add_argument(
        "--forward-only", action="store_true",
        help=(
            "Run one forward simulation with a fixed control (v = 0, or "
            "--initial-control), write the state series, and exit. The "
            "trajectory is not stored."
        ),
    )
    mode.add_argument(
        "--heuristic", metavar="EFFORT",
        help=(
            "Score the rule v = min(v_max, k*R), set at the start of each "
            "control block, instead of optimizing; k is found so the mean "
            "effort matches EFFORT. EFFORT is a rate in 1/yr, an "
            "optimization_history.json, or a sweep output root containing "
            "<output-folder name>/optimization_history.json. Writes "
            "heuristic_summary.json and heuristic_control.npy."
        ),
    )
    return parser


############################
# reporting helpers
############################
class Reporter:
    """Root-rank-only printing with an in-place sweep progress bar."""

    def __init__(self, comm):
        self.is_root = comm.rank == 0
        self._bar_open = False

    def line(self, message=""):
        if not self.is_root:
            return
        if self._bar_open:
            sys.stdout.write("\n")
            self._bar_open = False
        print(message, flush=True)

    def bar(self, label, width=32):
        def update(completed, total):
            if not self.is_root:
                return
            if completed != total and completed % 10 != 0:
                return
            fraction = completed / total if total else 1.0
            filled = min(width, int(width * fraction))
            sys.stdout.write(
                f"\r    {label:<22s}[{'=' * filled}{'-' * (width - filled)}] "
                f"{fraction:5.1%}"
            )
            sys.stdout.flush()
            self._bar_open = True
        return update

    def close_bar(self):
        if self.is_root and self._bar_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._bar_open = False


TERM_LABELS = {
    "susceptible_deficit": "susceptible deficit  c1*(S_inf - S)",
    "environment": "environmental load   W",
    "exposed": "exposed              E",
    "preclinical": "preclinical          I",
    "clinical": "clinical             R",
    "targeted_quadratic": "targeted effort      c3*v^2",
    "targeted_l1": "targeted effort      c4*v",
}

TERMINAL_LABELS = dict(TERM_LABELS, susceptible_deficit=
                       "susceptible deficit  mS*(S_inf - S)")


def report_cost_breakdown(reporter, breakdown):
    """Print the term-by-term cost decomposition of J."""
    if not breakdown:
        return
    sections = (
        ("Running state cost", "running_state", TERM_LABELS),
        ("Running control cost", "running_control", TERM_LABELS),
        ("Terminal cost", "terminal", TERMINAL_LABELS),
    )
    reporter.line("")
    reporter.line("Cost decomposition")
    reporter.line(f"  {'term':<38} {'integral':>13} {'weight':>9} "
                  f"{'cost':>14} {'share':>7}")
    reporter.line("  " + "-" * 84)
    for title, key, labels in sections:
        section = breakdown.get(key)
        if not section:
            continue
        reporter.line(f"  {title}")
        for name, entry in section["terms"].items():
            reporter.line(
                f"    {labels.get(name, name):<36} "
                f"{entry['integral']:>13.6e} {entry['weight']:>9.4g} "
                f"{entry['cost']:>14.6e} {100.0 * entry['share']:>6.2f}%"
            )
        reporter.line(
            f"    {'subtotal':<36} {'':>13} {'':>9} "
            f"{section['total']:>14.6e} {100.0 * section['share']:>6.2f}%"
        )
        scale = max(abs(section["total"]), 1.0)
        if abs(section["residual"]) > 1e-9 * scale:
            reporter.line(
                f"    WARNING unaccounted residual {section['residual']:.6e} "
                f"({section['residual'] / scale:.2e} relative)"
            )
    reporter.line("  " + "-" * 84)
    reporter.line(f"  {'J':<38} {'':>13} {'':>9} "
                  f"{breakdown['objective']:>14.6e} {100.0:>6.2f}%")


def report_diagnostics(reporter, diagnostics):
    """Print the outcome diagnostics (incidence, infected area, front)."""
    if not diagnostics:
        return
    reporter.line("")
    reporter.line("Outcome diagnostics")
    reporter.line(f"  {'domain area':<38} "
                  f"{diagnostics['domain_area_km2']:>13.6e} km^2")
    reporter.line(f"  {'cumulative incidence':<38} "
                  f"{diagnostics['cumulative_incidence_deer']:>13.6e} deer "
                  f"infected over the horizon")
    reporter.line(f"  {'infected animals at T':<38} "
                  f"{diagnostics['infected_animals_at_T']:>13.6e} deer")
    reporter.line(f"  {'mean infection radius at T':<38} "
                  f"{diagnostics['mean_infection_radius_km']:>13.6e} km "
                  f"(threshold-free)")
    for threshold, area, front in zip(
        diagnostics["extent_thresholds_deer_per_km2"],
        diagnostics["infected_area_km2"],
        diagnostics["front_radius_km"],
    ):
        reporter.line(
            f"    above {threshold:g} deer/km^2: area {area:>12.6e} km^2, "
            f"front {front:>10.4f} km"
        )


def format_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.1f} {unit}"
        count /= 1024


############################
# gradient verification
############################
def _taylor_test(problem, reporter, control, direction, trajectory, label,
                 budget=None):
    """One Taylor test along ``direction``; returns True if the rate reaches 2.

    R0(eps) = |F(w + eps*d) - F(w)|                  should be O(eps)
    R1(eps) = |F(w + eps*d) - F(w) - eps*<g, d>|     should be O(eps^2)

    F is J, or with a budget the augmented Lagrangian at the current
    multipliers. Returns (passed, number of eps values judged).
    """
    def evaluate(field, store=None):
        cost = problem.forward(field, store)[0]
        if budget is not None:
            cost += budget.value(problem.block_means(field))
        return cost

    base_cost = evaluate(control, trajectory)
    gradient = problem.backward(control, trajectory)
    if budget is not None:
        problem.add_budget_gradient(
            gradient, budget, problem.block_means(control)
        )
    slope = problem.inner_product(gradient, direction)

    reporter.line(f"  [{label}]")
    reporter.line(f"    J(w)            = {base_cost:.12e}")
    reporter.line(f"    <grad J(w), d>  = {slope:.12e}")
    if slope == 0.0:
        reporter.line("    slope is exactly zero along this direction; "
                      "the test is vacuous.")
        return False
    reporter.line("        eps          R0            rate       R1            rate")

    previous = None
    ok = True
    judged = 0
    headroom = 0.0
    noise_floor = 1e-13 * max(1.0, abs(base_cost))
    reporter.line(f"    perturbation at eps: eps * 1.0 in control units "
                  f"(RMS), i.e. 1e-1 moves v by ~0.1")
    for exponent in range(1, 7):
        eps = 10.0 ** (-exponent)
        perturbed_cost = evaluate(control + eps * direction)
        r0 = abs(perturbed_cost - base_cost)
        r1 = abs(perturbed_cost - base_cost - eps * slope)
        if previous is None:
            reporter.line(f"      {eps:8.1e}  {r0:.4e}     --     {r1:.4e}     --")
        else:
            rate0 = np.log(previous[0] / r0) / np.log(10) if r0 > 0 else np.inf
            rate1 = np.log(previous[1] / r1) / np.log(10) if r1 > 0 else np.inf
            reporter.line(
                f"      {eps:8.1e}  {r0:.4e}  {rate0:6.2f}   {r1:.4e}  {rate1:6.2f}"
            )
            # only judge the range where round-off has not yet taken over
            if exponent <= 4 and r1 > noise_floor:
                judged += 1
                headroom = max(headroom, r1 / noise_floor)
                ok = ok and rate1 > 1.6
        previous = (r0, r1)
    if judged > 0:
        reporter.line(
            f"    {judged} eps judged; best R1 stood {headroom:.1e}x above the "
            f"round-off floor {noise_floor:.2e}."
        )
        if headroom < 10.0:
            ok = False
            reporter.line(
                "    MARGINAL: less than a decade above round-off; treat as "
                "inconclusive."
            )
    if judged == 0:
        reporter.line(
            f"    INCONCLUSIVE: every R1 fell below the round-off floor "
            f"{noise_floor:.2e} (= 1e-13*|J|), so no epsilon was judged."
        )
    reporter.line("")
    return ok, judged


def gradient_check(problem, reporter, budget=None, trajectory_file=None,
                   which="all"):
    """Taylor tests of the reduced gradient at the midpoint of the box.

    Tests a random full-support direction and the same direction restricted
    to the last control block. With a budget, the augmented Lagrangian is
    tested with the multipliers seeded away from zero.
    """
    trajectory = problem.new_trajectory(trajectory_file)
    control = problem.new_control(0.5 * (problem.v_min + problem.v_max))

    # Directions are scaled to unit RMS rather than unit L2 norm, so that eps
    # is the perturbation size in the control's own units.
    rng = np.random.default_rng(0)
    ones = np.ones_like(control)
    measure = problem.inner_product(ones, ones)          # |Gamma| * T
    root_measure = float(np.sqrt(max(measure, 1e-30)))

    def unit_rms(field):
        """Scale so the field has RMS 1 over space-time, or None if it is zero."""
        scale = problem.norm(field)
        if scale <= 1e-30:
            return None
        return field * (root_measure / scale)

    joint = unit_rms(rng.standard_normal(control.shape))
    if joint is None:
        raise SystemExit("random direction has zero norm; check the metric.")

    directions = [("full support", joint)]

    # Last block only: the continuation tail reaches the gradient only there.
    if control.shape[0] > 1:
        last = np.zeros_like(joint)
        last[-1] = joint[-1]
        scaled = unit_rms(last)
        if scaled is not None:
            directions.append(
                (f"{CONTROL_SYMBOL}, last block only", scaled)
            )

    if which == "core":
        keep = ("full support", f"{CONTROL_SYMBOL}, last block only")
        directions = [d for d in directions if d[0] in keep]
    elif which != "all":
        raise SystemExit(
            f"unknown --gradient-check-directions {which!r}: use 'all' or 'core'."
        )

    reporter.line(
        f"  Directions scaled to unit RMS (measure |Gamma|*T = {measure:.3e}), "
        f"so eps is the perturbation size in the control's own units."
    )
    reporter.line(
        f"  Testing {len(directions)} direction(s): "
        + ", ".join(label for label, _ in directions)
    )

    if budget is not None:
        means = problem.block_means(control)
        # Nonzero multipliers so every part of the penalty term is tested.
        budget.multipliers = np.maximum(0.25, budget.slopes(means) * 0.5)
        reporter.line(
            f"  Budget active: testing the augmented Lagrangian, not J alone. "
            f"worst overspend at the test control "
            f"{budget.worst_violation(means):+.4e} /yr, "
            f"mu seeded at {float(budget.multipliers.mean()):.4e}"
        )
        reporter.line("")

    results = []
    for label, direction in directions:
        results.append(
            _taylor_test(problem, reporter, control, direction, trajectory,
                         label, budget=budget)
        )

    ok = all(passed for passed, _ in results)
    judged = sum(count for _, count in results)
    if not ok:
        reporter.line(
            "  Taylor test FAILED: an R1 rate fell below 2 (see above)."
        )
    elif judged == 0:
        reporter.line(
            "  Taylor test INCONCLUSIVE: no R1 rose above the round-off floor."
        )
    else:
        reporter.line(
            f"  Taylor test PASSED on every direction ({judged} epsilon "
            f"value(s) judged; first-order remainder converges at ~2)."
        )
    return ok and judged > 0


############################
# crash-safe control checkpoints
############################
def control_path(path, comm):
    """This rank's file for a control array (unchanged in serial)."""
    path = Path(path)
    if comm.size > 1:
        path = path.with_name(f"{path.stem}_rank{comm.rank}{path.suffix}")
    return path


def save_control_array(path, control, comm):
    """Write a control array atomically (temporary file, fsync, rename)."""
    path = control_path(path, comm)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as stream:
        np.save(stream, control, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


class CheckpointWriter:
    """Saves the current control every ``every`` optimizer iterations.

    Writes control_checkpoint.npy and a JSON sidecar with the iteration, cost
    and history. A failed write is reported and does not stop the run.
    """

    def __init__(self, problem, reporter, output_folder, comm, every=1,
                 enabled=True, run_metadata=None):
        self.problem = problem
        self.reporter = reporter
        self.comm = comm
        self.every = max(1, int(every))
        self.enabled = bool(enabled)
        self.control_file = Path(output_folder) / "control_checkpoint.npy"
        self.metadata_file = Path(output_folder) / "control_checkpoint.json"
        self.run_metadata = dict(run_metadata or {})
        self.saved = 0

    def announce(self):
        if not self.enabled:
            self.reporter.line("Checkpoints: disabled (--no-checkpoint)")
            return
        cadence = ("every iteration" if self.every == 1
                   else f"every {self.every} iterations")
        self.reporter.line(
            "Checkpoints: "
            f"{cadence} to {control_path(self.control_file, self.comm).name} "
            f"({self.problem.control_bytes() / 2**20:.1f} MiB), resume with "
            "--initial-control"
        )

    def save(self, control, iteration, costs, history, started=None,
             force=False):
        """Persist ``control`` if this iteration is due for a checkpoint."""
        if not self.enabled:
            return
        if not force and iteration % self.every:
            return
        try:
            path = save_control_array(self.control_file, control, self.comm)
            if self.comm.rank == 0:
                record = dict(self.run_metadata)
                record.update({
                    "iteration": iteration,
                    "cost": costs[0],
                    "state_cost": costs[1],
                    "control_cost": costs[2],
                    "terminal_cost": costs[3],
                    "control_shape": list(control.shape),
                    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": (
                        None if started is None else time.time() - started
                    ),
                    "control_file": path.name,
                    "resume_with": f"--initial-control {path}",
                    "iterations": history,
                })
                self._write_metadata(record)
        except OSError as error:
            self.reporter.line(
                f"    WARNING: could not write checkpoint ({error})"
            )
            return
        self.saved += 1

    def _write_metadata(self, record):
        temporary = self.metadata_file.with_name(
            self.metadata_file.name + ".partial"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.metadata_file)


############################
# L-BFGS-B on the reduced problem
############################
def run_lbfgs(problem, reporter, control, trajectory, gradient, settings,
              history, checkpoint=None, started=None,
              budget=None, iteration_offset=0, ftol=None):
    """Minimize the reduced objective with SciPy's L-BFGS-B.

    Each evaluation is one forward solve and one backward sweep. SciPy works in
    the scaled variable z = D v, D = sqrt(m_i * w_b), so its Euclidean product
    is the L2 product; the gradient passed to it is D g and the box becomes
    D*v_min <= z <= D*v_max.

    Returns (control, stop_reason) and appends to ``history`` in place.
    """
    try:
        from scipy.optimize import Bounds, minimize
    except ImportError as error:
        raise SystemExit(
            "optimizer = 'lbfgs' needs SciPy, which is not importable "
            f"({error}). Install it, or set optimal_control.optimizer to "
            "'projected-gradient' (or pass --optimizer projected-gradient)."
        )

    if problem.comm.size > 1:
        raise SystemExit(
            "optimizer = 'lbfgs' is serial only. Run on one rank, or use "
            "--optimizer projected-gradient."
        )

    shape = control.shape

    max_iterations = int(settings["max_iterations"])
    latest = {"costs": None, "evaluations": 0, "iterations": 0,
              "penalty": 0.0, "means": None}

    # z = scale * v. The metric is (n_blocks, n_owned), which equals the
    # control shape on one rank.
    try:
        scale = np.broadcast_to(problem.control_metric_scale(), shape)
    except ValueError as error:
        raise SystemExit(
            f"control metric has shape {problem.control_metric_scale().shape} "
            f"and does not broadcast to the control shape {shape} ({error})."
        ) from error
    inverse_scale = 1.0 / scale

    lower = problem.v_min * scale
    upper = problem.v_max * scale

    def objective(z):
        """(value, gradient) at z: J, or with a budget the augmented Lagrangian
        at the current multipliers. latest["costs"] keeps J itself."""
        candidate = z.reshape(shape) * inverse_scale
        costs = problem.forward(candidate, trajectory)
        problem.backward(candidate, trajectory, gradient)
        penalty = 0.0
        if budget is not None:
            means = problem.block_means(candidate)
            penalty = COST_SCALE * budget.value(means)
            problem.add_budget_gradient(gradient, budget, means)
            latest["means"] = means
        latest["penalty"] = penalty
        latest["costs"] = costs
        latest["evaluations"] += 1
        # dJ/dz = D * g; the product is a new array, so no copy is needed.
        return costs[0] + penalty, (gradient * scale).ravel()

    def callback(xk):
        # One-argument form, which SciPy accepts across versions.
        latest["iterations"] += 1
        cost, state_cost, control_cost, terminal_cost = latest["costs"]
        # xk is the accepted iterate (not necessarily the last evaluated
        # point), in the scaled variable.
        iterate = problem.project_control(
            xk.reshape(shape) * inverse_scale
        )
        entry_budget = None
        if budget is not None and latest["means"] is not None:
            iterate_means = problem.block_means(iterate)
            entry_budget = {
                "realized": iterate_means.tolist(),
                "worst_violation": budget.worst_violation(iterate_means),
                "penalty_value": budget.value(iterate_means),
                "multipliers": budget.multipliers.tolist(),
            }
        history.append({
            "iteration": iteration_offset + latest["iterations"],
            "cost": cost,
            "budget": entry_budget,
            "state_cost": state_cost,
            "control_cost": control_cost,
            "terminal_cost": terminal_cost,
            "gradient_norm": problem.norm(gradient),
            "projected_gradient_norm": problem.norm(
                problem.project_control(iterate - gradient) - iterate
            ),
            "step_size": None,
            "line_search_trials": latest["evaluations"],
        })
        reporter.line(
            f"  iteration {iteration_offset + latest['iterations']:3d}   "
            f"J = {cost:.10e}   "
            f"||grad L|| = {history[-1]['gradient_norm']:.6e}   "
            f"({latest['evaluations']} evaluations so far)"
        )
        if entry_budget is not None:
            reporter.line(
                f"      budget: worst overspend "
                f"{entry_budget['worst_violation']:+.4e} /yr, "
                f"mean effort "
                f"{float(np.mean(entry_budget['realized'])):.4f} /yr"
            )
        if checkpoint is not None:
            checkpoint.save(iterate, latest["iterations"], latest["costs"],
                            history, started)

    reporter.line(
        f"L-BFGS-B: up to {max_iterations} iterations, "
        f"{int(settings['lbfgs_memory'])} correction pairs"
    )
    result = minimize(
        objective,
        (control * scale).ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(lower.ravel(), upper.ravel()),
        callback=callback,
        options={
            "maxiter": max_iterations,
            "maxcor": int(settings["lbfgs_memory"]),
            "ftol": (float(settings["relative_cost_tolerance"])
                     if ftol is None else float(ftol)),
            # Max-norm of the projected gradient in the scaled variable z.
            "gtol": float(settings["gradient_tolerance"]),
            "maxls": int(settings["armijo_max_backtracks"]),
        },
    )

    control[:] = problem.project_control(
        result.x.reshape(shape) * inverse_scale
    )

    # Optimizer counters and active-set sizes, for the history file.
    scaled_gradient = np.asarray(result.jac, dtype=np.float64).ravel()
    at_lower = int(np.count_nonzero(result.x <= lower.ravel() + 1e-12))
    at_upper = int(np.count_nonzero(result.x >= upper.ravel() - 1e-12))
    optimizer_report = {
        "method": "L-BFGS-B",
        "status": int(result.status),
        "message": (result.message if isinstance(result.message, str)
                    else result.message.decode()),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "evaluations_per_iteration": (float(result.nfev) / result.nit
                                      if result.nit else None),
        "maxcor": int(settings["lbfgs_memory"]),
        "ftol": float(settings["relative_cost_tolerance"]),
        "gtol": float(settings["gradient_tolerance"]),
        "maxls": int(settings["armijo_max_backtracks"]),
        "final_scaled_gradient_max_norm": float(np.abs(scaled_gradient).max())
        if scaled_gradient.size else None,
        "variables": int(result.x.size),
        "at_lower_bound": at_lower,
        "at_upper_bound": at_upper,
        "free_fraction": 1.0 - (at_lower + at_upper) / float(result.x.size),
    }
    reporter.line("")
    reporter.line(
        f"Optimizer: {result.nit} iteration(s), {result.nfev} evaluation(s)"
        + (f" ({optimizer_report['evaluations_per_iteration']:.1f} per "
           "iteration)" if result.nit else "")
    )
    reporter.line(
        f"  scaled |g|_inf = "
        f"{optimizer_report['final_scaled_gradient_max_norm']:.6e}  "
        f"against gtol {optimizer_report['gtol']:.1e}"
    )
    reporter.line(
        f"  active set: {at_lower} at lower bound, {at_upper} at upper, "
        f"{100.0 * optimizer_report['free_fraction']:.1f}% free"
    )

    # The returned iterate may not be the last one evaluated, so re-solve to
    # make the stored trajectory match it.
    costs = problem.forward(control, trajectory, reporter.bar("final forward"),
                            breakdown=True)
    reporter.close_bar()
    history.append({
        "iteration": latest["iterations"] + 1,
        "cost": costs[0],
        "state_cost": costs[1],
        "control_cost": costs[2],
        "terminal_cost": costs[3],
        "gradient_norm": None,
        "projected_gradient_norm": None,
        "step_size": None,
        "line_search_trials": latest["evaluations"],
        "optimizer_report": optimizer_report,
        "cost_breakdown": problem.last_breakdown,
        "diagnostics": problem.last_diagnostics,
    })
    if checkpoint is not None:
        checkpoint.save(control, latest["iterations"] + 1, costs, history,
                        started, force=True)

    message = result.message
    if isinstance(message, bytes):
        message = message.decode("utf-8", "replace")
    return control, (
        f"L-BFGS-B: {message} "
        f"({result.nit} iterations, {latest['evaluations']} evaluations)"
    )


############################
# effort budget: augmented Lagrangian
############################
def run_augmented_lagrangian(problem, reporter, control, trajectory, gradient,
                             settings, history, budget, checkpoint=None,
                             started=None):
    """Outer augmented-Lagrangian loop around warm-started L-BFGS-B solves.

    Each outer iteration minimizes at fixed multipliers, then updates
    mu_b <- max(0, mu_b + rho*g_b). Stops when every block meets its budget
    within the relative tolerance, or on the outer iteration cap, in which case
    the stop reason says the budget was not met.
    """
    outer_maximum = int(settings["control_budget_outer_iterations"])
    tolerance = float(settings["control_budget_tolerance"])
    final_ftol = float(settings["relative_cost_tolerance"])
    # Inner ftol starts loose and tightens geometrically to final_ftol.
    loose_ftol = max(final_ftol, 1.0e-6)

    reporter.line(
        f"Augmented Lagrangian: up to {outer_maximum} outer iterations, "
        f"rho = {budget.penalty:g}, "
        f"relative budget tolerance {tolerance:g}"
    )
    targets = np.asarray(budget.targets, dtype=np.float64)
    if float(targets.min()) == float(targets.max()):
        reporter.line(f"  budget: {float(targets[0]):.4f} /yr in every block")
    else:
        reporter.line(
            f"  budget: {float(targets.min()):.4f}..{float(targets.max()):.4f}"
            f" /yr, per block"
        )

    stop_reason = (
        f"augmented Lagrangian reached its outer iteration cap "
        f"({outer_maximum}) without meeting the budget"
    )
    for outer in range(1, outer_maximum + 1):
        reporter.line("")
        reporter.line(f"Outer iteration {outer} of {outer_maximum}")
        # The last outer iteration always runs at the final tolerance.
        inner_ftol = max(final_ftol, loose_ftol * (0.01 ** (outer - 1)))
        if outer == outer_maximum:
            inner_ftol = final_ftol
        reporter.line(f"  inner ftol {inner_ftol:.1e}")
        control, inner_stop = run_lbfgs(
            problem, reporter, control, trajectory, gradient, settings,
            history, checkpoint, started, budget=budget,
            iteration_offset=len(history) - 1, ftol=inner_ftol,
        )
        means = problem.block_means(control)
        worst = budget.worst_violation(means)
        satisfied = budget.satisfied(means, tolerance)
        violation, multipliers = budget.update(means)
        reporter.line(
            f"  outer {outer}: worst overspend {worst:+.6e} /yr "
            f"({'within' if satisfied else 'outside'} tolerance), "
            f"mean effort {float(np.mean(means)):.4f} /yr"
        )
        reporter.line(
            f"    shadow prices mu: min {float(multipliers.min()):.4e}, "
            f"max {float(multipliers.max()):.4e}, "
            f"mean {float(multipliers.mean()):.4e}"
        )
        reporter.line(f"    inner solve: {inner_stop}")
        if history:
            history[-1]["outer_iteration"] = outer
            history[-1]["budget_after_update"] = budget.report(
                means, CONTROL_NAME
            )
        if satisfied:
            stop_reason = (
                f"augmented Lagrangian met the budget in every block after "
                f"{outer} outer iteration(s) (worst overspend "
                f"{worst:+.3e} /yr, tolerance {tolerance:g} relative)"
            )
            break

    return control, stop_reason


############################
# projected gradient with Armijo backtracking
############################
def armijo_line_search(problem, reporter, control, cost, gradient,
                       step_size, settings, trajectory):
    """Projected-gradient backtracking search.

    Accepts the first step for which

        J(P[v - alpha*g]) <= J(v) + eta * <g, P[v - alpha*g] - v>,

    where P projects onto [v_min, v_max]. On return ``trajectory`` holds the
    states of the returned control. Returns (new_control, costs,
    accepted_step, trials, accepted).
    """
    eta = float(settings["armijo_sufficient_decrease"])
    tau = float(settings["armijo_backtrack_factor"])
    max_backtracks = int(settings["armijo_max_backtracks"])
    minimum_step = float(settings["minimum_step_size"])

    for trial in range(1, max_backtracks + 1):
        candidate = problem.project_control(control - step_size * gradient)
        difference = candidate - control
        predicted = problem.inner_product(gradient, difference)

        candidate_costs = problem.forward(candidate, trajectory)

        if candidate_costs[0] <= cost + eta * predicted:
            return candidate, candidate_costs, step_size, trial, True

        reporter.line(
            f"    backtrack {trial:2d}: alpha = {step_size:.4e}  "
            f"J = {candidate_costs[0]:.10e}  (target <= "
            f"{cost + eta * predicted:.10e})"
        )
        step_size *= tau
        if step_size < minimum_step:
            break

    # No acceptable step: restore the trajectory belonging to the incumbent.
    incumbent_costs = problem.forward(control, trajectory)
    return control, incumbent_costs, step_size, max_backtracks, False


############################
# output
############################
class ControlOutputWriter:
    """XDMF time series for the optimal state, control, and adjoint fields."""

    def __init__(self, problem, output_folder, write_adjoint):
        self.problem = problem
        self.write_adjoint = write_adjoint
        domain = problem.domain
        comm = domain.comm

        def series(name):
            handle = io.XDMFFile(comm, str(output_folder / f"{name}.xdmf"), "w")
            handle.write_mesh(domain)
            return handle

        # Compartment order matches the mixed space: S, E, I, R, W.
        self.state_keys = ("S", "E", "I", "R", "W")
        labels = {
            "S": "Susceptible",
            "E": "Exposed",
            "I": "Preclinical",
            "R": "Clinical",
            "W": "Environment",
        }
        self.files = {key: series(f"CWD_{key}") for key in self.state_keys}
        self.fields = {
            key: Function(problem.V_scal, name=labels[key])
            for key in self.state_keys
        }
        self.control_key = "V_" + CONTROL_NAME
        self.files[self.control_key] = series(f"CWD_control_{CONTROL_SYMBOL}")
        self.fields[self.control_key] = Function(
            problem.V_scal, name=f"{CONTROL_NAME.capitalize()}_Control"
        )

        # S_infinity is constant in time, so it gets one file and one frame.
        with io.XDMFFile(comm, str(output_folder / "CWD_S_infinity.xdmf"),
                         "w") as handle:
            handle.write_mesh(domain)
            handle.write_function(problem.S_infinity)

        self.adjoint_keys = tuple("L" + key for key in self.state_keys)
        if write_adjoint:
            for key in self.adjoint_keys:
                compartment = key[1:]
                self.files[key] = series(f"CWD_adjoint_{compartment}")
                self.fields[key] = Function(
                    problem.V_scal, name=f"Adjoint_{compartment}"
                )

    def write_state(self, state_function, control_slab, t):
        for index, key in enumerate(self.state_keys):
            self.fields[key].interpolate(state_function.sub(index))
            self.files[key].write_function(self.fields[key], t)
        if control_slab is not None:
            key = self.control_key
            self.fields[key].x.array[:] = control_slab
            self.fields[key].x.scatter_forward()
            self.files[key].write_function(self.fields[key], t)

    def write_adjoint_fields(self, adjoint_function, t):
        if not self.write_adjoint:
            return
        for index, key in enumerate(self.adjoint_keys):
            self.fields[key].interpolate(adjoint_function.sub(index))
            self.files[key].write_function(self.fields[key], t)

    def close(self):
        for handle in self.files.values():
            handle.close()


def _state_series_observer(problem, control, writer, output_every):
    """Observer that writes every output_every-th step, plus the endpoints."""

    def observer(step, t):
        if step == 0 or step % output_every == 0 or step == problem.total_steps:
            writer.write_state(problem.Y, problem.control_slab(control, step), t)

    return observer


def write_final_series(problem, control, writer, output_every, reporter,
                       trajectory):
    """Replay the optimal control once, writing the state and control series."""
    observer = _state_series_observer(problem, control, writer, output_every)
    progress = reporter.bar("writing solution")
    problem.forward(control, trajectory, progress, observer)
    reporter.close_bar()


def run_forward_only(problem, reporter, control, output_folder, output_every,
                     control_source):
    """One forward simulation with a fixed control; writes forward_summary.json.

    No adjoint is solved, so no trajectory is stored.
    """
    writer = ControlOutputWriter(problem, output_folder, write_adjoint=False)
    observer = _state_series_observer(problem, control, writer, output_every)

    started = time.time()
    progress = reporter.bar("forward solve")
    cost, state_cost, control_cost, terminal_cost = problem.forward(
        control, None, progress, observer, breakdown=True
    )
    reporter.close_bar()
    writer.close()
    elapsed = time.time() - started

    reporter.line("")
    reporter.line(f"  J = {cost:.10e}   "
                  f"(state {state_cost:.6e} + control {control_cost:.6e}"
                  f" + terminal {terminal_cost:.6e})")
    if control_source == "zero":
        reporter.line("  (uncontrolled baseline)")
    reporter.line(f"  Elapsed: {elapsed:.1f} s")
    report_cost_breakdown(reporter, problem.last_breakdown)
    report_diagnostics(reporter, problem.last_diagnostics)

    summary = {
        "mode": "forward_only",
        "control_source": control_source,
        "final_time_years": problem.T,
        "time_step_years": problem.dt,
        "total_steps": problem.total_steps,
        "objective": cost,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "terminal_cost": terminal_cost,
        "cost_breakdown": problem.last_breakdown,
        "diagnostics": problem.last_diagnostics,
        "cost_weights": {"c1_susceptible_deficit": problem.c1,
                         "c2_environment": problem.c2,
                         "c3_targeted": problem.c3,
                         "c4_targeted_l1": problem.c4,
                         **problem.terminal_weight_summary()},
        "seconds": elapsed,
    }
    if problem.comm.rank == 0:
        with (output_folder / "forward_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(summary, stream, indent=2)
    problem.comm.barrier()
    reporter.line(f"  Wrote {output_folder / 'forward_summary.json'}")
    return 0


def heuristic_target(source, output_folder):
    """The realized mean effort the heuristic must match, and where it came from.

    `source` is a rate, an optimization_history.json, or a sweep output root
    holding one folder per point named like this run's output folder.
    """
    try:
        return float(source), "given"
    except ValueError:
        pass
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / output_folder.name / "optimization_history.json"
    if not path.is_file():
        raise SystemExit(f"--heuristic: no optimization history at {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    return float(record["controls"][CONTROL_NAME]["space_time_mean"]), str(path)


def clinical_policy(problem, control, gain):
    """Observer that sets each block to min(v_max, gain*R) at the block's start.

    forward() calls the observer after each state update and before loading
    the next step's control, so writing control[b] when step t_{b-1} is
    reached makes block b act on the clinical field a survey would see then.
    """
    control[:] = 0.0

    def observer(step, t):
        if step >= problem.control_steps:
            return
        block = problem.block_of_step[step]
        if step == 0 or problem.block_of_step[step - 1] != block:
            clinical = problem.Y.x.array[problem.dof_map_R]
            control[block] = np.clip(gain * clinical, problem.v_min,
                                     problem.v_max)

    return observer


def run_heuristic(problem, reporter, source, output_folder, output_every):
    """Score v = min(v_max, k*R) at a target mean effort.

    k is found by bracketing and a log-space secant with bisection fallback,
    one forward solve per evaluation.
    """
    target, target_source = heuristic_target(source, output_folder)
    if target <= 0.0:
        raise SystemExit(f"--heuristic: target effort must be positive, got "
                         f"{target}")
    reporter.line(f"Clinical-density heuristic, matching mean effort "
                  f"{target:.6f} /yr ({target_source})")

    control = problem.new_control(0.0)
    ones = np.ones_like(control)
    measure = problem.inner_product(ones, ones)
    evaluations = []

    def mean_effort(gain):
        problem.forward(control, None, None,
                        clinical_policy(problem, control, gain))
        mean = problem.inner_product(control, ones) / measure
        evaluations.append({"gain": gain, "mean_effort": mean})
        reporter.line(f"    k = {gain:.6e}   mean v = {mean:.6f}")
        return mean

    started = time.time()
    # First guess: the gain that spends the target on the handover field.
    clinical = problem.Y_initial.x.array[problem.dof_map_R][: problem.n_owned]
    mean_clinical = problem._global_sum(
        float(clinical @ problem.lumped_mass)
    ) / problem.domain_measure
    gain = target / max(mean_clinical, 1e-300)

    low, high = (0.0, 0.0), None
    value = mean_effort(gain)
    while value < target:
        low = (gain, value)
        if len(evaluations) > 20:
            raise SystemExit(
                f"--heuristic: mean effort {value:.6f} at k = {gain:.3e} is "
                f"still below the target {target:.6f}; the target is not "
                f"reachable under v_max = {problem.v_max:g}."
            )
        gain *= 4.0
        value = mean_effort(gain)
    high = (gain, value)
    if low[0] == 0.0:
        gain = high[0] / 4.0
        while True:
            value = mean_effort(gain)
            if value < target:
                low = (gain, value)
                break
            high = (gain, value)
            gain /= 4.0

    tolerance = 2e-3
    for _ in range(25):
        if abs(high[1] / target - 1.0) <= tolerance:
            gain = high[0]
            break
        if abs(low[1] / target - 1.0) <= tolerance:
            gain = low[0]
            break
        # secant in log k against log mean, kept inside the bracket
        log_low, log_high = np.log(low[0]), np.log(high[0])
        slope = (np.log(high[1]) - np.log(low[1])) / (log_high - log_low)
        trial = log_low + (np.log(target) - np.log(low[1])) / slope
        if not log_low < trial < log_high:
            trial = 0.5 * (log_low + log_high)
        gain = float(np.exp(trial))
        value = mean_effort(gain)
        if value < target:
            low = (gain, value)
        else:
            high = (gain, value)
    else:
        reporter.line("  WARNING: effort match did not reach tolerance; "
                      "reporting the last gain.")

    writer = ControlOutputWriter(problem, output_folder, write_adjoint=False)
    policy = clinical_policy(problem, control, gain)
    series = _state_series_observer(problem, control, writer, output_every)

    def observer(step, t):
        policy(step, t)
        series(step, t)

    cost, state_cost, control_cost, terminal_cost = problem.forward(
        control, None, reporter.bar("final forward solve"), observer,
        breakdown=True,
    )
    reporter.close_bar()
    writer.close()
    elapsed = time.time() - started

    mean = problem.inner_product(control, ones) / measure
    at_zero = (control <= problem.v_min + 1e-12).astype(np.float64)
    at_bound = (control >= problem.v_max - 1e-12).astype(np.float64)
    reporter.line("")
    reporter.line(f"  k = {gain:.6e}   mean v = {mean:.6f} /yr "
                  f"(target {target:.6f})")
    reporter.line(f"  J = {cost:.10e}   (state {state_cost:.6e} + control "
                  f"{control_cost:.6e} + terminal {terminal_cost:.6e})")
    report_cost_breakdown(reporter, problem.last_breakdown)
    report_diagnostics(reporter, problem.last_diagnostics)

    save_control_array(output_folder / "heuristic_control.npy", control,
                       problem.comm)
    summary = {
        "mode": "clinical_density_heuristic",
        "policy": "v_b = min(v_max, k * R(t_{b-1})), block-constant",
        "target_mean_effort": target,
        "target_source": target_source,
        "gain": gain,
        "controls": {CONTROL_NAME: {
            "space_time_mean": mean,
            "fraction_at_lower_bound":
                problem.inner_product(at_zero, ones) / measure,
            "fraction_at_upper_bound":
                problem.inner_product(at_bound, ones) / measure,
            "bounds": [problem.v_min, problem.v_max],
        }},
        "objective": cost,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "terminal_cost": terminal_cost,
        "cost_breakdown": problem.last_breakdown,
        "diagnostics": problem.last_diagnostics,
        "evaluations": evaluations,
        "seconds": elapsed,
    }
    if problem.comm.rank == 0:
        with (output_folder / "heuristic_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(summary, stream, indent=2)
    problem.comm.barrier()
    reporter.line(f"  Wrote {output_folder / 'heuristic_summary.json'}")
    return 0


############################
# main
############################
def main():
    comm = MPI.COMM_WORLD
    args = build_parser().parse_args()
    reporter = Reporter(comm)

    mesh_path, classes_path, diffusivity_path, parameter_path = \
        resolve_input_files(args)
    spatial_parameters = load_parameters(parameter_path)
    settings = spatial_parameters["optimal_control"]

    if args.max_iterations is not None:
        settings["max_iterations"] = args.max_iterations
    if args.initial_step_size is not None:
        settings["initial_step_size"] = args.initial_step_size
    if args.optimizer is not None:
        settings["optimizer"] = args.optimizer
    if args.control_block_years is not None:
        settings["control_block_years"] = (
            None if args.control_block_years <= 0.0 else args.control_block_years
        )
    if args.control_horizon_years is not None:
        settings["control_horizon_years"] = (
            None if args.control_horizon_years <= 0.0
            else args.control_horizon_years
        )
    optimizer_name = settings["optimizer"]

    output_folder = Path(args.output_folder).expanduser().resolve()
    if comm.rank == 0:
        output_folder.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    reporter.line(f"Mesh:                   {mesh_path}")
    reporter.line(f"Land-cover classes:     {classes_path}")
    reporter.line(f"Land-cover diffusivity: {diffusivity_path}")
    reporter.line(f"Model parameters:       {parameter_path}")
    reporter.line(f"Outputs:                {output_folder}")
    reporter.line("")

    problem = CWDControlProblem(
        mesh_path, classes_path, diffusivity_path, spatial_parameters, comm,
        equilibrium_array_path=args.susceptible_equilibrium,
        recompute_equilibrium=args.recompute_equilibrium,
        disease_state_path=args.disease_state,
        recompute_disease_state=args.recompute_disease_state,
        spinup_series_folder=(
            None if args.no_spinup_series
            else Path(args.spinup_series).expanduser().resolve()
            if args.spinup_series else output_folder / "spinup"
        ),
        spinup_log_every=args.spinup_log_every,
    )

    reporter.line(
        f"Scalar degrees of freedom: {problem.n_global}   "
        f"time steps: {problem.total_steps}   dt = {problem.dt} yr   "
        f"T = {problem.T} yr"
    )
    if problem.tail_steps:
        reporter.line(
            f"Control horizon: {problem.control_horizon:g} yr "
            f"({problem.control_steps} steps); continuation tail "
            f"{problem.tail_years:g} yr ({problem.tail_steps} steps) "
            "simulated at v = 0."
        )
    reporter.line(
        f"Implicit operator: {problem.solver.distinct_factorizations} "
        f"factorization(s) of {problem.n_global} dofs "
        f"({problem.solver.factor_package}), W block diagonal"
    )
    reporter.line(
        f"Cost weights: c1 (susceptible deficit) = {problem.c1}, "
        f"c2 (environment) = {problem.c2}"
    )
    reporter.line(
        f"  targeted v:     c3 = {problem.c3}"
        + (f", c4 (L1) = {problem.c4}" if problem.c4 > 0 else "")
        + f"   box [{problem.v_min}, {problem.v_max}] per year, acting on R"
    )
    if problem.c4 > 0 and problem.c3 > 0:
        reporter.line(
            f"  elastic net on v: zero wherever lambda_R*R < {problem.c4}, "
            f"then ramping at 1/(2*c3) = {0.5 / problem.c3:.4g}"
        )
    elif problem.c4 > 0:
        reporter.line(
            f"  pure L1 on v (bang-bang): v_max wherever lambda_R*R > "
            f"{problem.c4}, else 0"
        )
    if problem.n_blocks == problem.total_steps:
        blocking = "one value per time step"
    else:
        blocking = (
            f"piecewise constant on {problem.block_years:g}-year blocks "
            f"({problem.block_step_counts[0]:.0f} steps each)"
        )
    reporter.line(
        f"Control: {problem.n_blocks} blocks "
        f"x {problem.n_global} nodes "
        f"= {problem.control_bytes() / 2**20:.1f} MiB per copy, {blocking}"
    )
    if optimizer_name == "lbfgs":
        pairs = int(settings["lbfgs_memory"])
        reporter.line(
            f"Optimizer: L-BFGS-B, {pairs} correction pairs "
            f"(~{(2 * pairs + 5) * problem.control_bytes() / 2**30:.2f} GiB "
            "workspace)"
        )
    else:
        reporter.line("Optimizer: projected gradient with Armijo line search")
    if args.forward_only or args.heuristic is not None:
        reporter.line(
            "Stored forward trajectory: none "
            f"(no adjoint is solved, saving "
            f"{format_bytes(problem.trajectory_bytes())} per rank)"
        )
    else:
        reporter.line(
            "Stored forward trajectory: "
            f"{format_bytes(problem.trajectory_bytes())} per rank"
            + (f" (memory-mapped to {args.trajectory_file})"
               if args.trajectory_file else " (in memory)")
        )
    reporter.line("")

    ############################
    # initial control
    ############################
    initial_value = float(settings["initial_control"])
    control = problem.new_control(initial_value)
    if initial_value == 0.0:
        control_source = "zero"
    else:
        control_source = f"uniform {CONTROL_SYMBOL} = {initial_value:g}"
    if args.initial_control is not None:
        initial_control_path = control_path(
            Path(args.initial_control).expanduser().resolve(), comm
        )
        loaded = np.load(initial_control_path)
        if loaded.shape != control.shape:
            raise ValueError(
                f"--initial-control has shape {loaded.shape}, but this "
                f"problem expects {control.shape}: {problem.n_blocks} blocks "
                f"of {problem.block_years:g} yr x {problem.n_local} nodes. A "
                "different leading axis means a different "
                "optimal_control.control_block_years."
            )
        control = problem.project_control(loaded.astype(np.float64))
        control_source = str(args.initial_control)
        reporter.line(f"Warm start from {initial_control_path}")

    ############################
    # forward-only mode
    ############################
    if args.forward_only:
        if control_source == "zero":
            reporter.line(
                "Forward simulation only, uncontrolled (v = 0)"
            )
        else:
            reporter.line(f"Forward simulation only, replaying {control_source}")
        reporter.line("")
        return run_forward_only(problem, reporter, control, output_folder,
                                args.output_every, control_source)
    if args.heuristic is not None:
        return run_heuristic(problem, reporter, args.heuristic, output_folder,
                             max(1, args.output_every))

    ############################
    # effort budget (off when control_budget is null and no --budget)
    ############################
    budget_setting = settings.get("control_budget")
    if args.budget is not None:
        budget_setting = args.budget if args.budget > 0.0 else None
    budget = None
    if budget_setting is not None:
        if optimizer_name != "lbfgs":
            raise SystemExit(
                "an effort budget needs optimizer = 'lbfgs'."
            )
        initial_multipliers = None
        if args.budget_multipliers:
            source = Path(args.budget_multipliers).expanduser()
            if not source.exists():
                raise SystemExit(
                    f"--budget-multipliers: no such file {source}"
                )
            if source.suffix == ".npy":
                initial_multipliers = np.load(source)
            else:
                record = json.loads(source.read_text(encoding="utf-8"))
                previous = record.get("effort_budget")
                if not previous or "multipliers" not in previous:
                    raise SystemExit(
                        f"{source} has no effort_budget.multipliers."
                    )
                initial_multipliers = np.asarray(
                    previous["multipliers"], dtype=np.float64
                )
                reporter.line(
                    f"Warm-starting budget multipliers from {source} "
                    f"(previous budget {previous['targets'][0]:g}/yr, "
                    f"mean mu {float(np.mean(initial_multipliers)):.4f})"
                )
        budget = EffortBudget(
            budget_setting,
            penalty=float(settings["control_budget_penalty"]),
            multipliers=initial_multipliers,
        ).resolve(problem.n_blocks).set_measure(
            problem.domain_measure, problem.block_weights
        )
        low = float(np.min(budget.targets))
        high = float(np.max(budget.targets))
        span = (f"{low:g} /yr in every block" if low == high
                else f"{low:g}..{high:g} /yr, per block")
        reporter.line(
            f"Effort budget on v: mass-weighted spatial mean <= {span}"
        )
        reporter.line(
            f"  augmented Lagrangian, rho = {budget.penalty:g}, "
            f"up to {int(settings['control_budget_outer_iterations'])} outer "
            f"iterations, relative tolerance "
            f"{float(settings['control_budget_tolerance']):g}"
        )

    ############################
    # gradient check mode
    ############################
    if args.gradient_check:
        reporter.line("Gradient check (Taylor test of the discrete adjoint)")
        reporter.line("")
        passed = gradient_check(problem, reporter, budget=budget,
                                trajectory_file=args.trajectory_file,
                                which=args.gradient_check_directions)
        return 0 if passed else 1

    ############################
    # forward/backward optimization loop
    ############################
    trajectory = problem.new_trajectory(args.trajectory_file)
    gradient = np.empty_like(control)

    checkpoint = CheckpointWriter(
        problem, reporter, output_folder, comm,
        every=args.checkpoint_every,
        enabled=not args.no_checkpoint,
        run_metadata={
            "optimizer": optimizer_name,
            "control_bounds": {
                CONTROL_NAME: [problem.v_min, problem.v_max]
            },
            "control_blocks": {
                "count": problem.n_blocks,
                "block_years": problem.block_years,
            },
            "initial_control": control_source,
            "completed": False,
        },
    )
    checkpoint.announce()
    reporter.line("")

    reporter.line("Initial forward solve")
    started = time.time()
    cost, state_cost, control_cost, terminal_cost = problem.forward(
        control, trajectory, reporter.bar("forward solve")
    )
    reporter.close_bar()
    reporter.line(
        f"  J = {cost:.10e}   (state {state_cost:.6e} + control "
        f"{control_cost:.6e} + terminal {terminal_cost:.6e})"
    )
    reporter.line("")

    step_size = float(settings["initial_step_size"])
    growth = float(settings["armijo_step_growth"])
    gradient_tolerance = float(settings["gradient_tolerance"])
    cost_tolerance = float(settings["relative_cost_tolerance"])
    max_iterations = int(settings["max_iterations"])

    history = [{
        "iteration": 0,
        "cost": cost,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "terminal_cost": terminal_cost,
        "gradient_norm": None,
        "projected_gradient_norm": None,
        "step_size": None,
        "line_search_trials": 0,
    }]
    stop_reason = f"reached max_iterations = {max_iterations}"

    if optimizer_name == "lbfgs" and budget is not None:
        control, stop_reason = run_augmented_lagrangian(
            problem, reporter, control, trajectory, gradient, settings,
            history, budget, checkpoint, started,
        )
        cost = history[-1]["cost"]
        gradient_iterations = ()
    elif optimizer_name == "lbfgs":
        control, stop_reason = run_lbfgs(
            problem, reporter, control, trajectory, gradient, settings,
            history, checkpoint, started,
        )
        cost = history[-1]["cost"]
        gradient_iterations = ()
    else:
        gradient_iterations = range(1, max_iterations + 1)

    for iteration in gradient_iterations:
        reporter.line(f"Iteration {iteration}")

        problem.backward(control, trajectory, gradient,
                         reporter.bar("backward solve"))
        reporter.close_bar()

        gradient_norm = problem.norm(gradient)
        # Stationarity measure ||P[v - g] - v||.
        projected = problem.project_control(control - gradient) - control
        projected_norm = problem.norm(projected)
        reporter.line(
            f"    ||grad J|| = {gradient_norm:.6e}   "
            f"||P[v - grad J] - v|| = {projected_norm:.6e}"
        )

        if projected_norm <= gradient_tolerance:
            stop_reason = (
                f"projected gradient norm {projected_norm:.3e} <= "
                f"gradient_tolerance {gradient_tolerance:.3e}"
            )
            reporter.line(f"  Converged: {stop_reason}")
            break

        if iteration == 1 and args.auto_initial_step:
            # First step moves the most sensitive node across the whole box.
            local_peak = (
                float(np.abs(gradient[:, : problem.n_owned]).max())
                if problem.n_owned else 0.0
            )
            peak = comm.allreduce(local_peak, op=MPI.MAX)
            if peak > 0.0:
                step_size = (problem.v_max - problem.v_min) / peak
                reporter.line(f"    auto initial step alpha = {step_size:.4e}")
        elif iteration > 1:
            step_size *= growth
        control, costs, step_size, trials, accepted = armijo_line_search(
            problem, reporter, control, cost, gradient, step_size,
            settings, trajectory
        )
        new_cost, state_cost, control_cost, terminal_cost = costs

        if not accepted:
            stop_reason = (
                "Armijo line search failed to find a decreasing step "
                f"(alpha fell below {settings['minimum_step_size']:.1e})"
            )
            reporter.line(f"  Stopping: {stop_reason}")
            break

        relative_decrease = abs(cost - new_cost) / max(abs(cost), 1e-300)
        reporter.line(
            f"    accepted alpha = {step_size:.4e} after {trials} "
            f"evaluation(s);  J = {new_cost:.10e}  "
            f"(relative decrease {relative_decrease:.3e})"
        )

        history.append({
            "iteration": iteration,
            "cost": new_cost,
            "state_cost": state_cost,
            "control_cost": control_cost,
            "terminal_cost": terminal_cost,
            "gradient_norm": gradient_norm,
            "projected_gradient_norm": projected_norm,
            "step_size": step_size,
            "line_search_trials": trials,
        })
        cost = new_cost
        checkpoint.save(control, iteration, costs, history, started)

        if relative_decrease <= cost_tolerance:
            stop_reason = (
                f"relative cost decrease {relative_decrease:.3e} <= "
                f"relative_cost_tolerance {cost_tolerance:.3e}"
            )
            reporter.line(f"  Converged: {stop_reason}")
            break

    elapsed = time.time() - started
    reporter.line("")
    reporter.line(f"Stopped: {stop_reason}")

    ############################
    # final report and output
    ############################
    # `trajectory` already belongs to `control` on every exit path.
    final = history[-1]
    final_cost = final["cost"]
    initial_cost = history[0]["cost"]

    if initial_cost > 0:
        reduction = f"{100.0 * (1.0 - final_cost / initial_cost):.2f}% reduction"
    else:
        reduction = "no positive baseline to compare against"
    reporter.line(
        f"J(initial) = {initial_cost:.10e}  ->  J(final) = {final_cost:.10e}   "
        f"({reduction})"
    )
    reporter.line(
        f"  final split: state {final['state_cost']:.6e} + "
        f"control {final['control_cost']:.6e} + "
        f"terminal {final.get('terminal_cost', 0.0):.6e}"
    )
    report_cost_breakdown(reporter, problem.last_breakdown)
    report_diagnostics(reporter, problem.last_diagnostics)

    # Control summary: peak, mean, and space-time fractions at each bound.
    ones_slab = np.ones_like(control)
    lower = problem.v_min
    upper = problem.v_max
    local_max = (
        float(control[:, : problem.n_owned].max()) if problem.n_owned else 0.0
    )
    peak = comm.allreduce(local_max, op=MPI.MAX)
    measure = problem.inner_product(ones_slab, ones_slab)   # |Gamma| * T
    mean = problem.inner_product(control, ones_slab) / measure
    at_zero = (control <= lower + 1e-12).astype(np.float64)
    at_bound = (control >= upper - 1e-12).astype(np.float64)
    zero_fraction = problem.inner_product(at_zero, ones_slab) / measure
    bound_fraction = problem.inner_product(at_bound, ones_slab) / measure

    # Area at v_min in every block (never treated), as a fraction and in km^2.
    never_area = problem.never_treated_area(control)
    never_fraction = never_area / problem.domain_measure
    never_area_km2 = never_area / SQUARE_METRES_PER_SQUARE_KM
    domain_km2 = problem.domain_measure / SQUARE_METRES_PER_SQUARE_KM

    control_report = {
        CONTROL_NAME: {
            "peak": peak,
            "space_time_mean": mean,
            "fraction_at_lower_bound": zero_fraction,
            "fraction_at_upper_bound": bound_fraction,
            "never_treated_area_fraction": never_fraction,
            "never_treated_area_km2": never_area_km2,
            "domain_area_km2": domain_km2,
            "bounds": [lower, upper],
        }
    }
    reporter.line(
        f"  {CONTROL_SYMBOL} ({CONTROL_NAME}): peak {peak:.4f} /yr, "
        f"space-time mean {mean:.4f} /yr"
    )
    reporter.line(
        f"      {100.0 * zero_fraction:.1f}% of space-time at "
        f"{CONTROL_SYMBOL} = {lower:g} (no effort), "
        f"{100.0 * bound_fraction:.1f}% at {CONTROL_SYMBOL} = {upper:g}"
    )
    reporter.line(
        f"      {100.0 * never_fraction:.1f}% of the domain "
        f"({never_area_km2:.1f} of {domain_km2:.1f} km^2) is at "
        f"{CONTROL_SYMBOL} = {lower:g} in EVERY block -- never visited"
    )
    reporter.line(f"  elapsed: {elapsed:.1f} s")
    reporter.line("")

    save_control_array(output_folder / "optimal_control.npy", control, comm)
    if comm.rank == 0:
        with (output_folder / "optimization_history.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "stop_reason": stop_reason,
                    "elapsed_seconds": elapsed,
                    "cost_weights": {
                        "c1_susceptible_deficit": problem.c1,
                        "c2_environment": problem.c2,
                        "c3_targeted": problem.c3,
                        "c4_targeted_l1": problem.c4,
                    },
                    # None without a budget; otherwise includes the
                    # multipliers and the worst violation.
                    "effort_budget": (
                        budget.report(
                            problem.block_means(control),
                            CONTROL_NAME,
                        ) if budget is not None else None
                    ),
                    "terminal_weights": problem.terminal_weight_summary(),
                    "controls": control_report,
                    "optimizer": optimizer_name,
                    "control_blocks": {
                        "count": problem.n_blocks,
                        "block_years": problem.block_years,
                        "steps_per_block": problem.block_step_counts.tolist(),
                    },
                    "time": {"final_time_years": problem.T,
                             "time_step_years": problem.dt,
                             "total_steps": problem.total_steps,
                             "control_horizon_years": problem.control_horizon,
                             "control_steps": problem.control_steps,
                             "tail_years": problem.tail_years,
                             "tail_steps": problem.tail_steps},
                    "cost_breakdown": problem.last_breakdown,
                    "diagnostics": problem.last_diagnostics,
                    "iterations": history,
                },
                stream,
                indent=2,
            )
    comm.barrier()

    # Mark the checkpoint as belonging to a completed run.
    checkpoint.run_metadata["completed"] = True
    checkpoint.run_metadata["stop_reason"] = stop_reason
    checkpoint.save(
        control, final["iteration"],
        (final_cost, final["state_cost"], final["control_cost"],
         final.get("terminal_cost", 0.0)),
        history, started, force=True,
    )

    output_every = max(1, args.output_every)
    writer = ControlOutputWriter(problem, output_folder, args.write_adjoint)
    write_final_series(problem, control, writer, output_every, reporter,
                       trajectory)

    if args.write_adjoint:
        def adjoint_observer(step, t):
            if step % output_every == 0 or step == problem.total_steps:
                writer.write_adjoint_fields(problem.Lam, t)

        progress = reporter.bar("writing adjoint")
        problem.backward(control, trajectory, gradient, progress,
                         adjoint_observer)
        reporter.close_bar()

    writer.close()

    reporter.line(f"Done. Results written to {output_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
