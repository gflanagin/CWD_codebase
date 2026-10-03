"""Load and validate model parameters shared by meshing, the PDE solver, and the
optimal-control driver."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMETERS = ROOT / "parameters.json"


# Defaults filled in for keys missing from optimal_control.
OPTIMAL_CONTROL_DEFAULTS = {
    "cost_susceptible_deficit": 10.0,
    "cost_environment": 0.5,
    "cost_control": 2.0,
    "cost_control_l1": 0.0,
    # Effort budget (1/yr): null, a scalar, or one value per block.
    "control_budget": None,
    # Augmented-Lagrangian penalty rho.
    "control_budget_penalty": 40.0,
    "control_budget_outer_iterations": 8,
    # Relative tolerance on each block's budget.
    "control_budget_tolerance": 1.0e-3,
    # Scale on the closed-form terminal cost.
    "terminal_cost_scale": 1.0,
    # Generations of onward transmission in the terminal weights (0 or 1).
    "terminal_tail_generations": 0,
    # null: one control value per time step.
    "control_block_years": 1.0,
    # null: the control is a variable over the whole horizon.
    "control_horizon_years": None,
    # "lbfgs" or "projected-gradient".
    "optimizer": "lbfgs",
    "lbfgs_memory": 10,
    "control_minimum": 0.0,
    "control_maximum": 2.0,
    "initial_control": 0.0,
    "max_iterations": 40,
    "initial_step_size": 1.0,
    "armijo_sufficient_decrease": 1.0e-4,
    "armijo_backtrack_factor": 0.5,
    "armijo_step_growth": 2.0,
    "armijo_max_backtracks": 25,
    "minimum_step_size": 1.0e-12,
    "gradient_tolerance": 1.0e-8,
    "relative_cost_tolerance": 1.0e-8,
}


# Defaults filled in for keys missing from susceptible_spinup.
SUSCEPTIBLE_SPINUP_DEFAULTS = {
    "enabled": True,
    "max_duration_years": 60.0,
    # null: use time.time_step_years.
    "time_step_years": None,
    # Stop when max|dS/dt| falls below this fraction of peak K per year.
    "drift_tolerance": 1.0e-5,
}


# Defaults filled in for keys missing from disease_spinup.
DISEASE_SPINUP_DEFAULTS = {
    "enabled": True,
    # "duration", "prevalence" or "detection".
    "mode": "duration",
    "duration_years": 20.0,
    # Used only when mode == "prevalence".
    "target_prevalence": 0.05,
    # Used only when mode == "detection": stop when
    # int int R^detection_exponent dx dt >= detection_threshold.
    "detection_threshold": 61.0,
    "detection_exponent": 1.0,
    "max_duration_years": 400.0,
    # null means "use time.time_step_years".
    "time_step_years": None,
}


def _validate_disease_spinup(parameters):
    """Fill in disease-spinup defaults in place, then range-check them."""
    spinup = dict(DISEASE_SPINUP_DEFAULTS)
    spinup.update(parameters.get("disease_spinup", {}))
    parameters["disease_spinup"] = spinup

    target = float(spinup["target_prevalence"])
    if not 0.0 < target < 1.0:
        raise ValueError(
            "disease_spinup.target_prevalence must lie strictly between 0 and 1"
        )
    if spinup["mode"] not in ("duration", "prevalence", "detection"):
        raise ValueError(
            "disease_spinup.mode must be 'duration', 'prevalence' or "
            f"'detection', not {spinup['mode']!r}"
        )
    if float(spinup["detection_threshold"]) <= 0:
        raise ValueError(
            "disease_spinup.detection_threshold must be positive"
        )
    if float(spinup["detection_exponent"]) < 1.0:
        raise ValueError(
            "disease_spinup.detection_exponent must be at least 1"
        )
    if float(spinup["duration_years"]) <= 0:
        raise ValueError("disease_spinup.duration_years must be positive")
    if float(spinup["max_duration_years"]) <= 0:
        raise ValueError("disease_spinup.max_duration_years must be positive")
    if spinup["time_step_years"] is not None:
        if float(spinup["time_step_years"]) <= 0:
            raise ValueError(
                "disease_spinup.time_step_years must be positive, or null to "
                "reuse time.time_step_years"
            )
    return spinup


def _validate_susceptible_spinup(parameters):
    """Fill in spin-up defaults in place, then range-check them."""
    spinup = dict(SUSCEPTIBLE_SPINUP_DEFAULTS)
    spinup.update(parameters.get("susceptible_spinup", {}))
    parameters["susceptible_spinup"] = spinup

    if float(spinup["max_duration_years"]) <= 0:
        raise ValueError(
            "susceptible_spinup.max_duration_years must be positive"
        )
    if spinup["time_step_years"] is not None:
        if float(spinup["time_step_years"]) <= 0:
            raise ValueError(
                "susceptible_spinup.time_step_years must be positive, or null "
                "to reuse time.time_step_years"
            )
    if float(spinup["drift_tolerance"]) <= 0:
        raise ValueError("susceptible_spinup.drift_tolerance must be positive")
    return spinup


def _validate_optimal_control(parameters):
    """Fill in optimal-control defaults in place, then range-check them."""
    supplied = parameters.get("optimal_control", {})

    control = dict(OPTIMAL_CONTROL_DEFAULTS)
    control.update(supplied)
    parameters["optimal_control"] = control

    for name in ("cost_susceptible_deficit", "cost_environment"):
        if float(control[name]) < 0:
            raise ValueError(f"optimal_control.{name} cannot be negative")

    control_costs = (
        ("cost_control", "cost_control_l1", "targeted control v"),
    )
    for quadratic, linear, label in control_costs:
        if float(control[quadratic]) < 0:
            raise ValueError(f"optimal_control.{quadratic} cannot be negative")
        if float(control[linear]) < 0:
            raise ValueError(f"optimal_control.{linear} cannot be negative")
        if float(control[quadratic]) + float(control[linear]) <= 0:
            raise ValueError(
                f"at least one of optimal_control.{quadratic} (quadratic) and "
                f"optimal_control.{linear} must be positive; with neither, "
                f"nothing penalizes effort on the {label} and the optimizer "
                "will simply saturate its bound"
            )

    # Budget settings are checked here so a bad file fails before meshing.
    budget = control["control_budget"]
    if budget is not None:
        values = budget if isinstance(budget, (list, tuple)) else [budget]
        if not values:
            raise ValueError(
                "optimal_control.control_budget cannot be an empty list; use "
                "null for an unconstrained run"
            )
        for entry in values:
            if float(entry) < 0:
                raise ValueError(
                    "optimal_control.control_budget cannot be negative"
                )
        if float(control["control_budget_penalty"]) <= 0:
            raise ValueError(
                "optimal_control.control_budget_penalty (rho) must be positive"
            )
        if int(control["control_budget_outer_iterations"]) < 1:
            raise ValueError(
                "optimal_control.control_budget_outer_iterations must be at "
                "least 1"
            )
        if float(control["control_budget_tolerance"]) <= 0:
            raise ValueError(
                "optimal_control.control_budget_tolerance must be positive"
            )
        if control["optimizer"] != "lbfgs":
            raise ValueError(
                "optimal_control.control_budget needs optimizer = 'lbfgs'"
            )
        if float(control["cost_control_l1"]) > 0:
            raise ValueError(
                "optimal_control.control_budget cannot be combined with "
                "cost_control_l1 > 0; set cost_control_l1 to 0 for a budgeted "
                "run."
            )

    if float(control["terminal_cost_scale"]) < 0:
        raise ValueError(
            "optimal_control.terminal_cost_scale cannot be negative"
        )

    generations = control["terminal_tail_generations"]
    if generations not in (0, 1) or isinstance(generations, bool):
        raise ValueError(
            "optimal_control.terminal_tail_generations must be 0 or 1, not "
            f"{generations!r}"
        )
    if generations == 1 and float(control["cost_environment"]) > 0:
        warnings.warn(
            "optimal_control: terminal_tail_generations = 1 together with "
            "cost_environment > 0 prices W twice.",
            stacklevel=2,
        )

    # Box constraint and starting iterate.
    boxes = (
        ("control_minimum", "control_maximum", "initial_control",
         "targeted control v"),
    )
    for lower, upper, initial, label in boxes:
        if float(control[lower]) < 0:
            raise ValueError(
                f"optimal_control.{lower} cannot be negative"
            )
        if float(control[upper]) < float(control[lower]):
            raise ValueError(
                f"optimal_control.{upper} must not be below {lower}"
            )
        # lower == upper is allowed and pins the control.
        if not float(control[lower]) <= float(control[initial]) <= float(
            control[upper]
        ):
            raise ValueError(
                f"optimal_control.{initial} must lie in [{lower}, {upper}]"
            )

    block_years = control["control_block_years"]
    if block_years is not None and float(block_years) <= 0:
        raise ValueError(
            "optimal_control.control_block_years must be positive, or null to "
            "give the control one value per time step"
        )
    horizon = control["control_horizon_years"]
    if horizon is not None:
        if float(horizon) <= 0:
            raise ValueError(
                "optimal_control.control_horizon_years must be positive, or "
                "null to make the control a variable over the whole horizon"
            )
        # Checked against T and dt in cwd_control_problem.py.
        if float(control["terminal_cost_scale"]) > 0:
            warnings.warn(
                "optimal_control: control_horizon_years is set and "
                "terminal_cost_scale > 0, so the continuation tail and the "
                "closed-form terminal cost are both applied. Set "
                "terminal_cost_scale to 0.",
                stacklevel=2,
            )

    if control["optimizer"] not in ("lbfgs", "projected-gradient"):
        raise ValueError(
            "optimal_control.optimizer must be 'lbfgs' or "
            f"'projected-gradient', not {control['optimizer']!r}"
        )
    if int(control["lbfgs_memory"]) < 1:
        raise ValueError("optimal_control.lbfgs_memory must be at least 1")
    if int(control["max_iterations"]) < 1:
        raise ValueError("optimal_control.max_iterations must be at least 1")
    if float(control["initial_step_size"]) <= 0:
        raise ValueError("optimal_control.initial_step_size must be positive")
    if not 0 < float(control["armijo_sufficient_decrease"]) < 1:
        raise ValueError(
            "optimal_control.armijo_sufficient_decrease must be in (0, 1)"
        )
    if not 0 < float(control["armijo_backtrack_factor"]) < 1:
        raise ValueError(
            "optimal_control.armijo_backtrack_factor must be in (0, 1)"
        )
    if float(control["armijo_step_growth"]) < 1:
        raise ValueError("optimal_control.armijo_step_growth must be at least 1")
    if int(control["armijo_max_backtracks"]) < 1:
        raise ValueError(
            "optimal_control.armijo_max_backtracks must be at least 1"
        )
    if float(control["minimum_step_size"]) <= 0:
        raise ValueError("optimal_control.minimum_step_size must be positive")
    if float(control["gradient_tolerance"]) < 0:
        raise ValueError("optimal_control.gradient_tolerance cannot be negative")
    if float(control["relative_cost_tolerance"]) < 0:
        raise ValueError(
            "optimal_control.relative_cost_tolerance cannot be negative"
        )
    return control


def _validate_compartment_scales(parameters):
    """Check that the four mobility multipliers are present and positive."""
    scales = parameters["diffusion_tensor"]["compartment_scales"]

    required = ("susceptible", "exposed", "preclinical", "clinical")
    missing = [name for name in required if name not in scales]
    if missing:
        raise ValueError(
            "diffusion_tensor.compartment_scales is missing "
            + ", ".join(missing)
            + "; the diffusive compartments are " + ", ".join(required)
            + " (W carries no diffusion term and takes no scale)"
        )
    for name in required:
        if float(scales[name]) <= 0:
            raise ValueError(
                f"diffusion_tensor.compartment_scales.{name} must be "
                "positive"
            )
    return scales


def _validate_reaction(parameters):
    """Check that the reaction rates are present and in range."""
    reaction = parameters["reaction"]

    positive = (
        "exposed_to_shedding_rate",
        "preclinical_to_clinical_rate",
        "clinical_removal_rate",
        "environmental_decay_rate",
    )
    for name in positive:
        if name not in reaction:
            raise ValueError(
                f"reaction.{name} is missing"
            )
        if float(reaction[name]) <= 0:
            raise ValueError(f"reaction.{name} must be positive")

    nonnegative = (
        "direct_transmission_rate_preclinical",
        "direct_transmission_rate_clinical",
        "intrinsic_growth_rate",
        "environmental_shedding_rate_preclinical",
        "environmental_shedding_rate_clinical",
        "environmental_transmission_rate",
    )
    for name in nonnegative:
        if name not in reaction:
            raise ValueError(
                f"reaction.{name} is missing"
            )
        if float(reaction[name]) < 0:
            raise ValueError(f"reaction.{name} cannot be negative")

    return reaction


def reproduction_numbers(parameters):
    """(R0_direct, R0_environmental) at the best-habitat carrying capacity."""
    reaction = parameters["reaction"]
    K0 = float(parameters["land_cover"]["carrying_capacity_base"])

    sigma_2 = float(reaction["preclinical_to_clinical_rate"])
    alpha = float(reaction["clinical_removal_rate"])
    delta = float(reaction["environmental_decay_rate"])
    beta_2 = float(reaction["environmental_transmission_rate"])
    beta_1I = float(reaction["direct_transmission_rate_preclinical"])
    beta_1R = float(reaction["direct_transmission_rate_clinical"])
    rho_I = float(reaction["environmental_shedding_rate_preclinical"])
    rho_R = float(reaction["environmental_shedding_rate_clinical"])

    R0_direct = K0 * (beta_1I / sigma_2 + beta_1R / alpha)
    R0_environmental = (K0 * beta_2 / delta) * (rho_I / sigma_2 + rho_R / alpha)
    return R0_direct, R0_environmental


def _load_json(path):
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_parameters(path=DEFAULT_PARAMETERS):
    parameters = _load_json(path)
    mesh = parameters["mesh"]
    tensor = parameters["diffusion_tensor"]
    land_cover = parameters["land_cover"]
    time = parameters["time"]
    infected_gaussians = parameters["initial_conditions"]["infected_gaussians"]

    if float(time["final_time_years"]) <= 0:
        raise ValueError("time.final_time_years must be positive")
    if float(time["time_step_years"]) <= 0:
        raise ValueError("time.time_step_years must be positive")

    seed_mode = parameters["initial_conditions"]["seed_mode"]
    if seed_mode not in ("staged", "exposed"):
        raise ValueError(
            f"initial_conditions.seed_mode = {seed_mode!r} must be "
            "'staged' or 'exposed'"
        )

    if not isinstance(infected_gaussians, list):
        raise ValueError("initial_conditions.infected_gaussians must be a list")
    for index, gaussian in enumerate(infected_gaussians):
        center = gaussian.get("center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].center "
                "must contain [x, y]"
            )
        if not all(np.isfinite(float(value)) for value in center):
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].center "
                "must contain finite values"
            )
        if not np.isfinite(float(gaussian["mean"])) or float(gaussian["mean"]) < 0:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].mean "
                "must be finite and nonnegative"
            )
        if not np.isfinite(float(gaussian["standard_deviation"])) or float(
            gaussian["standard_deviation"]
        ) <= 0:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].standard_deviation "
                "must be finite and positive"
            )

    if int(mesh["downsample"]) < 1:
        raise ValueError("mesh.downsample must be at least 1")
    if float(mesh["smoothing_sigma"]) < 0:
        raise ValueError("mesh.smoothing_sigma cannot be negative")
    if float(mesh["hmax"]) <= 0:
        raise ValueError("mesh.hmax must be positive")
    if not 0 < float(tensor["isotropy"]) <= 1:
        raise ValueError("diffusion_tensor.isotropy must be in (0, 1]")
    if not isinstance(tensor.get("trace_preserving", False), bool):
        raise ValueError("diffusion_tensor.trace_preserving must be true or false")
    if float(tensor["kappa"]) <= 0:
        raise ValueError("diffusion_tensor.kappa must be positive")
    if float(tensor["slope_penalty"]) < 0:
        raise ValueError("diffusion_tensor.slope_penalty cannot be negative")
    if float(land_cover["carrying_capacity_base"]) < 0:
        raise ValueError("land_cover.carrying_capacity_base cannot be negative")

    class_codes = set(land_cover["class_names"])
    for mapping_name in ("diffusivity", "carrying_capacity_fractions"):
        missing = set(land_cover[mapping_name]) - class_codes
        if missing:
            raise ValueError(
                f"land_cover.class_names is missing codes used by {mapping_name}: "
                + ", ".join(sorted(missing))
            )

    _validate_reaction(parameters)
    _validate_compartment_scales(parameters)
    _validate_susceptible_spinup(parameters)
    _validate_disease_spinup(parameters)
    _validate_optimal_control(parameters)

    return parameters


def land_cover_to_diffusivity(land_cover_array, spatial_parameters):
    config = spatial_parameters["land_cover"]
    mapping = {int(code): float(value) for code, value in config["diffusivity"].items()}
    result = np.full(
        np.asarray(land_cover_array).shape,
        float(config["diffusivity_default"]),
        dtype=np.float64,
    )
    for code, value in mapping.items():
        result[np.asarray(land_cover_array) == code] = value
    return result


def land_cover_to_carrying_capacity(land_cover_array, spatial_parameters):
    config = spatial_parameters["land_cover"]
    base = float(config["carrying_capacity_base"])
    fractions = {
        int(code): float(value)
        for code, value in config["carrying_capacity_fractions"].items()
    }
    result = np.full(
        np.asarray(land_cover_array).shape,
        base * float(config["carrying_capacity_fraction_default"]),
        dtype=np.float64,
    )
    for code, fraction in fractions.items():
        result[np.asarray(land_cover_array) == code] = base * fraction
    return result
