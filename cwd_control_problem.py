"""Controlled CWD state system, its discrete adjoint, and the reduced gradient.

Builds the mesh, the five-component mixed space (S, E, I, R, W), the
land-cover fields, the diffusion tensor and every variational form, and runs
the forward and backward sweeps. CWD_optimal_control.py drives it.

Identifiers follow the paper: uS..uW are the state components, psiS..psiW the
test functions, lamS..lamW the adjoint components, and beta1I, beta1R, beta2,
sigma1, sigma2, alpha, rhoI, rhoR, delta the rates. The linear system of each
step is solved blockwise by utilities/block_solver.py, whose factorizations
are reused by the backward sweep.
"""

from __future__ import annotations


import sys
from pathlib import Path

import numpy as np
import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io.gmsh import read_from_msh
from ufl import (
    CellNormal, Identity, TestFunction,
    as_vector, dot, dx, exp, outer, split,
)

UTILITIES_DIR = Path(__file__).resolve().parent / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from block_solver import CompartmentBlockSolver  # noqa: E402
from mesh_bundle import explain_nodal_array_mismatch  # noqa: E402
from shared_parameters import land_cover_to_carrying_capacity  # noqa: E402
from susceptible_spinup import (  # noqa: E402
    default_equilibrium_paths,
    susceptible_initial_condition,
)
from spinup_series import SpinupSeriesWriter  # noqa: E402
from effort_budget import EffortBudget  # noqa: E402
from disease_spinup import (  # noqa: E402
    default_disease_paths,
    disease_initial_state,
    disease_signature,
    run_disease_spinup,
)


# Quadrature degree shared by the forward, adjoint, cost and gradient forms.
QUADRATURE_DEGREE = 4


# Mesh coordinates are in metres and densities per km^2.
SQUARE_METRES_PER_SQUARE_KM = 1.0e6

# J and its gradient are multiplied by this so that J is in deer*km^2*yr.
COST_SCALE = 1.0 / SQUARE_METRES_PER_SQUARE_KM
COST_UNITS = "deer*km^2*yr (J divided by SQUARE_METRES_PER_SQUARE_KM)"


CONTROL_NAME = "targeted"
CONTROL_SYMBOL = "v"


class CWDControlProblem:
    """Everything the optimization loop needs, assembled once."""

    def __init__(self, mesh_path, classes_path, diffusivity_path,
                 spatial_parameters, comm=MPI.COMM_WORLD,
                 equilibrium_array_path=None, equilibrium_signature_path=None,
                 recompute_equilibrium=False, disease_state_path=None,
                 recompute_disease_state=False, spinup_series_folder=None,
                 spinup_log_every=10):
        self.comm = comm
        self.parameters = spatial_parameters

        # Right-hand-side vectors, one per compiled form, reassembled in place.
        self._rhs_buffers = {}

        # Cached disease-free susceptible field, beside the mesh by default.
        default_array, default_signature = default_equilibrium_paths(mesh_path)
        self.equilibrium_array_path = (
            Path(equilibrium_array_path).expanduser().resolve()
            if equilibrium_array_path is not None
            else default_array
        )
        self.equilibrium_signature_path = (
            Path(equilibrium_signature_path).expanduser().resolve()
            if equilibrium_signature_path is not None
            else (
                default_signature
                if equilibrium_array_path is None
                else self.equilibrium_array_path.with_suffix(".json")
            )
        )
        self.recompute_equilibrium = bool(recompute_equilibrium)

        # Cached spun-up epizootic, beside the mesh by default.
        default_disease_array, default_disease_signature = (
            default_disease_paths(mesh_path)
        )
        self.disease_array_path = (
            Path(disease_state_path).expanduser().resolve()
            if disease_state_path is not None
            else default_disease_array
        )
        self.disease_signature_path = (
            default_disease_signature
            if disease_state_path is None
            else self.disease_array_path.with_suffix(".json")
        )
        self.recompute_disease_state = bool(recompute_disease_state)
        self.spinup_series_folder = (
            None if spinup_series_folder is None
            else Path(spinup_series_folder)
        )
        self.spinup_log_every = int(spinup_log_every)

        tensor_parameters = spatial_parameters["diffusion_tensor"]
        time_parameters = spatial_parameters["time"]
        kinetic_parameters = spatial_parameters["reaction"]
        # _build_initial_state derives the staged seeding ratios from these.
        self.reaction_parameters = kinetic_parameters
        control_parameters = spatial_parameters["optimal_control"]
        self.initial_condition_parameters = spatial_parameters["initial_conditions"]

        ############################
        # model, time, and cost parameters
        ############################
        self.T = float(time_parameters["final_time_years"])
        self.dt = float(time_parameters["time_step_years"])
        self.total_steps = int(self.T / self.dt)
        if self.total_steps < 1:
            raise ValueError(
                "time.final_time_years / time.time_step_years must be at least 1"
            )

        isotropy = float(tensor_parameters["isotropy"])
        kappa = float(tensor_parameters["kappa"])
        scales = tensor_parameters["compartment_scales"]
        DS = float(scales["susceptible"])
        DE = float(scales["exposed"])
        DI = float(scales["preclinical"])
        DR = float(scales["clinical"])
        slope_penalty = float(tensor_parameters["slope_penalty"])
        trace_preserving = bool(tensor_parameters.get("trace_preserving", False))

        sigma1 = float(kinetic_parameters["exposed_to_shedding_rate"])
        sigma2 = float(kinetic_parameters["preclinical_to_clinical_rate"])
        alpha = float(kinetic_parameters["clinical_removal_rate"])
        r = float(kinetic_parameters["intrinsic_growth_rate"])

        beta1I = float(kinetic_parameters["direct_transmission_rate_preclinical"])
        beta1R = float(kinetic_parameters["direct_transmission_rate_clinical"])
        rhoI = float(kinetic_parameters["environmental_shedding_rate_preclinical"])
        rhoR = float(kinetic_parameters["environmental_shedding_rate_clinical"])
        delta = float(kinetic_parameters["environmental_decay_rate"])
        beta2 = float(kinetic_parameters["environmental_transmission_rate"])

        ############################
        # control time blocks: v is constant over blocks of steps
        ############################
        block_years = control_parameters["control_block_years"]
        self.block_years = self.dt if block_years is None else float(block_years)

        ############################
        # control horizon: v is a variable on [0, T_c] and zero on (T_c, T]
        ############################
        horizon = control_parameters["control_horizon_years"]
        self.control_horizon = self.T if horizon is None else float(horizon)
        if not 0.0 < self.control_horizon <= self.T + 1.0e-9:
            raise ValueError(
                "optimal_control.control_horizon_years must be positive and no "
                f"greater than time.final_time_years ({self.T}), not "
                f"{self.control_horizon}"
            )
        self.control_steps = int(round(self.control_horizon / self.dt))
        if abs(self.control_steps * self.dt - self.control_horizon) > 1.0e-9:
            raise ValueError(
                "optimal_control.control_horizon_years "
                f"({self.control_horizon}) must land on a time-step boundary: "
                f"it is not a whole multiple of time.time_step_years "
                f"({self.dt}). A control block that switches inside a step "
                "would be applied to the wrong interval."
            )
        self.control_steps = min(self.control_steps, self.total_steps)
        self.tail_steps = self.total_steps - self.control_steps
        self.tail_years = self.T - self.control_horizon

        step_start_times = np.arange(self.control_steps) * self.dt
        controlled_blocks = np.floor(
            step_start_times / self.block_years + 1e-9
        ).astype(np.int64)
        self.n_blocks = int(controlled_blocks[-1]) + 1
        # Steps past the control horizon carry block -1: they load v = 0 and
        # contribute to no gradient.
        self.block_of_step = np.full(self.total_steps, -1, dtype=np.int64)
        self.block_of_step[:self.control_steps] = controlled_blocks
        self.block_step_counts = np.bincount(
            controlled_blocks, minlength=self.n_blocks
        ).astype(np.float64)
        # Each block's length in years, the time weight of the space-time L2
        # inner product.
        self.block_weights = self.block_step_counts * self.dt

        self.c1 = float(control_parameters["cost_susceptible_deficit"])
        self.c2 = float(control_parameters["cost_environment"])
        self.c3 = float(control_parameters["cost_control"])
        self.c4 = float(control_parameters["cost_control_l1"])

        self.v_min = float(control_parameters["control_minimum"])
        self.v_max = float(control_parameters["control_maximum"])
        self.v_initial = float(control_parameters["initial_control"])

        ############################
        # optional closed-form terminal cost, scaled by terminal_cost_scale
        # (0 when the continuation tail is used). The weights are checked by
        # utilities/check_reaction_adjoint.py.
        ############################
        terminal_scale = float(control_parameters["terminal_cost_scale"])
        self.terminal_scale = terminal_scale
        self.terminal_generations = int(
            control_parameters["terminal_tail_generations"]
        )
        environment_tail = self.c2 / delta
        susceptible_tail = self.c1 / r
        clinical_tail = (self.c1 + rhoR * environment_tail) / alpha
        preclinical_tail = (
            (self.c1 + rhoI * environment_tail) / sigma2 + clinical_tail
        )
        exposed_tail = self.c1 / sigma1 + preclinical_tail
        # Unscaled weights; the one-generation correction below is built on them.
        self.tail_zero_generation = {
            "susceptible": susceptible_tail,
            "exposed": exposed_tail,
            "preclinical": preclinical_tail,
            "clinical": clinical_tail,
            "environment": environment_tail,
        }
        self.m_susceptible = terminal_scale * susceptible_tail
        self.m_exposed = terminal_scale * exposed_tail
        self.m_preclinical = terminal_scale * preclinical_tail
        self.m_clinical = terminal_scale * clinical_tail
        self.m_environment = terminal_scale * environment_tail

        ############################
        # domain and function spaces
        ############################
        mesh_data = read_from_msh(str(mesh_path), comm, gdim=3)
        domain = mesh_data.mesh
        self.domain = domain

        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        mixed = basix.ufl.mixed_element([P1, P1, P1, P1, P1])
        self.P = fem.functionspace(domain, mixed)
        self.V_scal = fem.functionspace(domain, ("CG", 1))

        index_map = self.V_scal.dofmap.index_map
        block_size = self.V_scal.dofmap.index_map_bs
        self.n_owned = index_map.size_local * block_size
        self.n_local = self.n_owned + index_map.num_ghosts * block_size
        self.n_global = index_map.size_global * block_size

        # Load vectors only; the bilinear form lives in CompartmentBlockSolver.
        V = TestFunction(self.P)
        psiS, psiE, psiI, psiR, psiW = split(V)

        # State at the current time level, also the adjoint linearization point.
        self.Y = Function(self.P)
        uS0, uE0, uI0, uR0, uW0 = split(self.Y)
        self.Y_next = Function(self.P)

        # Adjoint at the *later* time level, lambda^(n+1).
        self.Lam = Function(self.P)
        lamS, lamE, lamI, lamR, lamW = split(self.Lam)
        self.Lam_next = Function(self.P)

        # The control on one step, v^n.
        self.v_func = Function(self.V_scal)
        self.v_func.name = "Targeted_Control"

        # Disease-free equilibrium, filled in by _build_initial_state before
        # the disease spin-up overwrites the initial state.
        self.S_infinity = Function(self.V_scal)
        self.S_infinity.name = "Disease_Free_Equilibrium"

        ############################
        # land cover, diffusivity, carrying capacity
        ############################
        land_cover_classes = np.load(classes_path).astype(np.int32)
        self.land_cover_func = Function(self.V_scal)
        self.land_cover_func.name = "Land_Cover_Class"
        self._check_nodal_array(land_cover_classes, "land_cover_classes.npy")
        self.land_cover_func.x.array[:] = land_cover_classes
        self.land_cover_func.x.scatter_forward()

        lcD_values = np.load(diffusivity_path)
        self.lcD_values = lcD_values
        lcD_scale = Function(self.V_scal)
        self._check_nodal_array(lcD_values, "land_cover_diffusivity.npy")
        lcD_scale.x.array[:] = lcD_values
        lcD_scale.x.scatter_forward()

        K_values = land_cover_to_carrying_capacity(
            land_cover_classes, spatial_parameters
        )
        K_func = Function(self.V_scal)
        K_func.x.array[:] = K_values
        K_func.x.scatter_forward()
        self.K_func = K_func

        ############################
        # diffusion tensor
        ############################
        n = CellNormal(domain)
        g = as_vector((0, 0, -1))
        cos_theta = dot(g, n)

        es_raw = g - dot(g, n) * n
        es_norm = ufl.sqrt(dot(es_raw, es_raw))

        xdir = as_vector((1, 0, 0))
        x_proj_raw = xdir - dot(xdir, n) * n
        x_proj = x_proj_raw / ufl.sqrt(dot(x_proj_raw, x_proj_raw))

        es = ufl.conditional(ufl.gt(es_norm, 1e-6), es_raw / es_norm, x_proj)

        # es_norm = sin(theta), so grade = tan(theta).
        grade = es_norm / abs(cos_theta)
        eta = isotropy + (1 - isotropy) * exp(-slope_penalty * grade)
        Ds = kappa * eta
        Dc = kappa * (2 - eta) if trace_preserving else kappa
        DTens = lcD_scale * (
            (Ds - Dc) * outer(es, es) + Dc * (Identity(3) - outer(n, n))
        )
        self.DTens = DTens

        ############################
        # quadrature measures
        #
        # dq : Gauss rule of degree QUADRATURE_DEGREE, for the reaction,
        #      cost and gradient terms.
        # dl : vertex rule, for the mass terms on both sides of the step; it
        #      yields the lumped mass matrix.
        ############################
        dq = ufl.Measure(
            "dx", domain=domain,
            metadata={"quadrature_degree": QUADRATURE_DEGREE},
        )
        dl = ufl.Measure(
            "dx", domain=domain,
            metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
        )
        self.dq = dq
        self.dl = dl

        ############################
        # bilinear form, assembled and factorized per compartment by
        # CompartmentBlockSolver:
        #
        #   a = (uS*psiS + uE*psiE + uI*psiI + uR*psiR + uW*psiW) * dl
        #     + dt * ( DS*dot(DTens*grad(uS), grad(psiS))
        #            + DE*dot(DTens*grad(uE), grad(psiE))
        #            + DI*dot(DTens*grad(uI), grad(psiI))
        #            + DR*dot(DTens*grad(uR), grad(psiR)) ) * dx
        ############################
        self.solver = CompartmentBlockSolver(
            self.P, self.dt, DTens, (DS, DE, DI, DR),
            mass_measure=dl, stiffness_measure=dx, comm=comm,
        )

        ############################
        # forward right-hand side L(y^n, v^n): mass terms on dl, reaction
        # terms on dq. The adjoint form below uses the same measure per term.
        ############################
        # Shared with the adjoint form.
        zero_c = fem.Constant(domain, PETSc.ScalarType(0.0))
        r_c = fem.Constant(domain, PETSc.ScalarType(r))

        water = ufl.lt(K_func, 0.01)
        growth = ufl.conditional(
            water,
            -r_c * uS0,                                          # decay in water
            r_c * uS0 * (1 - (uS0 + uE0 + uI0 + uR0) / K_func),  # logistic on land
        )

        # Force of infection per susceptible, used in both the S and E rows.
        infection_density = beta1I * uI0 + beta1R * uR0 + beta2 * uW0
        infection = uS0 * infection_density

        self.state_rhs_form = fem.form(
            # ---- M_L y^n
            (uS0 * psiS + uE0 * psiE + uI0 * psiI
             + uR0 * psiR + uW0 * psiW) * dl

            # ---- dt * F(y^n, v^n)
            + self.dt * (
                psiS * (growth - infection)

                + psiE * (
                      infection
                    - sigma1 * uE0
                )

                + psiI * (
                      sigma1 * uE0
                    - sigma2 * uI0
                )

                # the control enters only the R row
                + psiR * (
                      sigma2 * uI0
                    - alpha * uR0
                    - self.v_func * uR0
                )

                + psiW * (
                      rhoI * uI0
                    + rhoR * uR0
                    - delta * uW0
                )
            ) * dq
        )

        ############################
        # adjoint right-hand side   (d L / d y^n)^T lambda^(n+1)  +  dt * g
        #
        # Row k collects sum_j lambda_j * d(L_j)/d(y_k). The growth
        # derivatives use the same water branch as the forward form.
        ############################
        growth_dS = ufl.conditional(
            water,
            -r_c,
            r_c * (1 - (uS0 + uE0 + uI0 + uR0) / K_func) - r_c * uS0 / K_func,
        )
        growth_dN = ufl.conditional(water, zero_c, -r_c * uS0 / K_func)

        # Derivative of the running state cost.
        running_cost_rhs = self.dt * (-self.c1 * psiS + self.c2 * psiW)

        self.adjoint_rhs_form = fem.form(
            # ---- M_L lambda^(n+1), on the same rule as the forward mass terms
            (lamS * psiS + lamE * psiE + lamI * psiI
             + lamR * psiR + lamW * psiW) * dl

            # ---- dt * (dF/dy)^T lambda^(n+1)
            + self.dt * (
                # row S
                psiS * (
                      lamS * (growth_dS - infection_density)
                    + lamE * infection_density
                )

                # row E
                + psiE * (
                      lamS * growth_dN
                    - lamE * sigma1
                    + lamI * sigma1
                )

                # row I
                + psiI * (
                      lamS * (growth_dN - beta1I * uS0)
                    + lamE * beta1I * uS0
                    - lamI * sigma2
                    + lamR * sigma2
                    + lamW * rhoI
                )

                # row R  (the control enters this diagonal)
                + psiR * (
                      lamS * (growth_dN - beta1R * uS0)
                    + lamE * beta1R * uS0
                    - lamR * (alpha + self.v_func)
                    + lamW * rhoR
                )

                # row W
                + psiW * (
                    - lamS * beta2 * uS0
                    + lamE * beta2 * uS0
                    - lamW * delta
                )
            ) * dq

            # ---- dt * g
            + running_cost_rhs * dq
        )

        ############################
        # terminal_tail_generations >= 1 adds one round of onward
        # transmission to the closed-form terminal weights, which become
        # spatial fields through S_infinity.
        ############################
        if self.terminal_generations >= 1:
            S_inf = self.S_infinity
            # Secondary infections per unit of each compartment.
            per_W = S_inf * (beta2 / delta)
            per_R = S_inf * (beta1R / alpha + beta2 * rhoR / (alpha * delta))
            per_I = S_inf * (
                beta1I / sigma2 + beta1R / alpha
                + beta2 * (rhoI / sigma2 + rhoR / alpha) / delta
            )
            secondary = exposed_tail
            self.m_exposed = terminal_scale * (
                exposed_tail + per_I * secondary
            )
            self.m_preclinical = terminal_scale * (
                preclinical_tail + per_I * secondary
            )
            self.m_clinical = terminal_scale * (
                clinical_tail + per_R * secondary
            )
            self.m_environment = terminal_scale * (
                environment_tail + per_W * secondary
            )
            self.terminal_secondary_counts = {
                "exposed": per_I,
                "preclinical": per_I,
                "clinical": per_R,
                "environment": per_W,
            }
        else:
            self.terminal_secondary_counts = {}

        # Terminal adjoint load: dt * g + dPhi/dy^N.
        self.terminal_rhs_form = fem.form(
            running_cost_rhs * dq
            + (-self.m_susceptible * psiS
               + self.m_exposed * psiE
               + self.m_preclinical * psiI
               + self.m_clinical * psiR
               + self.m_environment * psiW) * dq
        )

        ############################
        # reduced objective and reduced gradient
        ############################
        q = TestFunction(self.V_scal)
        # The deficit is not clamped at zero, so its derivative stays constant.
        self.state_cost_form = fem.form(
            (self.c1 * (self.S_infinity - uS0) + self.c2 * uW0) * dq
        )
        self.terminal_cost_form = fem.form(
            (self.m_susceptible * (self.S_infinity - uS0)
             + self.m_exposed * uE0
             + self.m_preclinical * uI0
             + self.m_clinical * uR0
             + self.m_environment * uW0) * dq
        )
        self.control_cost_form = fem.form(
            (self.c3 * self.v_func**2 + self.c4 * self.v_func) * dq
        )

        # ------------------------------------------------ cost decomposition
        # Per-term cost forms for reporting, assembled only by
        # forward(breakdown=True). Entries are (name, weight, unweighted form).
        self.state_cost_terms = (
            ("susceptible_deficit", self.c1,
             fem.form((self.S_infinity - uS0) * dq)),
            ("environment", self.c2, fem.form(uW0 * dq)),
        )
        self.control_cost_terms = (
            ("targeted_quadratic", self.c3, fem.form(self.v_func**2 * dq)),
            ("targeted_l1", self.c4, fem.form(self.v_func * dq)),
        )
        # Terminal entries add a fourth element, the weighted form, because the
        # weights may be spatial fields; the reported weight is cost/integral.
        terminal_state_forms = (
            ("susceptible_deficit", self.m_susceptible,
             (self.S_infinity - uS0)),
            ("exposed", self.m_exposed, uE0),
            ("preclinical", self.m_preclinical, uI0),
            ("clinical", self.m_clinical, uR0),
            ("environment", self.m_environment, uW0),
        )
        self.terminal_cost_terms = tuple(
            (name, weight, fem.form(expr * dq), fem.form(weight * expr * dq))
            for name, weight, expr in terminal_state_forms
        )
        self.last_breakdown = None

        # dJ/dv on one step in weak form; dividing by the lumped mass gives
        # the L2 Riesz representative.
        self.gradient_form = fem.form(
            (2.0 * self.c3 * self.v_func + self.c4 - uR0 * lamR) * q * dq
        )
        self.lumped_mass = self._assemble_scalar_vector(fem.form(q * dq))
        # |Gamma|, used to normalize the budget constraint to a spatial mean.
        self.domain_measure = self._global_sum(float(self.lumped_mass.sum()))

        # Domain-wide prevalence (E + I + R over all deer), for the
        # prevalence-mode spin-up.
        uS_c, uE_c, uI_c, uR_c, uW_c = ufl.split(self.Y)
        self.infected_mass_form = fem.form((uE_c + uI_c + uR_c) * dq)
        self.deer_mass_form = fem.form((uS_c + uE_c + uI_c + uR_c) * dq)

        # Integrand of the clinical burden, for the detection-mode spin-up.
        detection_exponent = float(
            self.parameters["disease_spinup"]["detection_exponent"]
        )
        clinical_integrand = (
            uR_c if detection_exponent == 1.0 else uR_c ** detection_exponent
        )
        self.clinical_burden_form = fem.form(clinical_integrand * dq)

        ############################
        # containment and incidence diagnostics
        ############################
        # Reporting only; nothing here enters J or the gradient.
        #
        # Cumulative incidence, accumulated per step like the running cost.
        self.incidence_form = fem.form(
            uS_c * (beta1I * uI_c + beta1R * uR_c + beta2 * uW_c) * dq
        )

        # Planar (x, y) distance from the nearest seeded focus.
        focus_centers = np.array(
            [[float(v) for v in gaussian["center"]]
             for gaussian in self.initial_condition_parameters
             ["infected_gaussians"]],
            dtype=np.float64,
        )
        coordinates = self.V_scal.tabulate_dof_coordinates()
        if focus_centers.size:
            offsets = (coordinates[:, None, :2] - focus_centers[None, :, :])
            self.seed_distance = np.min(
                np.linalg.norm(offsets, axis=2), axis=1
            )
        else:
            self.seed_distance = np.zeros(coordinates.shape[0])
        self.seed_distance_function = Function(self.V_scal)
        self.seed_distance_function.x.array[:] = self.seed_distance
        self.seed_distance_function.x.scatter_forward()

        # Numerator of the infected-mass-weighted mean distance from the focus.
        self.infection_radius_form = fem.form(
            self.seed_distance_function * (uE_c + uI_c + uR_c) * dq
        )

        # Infected area and front position at two density thresholds (deer/km^2).
        self.extent_thresholds = (0.01, 0.1)
        self.infected_area_forms = tuple(
            fem.form(
                ufl.conditional(
                    ufl.gt(uE_c + uI_c + uR_c, threshold), 1.0, 0.0
                ) * dq
            )
            for threshold in self.extent_thresholds
        )
        # Subspace dof maps for the nodal front-position reduction. collapse()
        # returns a (1, n_local) map for a scalar subspace, so flatten it.
        V_E, dof_map_E = self.P.sub(1).collapse()
        _, dof_map_I = self.P.sub(2).collapse()
        _, dof_map_R = self.P.sub(3).collapse()
        self.dof_map_E = np.asarray(dof_map_E, dtype=np.int32).ravel()
        self.dof_map_I = np.asarray(dof_map_I, dtype=np.int32).ravel()
        self.dof_map_R = np.asarray(dof_map_R, dtype=np.int32).ravel()

        # seed_distance is in V_scal order but indexed in the collapsed
        # subspace order; check that the two agree.
        if not np.allclose(V_E.tabulate_dof_coordinates(), coordinates):
            raise RuntimeError(
                "the collapsed E subspace does not share V_scal's dof "
                "ordering, so the containment diagnostics would pair nodal "
                "values with the wrong distances. Build seed_distance from "
                "V_E.tabulate_dof_coordinates() instead."
            )
        self.last_diagnostics = None

        ############################
        # initial state y^0 (control-independent)
        ############################
        self.V_S, self.dof_map_S = self.P.sub(0).collapse()
        self.Y_initial = Function(self.P)
        self._build_initial_state()

    # ------------------------------------------------------------------ setup

    def _check_nodal_array(self, values, name):
        if values.size != self.n_local:
            raise ValueError(explain_nodal_array_mismatch(
                name, values.size, self.n_local, self.domain, self.comm
            ))

    def _build_initial_state(self):
        """Build y^0: S at the disease-free equilibrium, infected classes seeded.

        Fills self.S_infinity, then seeds the Gaussian foci over E, I, R and W
        according to initial_conditions.seed_mode, then runs the disease
        spin-up if enabled.
        """
        S_values = susceptible_initial_condition(
            self.domain, self.V_scal, self.DTens, self.K_func, self.lcD_values,
            self.parameters, self.equilibrium_array_path,
            self.equilibrium_signature_path, comm=self.comm,
            recompute=self.recompute_equilibrium,
        )
        S_equilibrium = Function(self.V_scal)
        S_equilibrium.x.array[:] = S_values
        S_equilibrium.x.scatter_forward()

        # Captured before the disease spin-up replaces Y_initial.
        self.S_infinity.x.array[:] = S_values
        self.S_infinity.x.scatter_forward()

        S_initial = Function(self.V_S)
        S_initial.interpolate(
            fem.Expression(S_equilibrium, self.V_S.element.interpolation_points)
        )
        self.Y_initial.sub(0).x.array[self.dof_map_S] = S_initial.x.array

        infected_gaussians = self.initial_condition_parameters["infected_gaussians"]
        seed_mode = str(self.initial_condition_parameters["seed_mode"])

        def focus_shape(x):
            """Summed foci, amplitude 1 -- the spatial profile alone."""
            values = np.zeros(x.shape[1], dtype=np.float64)
            for gaussian in infected_gaussians:
                center_x, center_y = (float(value) for value in gaussian["center"])
                mean = float(gaussian["mean"])
                standard_deviation = float(gaussian["standard_deviation"])
                squared_distance = (x[0] - center_x) ** 2 + (x[1] - center_y) ** 2
                values += mean * np.exp(
                    -squared_distance / (2.0 * standard_deviation**2)
                )
            return values

        # "exposed": the whole amplitude goes into E, with I = R = W = 0.
        # "staged": the amplitude (the total infected density) is split over
        # E : I : R = 1 : sigma1/sigma2 : sigma1/alpha, and W is set to
        # (rhoI*I + rhoR*R) / delta.
        if seed_mode == "exposed":
            stage_fraction = (1.0, 0.0, 0.0)
        elif seed_mode == "staged":
            sigma1 = float(self.reaction_parameters["exposed_to_shedding_rate"])
            sigma2 = float(
                self.reaction_parameters["preclinical_to_clinical_rate"]
            )
            alpha = float(self.reaction_parameters["clinical_removal_rate"])
            weights = np.array([1.0, sigma1 / sigma2, sigma1 / alpha])
            stage_fraction = tuple(weights / weights.sum())
        else:
            raise ValueError(
                f"initial_conditions.seed_mode = {seed_mode!r} is not "
                "recognised; expected 'staged' or 'exposed'."
            )

        rhoI = float(
            self.reaction_parameters["environmental_shedding_rate_preclinical"]
        )
        rhoR = float(
            self.reaction_parameters["environmental_shedding_rate_clinical"]
        )
        delta = float(self.reaction_parameters["environmental_decay_rate"])
        reservoir_per_unit = (
            rhoI * stage_fraction[1] + rhoR * stage_fraction[2]
        ) / delta

        for index, factor in zip(
            (1, 2, 3, 4),
            stage_fraction + (reservoir_per_unit,),
        ):
            self.Y_initial.sub(index).interpolate(
                (lambda scale: lambda x: scale * focus_shape(x))(factor)
            )
        self.Y_initial.x.scatter_forward()

        self.seed_mode = seed_mode
        self.seed_stage_fraction = stage_fraction
        self.seed_reservoir_per_unit = reservoir_per_unit

        if bool(self.parameters["disease_spinup"]["enabled"]):
            self._apply_disease_spinup(S_values)

    def _current_prevalence(self):
        """Domain-wide (E + I + R) / (S + E + I + R) for the state in self.Y."""
        infected = self._global_sum(fem.assemble_scalar(self.infected_mass_form))
        total = self._global_sum(fem.assemble_scalar(self.deer_mass_form))
        return infected / total if total > 0.0 else 0.0

    def _current_clinical_burden(self):
        """int R^p dx over the domain, converted from m^2 to km^2 (deer at p = 1)."""
        return self._global_sum(
            fem.assemble_scalar(self.clinical_burden_form)
        ) / SQUARE_METRES_PER_SQUARE_KM

    def _apply_disease_spinup(self, susceptible_values):
        """Replace the seeded initial state with the spun-up epizootic.

        Runs the uncontrolled system (v = 0) until the disease_spinup stopping
        rule is met, or loads the cached result. See utilities/disease_spinup.py.
        """
        expected = disease_signature(
            self.domain, self.K_func.x.array, self.lcD_values,
            susceptible_values, self.parameters,
        )

        def compute():
            # Start from the seeded state and step with v = 0.
            self.Y.x.array[:] = self.Y_initial.x.array
            self.Y.x.scatter_forward()
            self.v_func.x.array[:] = 0.0
            self.v_func.x.scatter_forward()

            def advance():
                self._solve_block(self.state_rhs_form, self.Y_next, "state")
                self.Y.x.array[:] = self.Y_next.x.array
                self.Y.x.scatter_forward()

            writer = observer = None
            if self.spinup_series_folder is not None:
                writer = SpinupSeriesWriter(
                    self.domain, self.V_scal, self.spinup_series_folder
                )
                every = max(1, int(self.spinup_log_every))

                def observer(step, years):
                    if step % every == 0:
                        writer.write(self.Y, years)

            try:
                return run_disease_spinup(
                    advance, self._current_prevalence,
                    lambda: self.Y.x.array.copy(),
                    self.parameters, comm=self.comm, observer=observer,
                    clinical_burden=self._current_clinical_burden,
                )
            finally:
                if writer is not None:
                    writer.close()
                    if self.comm.rank == 0:
                        print(f"  spin-up series: {writer.frames} frames -> "
                              f"{self.spinup_series_folder}")

        values = disease_initial_state(
            expected, self.Y_initial.x.array.size, compute,
            self.disease_array_path, self.disease_signature_path,
            comm=self.comm, recompute=self.recompute_disease_state,
        )
        self.Y_initial.x.array[:] = values
        self.Y_initial.x.scatter_forward()

    # -------------------------------------------------------- linear algebra

    def _assemble_scalar_vector(self, form):
        """Assemble a V_scal form and return its owned entries as a NumPy array."""
        b = assemble_vector(form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        values = b.array_r[: self.n_owned].copy()
        b.destroy()
        return values

    def _solve_block(self, form, out_function, buffer_key):
        """Solve B x = assemble(form) into out_function, reusing the RHS buffer."""
        b = self._rhs_buffers.get(buffer_key)
        if b is None:
            b = assemble_vector(form)
            self._rhs_buffers[buffer_key] = b
        else:
            with b.localForm() as local:
                local.set(0.0)
            assemble_vector(b, form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.solver.solve(b, out_function)

    def _global_sum(self, local_value):
        return self.comm.allreduce(local_value, op=MPI.SUM)

    def terminal_weight_summary(self):
        """The five terminal weights as area-weighted mean, min and max."""
        summary = {}
        for name, weight in (
            ("mS_terminal_susceptible", self.m_susceptible),
            ("mE_terminal_exposed", self.m_exposed),
            ("mI_terminal_preclinical", self.m_preclinical),
            ("mR_terminal_clinical", self.m_clinical),
            ("mW_terminal_environment", self.m_environment),
        ):
            # A zero or scalar weight has no Function for fem.Expression to
            # take a communicator from, so report it directly.
            if self.terminal_scale == 0.0 or isinstance(weight, (int, float)):
                value = 0.0 if self.terminal_scale == 0.0 else float(weight)
                summary[name] = {"mean": value, "min": value, "max": value,
                                 "uniform": True}
                continue
            field = Function(self.V_scal)
            field.interpolate(
                fem.Expression(weight, self.V_scal.element.interpolation_points)
            )
            owned = field.x.array[:self.n_owned]
            total = self._global_sum(
                float(np.dot(owned, self.lumped_mass[:self.n_owned]))
            )
            area = self._global_sum(float(self.lumped_mass[:self.n_owned].sum()))
            summary[name] = {
                "mean": total / area if area else 0.0,
                "min": self.comm.allreduce(float(owned.min()), op=MPI.MIN),
                "max": self.comm.allreduce(float(owned.max()), op=MPI.MAX),
                "uniform": False,
            }
        summary["generations"] = self.terminal_generations
        summary["scale"] = self.terminal_scale
        return summary

    # --------------------------------------------------- control array helpers

    def new_control(self, value=None):
        """A control array of shape (n_blocks, n_local).

        ``value`` is a scalar, defaulting to the configured initial iterate.
        """
        if value is None:
            value = self.v_initial
        control = np.empty((self.n_blocks, self.n_local), dtype=np.float64)
        control[:] = float(value)
        return control

    def control_bytes(self):
        return self.n_blocks * self.n_local * 8

    def control_slab(self, control, step):
        """The control field in force during time step ``step``.

        Returns an array of shape (n_local,). Steps in the uncontrolled
        continuation tail return zeros.
        """
        block = self.block_of_step[min(step, self.total_steps - 1)]
        if block < 0:
            return np.zeros(self.n_local, dtype=np.float64)
        return control[block]

    @staticmethod
    def _trajectory_cache_release(trajectory):
        """Return a flush callable for a disk-backed trajectory, else None.

        Flushing bounds dirty pages; the clean pages stay in the page cache.
        """
        return getattr(trajectory, "flush", None)

    def new_trajectory(self, path=None):
        """Storage for y^0 .. y^N, in memory or memory-mapped to disk."""
        shape = (self.total_steps + 1, self.Y.x.array.size)
        if path is None:
            return np.empty(shape, dtype=np.float64)
        path = Path(path)
        if self.comm.size > 1:
            # one backing file per rank
            path = path.with_name(f"{path.stem}_rank{self.comm.rank}{path.suffix}")
        return np.lib.format.open_memmap(
            str(path), mode="w+", dtype=np.float64, shape=shape
        )

    def trajectory_bytes(self):
        return (self.total_steps + 1) * self.Y.x.array.size * 8

    def project_control(self, control):
        """Pointwise projection onto the admissible box [v_min, v_max]."""
        return np.clip(control, self.v_min, self.v_max)

    def inner_product(self, first, second):
        """Space-time L2 inner product: lumped mass in space, block length in time."""
        local = float(np.einsum(
            "bi,bi,i,b->",
            first[:, : self.n_owned],
            second[:, : self.n_owned],
            self.lumped_mass,
            self.block_weights,
        ))
        return self._global_sum(local)

    def norm(self, field):
        return float(np.sqrt(max(0.0, self.inner_product(field, field))))

    def block_means(self, control):
        """Mass-weighted spatial mean of the control in each block (1/yr)."""
        owned = control[:, : self.n_owned]
        local = owned @ self.lumped_mass
        total = np.asarray(
            self.comm.allreduce(local, op=MPI.SUM), dtype=np.float64
        )
        return total / self.domain_measure

    def never_treated_area(self, control, tolerance=1e-12):
        """Mass-weighted area, in m^2, at the lower bound in every block."""
        owned = control[:, : self.n_owned]
        never = np.all(owned <= self.v_min + tolerance, axis=0)
        local = float(np.dot(never, self.lumped_mass[: self.n_owned]))
        return self._global_sum(local)

    def add_budget_gradient(self, gradient, budget, means):
        """Add the budget's augmented-Lagrangian term to `gradient` in place.

        The per-block term comes from EffortBudget.gradient_term and is scaled
        by COST_SCALE like the rest of the gradient.
        """
        gradient += COST_SCALE * budget.gradient_term(means)[:, None]
        return gradient

    def control_metric_scale(self):
        """D = sqrt(m_i * w_b), shape (n_blocks, n_owned).

        inner_product is x^T D^2 y, so L-BFGS-B works in z = D x, where its
        Euclidean product is the L2 product and the gradient it needs is D g.
        """
        if np.any(self.lumped_mass <= 0.0):
            raise ValueError(
                "lumped mass has non-positive entries, so the L2 metric is "
                "singular; check the mesh for degenerate or duplicated cells."
            )
        return np.sqrt(
            self.lumped_mass[None, :] * self.block_weights[:, None]
        )

    def _set_control_slab(self, control, step):
        """Load the control for step ``step`` into v_func (zero in the tail)."""
        block = self.block_of_step[step]
        if block < 0:
            self.v_func.x.array[:] = 0.0
        else:
            self.v_func.x.array[:] = control[block]
        self.v_func.x.scatter_forward()

    # --------------------------------------------------------- forward solve

    def forward(self, control, trajectory=None, progress=None, observer=None,
                breakdown=False):
        """Advance the controlled state system and evaluate J(v).

        control has shape (n_blocks, n_local); the field is loaded into its
        Function once per step by _set_control_slab, and the running cost form
        reads it. trajectory (optional) must have shape
        (total_steps + 1, mixed local size); it is filled with y^0 .. y^N.
        observer (optional) is called as observer(step, t) after each state
        update, with self.Y holding y^step; it is the hook the driver uses to
        write the XDMF series. Returns
        (cost, state_cost, control_cost, terminal_cost).

        breakdown=True also decomposes each cost into its terms
        (self.last_breakdown) and computes the outcome diagnostics
        (self.last_diagnostics); the driver uses it on the final pass only.
        """
        self.Y.x.array[:] = self.Y_initial.x.array
        self.Y.x.scatter_forward()
        # Flush a memory-mapped trajectory every 256 MiB so dirty pages stay
        # bounded.
        trajectory_release = self._trajectory_cache_release(trajectory)
        row_bytes = self.Y.x.array.size * 8
        flush_every = max(1, (256 << 20) // max(row_bytes, 1))
        if trajectory is not None:
            trajectory[0] = self.Y.x.array
        if observer is not None:
            observer(0, 0.0)

        state_cost = 0.0
        control_cost = 0.0

        # Per-term and incidence accumulators, used only when breakdown=True.
        state_terms = [0.0] * len(self.state_cost_terms)
        control_terms = [0.0] * len(self.control_cost_terms)
        slab_terms = None
        slab_block = -1
        incidence = 0.0

        for step in range(self.total_steps):
            self._set_control_slab(control, step)
            control_cost += fem.assemble_scalar(self.control_cost_form)

            if breakdown:
                # Reassemble the control terms only when the block changes.
                block = int(self.block_of_step[step])
                if block != slab_block:
                    slab_terms = [fem.assemble_scalar(form)
                                  for _, _, form in self.control_cost_terms]
                    slab_block = block
                for index, value in enumerate(slab_terms):
                    control_terms[index] += value

            self._solve_block(self.state_rhs_form, self.Y_next, "state")

            self.Y.x.array[:] = self.Y_next.x.array
            self.Y.x.scatter_forward()
            if trajectory is not None:
                trajectory[step + 1] = self.Y.x.array
                if trajectory_release is not None and (step + 1) % flush_every == 0:
                    trajectory_release()

            state_cost += fem.assemble_scalar(self.state_cost_form)
            if breakdown:
                for index, (_, _, form) in enumerate(self.state_cost_terms):
                    state_terms[index] += fem.assemble_scalar(form)
                incidence += fem.assemble_scalar(self.incidence_form)

            if observer is not None:
                observer(step + 1, (step + 1) * self.dt)
            if progress is not None:
                progress(step + 1, self.total_steps)

        state_cost = COST_SCALE * self.dt * self._global_sum(state_cost)
        control_cost = COST_SCALE * self.dt * self._global_sum(control_cost)
        terminal_cost = COST_SCALE * self._global_sum(
            fem.assemble_scalar(self.terminal_cost_form)
        )

        if breakdown:
            self.last_breakdown = self._assemble_breakdown(
                state_terms, control_terms,
                (state_cost, control_cost, terminal_cost),
            )
            self.last_diagnostics = self._assemble_diagnostics(
                self.dt * self._global_sum(incidence)
            )

        return (state_cost + control_cost + terminal_cost,
                state_cost, control_cost, terminal_cost)

    def _assemble_breakdown(self, state_terms, control_terms, totals):
        """Package the per-term integrals accumulated by a breakdown pass.

        Each term reports its ``integral``, ``weight``, ``cost`` and ``share``
        of J. Each group reports a ``residual``, its total minus the sum of its
        terms, which should be at rounding level.
        """
        def group(terms, values, total, costs=None):
            entries = {}
            for index, (term, value) in enumerate(zip(terms, values)):
                name, weight = term[0], term[1]
                if costs is None:
                    cost = float(weight) * float(value)
                    effective = float(weight)
                else:
                    # Spatial weight: cost assembled directly.
                    cost = float(costs[index])
                    effective = cost / float(value) if value else 0.0
                entries[name] = {
                    "integral": float(value),
                    "weight": effective,
                    "cost": cost,
                }
            parts = sum(entry["cost"] for entry in entries.values())
            return {
                "total": float(total),
                "terms": entries,
                "residual": float(total) - parts,
            }

        state_total, control_total, terminal_total = totals
        # Scaled by COST_SCALE to match the totals.
        terminal_values = [
            COST_SCALE * self._global_sum(fem.assemble_scalar(term[2]))
            for term in self.terminal_cost_terms
        ]
        terminal_costs = [
            COST_SCALE * self._global_sum(fem.assemble_scalar(term[3]))
            for term in self.terminal_cost_terms
        ]
        state_values = [COST_SCALE * self.dt * self._global_sum(x)
                        for x in state_terms]
        control_values = [COST_SCALE * self.dt * self._global_sum(x)
                          for x in control_terms]

        objective = state_total + control_total + terminal_total
        breakdown = {
            "objective": float(objective),
            "cost_units": COST_UNITS,
            "running_state": group(self.state_cost_terms, state_values,
                                   state_total),
            "running_control": group(self.control_cost_terms, control_values,
                                     control_total),
            "terminal": group(self.terminal_cost_terms, terminal_values,
                              terminal_total, costs=terminal_costs),
        }
        for section in ("running_state", "running_control", "terminal"):
            for entry in breakdown[section]["terms"].values():
                entry["share"] = (entry["cost"] / objective
                                  if objective else 0.0)
            breakdown[section]["share"] = (
                breakdown[section]["total"] / objective if objective else 0.0
            )
        return breakdown

    def _assemble_diagnostics(self, cumulative_incidence):
        """Outcome measures at t = T, plus the run's cumulative incidence."""
        infected = self._global_sum(
            fem.assemble_scalar(self.infected_mass_form)
        )
        radius_moment = self._global_sum(
            fem.assemble_scalar(self.infection_radius_form)
        )
        areas = [
            self._global_sum(fem.assemble_scalar(form))
            for form in self.infected_area_forms
        ]

        # Front position: the farthest owned node above each threshold.
        values = self.Y.x.array
        burden = (values[self.dof_map_E] + values[self.dof_map_I]
                  + values[self.dof_map_R])[: self.n_owned]
        distance = self.seed_distance[: self.n_owned]
        fronts = []
        for threshold in self.extent_thresholds:
            above = distance[burden > threshold]
            local = float(above.max()) if above.size else 0.0
            fronts.append(self.comm.allreduce(local, op=MPI.MAX))

        # Convert areal integrals from m^2 and lengths from m to km.
        square_metres_per_square_km = SQUARE_METRES_PER_SQUARE_KM
        metres_per_km = 1.0e3
        infected_deer = infected / square_metres_per_square_km
        return {
            "cumulative_incidence_deer":
                float(cumulative_incidence / square_metres_per_square_km),
            "infected_animals_at_T": float(infected_deer),
            "mean_infection_radius_km": (
                float(radius_moment / infected / metres_per_km)
                if infected > 0.0 else 0.0
            ),
            "extent_thresholds_deer_per_km2": list(self.extent_thresholds),
            "infected_area_km2": [
                float(a / square_metres_per_square_km) for a in areas
            ],
            "front_radius_km": [float(r / metres_per_km) for r in fronts],
            "domain_area_km2":
                float(self.domain_measure / square_metres_per_square_km),
        }

    # -------------------------------------------------------- backward solve

    def backward(self, control, trajectory, gradient=None, progress=None,
                 observer=None):
        """Backward adjoint sweep; returns the reduced gradient array.

        Solves B lambda^N = dt*g + dPhi/dy and, for n = N-1 down to 1,
        B lambda^n = (dL(y^n, v^n)/dy)^T lambda^(n+1) + dt*g, accumulating the
        gradient of each step into its control block.

        The returned array has the same shape as the control,
        (n_blocks, n_local). trajectory must hold the states forward()
        produced for this same control. observer (optional) is called as
        observer(step + 1, t) with self.Lam holding lambda^(step+1); times
        therefore arrive in decreasing order.
        """
        if gradient is None:
            gradient = np.empty(
                (self.n_blocks, self.n_local), dtype=np.float64
            )
        gradient[:] = 0.0

        # lambda^N
        self._solve_block(self.terminal_rhs_form, self.Lam, "terminal")

        for step in range(self.total_steps - 1, -1, -1):
            # self.Lam currently holds lambda^(step+1)
            if observer is not None:
                observer(step + 1, (step + 1) * self.dt)
            self.Y.x.array[:] = trajectory[step]
            self.Y.x.scatter_forward()
            self._set_control_slab(control, step)

            # Sum this step's gradient into its block (tail steps have none);
            # the adjoint solve still runs over tail steps.
            block = self.block_of_step[step]
            if block >= 0:
                raw = self._assemble_scalar_vector(self.gradient_form)
                gradient[block, : self.n_owned] += raw / self.lumped_mass

            if step >= 1:
                self._solve_block(self.adjoint_rhs_form, self.Lam_next, "adjoint")
                self.Lam.x.array[:] = self.Lam_next.x.array
                self.Lam.x.scatter_forward()

            if progress is not None:
                progress(self.total_steps - step, self.total_steps)

        # Average over each block's steps and apply the same COST_SCALE as J.
        # Ghosts are filled after the loop because _scatter_slab uses v_func.
        gradient[:, : self.n_owned] /= self.block_step_counts[:, None]
        gradient[:, : self.n_owned] *= COST_SCALE
        for block in range(self.n_blocks):
            self._scatter_slab(gradient, block)
        return gradient

    def _scatter_slab(self, array, block):
        """Fill the ghost entries of one control-shaped slab."""
        if self.n_local == self.n_owned:
            return
        self.v_func.x.array[: self.n_owned] = array[block, : self.n_owned]
        self.v_func.x.scatter_forward()
        array[block] = self.v_func.x.array
