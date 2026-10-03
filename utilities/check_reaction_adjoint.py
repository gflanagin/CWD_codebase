"""Verify the reaction algebra of the five-compartment model without dolfinx.

Transcribes the reaction terms, adjoint rows and control gradient from the UFL
forms in cwd_control_problem.py and checks them against independent references:

  1. adjoint rows against J^T lambda + dl/dy, with J and dl/dy by central
     differences, at random states on land, poor habitat and water;
  2. the control gradient against dH/dv of the Hamiltonian;
  3. that v acts on R only;
  4. the endemic equilibrium against S* = K0/R0;
  5. the closed-form terminal weights against a numerically integrated drain;
  6. horizon complementarity of the running and terminal W prices;
  7. the effort-budget gradient against finite differences.

    python utilities/check_reaction_adjoint.py

Needs numpy only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

UTILITIES_DIR = Path(__file__).resolve().parent
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import load_parameters, reproduction_numbers  # noqa: E402
from effort_budget import EffortBudget  # noqa: E402


# Central differences with h = 1e-6 are accurate to about h^2.
DERIVATIVE_TOLERANCE = 1.0e-7
EQUILIBRIUM_TOLERANCE = 1.0e-10
DRAIN_TOLERANCE = 1.0e-4        # explicit Euler over the drain is first order


class Rates:
    """The reaction rates and cost weights, unpacked once."""

    def __init__(self, parameters):
        reaction = parameters["reaction"]
        control = parameters["optimal_control"]
        self.sigma1 = float(reaction["exposed_to_shedding_rate"])
        self.sigma2 = float(reaction["preclinical_to_clinical_rate"])
        self.alpha = float(reaction["clinical_removal_rate"])
        self.r = float(reaction["intrinsic_growth_rate"])
        self.beta1I = float(reaction["direct_transmission_rate_preclinical"])
        self.beta1R = float(reaction["direct_transmission_rate_clinical"])
        self.rhoI = float(reaction["environmental_shedding_rate_preclinical"])
        self.rhoR = float(reaction["environmental_shedding_rate_clinical"])
        self.delta = float(reaction["environmental_decay_rate"])
        self.beta2 = float(reaction["environmental_transmission_rate"])
        self.K0 = float(parameters["land_cover"]["carrying_capacity_base"])
        self.c1 = float(control["cost_susceptible_deficit"])
        self.c2 = float(control["cost_environment"])
        self.c3 = float(control["cost_control"])
        self.c4 = float(control["cost_control_l1"])
        self.terminal_scale = float(control["terminal_cost_scale"])


# --------------------------------------------------------------------------
# transcriptions of the UFL forms
# --------------------------------------------------------------------------

def reaction(y, v, K, p):
    """F(y, v), transcribed from state_rhs_form, including the K < 0.01 branch."""
    S, E, I, R, W = y
    if K < 0.01:
        growth = -p.r * S
    else:
        growth = p.r * S * (1.0 - (S + E + I + R) / K)

    infection_density = p.beta1I * I + p.beta1R * R + p.beta2 * W
    infection = S * infection_density

    return np.array([
        growth - infection,                                       # S
        infection - p.sigma1 * E,                                 # E
        p.sigma1 * E - p.sigma2 * I,                              # I
        p.sigma2 * I - p.alpha * R - v * R,                       # R
        p.rhoI * I + p.rhoR * R - p.delta * W,                    # W
    ])


def running_cost(y, v, S_infinity, p):
    """l(y, v), transcribed from state_cost_form and control_cost_form."""
    S, _E, _I, _R, W = y
    return (p.c1 * (S_infinity - S) + p.c2 * W
            + p.c3 * v**2 + p.c4 * v)


def adjoint_rows(y, lam, v, K, p):
    """The coefficient of each psi_k in adjoint_rhs_form, including the
    running-cost source; transcribed, not derived."""
    S, E, I, R, W = y
    lamS, lamE, lamI, lamR, lamW = lam

    if K < 0.01:
        growth_dS = -p.r
        growth_dN = 0.0
    else:
        growth_dS = p.r * (1.0 - (S + E + I + R) / K) - p.r * S / K
        growth_dN = -p.r * S / K

    infection_density = p.beta1I * I + p.beta1R * R + p.beta2 * W

    row_S = (lamS * (growth_dS - infection_density)
             + lamE * infection_density
             - p.c1)                       # d/dS of c1*(S_inf - S)

    row_E = (lamS * growth_dN
             - lamE * p.sigma1
             + lamI * p.sigma1)

    row_I = (lamS * (growth_dN - p.beta1I * S)
             + lamE * p.beta1I * S
             - lamI * p.sigma2
             + lamR * p.sigma2
             + lamW * p.rhoI)

    row_R = (lamS * (growth_dN - p.beta1R * S)
             + lamE * p.beta1R * S
             - lamR * (p.alpha + v)
             + lamW * p.rhoR)

    row_W = (-lamS * p.beta2 * S
             + lamE * p.beta2 * S
             - lamW * p.delta
             + p.c2)                       # d/dW of c2*W

    return np.array([row_S, row_E, row_I, row_R, row_W])


def gradient_expression(y, lam, v, p):
    """dJ/dv, transcribed from gradient_form."""
    _S, _E, _I, R, _W = y
    _lamS, _lamE, _lamI, lamR, _lamW = lam
    return 2.0 * p.c3 * v + p.c4 - R * lamR


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------

def numerical_jacobian(y, v, K, p, h=1.0e-6):
    """dF/dy by central differences."""
    jacobian = np.zeros((5, 5))
    for column in range(5):
        step = np.zeros(5)
        step[column] = h
        jacobian[:, column] = (
            reaction(y + step, v, K, p) - reaction(y - step, v, K, p)
        ) / (2.0 * h)
    return jacobian


def numerical_cost_gradient(y, v, S_infinity, p, h=1.0e-6):
    """dl/dy by central differences."""
    out = np.zeros(5)
    for index in range(5):
        step = np.zeros(5)
        step[index] = h
        out[index] = (
            running_cost(y + step, v, S_infinity, p)
            - running_cost(y - step, v, S_infinity, p)
        ) / (2.0 * h)
    return out


def numerical_hamiltonian_control_gradient(y, lam, v, S_infinity, K, p,
                                           h=1.0e-6):
    """dH/dv by central differences on the Hamiltonian."""
    def hamiltonian(vv):
        return (running_cost(y, vv, S_infinity, p)
                + float(lam @ reaction(y, vv, K, p)))
    return (hamiltonian(v + h) - hamiltonian(v - h)) / (2.0 * h)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_adjoint_and_gradients(p):
    """Adjoint rows and the control gradient, at many random states."""
    rng = np.random.default_rng(20260904)
    worst_adjoint = 0.0
    worst_gradient = 0.0
    cases = 0
    for K in (p.K0, 0.5 * p.K0, 0.0):        # land, poor habitat, and water
        for _ in range(200):
            y = rng.uniform(0.0, 3.0, size=5)
            lam = rng.standard_normal(5) * 10.0
            v = rng.uniform(0.0, 1.0)
            S_infinity = max(K, 1.0e-3)

            expected = (numerical_jacobian(y, v, K, p).T @ lam
                        + numerical_cost_gradient(y, v, S_infinity, p))
            actual = adjoint_rows(y, lam, v, K, p)
            scale = max(1.0, float(np.max(np.abs(expected))))
            worst_adjoint = max(
                worst_adjoint, float(np.max(np.abs(actual - expected))) / scale
            )

            expected_g = numerical_hamiltonian_control_gradient(
                y, lam, v, S_infinity, K, p
            )
            actual_g = gradient_expression(y, lam, v, p)
            scale_g = max(1.0, abs(expected_g))
            worst_gradient = max(
                worst_gradient, abs(actual_g - expected_g) / scale_g,
            )
            cases += 1

    adjoint_ok = worst_adjoint < DERIVATIVE_TOLERANCE
    gradient_ok = worst_gradient < DERIVATIVE_TOLERANCE
    print(f"  adjoint transpose      {'PASS' if adjoint_ok else 'FAIL'}   "
          f"max relative residual {worst_adjoint:.2e} over {cases} random "
          f"states (land, poor habitat, water)")
    print(f"  control gradient       {'PASS' if gradient_ok else 'FAIL'}   "
          f"max relative residual {worst_gradient:.2e} on dH/dv over "
          f"the same states")
    return adjoint_ok and gradient_ok


def check_control_reaches_clinical_only(p):
    """v must change the R row only."""
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    base = reaction(y, 0.0, p.K0, p)
    only_v = reaction(y, 0.1, p.K0, p)

    v_touches = np.abs(only_v - base) > 1e-12
    # index order S, E, I, R, W
    ok = (
        not v_touches[0] and not v_touches[1] and not v_touches[2]
        and v_touches[3] and not v_touches[4]
    )
    labels = "SEIRW"
    print(f"  control reach          {'PASS' if ok else 'FAIL'}   "
          f"v acts on [{''.join(l for l, t in zip(labels, v_touches) if t)}]"
          "   (expected v on [R])")
    return ok


def endemic_equilibrium(p):
    """Newton on F(y, 0) = 0 at K = K0, from a seeded epizootic."""
    y = np.array([0.5 * p.K0, 0.1, 0.1, 0.05, 0.5])
    for _ in range(200):
        residual = reaction(y, 0.0, p.K0, p)
        jacobian = numerical_jacobian(y, 0.0, p.K0, p)
        y = y - np.linalg.solve(jacobian, residual)
    return y


def check_equilibrium(p, R0):
    y = endemic_equilibrium(p)
    S, E, I, R, W = y
    residual = float(np.max(np.abs(reaction(y, 0.0, p.K0, p))))
    predicted = p.K0 / R0
    drift = abs(S - predicted)
    ok = residual < EQUILIBRIUM_TOLERANCE and drift < 1.0e-9
    N = S + E + I + R
    print(f"  endemic equilibrium    {'PASS' if ok else 'FAIL'}   "
          f"S* = {S:.6f} vs K0/R0 = {predicted:.6f} (drift {drift:.2e}), "
          f"residual {residual:.2e}")
    print(f"                                E* = {E:.5f}  I* = {I:.5f}  "
          f"R* = {R:.5f}  W* = {W:.5f}  N* = {N:.5f}")
    print(f"                                prevalence {100*(E+I+R)/N:.1f}%, "
          f"decline {100*(1-N/p.K0):.1f}%, deficit S_inf - S* = "
          f"{p.K0 - S:.5f}")
    return ok, y


def drain_cost(p, y0, horizon=400.0, steps=2_000_000):
    """Running cost accumulated by the drain from y0, with no transmission.

    y0 is (eps, E, I, R, W) with eps = S_inf - S; explicit Euler on
    d(eps)/dt = -r*(eps - (E + I + R)) and the linear E, I, R, W drain.
    """
    dt = horizon / steps
    eps, E, I, R, W = y0
    total = 0.0
    for _ in range(steps):
        total += (p.c1 * eps + p.c2 * W) * dt
        J = E + I + R
        d_eps = -p.r * (eps - J)
        dE = -p.sigma1 * E
        dI = p.sigma1 * E - p.sigma2 * I
        dR = p.sigma2 * I - p.alpha * R
        dW = p.rhoI * I + p.rhoR * R - p.delta * W
        eps, E, I, R, W = (eps + dt * d_eps, E + dt * dE, I + dt * dI,
                           R + dt * dR, W + dt * dW)
    return total


def check_terminal_weights(p):
    """Integrate the drain and compare against the closed-form weights."""
    environment_tail = p.c2 / p.delta
    susceptible_tail = p.c1 / p.r
    clinical_tail = (p.c1 + p.rhoR * environment_tail) / p.alpha
    preclinical_tail = (
        (p.c1 + p.rhoI * environment_tail) / p.sigma2 + clinical_tail
    )
    exposed_tail = p.c1 / p.sigma1 + preclinical_tail

    closed_form = {
        "m_susceptible": susceptible_tail,
        "m_exposed": exposed_tail,
        "m_preclinical": preclinical_tail,
        "m_clinical": clinical_tail,
        "m_environment": environment_tail,
    }
    seeds = {
        "m_susceptible": (1.0, 0.0, 0.0, 0.0, 0.0),
        "m_exposed": (0.0, 1.0, 0.0, 0.0, 0.0),
        "m_preclinical": (0.0, 0.0, 1.0, 0.0, 0.0),
        "m_clinical": (0.0, 0.0, 0.0, 1.0, 0.0),
        "m_environment": (0.0, 0.0, 0.0, 0.0, 1.0),
    }
    ok = True
    for name, expected in closed_form.items():
        got = drain_cost(p, np.array(seeds[name]))
        relative = abs(got - expected) / max(1.0, abs(expected))
        good = relative < DRAIN_TOLERANCE
        ok = ok and good
        print(f"  {name:<21s}{'PASS' if good else 'FAIL'}   "
              f"closed form {expected * p.terminal_scale:10.4f}   "
              f"integrated drain {got * p.terminal_scale:10.4f}   "
              f"({relative:.1e})")

    if closed_form["m_exposed"] <= closed_form["m_preclinical"]:
        ok = False
        print("  m_exposed must exceed m_preclinical.")
    return ok


def check_horizon_complementarity(p, horizon=20.0):
    """(c2/delta)*(1 - exp(-delta*(T-s))) + mW*exp(-delta*(T-s)) must not
    depend on s when mW = c2/delta."""
    mW = p.c2 / p.delta
    totals = []
    for s in np.linspace(0.0, horizon, 21):
        decay = np.exp(-p.delta * (horizon - s))
        totals.append((p.c2 / p.delta) * (1.0 - decay) + mW * decay)
    totals = np.array(totals)
    spread = float(totals.max() - totals.min()) / max(1.0, float(totals.mean()))
    ok = spread < 1.0e-12

    terminal_only = np.array([
        mW * np.exp(-p.delta * (horizon - s))
        for s in (0.0, horizon)
    ])
    discount = terminal_only[0] / terminal_only[1]
    print(f"  {'horizon complementarity':<21s} {'PASS' if ok else 'FAIL'}  "
          f"total price of a unit of W is {totals.mean():.4f} regardless of "
          f"deposition time (spread {spread:.1e})")
    print(f"                                without the running half, year-0 "
          f"contamination would price at {100*discount:.1f}% of year-{horizon:.0f}")
    return ok


def check_effort_budget(seed=20260907):
    """Finite-difference the augmented-Lagrangian budget term.

    Synthetic arrays with lumped mass spanning five decades and a short final
    block. Checks: the gradient with cold, warm and mixed multipliers; that a
    slack budget with mu = 0 is exactly zero; that a frozen multiplier equals a
    linear weight c4; and that the update equals the current slope.
    """
    rng = np.random.default_rng(seed)
    n_nodes, n_blocks = 37, 6
    mass = 10.0 ** rng.uniform(-2.0, 3.0, n_nodes)
    measure = float(mass.sum())
    # A short final block.
    weights = np.array([20, 20, 20, 20, 20, 7], dtype=float) * 0.05

    def inner(a, b):
        return float(np.einsum("bi,bi,i,b->", a, b, mass, weights))

    def means_of(field):
        return (field @ mass) / measure

    def directional(budget, field, trials=6):
        """Worst relative error between the analytic and numeric derivative."""
        means = means_of(field)
        gradient = np.broadcast_to(
            budget.gradient_term(means)[:, None], field.shape
        )
        worst = 0.0
        for _ in range(trials):
            d = rng.normal(size=field.shape)
            analytic = inner(gradient, d)
            h = 1.0e-6 * max(1.0, float(np.abs(field).max()))
            numeric = (budget.value(means_of(field + h * d))
                       - budget.value(means_of(field - h * d))) / (2.0 * h)
            denom = max(abs(analytic), abs(numeric), 1.0e-30)
            worst = max(worst, abs(analytic - numeric) / denom)
        return worst

    v = rng.uniform(0.5, 2.0, (n_blocks, n_nodes))
    targets = np.array([0.2, 5.0, 0.4, 9.0, 0.1, 0.3])
    errors = []
    checks = []

    def build(targets, penalty, multipliers=None):
        return EffortBudget(
            targets, penalty=penalty, multipliers=multipliers
        ).resolve(n_blocks).set_measure(measure, weights)

    cold = build(0.2, 3.0)
    errors.append(directional(cold, v))

    warm = build(0.2, 3.0, rng.uniform(0.1, 2.0, n_blocks))
    errors.append(directional(warm, v))

    slack = build(50.0, 3.0)
    slack_means = means_of(v)
    checks.append(
        slack.value(slack_means) == 0.0
        and float(np.abs(slack.gradient_term(slack_means)).max()) == 0.0
    )

    mixed = build(targets, 7.0, np.array([0.0, 0.0, 1.5, 0.0, 0.4, 0.0]))
    errors.append(directional(mixed, v))

    # Frozen multiplier (negligible rho) against a linear weight c4.
    prescribed = 0.37
    frozen = build(targets, 1.0e-12, np.full(n_blocks, prescribed))
    # Compared entry by entry, since c4*v contributes exactly c4.
    from_budget = frozen.gradient_term(means_of(v))
    linear_error = float(
        np.max(np.abs(from_budget - prescribed)) / prescribed
    )
    checks.append(linear_error < 1.0e-9)

    stepped = build(targets, 7.0, np.array([0.0, 0.0, 1.5, 0.0, 0.4, 0.0]))
    stepped_means = means_of(v)
    expected = stepped.slopes(stepped_means).copy()
    _, updated = stepped.update(stepped_means)
    checks.append(bool(np.array_equal(updated, expected)))

    worst = max(errors)
    ok = worst < 2.0e-6 and all(checks)
    print(f"  {'effort budget':<21s} {'PASS' if ok else 'FAIL'}   "
          f"worst gradient error {worst:.2e} over cold, warm and mixed "
          f"active sets")
    print(f"                                slack case exactly off; "
          f"prescribed mu reproduces a linear c4 to {linear_error:.1e}")
    return ok


def main():
    parameters = load_parameters()
    p = Rates(parameters)
    R0_direct, R0_environmental = reproduction_numbers(parameters)
    R0 = R0_direct + R0_environmental

    print("Five-compartment, single-control reaction check (no dolfinx required)")
    print(f"  R0_direct = {R0_direct:.6f}   R0_environmental = "
          f"{R0_environmental:.6f}   R0_nom = {R0:.6f}")
    print(f"  stage durations: E {1/p.sigma1:.5f} yr, I {1/p.sigma2:.5f} yr, "
          f"R {1/p.alpha:.5f} yr   (shedding window "
          f"{1/p.sigma2 + 1/p.alpha:.6f} yr)")
    print(f"  cost weights: c1 = {p.c1}, c2 = {p.c2}, c3 = {p.c3}, "
          f"c4 = {p.c4}")
    print()

    results = [
        check_adjoint_and_gradients(p),
        check_control_reaches_clinical_only(p),
        check_equilibrium(p, R0)[0],
        check_terminal_weights(p),
        check_horizon_complementarity(p),
        check_effort_budget(),
    ]
    print()
    if all(results):
        print("All reaction checks passed. Run CWD_optimal_control.py "
              "--gradient-check for the mesh-level Taylor test.")
        return 0
    print("FAILURES above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
