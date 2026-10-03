"""The per-block effort budget and its augmented Lagrangian.

Free of dolfinx and MPI so that utilities/check_reaction_adjoint.py can test it.

Constraint, one row per control block:

    g_b(v) = (1/|Gamma|) * int_Gamma v_b dGamma - B_b <= 0.

Each row is weighted by its space-time measure m_b = |Gamma| * w_b, so the
penalty scales like J:

    L_rho(v, mu) = J(v) + sum_b m_b * (rho/2) * [ max(0, g_b + mu_b/rho)^2
                                                  - (mu_b/rho)^2 ],

with mu_b <- max(0, mu_b + rho*g_b) after each inner minimization.
"""

from __future__ import annotations

import numpy as np


class EffortBudget:
    """Per-block effort budget on the control, as an augmented Lagrangian.

    Call ``resolve`` and then ``set_measure`` before ``value`` or
    ``gradient_term``.
    """

    def __init__(self, targets, penalty, multipliers=None):
        self.targets = np.atleast_1d(np.asarray(targets, dtype=np.float64))
        if np.any(self.targets < 0.0):
            raise ValueError("effort budgets cannot be negative")
        if not np.all(np.isfinite(self.targets)):
            raise ValueError("effort budgets must be finite")
        self.penalty = float(penalty)
        if not self.penalty > 0.0:
            raise ValueError(
                "the augmented-Lagrangian penalty rho must be positive"
            )
        if multipliers is None:
            self.multipliers = np.zeros_like(self.targets)
        else:
            self.multipliers = np.atleast_1d(
                np.asarray(multipliers, dtype=np.float64)
            ).astype(np.float64, copy=True)
            if np.any(self.multipliers < 0.0):
                raise ValueError("budget multipliers cannot be negative")
        self.measure_weights = None

    # ------------------------------------------------------------------ setup

    def resolve(self, n_blocks):
        """Broadcast scalar budgets and multipliers to one value per block."""
        if self.targets.size == 1 and n_blocks != 1:
            self.targets = np.repeat(self.targets, n_blocks)
        if self.multipliers.size == 1 and n_blocks != 1:
            self.multipliers = np.repeat(self.multipliers, n_blocks)
        if self.targets.size != n_blocks:
            raise ValueError(
                f"control_budget has {self.targets.size} entries but the "
                f"horizon has {n_blocks} control blocks; give one value per "
                "block, or a single value to apply to all of them"
            )
        if self.multipliers.shape != self.targets.shape:
            raise ValueError(
                "budget multipliers and budgets must have the same shape"
            )
        return self

    def set_measure(self, domain_measure, block_weights):
        """Record m_b = |Gamma| * w_b. Returns self."""
        measure = float(domain_measure)
        if not measure > 0.0:
            raise ValueError("domain measure must be positive")
        weights = np.asarray(block_weights, dtype=np.float64)
        if np.any(weights <= 0.0):
            raise ValueError("block weights must be positive")
        if weights.shape != self.targets.shape:
            raise ValueError(
                f"block weights have shape {weights.shape} but the budget has "
                f"{self.targets.shape}; call resolve(n_blocks) first"
            )
        self.measure_weights = measure * weights
        return self

    def _require_measure(self):
        if self.measure_weights is None:
            raise RuntimeError(
                "EffortBudget.set_measure has not been called."
            )
        return self.measure_weights

    # ------------------------------------------------------------- the algebra

    def violation(self, means):
        """g_b: how far block b overspends its budget, in 1/yr."""
        return np.asarray(means, dtype=np.float64) - self.targets

    def slopes(self, means):
        """max(0, mu_b + rho*g_b), also the next multiplier value."""
        return np.maximum(
            0.0, self.multipliers + self.penalty * self.violation(means)
        )

    def value(self, means):
        """The augmented-Lagrangian penalty added to J."""
        weights = self._require_measure()
        shifted = self.slopes(means) / self.penalty
        offset = self.multipliers / self.penalty
        return 0.5 * self.penalty * float(
            np.sum(weights * (shifted ** 2 - offset ** 2))
        )

    def gradient_term(self, means):
        """The budget's contribution to the L2 reduced gradient, per block.

        With the measure weighting, the lumped mass and |Gamma| cancel and the
        contribution is the per-block constant slopes(means), shape
        (n_blocks,); the caller broadcasts it over nodes.
        """
        self._require_measure()
        return self.slopes(means)

    # -------------------------------------------------------------- outer loop

    def update(self, means):
        """First-order multiplier update. Returns (violation, new multipliers)."""
        violation = self.violation(means)
        self.multipliers = self.slopes(means)
        return violation, self.multipliers.copy()

    def worst_violation(self, means):
        """The largest overspend across blocks, in 1/yr. Negative if slack."""
        violation = self.violation(means)
        return float(violation.max()) if violation.size else 0.0

    def satisfied(self, means, tolerance):
        """True if every block's overspend is within `tolerance` * its budget."""
        scale = np.maximum(self.targets, 1e-12)
        return bool(np.all(self.violation(means) <= tolerance * scale))

    def report(self, means, control_name=None):
        """Diagnostics for the driver log and the history file."""
        violation = self.violation(means)
        return {
            "control": control_name,
            "targets": self.targets.tolist(),
            "realized": np.asarray(means, dtype=np.float64).tolist(),
            "violation": violation.tolist(),
            "worst_violation": self.worst_violation(means),
            "multipliers": self.multipliers.tolist(),
            "penalty": self.penalty,
        }
