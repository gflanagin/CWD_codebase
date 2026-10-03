"""Compartment-blocked direct solver for the implicit operator B.

The bilinear form has no cross-compartment terms, so

    B = M_L + dt * A,   A = blockdiag(D_S*A_D, D_E*A_D, D_I*A_D, D_R*A_D, 0)

is block diagonal. Each diffusive block is assembled and factorized once, and
blocks with equal mobility share a factorization; the W block is the diagonal
lumped mass and is solved by a multiply. The result is the same linear system
as the monolithic one.

``configure_direct_solver`` prefers MUMPS when available and calls
``setFromOptions`` last, so PETSC_OPTIONS can override it.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import TestFunction, TrialFunction, dot, grad


def configure_direct_solver(ksp, prefix=None):
    """Set up a KSP as a cached direct solve, preferring MUMPS when available."""
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")

    try:
        has_mumps = PETSc.Sys.hasExternalPackage("mumps")
    except Exception:
        # Older petsc4py builds lack the query.
        has_mumps = False
    if has_mumps:
        pc.setFactorSolverType("mumps")

    if prefix is not None:
        ksp.setOptionsPrefix(prefix)
    # Last, so that anything in PETSC_OPTIONS overrides the choices above.
    ksp.setFromOptions()
    return ksp


def _dofmap_array(V):
    """The cell-to-DOF connectivity of ``V`` as a flat array (any DOLFINx version)."""
    listing = V.dofmap.list
    return np.asarray(getattr(listing, "array", listing))


def lumped_mass_diagonal(V, mass_measure):
    """diag(M_L) on ``V``: the row sums int(phi_i) under ``mass_measure``."""
    v = TestFunction(V)
    b = assemble_vector(fem.form(v * mass_measure))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    values = b.array_r.copy()
    b.destroy()
    return values


class CompartmentBlockSolver:
    """Direct solves against B, one cached factorization per distinct block.

    ``solve(b, out)`` takes an assembled right-hand side on the mixed space
    (ghost-updated) and writes B^{-1} b into the mixed ``Function`` ``out``.
    """

    def __init__(self, P, dt, DTens, compartment_scales,
                 mass_measure, stiffness_measure, comm=MPI.COMM_WORLD):
        # compartment_scales lists the diffusive compartments in mixed-space
        # order; W is the last sub-space.
        self.comm = comm
        if comm.size > 1:
            # The gather/scatter below assumes no ghosts.
            raise NotImplementedError(
                "CompartmentBlockSolver supports serial runs only."
            )

        self.P = P
        self.n_diffusive = len(compartment_scales)
        n_compartments = self.n_diffusive + 1          # + W
        if P.num_sub_spaces != n_compartments:
            raise ValueError(
                f"the mixed space has {P.num_sub_spaces} sub-spaces but "
                f"{len(compartment_scales)} compartment scales were given, "
                f"implying {n_compartments} compartments. The scales must "
                "list every DIFFUSIVE compartment, in mixed-space order, and "
                "exclude W."
            )
        blocks = [P.sub(k).collapse() for k in range(n_compartments)]
        self.spaces = [space for space, _ in blocks]
        self.dof_maps = [np.asarray(dofs, dtype=np.int32) for _, dofs in blocks]

        V = self.spaces[0]
        # One assembled block serves every compartment only if the collapsed
        # sub-spaces share a DOF map; check it.
        reference = _dofmap_array(V)
        for k, space in enumerate(self.spaces[1:], start=1):
            if not np.array_equal(_dofmap_array(space), reference):
                raise RuntimeError(
                    "Collapsed sub-space %d has a different DOF map from "
                    "sub-space 0, so the compartment blocks cannot share an "
                    "assembled operator." % k
                )

        u = TrialFunction(V)
        v = TestFunction(V)

        # W block: pure lumped mass, hence diagonal. Stored as its reciprocal
        # so the per-step solve is a multiply.
        mass_diagonal = lumped_mass_diagonal(V, mass_measure)
        if np.any(mass_diagonal <= 0.0):
            raise RuntimeError(
                "The lumped mass matrix has a non-positive diagonal entry, so "
                "the mesh has a degenerate cell; the W block is not invertible."
            )
        self.inverse_mass_diagonal = 1.0 / mass_diagonal

        # One factorization per distinct mobility.
        self.matrices = {}
        self.solvers = {}
        self.block_keys = []
        for index, scale in enumerate(compartment_scales):
            key = repr(float(scale))
            if key not in self.solvers:
                a = (
                    u * v * mass_measure
                    + dt * float(scale)
                    * dot(DTens * grad(u), grad(v)) * stiffness_measure
                )
                matrix = assemble_matrix(fem.form(a))
                matrix.assemble()
                ksp = PETSc.KSP().create(V.mesh.comm)
                ksp.setOperators(matrix)
                configure_direct_solver(ksp, prefix="block%d_" % index)
                self.matrices[key] = matrix
                self.solvers[key] = ksp
            self.block_keys.append(key)

        # Scratch vectors, allocated once rather than per solve.
        any_matrix = self.matrices[self.block_keys[0]]
        self.block_rhs = any_matrix.createVecRight()
        self.block_solution = any_matrix.createVecRight()

    @property
    def distinct_factorizations(self):
        """How many LU factorizations are actually held (2 of 4 typically)."""
        return len(self.solvers)

    @property
    def factor_package(self):
        """Which package is doing the factorization, for the start-up banner."""
        ksp = self.solvers[self.block_keys[0]]
        try:
            return str(ksp.getPC().getFactorSolverType())
        except Exception:
            return "unknown"

    def solve(self, b, out_function):
        """B x = b, blockwise. ``b`` must already be ghost-updated."""
        rhs = b.array_r
        out = out_function.x.array

        for index, key in enumerate(self.block_keys):
            dof_map = self.dof_maps[index]
            self.block_rhs.array[:] = rhs[dof_map]
            self.solvers[key].solve(self.block_rhs, self.block_solution)
            out[dof_map] = self.block_solution.array_r

        # W: diagonal lumped mass, so no triangular solve at all.
        dof_map = self.dof_maps[self.n_diffusive]
        out[dof_map] = rhs[dof_map] * self.inverse_mass_diagonal

        out_function.x.scatter_forward()

    def destroy(self):
        for ksp in self.solvers.values():
            ksp.destroy()
        for matrix in self.matrices.values():
            matrix.destroy()
        self.block_rhs.destroy()
        self.block_solution.destroy()
        self.solvers.clear()
        self.matrices.clear()
