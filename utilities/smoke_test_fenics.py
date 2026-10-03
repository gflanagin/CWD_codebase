"""Small distributed DOLFINx/PETSc assembly test for a new environment."""

from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
import ufl


comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
space = fem.functionspace(domain, ("Lagrange", 1))
trial = ufl.TrialFunction(space)
test = ufl.TestFunction(space)

bilinear_form = fem.form(
    (ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * ufl.dx
)
matrix = assemble_matrix(bilinear_form)
matrix.assemble()

if comm.rank == 0:
    print(
        f"DOLFINx/PETSc smoke test passed with {comm.size} MPI rank(s); "
        f"global matrix size={matrix.getSize()}"
    )
