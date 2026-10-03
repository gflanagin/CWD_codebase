#!/usr/bin/env bash
# One-time setup on an HTCondor submit node with a shared filesystem.
# Idempotent: each step checks for its own result first.
#
#   export CWD_CLUSTER_ROOT=/path/to/shared/cwd     # holds repo/, conda/, jit_cache/
#   bash $CWD_CLUSTER_ROOT/repo/condor/bootstrap_cluster.sh
#
# Installs Miniforge and the cwd-fem environment under the root, checks that
# the PETSc/DOLFINx stack imports, and compiles one form to warm the JIT cache.
set -euo pipefail

ROOT=${CWD_CLUSTER_ROOT:?set CWD_CLUSTER_ROOT to the shared deployment root}
ENV_PREFIX="$ROOT/conda/cwd-fem"
CONDA_ROOT="$ROOT/conda/miniforge"
REPO=${CWD_CLUSTER_REPO:-$ROOT/repo}

say() { printf '\n=== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. sanity
say "checking the deployment root"
# HTCondor submit files cannot quote paths containing whitespace.
case "$ROOT$REPO" in
    *[[:space:]]*) die "deployment paths must not contain whitespace: $ROOT / $REPO" ;;
esac
mkdir -p "$ROOT"
[[ -w $ROOT ]] || die "$ROOT is not writable by $USER"

df -h "$ROOT" | tail -1 | awk '{print "  free on the share: " $4 " of " $2}'

# ---------------------------------------------------- 1. conda distribution
if [[ -x $CONDA_ROOT/bin/conda ]]; then
    say "miniforge already installed at $CONDA_ROOT"
else
    say "installing miniforge into $CONDA_ROOT"
    installer=$(mktemp "${TMPDIR:-/tmp}/miniforge.XXXXXX.sh")
    # Miniforge defaults to conda-forge, where fenics-dolfinx lives.
    curl -fsSL -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ------------------------------------------------------------ 2. the repo
say "checking the repository at $REPO"
if [[ -f $REPO/CWD_optimal_control.py ]]; then
    echo "  found the driver"
else
    die "no CWD_optimal_control.py under $REPO. Clone the repository there first."
fi

# ------------------------------------------------------- 3. the environment
if [[ -x $ENV_PREFIX/bin/python ]]; then
    say "environment already present at $ENV_PREFIX"
else
    say "creating the cwd-fem environment (this takes 10-20 minutes)"
    [[ -f $REPO/environment.yml ]] || die "no environment.yml in $REPO"
    # --prefix keeps the environment on the shared root.
    conda env create --prefix "$ENV_PREFIX" --file "$REPO/environment.yml"
fi

# ------------------------------------------------------------ 4. preflight
say "preflight"
"$ENV_PREFIX/bin/python" - <<'PY'
import importlib
import sys

# Import each package, which also checks its shared libraries load.
failed = []
for module in ("numpy", "scipy", "mpi4py", "petsc4py", "basix", "ufl",
               "dolfinx"):
    try:
        importlib.import_module(module)
    except Exception as error:                        # noqa: BLE001
        failed.append(f"{module}: {error}")
if failed:
    print("  IMPORTS FAILED:")
    for line in failed:
        print("   ", line)
    sys.exit(1)
import dolfinx
print(f"  python  {sys.version.split()[0]}")
print(f"  dolfinx {dolfinx.__version__}")
PY

# FFCx needs the environment's gcc, which run_solve.sh puts on PATH.
if [[ -x $ENV_PREFIX/bin/gcc || -n $(ls "$ENV_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null) ]]; then
    echo "  gcc      present next to the interpreter"
else
    echo "  WARNING: no gcc in $ENV_PREFIX/bin -- FFCx will not be able to JIT."
    echo "           conda install --prefix $ENV_PREFIX -c conda-forge gcc_linux-64"
fi

# Compile one form: tests the JIT and warms the shared cache.
say "compiling one form to prove the JIT works"
XDG_CACHE_HOME="$ROOT/jit_cache" PATH="$ENV_PREFIX/bin:$PATH" \
    "$ENV_PREFIX/bin/python" - <<'PY'
from mpi4py import MPI
import dolfinx.fem as fem
import ufl
from dolfinx.mesh import create_unit_square

mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(mesh, ("Lagrange", 1))
form = fem.form(ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx)
fem.assemble_matrix(form)
print("  JIT ok")
PY

say "done"
cat <<EOF

  environment : $ENV_PREFIX/bin/python
  repository  : $REPO
  JIT cache   : $ROOT/jit_cache

  Next: see condor/README.md.
EOF
