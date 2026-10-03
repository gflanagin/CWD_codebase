#!/usr/bin/env bash
# One HTCondor job: the uncontrolled baseline for a spin-up group, or one
# controlled sweep point.
#
#   run_solve.sh baseline <representative-point> <group>
#   run_solve.sh point    <point> <group>
#
# Settings come from CWD_SWEEP_ENV, written by utilities/make_condor_sweep.py.
# Assumes a shared filesystem.
set -uo pipefail

MODE=${1:?usage: run_solve.sh <baseline|point> <name> <group>}
NAME=${2:?missing name}
GROUP=${3:?missing group}

: "${CWD_SWEEP_ENV:?CWD_SWEEP_ENV is unset -- the submit file must set it}"
# shellcheck disable=SC1090
source "$CWD_SWEEP_ENV"

: "${CWD_PYTHON:?}" "${CWD_REPO:?}" "${CWD_MESH:?}" "${CWD_PARAMS:?}"
: "${CWD_OUTROOT:?}" "${CWD_DRIVER:?}"

cd "$CWD_REPO" || { echo "cannot cd to $CWD_REPO" >&2; exit 1; }

# FFCx needs the environment's gcc on PATH.
PATH="$(cd "$(dirname "$CWD_PYTHON")" && pwd):${PATH:-/usr/bin:/bin}"
export PATH
if ! command -v gcc >/dev/null 2>&1; then
    echo "no gcc on PATH next to $CWD_PYTHON -- FFCx cannot JIT-compile forms." >&2
    exit 1
fi

# One thread per job.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Shared FFCx JIT cache, warmed by the DAG's first node.
export XDG_CACHE_HOME=${CWD_JIT_CACHE:-$CWD_OUTROOT/.cache}
mkdir -p "$XDG_CACHE_HOME"

SPINUP_CACHE="$CWD_OUTROOT/spinup_cache"
mkdir -p "$SPINUP_CACHE"

# Trajectory memmap on the execute node's scratch.
TRAJ_DIR=${_CONDOR_SCRATCH_DIR:-${CWD_SCRATCH:-$CWD_OUTROOT/.trajectories}}
mkdir -p "$TRAJ_DIR"

started=$(date +%s)

if [[ $MODE == baseline ]]; then
    OUT="$CWD_OUTROOT/baseline_$GROUP"
    mkdir -p "$OUT"
    # Skip a completed baseline so retries and rescue DAGs are safe.
    if [[ -f $SPINUP_CACHE/$GROUP.npy && -f $OUT/forward_summary.json ]]; then
        echo "[skip] baseline $GROUP already complete"
        exit 0
    fi
    echo "[start] baseline $GROUP  (point $NAME, v = 0)"
    # Driver status from PIPESTATUS, since tee is last in the pipe.
    "$CWD_PYTHON" "$CWD_DRIVER" \
        --mesh-folder "$CWD_MESH" \
        --parameters "$CWD_PARAMS/$NAME.json" \
        --output-folder "$OUT" \
        --disease-state "$SPINUP_CACHE/$GROUP.npy" \
        --susceptible-equilibrium "$CWD_MESH/susceptible_equilibrium.npy" \
        --forward-only --output-every "${CWD_OUTPUT_EVERY:-40}" \
        2>&1 | tee "$OUT/baseline.log"
    status=${PIPESTATUS[0]}
    label="baseline $GROUP"
else
    OUT="$CWD_OUTROOT/$NAME"
    mkdir -p "$OUT"
    if [[ ${CWD_REDO:-0} != 1 && -f $OUT/optimization_history.json ]]; then
        echo "[skip] $NAME already complete"
        exit 0
    fi
    if [[ ! -f $SPINUP_CACHE/$GROUP.npy ]]; then
        echo "no spin-up cache $SPINUP_CACHE/$GROUP.npy for group $GROUP." >&2
        echo "  Its baseline node must run first; check the DAG edges." >&2
        exit 1
    fi
    echo "[start] $NAME  (group $GROUP)"
    # Warm-start an evicted point from its last checkpoint. L-BFGS-B
    # curvature pairs are not saved, so this is a restart from that iterate.
    RESUME=()
    if [[ -f $OUT/control_checkpoint.npy ]]; then
        done_iters=$("$CWD_PYTHON" -c "import json,sys
try:
    print(json.load(open(sys.argv[1]))['iteration'])
except Exception:
    print('?')" "$OUT/control_checkpoint.json" 2>/dev/null || echo '?')
        echo "[resume] $NAME from control_checkpoint.npy (${done_iters} iteration(s) already accepted)"
        RESUME=(--initial-control "$OUT/control_checkpoint.npy")
    fi
    # shellcheck disable=SC2086
    "$CWD_PYTHON" "$CWD_DRIVER" \
        --mesh-folder "$CWD_MESH" \
        --parameters "$CWD_PARAMS/$NAME.json" \
        --output-folder "$OUT" \
        --disease-state "$SPINUP_CACHE/$GROUP.npy" \
        --susceptible-equilibrium "$CWD_MESH/susceptible_equilibrium.npy" \
        --trajectory-file "$TRAJ_DIR/$NAME.npy" \
        "${RESUME[@]}" \
        ${CWD_EXTRA_ARGS:-} \
        2>&1 | tee -a "$OUT/solve.log"
    status=${PIPESTATUS[0]}
    # Remove the trajectory memmap.
    rm -f "$TRAJ_DIR/$NAME.npy"
    label="$NAME"
fi

elapsed=$(( $(date +%s) - started ))
if [[ $status -eq 0 ]]; then
    echo "[done] $label  (${elapsed}s)"
else
    echo "[FAIL] $label  (${elapsed}s, exit $status)" >&2
fi
exit $status
