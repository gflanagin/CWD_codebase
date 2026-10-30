"""Radial state and control at (delta = 0.25, R0 = 3.5) under three objectives.

The reference objective (c1 = 10, c2 = 0) against two arms that move the
state weight onto W, with c2 scaled so the uncontrolled J is unchanged.

    python utilities/make_transect_figure.py

Needs only h5py and matplotlib.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flat_transect import plot  # noqa: E402

CELL = "delta_0p25_R0_3p5__c4_0p5"
RUNS = [
    (f"cluster_runs/flat_grid/{CELL}", r"$c_1 = 10$, $c_2 = 0$"),
    (f"cluster_runs/flat_c2/{CELL}__c1_5_c2_12p21", r"$c_1 = 5$, $c_2 = 12.21$"),
    (f"cluster_runs/flat_c2/{CELL}__c1_0_c2_24p41", r"$c_1 = 0$, $c_2 = 24.41$"),
]
YEARS = [0, 5, 10, 15, 19]
OUT = Path("figures/flat_c2/transect_c1_c2.png")

if __name__ == "__main__":
    plot([Path(r) for r, _ in RUNS], OUT, YEARS, 0.25,
         labels=[lab for _, lab in RUNS])
