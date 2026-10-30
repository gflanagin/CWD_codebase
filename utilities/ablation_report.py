"""Report the landscape-ablation results for one ablation root.

    python utilities/ablation_report.py cluster_runs/ablation upper_arkansas --focus 39 -44
    python utilities/ablation_report.py cluster_runs/ablation estes_park carbondale

ROOT holds <region>_<arm> folders (arm in aniso_lc, aniso_nolc, iso_lc,
iso_nolc), each with one baseline_<cell> and one <cell>__c4_0p5 point; the four
arms of a region must share a mesh. Everything is read from the XDMF/HDF5
output, with the lumped mass as the P1 row sum; the recomputed space-time mean
of v is checked against optimization_history.json.

Per arm: deficit and J reduction against the arm's baseline, handover time,
realized effort, never-treated area, location of the top 5% of v in years 1-8,
mass-weighted |grad v| at year 5, elevation structure of the time-mean
control, and which domain edges the infection reaches. Between arms: the
|grad S_inf| ratio binned by relief, and the RMS difference and correlation of
the time-mean controls.
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

ARMS = ["aniso_lc", "aniso_nolc", "iso_lc", "iso_nolc"]
# Default foci in mesh coordinates (m); override with --focus.
FOCUS = {"park_range": (16.0, 1864.0), "upper_arkansas": (3829.0, -24.0),
         "estes_park": (7045.0, -7045.0), "carbondale": (45.0, -10045.0)}
BLOCKS = 20              # control blocks of one year on [0, T_c]
RELIEF_BINS = [0, 100, 250, 400, 600, np.inf]   # m/km
FRONT = 0.1              # E+I+R contour taken as the infected edge (deer/km^2)


def h5(d, name):
    return h5py.File(d / f"CWD_{name}.h5", "r")


def frame(d, name, fr):
    with h5(d, name) as f:
        group = f["Function"][next(iter(f["Function"]))]
        return group[str(fr)][:, 0]


def infected(d, fr):
    return sum(frame(d, n, fr) for n in ("E", "I", "R"))


class Mesh:
    def __init__(self, d):
        with h5(d, "control_v") as f:
            self.xyz = f["Mesh/mesh/geometry"][:]
            self.tri = f["Mesh/mesh/topology"][:]
        p = self.xyz[self.tri]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        self.area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        self.mass = np.bincount(self.tri.ravel(), np.repeat(self.area / 3, 3),
                                len(self.xyz))
        self.A = self.mass.sum()
        # surface gradient: g = a1 e1 + a2 e2 with g.e1, g.e2 the P1 increments
        G = np.stack([np.stack([(e1 * e1).sum(1), (e1 * e2).sum(1)], -1),
                      np.stack([(e1 * e2).sum(1), (e2 * e2).sum(1)], -1)], -2)
        self._Ginv, self._e = np.linalg.inv(G), (e1, e2)
        # relief: horizontal slope of the surface, m/km, per triangle
        H = np.stack([e1[:, :2], e2[:, :2]], 1)
        dz = np.stack([e1[:, 2], e2[:, 2]], 1)[..., None]
        self.relief = 1000 * np.linalg.norm(np.linalg.solve(H, dz)[..., 0], axis=1)
        self.z = self.xyz[:, 2]
        edges = np.sort(np.concatenate([self.tri[:, [0, 1]], self.tri[:, [1, 2]],
                                        self.tri[:, [2, 0]]]), 1)
        u, c = np.unique(edges, axis=0, return_counts=True)
        self.boundary = np.unique(u[c == 1])

    def grad_norm(self, f):
        """|surface gradient| per triangle, per km of the field's unit."""
        df = f[self.tri[:, 1:]] - f[self.tri[:, :1]]
        a = np.einsum("tij,tj->ti", self._Ginv, df)
        g = a[:, :1] * self._e[0] + a[:, 1:] * self._e[1]
        return 1000 * np.linalg.norm(g, axis=1)

    def mean(self, f):
        return (self.mass * f).sum() / self.A

    def tri_mean(self, g, sel=slice(None)):
        return (self.area[sel] * g[sel]).sum() / self.area[sel].sum()

    def quantile_bins(self, f, n):
        """Node index sets splitting the mesh area into n equal parts by f."""
        order = np.argsort(f)
        cum = np.cumsum(self.mass[order]) / self.A
        k = np.minimum((cum * n).astype(int), n - 1)
        return [order[k == i] for i in range(n)]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def deficit(record):
    return (record["cost_breakdown"]["running_state"]["terms"]
            ["susceptible_deficit"]["integral"])


def objective(record):
    return record.get("objective", record["cost_breakdown"].get("objective"))


def arm_dirs(root, region, arm):
    """(baseline, control point) folders, or None while the point is unfinished."""
    a = root / f"{region}_{arm}"
    p = next(a.glob("*__c4_0p5"), None)
    if p is None or not (p / "optimization_history.json").is_file():
        return None
    return (next(a.glob("baseline_*")), p)


def arm_stats(mesh, region, base_dir, point_dir):
    base = load(base_dir / "forward_summary.json")
    hist = load(point_dir / "optimization_history.json")
    control = next(iter(hist["controls"].values()))
    ho = re.search(r"after ([0-9.]+) yr, at ([0-9.]+)%",
                   (base_dir / "baseline.log").read_text(errors="ignore"))
    v = np.array([frame(point_dir, "control_v", b) for b in range(BLOCKS)])
    vbar = v.mean(0)
    effort = mesh.mean(vbar)
    zero = (v <= 1e-9).all(0)
    s = {
        "deficit_red": 100 * (1 - deficit(hist) / deficit(base)),
        "J_red": 100 * (1 - objective(hist) / objective(base)),
        "handover_yr": float(ho.group(1)), "handover_prev": float(ho.group(2)),
        "effort": effort, "effort_solver": control["space_time_mean"],
        "never_km2": mesh.mass[zero].sum() / 1e6,
        "never_frac": mesh.mass[zero].sum() / mesh.A,
        "at_vmax": 100 * control["fraction_at_upper_bound"],
    }
    s["def_per_effort"] = s["deficit_red"] / effort
    # top 5% of the mesh area by v, years 1-8: centroid relative to the focus
    fx, fy = FOCUS[region]
    off = []
    for b in range(8):
        order = np.argsort(-v[b])
        top = order[np.cumsum(mesh.mass[order]) <= 0.05 * mesh.A]
        w = mesh.mass[top]
        off.append(((w * mesh.xyz[top, 0]).sum() / w.sum() - fx,
                    (w * mesh.xyz[top, 1]).sum() / w.sum() - fy))
    s["top5_dx_km"] = np.array(off)[:, 0] / 1000
    s["top5_dy_km"] = np.array(off)[:, 1] / 1000
    s["gradv_y5"] = mesh.tri_mean(mesh.grad_norm(v[4]))
    def bin_means(n):
        return np.array([(mesh.mass[i] * vbar[i]).sum() / mesh.mass[i].sum()
                         for i in mesh.quantile_bins(mesh.z, n)])
    s["decile_rel"] = bin_means(10) / effort
    q = bin_means(4)
    s["quartile_low_high"] = q[0] / q[-1] if q[-1] > 0 else np.inf
    # first saved year the E+I+R > FRONT region reaches each domain edge
    bx, by = mesh.xyz[mesh.boundary, 0], mesh.xyz[mesh.boundary, 1]
    side = np.argmin(np.stack([bx - bx.min(), bx.max() - bx, by - by.min(),
                               by.max() - by]), 0)
    for tag, d in (("base", base_dir), ("ctrl", point_dir)):
        with h5(d, "R") as f:
            frames = sorted(map(int, f["Function/Clinical"]))
        first = {}
        for fr in frames:
            hit = infected(d, fr)[mesh.boundary] > FRONT
            for k, n in enumerate("WESN"):
                if n not in first and hit[side == k].any():
                    first[n] = fr
            if len(first) == 4:
                break
        s[f"edges_{tag}"] = " ".join(f"{n}{first.get(n, '-')}" for n in "WESN")
    return s, vbar, v


def pair(mesh, va, vi):
    """RMS difference as a share of the first control's mean, and correlation."""
    ma, mi = mesh.mean(va), mesh.mean(vi)
    rms = np.sqrt(mesh.mean((va - vi) ** 2)) / ma
    cov = mesh.mean((va - ma) * (vi - mi))
    corr = cov / np.sqrt(mesh.mean((va - ma) ** 2) * mesh.mean((vi - mi) ** 2))
    return 100 * rms, corr


def report(root, region):
    dirs = {a: arm_dirs(root, region, a) for a in ARMS}
    missing = [a for a, d in dirs.items() if d is None]
    dirs = {a: d for a, d in dirs.items() if d is not None}
    mesh = Mesh(dirs["aniso_lc"][1])
    for a, (b, p) in dirs.items():
        m = Mesh(p) if a != "aniso_lc" else mesh
        if not np.array_equal(m.xyz, mesh.xyz):
            raise SystemExit(f"{region}_{a}: mesh differs from aniso_lc")
    print(f"\n==== {region}: {len(mesh.xyz):,} nodes, {mesh.A / 1e6:.1f} km^2, "
          f"focus {FOCUS[region]} m")
    if missing:
        print(f"  skipped (no finished control point): {', '.join(missing)}")

    stats, vbar, v = {}, {}, {}
    for a, (b, p) in dirs.items():
        stats[a], vbar[a], v[a] = arm_stats(mesh, region, b, p)

    print(f"\n{'arm':<11}{'deficit%':>9}{'J%':>7}{'handover':>9}{'prev%':>7}"
          f"{'effort':>8}{'check':>8}{'at vmax%':>9}{'never km2':>10}{'never%':>7}"
          f"{'def/eff':>8}{'|gv|y5':>8}{'lowQ/hiQ':>9}")
    for a, s in stats.items():
        print(f"{a:<11}{s['deficit_red']:9.2f}{s['J_red']:7.2f}"
              f"{s['handover_yr']:9.2f}{s['handover_prev']:7.3f}{s['effort']:8.3f}"
              f"{s['effort'] / s['effort_solver'] - 1:+8.1e}{s['at_vmax']:9.2f}"
              f"{s['never_km2']:10.1f}{100 * s['never_frac']:7.1f}"
              f"{s['def_per_effort']:8.1f}{s['gradv_y5']:8.3f}"
              f"{s['quartile_low_high']:9.2f}")

    print("\ntop 5% of v, years 1-8, centroid minus focus (km): dx range, dy range")
    for a, s in stats.items():
        print(f"  {a:<11} dx {s['top5_dx_km'].min():6.1f} .. {s['top5_dx_km'].max():6.1f}"
              f"   dy {s['top5_dy_km'].min():6.1f} .. {s['top5_dy_km'].max():6.1f}"
              f"   (yr1-8 dx: {' '.join(f'{x:.1f}' for x in s['top5_dx_km'])})")

    print("\ntime-mean v by elevation decile, relative to its mean (low -> high)")
    for a, s in stats.items():
        print(f"  {a:<11}" + " ".join(f"{x:5.2f}" for x in s["decile_rel"])
              + f"   max |dev| {100 * np.abs(s['decile_rel'] - 1).max():.0f}%")

    print("\nfirst saved year E+I+R > 0.1 reaches each edge (t = 0 is the handover)")
    for a, s in stats.items():
        print(f"  {a:<11} baseline {s['edges_base']:<22} controlled {s['edges_ctrl']}")

    print("\naniso vs iso")
    lcs = [lc for lc in ("lc", "nolc")
           if f"aniso_{lc}" in dirs and f"iso_{lc}" in dirs]
    for lc in lcs:
        rms, corr = pair(mesh, vbar[f"aniso_{lc}"], vbar[f"iso_{lc}"])
        g = stats[f"aniso_{lc}"]["gradv_y5"] / stats[f"iso_{lc}"]["gradv_y5"]
        print(f"  land cover {lc:<5} |grad v| y5 ratio {g:.2f}, time-mean RMS "
              f"{rms:.0f}% of the aniso mean, correlation {corr:.3f}")

    # flattened land cover makes S_inf uniform, so its gradient is round-off
    lcs = [lc for lc in lcs if lc == "lc"]
    sa = {lc: mesh.grad_norm(frame(dirs[f"aniso_{lc}"][0], "S_infinity", 0))
          for lc in lcs}
    si = {lc: mesh.grad_norm(frame(dirs[f"iso_{lc}"][0], "S_infinity", 0))
          for lc in lcs}
    print("\n|grad S_inf| aniso / iso, by relief (m/km); area share of each bin")
    for lo, hi in zip(RELIEF_BINS[:-1], RELIEF_BINS[1:]):
        sel = (mesh.relief >= lo) & (mesh.relief < hi)
        if not sel.any():
            continue
        ratios = "  ".join(
            f"{lc} {mesh.tri_mean(sa[lc], sel) / mesh.tri_mean(si[lc], sel):5.2f}"
            for lc in lcs)
        print(f"  {lo:>4}-{hi:<5} {ratios}   area "
              f"{100 * mesh.area[sel].sum() / mesh.area.sum():5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Ablation output root.")
    parser.add_argument("regions", nargs="*",
                        help="Default: every region with arms under ROOT.")
    parser.add_argument("--focus", type=float, nargs=2, metavar=("X", "Y"),
                        help="Override FOCUS for a single region (mesh m), e.g. "
                             "upper_arkansas on its 100 km mesh: 39 -44.")
    args = parser.parse_args()
    root = Path(args.root)
    if args.focus:
        if len(args.regions) != 1:
            raise SystemExit("--focus needs exactly one region")
        FOCUS[args.regions[0]] = tuple(args.focus)
    for region in args.regions or [r for r in FOCUS
                                   if (root / f"{r}_aniso_lc").is_dir()]:
        report(root, region)


if __name__ == "__main__":
    main()
