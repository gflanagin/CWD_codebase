"""Summarize a sweep output root: one row per point.

Columns: iterations, J and its reduction against the group's baseline,
`last/tot` (the last step's decrease as a fraction of the total decrease,
(J_{k-1} - J_k) / (J_0 - J_k)), peak and mean v, the space-time percentage at
v = 0, the implied peak lambda_R*R = 2*c3*v_peak + c4, and the stop reason.

--diagnostics adds a table of outcome measures (incidence, infected animals at
T, mean radius, front, infected area); --breakdown adds each point's J as
percentage shares per cost term.

    python utilities/sweep_report.py <sweep-output-root> [--breakdown] [--diagnostics]
"""
import io
import json
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except (OSError, ValueError):
        return None


def short_stop(reason):
    """The informative tail of a SciPy stop message."""
    if not reason:
        return "-"
    text = str(reason)
    for marker, label in (
        ("RELATIVE REDUCTION OF F", "ftol (relative reduction)"),
        ("NORM OF PROJECTED GRADIENT", "gtol (projected gradient)"),
        ("TOTAL NO. of ITERATIONS", "max_iterations"),
        ("ABNORMAL", "ABNORMAL TERMINATION"),
    ):
        if marker in text.upper():
            return label
    return text[:34]


# (section, term key, column heading) for the --breakdown table.
BREAKDOWN_COLUMNS = (
    ("running_state", "susceptible_deficit", "run:Sdef"),
    ("running_state", "environment", "run:W"),
    ("running_control", "targeted_quadratic", "run:v^2"),
    ("running_control", "targeted_l1", "run:v"),
    ("terminal", "susceptible_deficit", "end:Sdef"),
    ("terminal", "exposed", "end:E"),
    ("terminal", "preclinical", "end:I"),
    ("terminal", "clinical", "end:R"),
    ("terminal", "environment", "end:W"),
)


def print_breakdown_table(rows):
    """Per-term percentage shares of J, one row per sweep point."""
    present = [(name, history) for name, history in rows
               if history and history.get("cost_breakdown")]
    if not present:
        print("No point recorded a cost_breakdown. It is written by the final")
        print("forward pass, so only completed runs carry one.")
        return
    print()
    print("Share of J by term (%)")
    header = f"{'point':<26}" + "".join(f"{h:>9}" for _, _, h in BREAKDOWN_COLUMNS)
    print(header)
    print("-" * len(header))
    for name, history in present:
        breakdown = history["cost_breakdown"]
        cells = []
        for section, term, _ in BREAKDOWN_COLUMNS:
            entry = breakdown.get(section, {}).get("terms", {}).get(term)
            cells.append(f"{100.0 * entry['share']:>9.2f}" if entry
                         else f"{'-':>9}")
        print(f"{name:<26}" + "".join(cells))


def print_diagnostics_table(rows, baseline_diagnostics=None):
    """Outcome measures per sweep point."""
    present = [(name, h) for name, h in rows if h and h.get("diagnostics")]
    if not present:
        print()
        print("No point recorded diagnostics. They are written by the final")
        print("forward pass, so only completed runs carry them.")
        return
    print()
    print("Outcome diagnostics (not optimized)")
    # Front and area at the lowest threshold on file.
    edge = present[0][1]["diagnostics"]["extent_thresholds_deer_per_km2"][0]
    print(f"  incidence and infected(T) in deer; radii in km; area in km^2; "
          f"front and area at > {edge:g} deer/km^2")
    header = (f"{'point':<26}{'incidence':>13}{'vs base':>9}{'infected(T)':>13}"
              f"{'meanRad':>9}{'front':>9}{'area':>12}")
    print(header)
    print("-" * len(header))
    for name, history in present:
        d = history["diagnostics"]
        base = None
        if baseline_diagnostics:
            base = baseline_diagnostics.get(name.split("__")[0])
        if base:
            reduction = 100.0 * (1.0 - d["cumulative_incidence_deer"]
                                 / base["cumulative_incidence_deer"])
            vs = f"{reduction:>8.2f}%"
        else:
            vs = f"{'-':>9}"
        print(f"{name:<26}{d['cumulative_incidence_deer']:>13.5e}{vs}"
              f"{d['infected_animals_at_T']:>13.5e}"
              f"{d['mean_infection_radius_km']:>9.3f}"
              f"{d['front_radius_km'][0]:>9.3f}"
              f"{d['infected_area_km2'][0]:>12.5e}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or flags - {"--breakdown", "--diagnostics"}:
        raise SystemExit(__doc__)
    root = Path(args[0]).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"no such sweep root: {root}")

    baselines = {}
    baseline_diagnostics = {}
    for folder in sorted(root.glob("baseline_*")):
        summary = load(folder / "forward_summary.json")
        if summary:
            key = folder.name[len("baseline_"):]
            baselines[key] = float(summary["objective"])
            if summary.get("diagnostics"):
                baseline_diagnostics[key] = summary["diagnostics"]

    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name
        # Skip baselines and the caches kept under the output root.
        if name.startswith("baseline_") or name in (
                "spinup_cache", ".trajectories", ".cache"):
            continue
        history = load(folder / "optimization_history.json")
        if history is None:
            rows.append((name, None))
            continue
        rows.append((name, history))

    if not rows:
        raise SystemExit(f"no sweep points found under {root}")

    header = (f"{'point':<26} {'c3':>6} {'c4':>8} {'iters':>5} "
              f"{'J_final':>12} {'vs base':>8} {'last/tot':>8} {'peak v':>9} "
              f"{'mean v':>9} {'zero%':>6} {'lamR*R':>9}  stop")
    print(header)
    print("-" * len(header))

    peaks = []
    for name, history in rows:
        if history is None:
            print(f"{name:<26} {'--- no optimization_history.json ---'}")
            continue
        weights = history.get("cost_weights", {})
        c3 = float(weights.get("c3_targeted", float("nan")))
        c4 = float(weights.get("c4_targeted_l1", float("nan")))
        iterations = history.get("iterations", [])
        n_iter = max((int(i.get("iteration", 0)) for i in iterations), default=0)

        # Drop the repeated final record before differencing.
        costs = []
        for entry in iterations:
            if entry.get("cost") is None:
                continue
            value = float(entry["cost"])
            if costs and abs(value - costs[-1]) <= 1e-14 * max(abs(value), 1.0):
                continue
            costs.append(value)
        cost = costs[-1] if costs else float("nan")

        # Last step's decrease as a fraction of the total decrease.
        if len(costs) >= 2 and abs(costs[0] - costs[-1]) > 0:
            last_over_total = abs(costs[-2] - costs[-1]) / abs(costs[0] - costs[-1])
            last_text = f"{last_over_total:.2e}"
        else:
            last_text = "-"

        targeted = history.get("controls", {}).get("targeted", {})
        peak = float(targeted.get("peak", float("nan")))
        mean = float(targeted.get("space_time_mean", float("nan")))
        zero = float(targeted.get("fraction_at_lower_bound", float("nan"))) * 100.0

        group = name.split("__")[0]
        base = baselines.get(group)
        versus = f"{100.0 * (cost - base) / base:+.2f}%" if base else "-"

        # Implied peak lambda_R*R; a lower bound if v reaches v_max.
        at_upper = float(targeted.get("fraction_at_upper_bound", 0.0))
        lam_peak = 2.0 * c3 * peak + c4
        if at_upper > 0.0:
            lam_text = f">{lam_peak:.4g}"
        else:
            lam_text = f"{lam_peak:.4g}"
            if n_iter >= 4:
                peaks.append(lam_peak)

        print(f"{name:<26} {c3:>6g} {c4:>8g} {n_iter:>5d} "
              f"{cost:>12.6g} {versus:>8} {last_text:>8} {peak:>9.4g} "
              f"{mean:>9.4g} {zero:>6.1f} {lam_text:>9}  "
              f"{short_stop(history.get('stop_reason'))}")

    if "--breakdown" in flags:
        print_breakdown_table(rows)
    if "--diagnostics" in flags:
        print_diagnostics_table(rows, baseline_diagnostics)

    stalled = [n for n, h in rows
               if h is not None
               and max((int(i.get("iteration", 0))
                        for i in h.get("iterations", [])), default=0) <= 3]
    print()
    if stalled:
        print(f"WARNING: {len(stalled)} point(s) stopped in <=3 iterations: "
              + ", ".join(stalled[:6]) + ("..." if len(stalled) > 6 else ""))
        print("  Check last/tot: small means no further progress was possible")
        print("  (for example a subcritical R0); large means the solve was cut off.")
        print()

    if peaks:
        top = max(peaks)
        print(f"measured peak lambda_R*R = {top:.5g} "
              f"(over {len(peaks)} point(s) that ran past 3 iterations)")
        print("  c4 >= this value gives v == 0 everywhere.")
        print("  c4 ladder:")
        print("    " + "  ".join(f"{f * top:.4g}" for f in (0.0, 0.1, 0.3, 0.6)))
    else:
        print("No point ran far enough to measure lambda_R*R.")


if __name__ == "__main__":
    main()
