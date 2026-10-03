"""Generate one parameter file per sweep point.

The grid is the product of the environmental decay rate (--delta), the
best-habitat reproduction number R0-tilde (--R0) and the effort axis (--c4, or
--budget). beta_2 is rescaled with delta so that R0 is the same along a delta
row, and a target R0 is reached by scaling beta_1I, beta_1R and beta_2 by one
factor, which preserves the base file's direct/environmental split.

--R0 is the best-habitat value; a landscape-average target R0 corresponds to
--R0 R0/c-bar.

Point names are delta_<d>[_R0_<r>]__c4_<c>. Everything before the first "__"
is the spin-up group, which make_condor_sweep.py uses to share one disease
spin-up per group.

    python utilities/sweep_setup.py \\
        --base BUNDLE/effective_parameters.json --overlay parameters.json \\
        --out sweep_params --delta 0.125 0.25 0.5 1.0 --R0 2 3.5 5 --c4 0.5
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_parameters import OPTIMAL_CONTROL_DEFAULTS  # noqa: E402


def r0_parts(reaction, K0):
    """(R0_direct, R0_environmental) for a reaction block."""
    s2 = float(reaction["preclinical_to_clinical_rate"])
    al = float(reaction["clinical_removal_rate"])
    rI = float(reaction["environmental_shedding_rate_preclinical"])
    rR = float(reaction["environmental_shedding_rate_clinical"])
    b1I = float(reaction["direct_transmission_rate_preclinical"])
    b1R = float(reaction["direct_transmission_rate_clinical"])
    delta = float(reaction["environmental_decay_rate"])
    b2 = float(reaction["environmental_transmission_rate"])
    deposit = rI / s2 + rR / al
    return K0 * (b1I / s2 + b1R / al), (K0 * b2 / delta) * deposit, deposit


def tag(value):
    """Filename-safe number: 0.15 -> 0p15."""
    return ("%g" % value).replace(".", "p").replace("-", "m")


def check_base_consistency(base, base_path, overlay):
    """Warn if the base reaction block breaks the model's fixed relations:
    equal direct rates, rho_I/sigma_2 + rho_R/alpha = 1, and rho_R/rho_I = 4.
    """
    rx = base["reaction"]
    warn = []

    b1I = float(rx["direct_transmission_rate_preclinical"])
    b1R = float(rx["direct_transmission_rate_clinical"])
    if abs(b1I - b1R) > 1e-12 * max(1.0, abs(b1I)):
        warn.append(
            f"direct transmission rates are not equal ({b1I:g} preclinical "
            f"against {b1R:g} clinical).")

    s2 = float(rx["preclinical_to_clinical_rate"])
    al = float(rx["clinical_removal_rate"])
    rI = float(rx["environmental_shedding_rate_preclinical"])
    rR = float(rx["environmental_shedding_rate_clinical"])
    deposit = rI / s2 + rR / al
    if abs(deposit - 1.0) > 1e-3:
        warn.append(
            f"rho_I/sigma_2 + rho_R/alpha is {deposit:.4f}, not 1.")
    ratio = rR / rI if rI else float("inf")
    if abs(ratio - 4.0) > 1e-6:
        warn.append(
            f"the deposition ratio rho_R/rho_I is {ratio:g}, not 4.")

    if not warn:
        return
    print("WARNING: %s does not match the model's fixed relations." % base_path)
    for w in warn:
        print("  - " + w)
    if overlay is None:
        print("  Pass --overlay parameters.json to sweep at the current "
              "parameter set; the mesh-derived sections stay from --base.")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True,
                        help="Bundle effective_parameters.json to start from.")
    parser.add_argument("--overlay", default=None,
                        help="Take the sections named by --overlay-sections "
                             "from this file (normally parameters.json) instead "
                             "of from --base; the mesh-derived sections stay "
                             "from the bundle.")
    parser.add_argument("--overlay-sections", nargs="+",
                        default=["reaction", "optimal_control",
                                 "disease_spinup", "time"],
                        help="Top-level sections --overlay replaces "
                             "(default: reaction optimal_control "
                             "disease_spinup time). initial_conditions holds "
                             "mesh-specific focus coordinates.")
    parser.add_argument("--out", required=True,
                        help="Folder to write the per-point parameter files into.")
    parser.add_argument("--delta", type=float, nargs="+", required=True,
                        help="Environmental decay rates to sweep.")
    parser.add_argument("--R0", type=float, nargs="+", default=None,
                        dest="R0",
                        help="Best-habitat reproduction numbers (R0-tilde) "
                             "to sweep. Omit to keep the base file's R0. Cost "
                             "weights are not rescaled with R0.")
    parser.add_argument("--c4", type=float, nargs="+", default=None,
                        help="Linear weights c4 on v (cost_control_l1).")
    parser.add_argument("--budget", type=float, nargs="+", default=None,
                        help="Per-block effort budgets on v "
                             "(control_budget), as a mass-weighted spatial mean "
                             "in 1/yr. Mutually exclusive with --c4.")
    parser.add_argument("--v-max", type=float, default=None,
                        help="Override optimal_control.control_maximum.")
    parser.add_argument("--budget-penalty", type=float, default=None,
                        help="Override control_budget_penalty (rho).")
    parser.add_argument("--c3", type=float, nargs="+", default=None,
                        help="Override cost_control (c3). Several values "
                             "make c3 a grid axis and tag the point names.")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override optimal_control.max_iterations in every "
                             "generated file.")
    parser.add_argument("--relative-cost-tolerance", type=float, default=None,
                        help="Override optimal_control.relative_cost_tolerance "
                             "(L-BFGS-B's ftol).")
    parser.add_argument("--gradient-tolerance", type=float, default=None,
                        help="Override optimal_control.gradient_tolerance "
                             "(SciPy's gtol, applied to the projected gradient "
                             "in the SCALED variable z = D*v).")
    parser.add_argument("--lbfgs-memory", type=int, default=None,
                        help="Override optimal_control.lbfgs_memory.")
    parser.add_argument("--shedding-ratio", type=float, default=None,
                        help="Set the deposition ratio rho_R/rho_I in every "
                             "file, keeping rho_I/sigma_2 + rho_R/alpha = 1. "
                             "Tagged into the spin-up group of every point "
                             "name.")
    parser.add_argument("--time-step-years", type=float, default=None,
                        help="Override time.time_step_years in every file.")
    parser.add_argument("--final-time-years", type=float, default=None,
                        help="Override time.final_time_years in every file.")
    args = parser.parse_args()

    if args.c4 is not None and args.budget is not None:
        raise SystemExit("--c4 and --budget cannot be combined.")

    base_path = Path(args.base).expanduser().resolve()
    base = json.loads(io.open(base_path, encoding="utf-8").read())

    if args.overlay is not None:
        overlay_path = Path(args.overlay).expanduser().resolve()
        overlay = json.loads(io.open(overlay_path, encoding="utf-8").read())
        print(f"overlay         : {overlay_path}")
        changed = 0
        for section in args.overlay_sections:
            if section not in overlay:
                raise SystemExit(
                    f"{overlay_path} has no '{section}' section.")
            old_section = base.get(section, {})
            new_section = overlay[section]
            for field in sorted(set(old_section) | set(new_section)):
                if field == "_notes":
                    continue
                a, b = old_section.get(field), new_section.get(field)
                if a != b:
                    print(f"  {section}.{field}: {a} -> {b}")
                    changed += 1
            base[section] = json.loads(json.dumps(new_section))
        print(f"  {changed} field(s) taken from the overlay\n")

    spinup = base.get("disease_spinup", {})
    mode = spinup.get("mode")
    if mode == "detection" and "detection_threshold" not in spinup:
        raise SystemExit(
            "disease_spinup.mode is 'detection' but the block carries no "
            "detection_threshold. Pass --overlay parameters.json.")
    if mode != "detection" and (args.R0 or len(args.delta) > 1):
        print(
            f"  WARNING: sweeping delta/R0 with a '{mode}' spin-up, so groups "
            "are handed over at\n"
            "    the same age rather than the same detected burden.\n")

    K0 = float(base["land_cover"]["carrying_capacity_base"])
    check_base_consistency(base, base_path, args.overlay)
    if args.shedding_ratio is not None:
        if args.shedding_ratio <= 0.0:
            raise SystemExit("--shedding-ratio must be positive")
        rx = base["reaction"]
        s2 = float(rx["preclinical_to_clinical_rate"])
        al = float(rx["clinical_removal_rate"])
        rho_I = 1.0 / (1.0 / s2 + args.shedding_ratio / al)
        rx["environmental_shedding_rate_preclinical"] = round(rho_I, 6)
        rx["environmental_shedding_rate_clinical"] = round(
            args.shedding_ratio * rho_I, 6)
        print(f"shedding ratio  : rho_R/rho_I = {args.shedding_ratio:g} "
              f"(rho_I = {rho_I:.6f}, rho_R = {args.shedding_ratio * rho_I:.6f})")
    R0_dir, R0_env_target, deposit = r0_parts(base["reaction"], K0)

    if "seed_mode" not in base.get("initial_conditions", {}):
        raise SystemExit(
            f"{base_path} has no initial_conditions.seed_mode. Add it "
            "(\"staged\" or \"exposed\") before sweeping, or load_parameters "
            "will reject every generated file."
        )

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    c4_values = args.c4 if args.c4 is not None else [
        float(base["optimal_control"]["cost_control_l1"])
    ]
    budget_values = args.budget if args.budget is not None else [None]
    c3_values = args.c3 if args.c3 is not None else [None]
    if args.budget is not None:
        c4_values = [0.0]

    R0_base = R0_dir + R0_env_target
    # None keeps the base file's R0 and leaves R0 out of the point names.
    R0_values = args.R0 if args.R0 is not None else [None]
    b1I_base = float(base["reaction"]["direct_transmission_rate_preclinical"])
    b1R_base = float(base["reaction"]["direct_transmission_rate_clinical"])

    print(f"base            : {base_path}")
    print(f"R0_direct       : {R0_dir:.5f}  (delta-independent)")
    print(f"R0_env          : {R0_env_target:.5f}")
    print(f"R0 (base file)  : {R0_base:.5f}")
    print(f"env. share      : {100 * R0_env_target / R0_base:.1f}%  "
          f"(preserved at every R0)")
    print(f"beta_2(delta)   : {R0_env_target / (K0 * deposit):.6f} * delta "
          f"at the base R0")
    if args.R0 is not None:
        print(f"R0 swept        : {', '.join('%g' % v for v in args.R0)}")
    print()
    manifest = []
    for delta in args.delta:
      for R0 in R0_values:
        scale = 1.0 if R0 is None else R0 / R0_base
        b2 = scale * R0_env_target * delta / (K0 * deposit)
        for c4 in c4_values:
          for budget in budget_values:
           for c3 in c3_values:
            # Everything before the first "__" is the spin-up group.
            name = f"delta_{tag(delta)}"
            if R0 is not None:
                name += f"_R0_{tag(R0)}"
            if args.shedding_ratio is not None:
                name += f"_rho_{tag(args.shedding_ratio)}"
            if args.c4 is not None:
                name += f"__c4_{tag(c4)}"
            if budget is not None:
                name += f"__B_{tag(budget)}"
            # c3 is tagged only when it is an axis.
            if c3 is not None and len(c3_values) > 1:
                name += f"__c3_{tag(c3)}"
            point = json.loads(json.dumps(base))     # deep copy
            point["reaction"]["environmental_decay_rate"] = delta
            point["reaction"]["environmental_transmission_rate"] = round(b2, 8)
            if R0 is not None:
                point["reaction"]["direct_transmission_rate_preclinical"] = \
                    round(scale * b1I_base, 8)
                point["reaction"]["direct_transmission_rate_clinical"] = \
                    round(scale * b1R_base, 8)
            point["optimal_control"]["cost_control_l1"] = c4
            point["optimal_control"]["control_budget"] = budget
            if budget is not None:
                # Record the augmented-Lagrangian settings explicitly.
                oc = point["optimal_control"]
                oc.setdefault("control_budget_penalty",
                              OPTIMAL_CONTROL_DEFAULTS["control_budget_penalty"])
                oc.setdefault("control_budget_outer_iterations",
                              OPTIMAL_CONTROL_DEFAULTS["control_budget_outer_iterations"])
                oc.setdefault("control_budget_tolerance",
                              OPTIMAL_CONTROL_DEFAULTS["control_budget_tolerance"])
                if args.budget_penalty is not None:
                    oc["control_budget_penalty"] = args.budget_penalty
            if args.budget_penalty is not None:
                point["optimal_control"]["control_budget_penalty"] = \
                    args.budget_penalty
            if c3 is not None:
                point["optimal_control"]["cost_control"] = c3
            if args.v_max is not None:
                point["optimal_control"]["control_maximum"] = args.v_max
            if args.max_iterations is not None:
                point["optimal_control"]["max_iterations"] = args.max_iterations
            if args.relative_cost_tolerance is not None:
                point["optimal_control"]["relative_cost_tolerance"] = \
                    args.relative_cost_tolerance
            if args.gradient_tolerance is not None:
                point["optimal_control"]["gradient_tolerance"] = \
                    args.gradient_tolerance
            if args.lbfgs_memory is not None:
                point["optimal_control"]["lbfgs_memory"] = args.lbfgs_memory
            if args.time_step_years is not None:
                point["time"]["time_step_years"] = args.time_step_years
            if args.final_time_years is not None:
                point["time"]["final_time_years"] = args.final_time_years

            check_dir, check_env, _ = r0_parts(point["reaction"], K0)
            check_R0 = check_dir + check_env
            want_R0 = R0_base if R0 is None else R0
            # Coefficients are rounded to 8 decimals above.
            if abs(check_R0 - want_R0) > 1e-6 * max(1.0, want_R0):
                raise SystemExit(
                    f"internal error: R0 is {check_R0:.8f} at {name}, "
                    f"wanted {want_R0:.8f}")
            split = check_env / check_R0
            base_split = R0_env_target / R0_base
            if abs(split - base_split) > 1e-6:
                raise SystemExit(
                    f"internal error: route split drifted at {name} "
                    f"({split:.6f} against {base_split:.6f})")

            path = out / f"{name}.json"
            io.open(path, "w", encoding="utf-8", newline="\n").write(
                json.dumps(point, indent=2, ensure_ascii=False) + "\n")
            manifest.append(name)
            effort = (f"B={budget:<6g}" if budget is not None
                      else f"c4={c4:<6g}")
            shown_c3 = point["optimal_control"]["cost_control"]
            print(f"  {name:<40} delta={delta:<6g} beta_2={b2:.6f} "
                  f"{effort} c3={shown_c3:<6g} R0={check_R0:.5f}")

    io.open(out / "manifest.txt", "w", encoding="utf-8", newline="\n").write(
        "\n".join(manifest) + "\n")
    print(f"\n{len(manifest)} point(s) -> {out}")
    print(f"manifest        : {out / 'manifest.txt'}")


if __name__ == "__main__":
    main()
