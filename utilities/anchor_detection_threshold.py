"""Measure the detection threshold B = int_0^t int_Gamma R^p dx ds.

The probe runs the ordinary detection-mode spin-up with the threshold out of
reach and the cap set to the anchor duration, so it stops on the cap (and warns
that it did) and reports the burden accumulated by then.

    # 1. write the probe parameter file from a sweep point of the anchor group
    python utilities/anchor_detection_threshold.py prepare \\
        --point sweep_params/delta_0p25_R0_2__c4_0p5.json \\
        --out sweep_params/_probe_B.json --years 20

    # 2. run it (one forward solve)
    python CWD_optimal_control.py --mesh-folder BUNDLE \\
        --parameters sweep_params/_probe_B.json \\
        --output-folder OUT/probe_B --forward-only \\
        --disease-state OUT/probe_B/state.npy

    # 3. read the answer, optionally writing it into a parameter file
    python utilities/anchor_detection_threshold.py read \\
        --state OUT/probe_B/state.npy --apply parameters.json
"""
import argparse
import io
import json
from pathlib import Path

# Out of reach, so the probe always stops on the cap.
UNREACHABLE = 1.0e12


def load(path):
    return json.loads(io.open(path, encoding="utf-8").read())


def dump(path, payload):
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, indent=2) + "\n"
    )


def prepare(args):
    point = load(args.point)
    spinup = point.setdefault("disease_spinup", {})
    spinup["enabled"] = True
    spinup["mode"] = "detection"
    spinup["detection_threshold"] = UNREACHABLE
    spinup["max_duration_years"] = float(args.years)

    exponent = float(spinup.get("detection_exponent", 1.0))

    # Only the spin-up matters, so keep the main horizon to two steps.
    time_block = point.setdefault("time", {})
    step = float(time_block.get("time_step_years", 0.025))
    time_block["final_time_years"] = step * 2.0

    dump(args.out, point)
    print(f"wrote {args.out}")
    print(f"  mode              detection, threshold {UNREACHABLE:g} "
          f"(unreachable by design)")
    print(f"  cap               {args.years:g} yr  <- the anchor duration")
    print(f"  exponent p        {exponent:g}")
    print(f"  control horizon   {time_block['final_time_years']:g} yr "
          f"(the probe is only here for its spin-up)")
    print()
    print("The run WILL print a warning that it hit the cap short of the")
    print("threshold. That is the intended outcome, not a failure.")


def read(args):
    # The spin-up result is in the sidecar beside the cached state.
    sidecar = Path(args.state).with_suffix(".json")
    if not sidecar.is_file():
        raise SystemExit(
            f"no sidecar at {sidecar}.\n"
            "  It is written next to the --disease-state file when the "
            "spin-up runs.\n"
            "  If the run reused a cached state, delete that state and its "
            "sidecar and re-run,\n"
            "  or point --state at the cache the probe actually wrote."
        )
    payload = load(sidecar)
    result = payload.get("result", {})
    inputs = payload.get("inputs", {})

    if result.get("mode") != "detection":
        raise SystemExit(
            f"{sidecar} records a '{result.get('mode')}' spin-up, not a "
            "detection one.\n"
            "  Only detection mode accumulates the burden. Re-run the probe "
            "with the file\n"
            "  that `prepare` wrote."
        )
    burden = result.get("accumulated_burden")
    if burden is None:
        raise SystemExit(f"{sidecar} carries no accumulated_burden")

    years = float(result.get("years_run", float("nan")))
    reached = bool(result.get("converged", False))

    print(f"accumulated burden      {burden:.4f} clinical deer-years")
    print(f"  over                  {years:g} yr")
    print(f"  exponent p            {result.get('detection_exponent')}")
    print(f"  final prevalence      {float(result.get('final_prevalence', 0)):.4%}")
    print(f"  delta                 "
          f"{inputs.get('reaction', {}).get('environmental_decay_rate')}")
    print()
    if reached:
        print("WARNING: this run STOPPED ON THE THRESHOLD rather than the cap,")
        print("  so the burden reported is the threshold, not the burden at")
        print(f"  {years:g} yr. The probe's threshold was not unreachable after")
        print("  all -- re-run with the file `prepare` writes.")
        return
    print(f"B = {burden:.4f}  anchored on this group at {years:g} yr.")

    if args.apply:
        target = load(args.apply)
        block = target.setdefault("disease_spinup", {})
        previous = block.get("detection_threshold")
        block["detection_threshold"] = round(float(burden), 4)
        dump(args.apply, target)
        print(f"  wrote detection_threshold = {block['detection_threshold']} "
              f"into {args.apply}")
        print(f"  (was {previous!r})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the probe parameter file")
    p.add_argument("--point", required=True,
                   help="A generated sweep point .json for the anchor group.")
    p.add_argument("--out", required=True, help="Probe parameter file to write.")
    p.add_argument("--years", type=float, default=20.0,
                   help="Anchor duration in years (default: %(default)s).")
    p.set_defaults(func=prepare)

    p = sub.add_parser("read", help="report B from a finished probe")
    p.add_argument("--state", required=True,
                   help="The --disease-state path the probe was run with; its "
                        ".json sidecar carries the result.")
    p.add_argument("--apply", default=None,
                   help="Parameters file to write detection_threshold into.")
    p.set_defaults(func=read)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
