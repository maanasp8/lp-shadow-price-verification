"""Does the BASIS-ARTIFACT confound get worse at scale? 2026-09-03.

WHY THIS IS A DIFFERENT QUESTION FROM scale_sweep.py. That script measured the
primal-invisible error class and found it thins out sharply with problem size:
1.7% at two variables, 0.24% at four, 0.03% at six, with no revival from adding
rows. This script measures the OTHER result in the paper, which does not depend
on that class being common.

THE CLAIM BEING TESTED. Section 3.2 reports that the certificate withdraws 96 of
the reference solve's 98 unique detections and 70 of its 75 answer-neutral
reports. That is a statement about how solution-level verifiers are EVALUATED: a
reference solve flags a candidate when the returned (z*, x*) differs, but if the
two optimal faces intersect then a solver could have returned the same point
from both, so the flag records vertex selection rather than anything the model
forces.

THE PREDICTION. Face ambiguity comes from degeneracy and multiple optima, and
those become MORE common as rows are added, not less. So this confound should
grow with instance size, which is the opposite direction from the error class.
If it does, the audit is the part of the paper that survives scale.

WHAT IS MEASURED. For each synthetic instance and each swap mutant whose
returned solution differs from the clean model's, which is exactly what a
reference solve flags:

    basis artifact  <=>  optimal_faces_intersect(clean, mutant, z_clean, z_mut)

reframe_measure.optimal_faces_intersect is the paper's own criterion, unmodified.
The reported rate is artifacts / reference-detections, per (n_vars, n_rows).

    venv/bin/python src/basis_artifact_scale.py --quick
    venv/bin/python src/basis_artifact_scale.py --vars 6 --rows 3,6,12,24 --instances 40

Writes data/basis_artifact_scale_v<vars>_<date>.{csv,json}.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks_common import REL_TOL
from mutations import solve
from reframe_measure import optimal_faces_intersect
from scale_sweep import (FAMILIES, apply_swap, is_inert, same_point,
                         swap_candidates)

TODAY = datetime.date.today().isoformat()


def score_cell(n_vars, n_rows, n_inst, seed, cap, verbose=False,
               family="resource"):
    import random
    rng = random.Random(seed)
    mk = FAMILIES[family]
    stat = Counter()

    for t in range(n_inst):
        prob = mk(n_vars, n_rows, rng, f"syn_{n_vars}x{n_rows}_{t}")
        ref = solve(prob)
        if not ref.optimal:
            continue
        stat["instances"] += 1

        for etype, desc, op in swap_candidates(prob, rng, cap):
            mut = copy.deepcopy(prob)
            apply_swap(mut, op)
            sol = solve(mut)
            if not sol.optimal:
                continue
            if is_inert(prob, mut, ref, sol, op):
                continue
            stat["evaluable"] += 1

            # what a reference solve flags: the returned answer differs
            z_same = (abs(sol.obj_value - ref.obj_value)
                      <= REL_TOL * max(1.0, abs(ref.obj_value)))
            x_same = same_point(ref.values, sol.values)
            if z_same and x_same:
                stat["reference_silent"] += 1
                continue
            stat["reference_detects"] += 1
            if z_same:
                stat["reference_detects_answer_neutral"] += 1

            try:
                inter = optimal_faces_intersect(prob, mut, ref.obj_value,
                                                sol.obj_value)
            except Exception:                                    # noqa: BLE001
                stat["face_error"] += 1
                continue
            if inter:
                stat["basis_artifact"] += 1
                if z_same:
                    stat["basis_artifact_answer_neutral"] += 1
            else:
                stat["model_forced"] += 1
        if verbose and (t + 1) % 10 == 0:
            print(f"      {t+1}/{n_inst}: {stat['reference_detects']} reference "
                  f"detections, {stat['basis_artifact']} artifacts", flush=True)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", type=int, default=6)
    ap.add_argument("--rows", default="3,6,12,24")
    ap.add_argument("--instances", type=int, default=40)
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--family", default="resource", choices=sorted(FAMILIES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.rows, args.instances, args.cap = "3,12", 6, 100

    rows = [int(x) for x in args.rows.split(",")]
    print(f"basis-artifact sweep: family={args.family}, n_vars={args.vars}, n_rows in {rows}, "
          f"{args.instances} instances per cell\n")

    out = {}
    summary = []
    for nr in rows:
        print(f"  n_rows={nr} ...", flush=True)
        st = score_cell(args.vars, nr, args.instances, args.seed + nr,
                        args.cap, verbose=True, family=args.family)
        det = st["reference_detects"]
        art = st["basis_artifact"]
        detn = st["reference_detects_answer_neutral"]
        artn = st["basis_artifact_answer_neutral"]
        row = {
            "n_vars": args.vars, "n_rows": nr,
            "instances": st["instances"], "evaluable": st["evaluable"],
            "reference_detects": det, "basis_artifact": art,
            "model_forced": st["model_forced"],
            "rate_artifact": (art / det if det else 0.0),
            "reference_detects_answer_neutral": detn,
            "basis_artifact_answer_neutral": artn,
            "rate_artifact_answer_neutral": (artn / detn if detn else 0.0),
        }
        summary.append(row)
        out[nr] = dict(st)
        print(f"    reference detections {det}, basis artifacts {art} "
              f"({100*row['rate_artifact']:.1f}%)\n", flush=True)

    print("=" * 78)
    print(f"SHARE OF A REFERENCE SOLVE'S DETECTIONS THAT ARE BASIS ARTIFACTS"
          f"  (n_vars={args.vars})")
    print("=" * 78)
    print("  n_rows rows/var | all detections        | ANSWER-NEUTRAL stratum")
    print("                   | det   artifact   rate | det   artifact   rate")
    for r in summary:
        print("  %5d  %6.2f   | %5d %8d %6.1f%% | %4d %8d %6.1f%%"
              % (r["n_rows"], r["n_rows"] / r["n_vars"],
                 r["reference_detects"], r["basis_artifact"],
                 100 * r["rate_artifact"],
                 r["reference_detects_answer_neutral"],
                 r["basis_artifact_answer_neutral"],
                 100 * r["rate_artifact_answer_neutral"]))
    print()
    print("  The answer-neutral stratum is the apples-to-apples comparison: the")
    print("  paper's 70-of-75 (93.3%) is that column, not the left one.")
    print()
    print("  NL4OPT for comparison: the certificate withdrew 96 of the reference")
    print("  solve's 98 unique detections (98.0%) and 70 of its 75 answer-neutral")
    print("  reports (93.3%).")

    os.makedirs("data", exist_ok=True)
    fam = "" if args.family == "resource" else f"_{args.family}"
    stem = f"data/basis_artifact_scale_v{args.vars}{fam}_{TODAY}"
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    json.dump({"stamp": TODAY, "family": args.family,
               "n_vars": args.vars, "rows": rows,
               "instances_per_cell": args.instances, "cap": args.cap,
               "seed": args.seed, "cells": out, "summary": summary},
              open(stem + ".json", "w"), indent=1)
    print(f"\nwrote {stem}.csv/.json")


if __name__ == "__main__":
    main()
