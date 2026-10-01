"""Solve-the-reference baseline: the comparator the paper never ran. 2026-08-27.

THE OBJECTION THIS ANSWERS
--------------------------
Checks A, B and C require every coefficient, limit and operator the text states,
aligned to the candidate. Anyone holding that much can BUILD the text's model and
solve it, then compare the candidate's optimum against the reference's. That
baseline is solution-level, so unlike a coefficient comparison it should also be
silent on differences that change no answer -- which is the one axis on which the
paper claims the checks win. If it is silent there AND detects more, the checks
are dominated on every reported axis and the paper must say so.

THE ORACLE
----------
Solve the text's model once. For a candidate, flag it when either
    |z*_cand - z*_ref| > tol      (the objective moved), or
    max_j |x*_cand,j - x*_ref,j| > tol   (the decision moved)
Both are needed: an error can change the recommended plan while leaving profit
identical, which the objective test alone misses.

WHAT IS MEASURED
----------------
Four populations, the same ones the checks are reported on:
  evaluable primary   detection    -- should be high
  inert               FALSE ALARMS -- the checks' claimed advantage
  restructured        FALSE ALARMS -- correct models written differently
  LLM-generated       both
The restructured arm is where the oracle can lose: a rescaled row can make the
solver return a different optimal VERTEX at the same objective value, and the
x* test would then flag a correct model. Whether that happens is measured, not
assumed.

Outputs solve_reference_<date>.{csv,json}.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json

from checks_common import REL_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, solve
import mutations
import restructure_invariance as RI
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
# x* tolerance: relative, and deliberately looser than the fire tolerance so the
# oracle is not credited with detections that are floating-point noise.
X_TOL = 1e-6


def differs(ref, cand, names) -> tuple[bool, bool]:
    """-> (objective moved, solution moved)."""
    dz = abs(cand.obj_value - ref.obj_value) > REL_TOL * max(1.0, abs(ref.obj_value))
    scale = max([abs(ref.values[v]) for v in names] + [1.0])
    dx = any(abs(cand.values[v] - ref.values[v]) > X_TOL * scale for v in names)
    return dz, dx


def main():
    problems, _ = load_clean_problems()
    rows = []

    # ------------------------------------------------ arms 1 and 2: mutants
    evaluable, inert = set(), set()
    seen = set()
    with open("data/mutants_log.csv") as fh:
        for r in csv.DictReader(fh):
            if r["mutant_id"] in seen:
                continue
            seen.add(r["mutant_id"])
            if r["population"] == "primary" and r["bucket"] == "3 evaluable":
                evaluable.add(r["mutant_id"])
            elif (r["bucket"] == "2 inert"
                  and r["population"] in ("primary", "extension")):
                inert.add(r["mutant_id"])

    tally = {k: [0, 0, 0] for k in ("evaluable", "inert")}   # n, oracle, checks
    for pid, _m in in_scope().items():
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        if not ref.optimal:
            continue
        for et in TYPE_ORDER:
            for scope in ("narrow", "wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    arm = ("evaluable" if mid in evaluable
                           else "inert" if mid in inert else None)
                    if arm is None:
                        continue
                    evaluable.discard(mid); inert.discard(mid)
                    mut = copy.deepcopy(prob)
                    cand.apply(mut)
                    sol = solve(mut)
                    if not sol.optimal:
                        continue
                    dz, dx = differs(ref, sol, prob.var_names)
                    chk = (check_a(text, sol)[0] or check_b(text, sol)[0]
                           or check_d(text, sol)[0])
                    t = tally[arm]
                    t[0] += 1; t[1] += (dz or dx); t[2] += chk
                    rows.append({"arm": arm, "id": mid, "oracle": dz or dx,
                                 "obj_moved": dz, "sol_moved": dx, "checks": chk})

    # ------------------------------------- arm 3: correct, restructured models
    rest = {}
    for pid in [p for p in RI.optimal_ids() if p in problems]:
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        if not ref.optimal:
            continue
        for name, fn in RI.TRANSFORMS:
            if name.endswith("row0") and not prob.constraints:
                continue
            variant = fn(prob)
            sol = solve(variant)
            if not sol.optimal:
                continue
            dz, dx = differs(ref, sol, prob.var_names)
            chk = (check_a(text, sol)[0] or check_b(text, sol)[0]
                   or check_d(text, sol)[0])
            d = rest.setdefault(name, [0, 0, 0, 0])
            d[0] += 1; d[1] += (dz or dx); d[2] += chk; d[3] += dx and not dz
            rows.append({"arm": f"restructure:{name}", "id": pid,
                         "oracle": dz or dx, "obj_moved": dz, "sol_moved": dx,
                         "checks": chk})

    # ----------------------------------------------------------------- report
    print("=" * 74)
    print("SOLVE-THE-REFERENCE ORACLE  vs  CHECKS A/B/C")
    print("=" * 74)
    print(f"{'population':<34} {'n':>7} {'oracle':>14} {'checks':>14}")
    print("-" * 74)
    for arm, label, want_silence in (("evaluable", "evaluable primary (detection)", False),
                                     ("inert", "inert (FALSE ALARMS)", True)):
        n, o, c = tally[arm]
        print(f"{label:<34} {n:>7} {o:>7} ({100*o/max(n,1):5.1f}%) "
              f"{c:>7} ({100*c/max(n,1):5.1f}%)")
    print(f"\n{'correct models, restructured (FALSE ALARMS)':<34}")
    for name, _ in RI.TRANSFORMS:
        if name not in rest:
            continue
        n, o, c, xonly = rest[name]
        print(f"  {name:<32} {n:>7} {o:>7} ({100*o/n:5.1f}%) "
              f"{c:>7} ({100*c/n:5.1f}%)"
              + (f"   [{xonly} are x*-only]" if xonly else ""))

    with open(f"solve_reference_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(f"solve_reference_{TODAY}.json", "w") as fh:
        json.dump({"stamp": TODAY, "x_tol": X_TOL,
                   "mutants": {k: {"n": v[0], "oracle": v[1], "checks": v[2]}
                               for k, v in tally.items()},
                   "restructure": {k: {"n": v[0], "oracle": v[1], "checks": v[2],
                                       "x_only": v[3]} for k, v in rest.items()}},
                  fh, indent=1)
    print(f"\nwrote solve_reference_{TODAY}.csv/.json")
    print("\nREADING: if the oracle detects at least as much on evaluable AND "
          "raises no more\nfalse alarms on inert and restructured, the checks "
          "are dominated on every axis\nthe paper reports and the framing must "
          "say so.")


if __name__ == "__main__":
    main()
