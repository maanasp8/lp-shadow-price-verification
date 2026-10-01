"""Row normalization -- does it restore the invariance the checks lack? 2026-08-25.

THE PROBLEM
-----------
Checks A and B pair a text-derived coefficient with a candidate-derived shadow
price. A candidate row scaled by k has its dual scaled by 1/k, so the product
a^txt_ij * pi^cand_i no longer reconstructs the quantity the identity needs, and
the check is violated by a model that is arithmetically identical to the text.
restructure_invariance.py measures this at 97.8% on rescaled or reoriented
correct models. Unit conversion (kg to lb) is exactly this transformation, so
this is not an edge case for a deployed autoformulator.

THE PROPOSED FIX
----------------
For each row, recover the scale factor k_i relating the candidate's row to the
text's, and correct the dual before use:

    pi_corrected_i = pi_cand_i * k_i

k_i is recovered by requiring that EVERY cell agree on it: the ratio
a^cand/a^txt must be the same for every structurally present variable, the ratio
b^cand/b^txt must match it, and the operator must flip exactly when k_i < 0. If
no single k_i satisfies all of those, the rows are not related by a scaling --
they genuinely differ -- and k_i = 1 is used, leaving the check to be violated
as before.

That last condition is what keeps the fix honest. A mutation that rescales one
coefficient and leaves the limit alone yields inconsistent ratios and is still
caught. Only a UNIFORM rescaling of the whole row, which preserves the row's
meaning exactly, is normalized away.

THE COST, STATED PLAINLY
------------------------
Recovering k_i requires reading the candidate's own coefficients and limit.
The unnormalized checks read only the candidate's SOLUTION. So the fix moves
the checks strictly closer to the coefficient-comparison baseline in what they
require, and any paper reporting it must say so.

WHAT THIS SCRIPT MEASURES
-------------------------
Both halves of the trade, on the frozen populations:
  arm 1  false positives on the semantics-preserving restructurings
         (should fall from 97.8% to ~0 if the fix works)
  arm 2  detection on the evaluable primary mutants
         (should NOT fall, if the fix suppresses only true rescalings)

Outputs row_normalization_<date>.{csv,json}.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json

from checks_common import REL_TOL, ZERO_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import solve
import mutations
from nl4opt_parser import ParsedProblem
from text_terms import load_clean_problems, text_terms_of
import restructure_invariance as RI

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
SCALE_TOL = 1e-9


# --------------------------------------------------------------------------
# scale recovery
# --------------------------------------------------------------------------

def row_scale(text, cand, rn) -> float:
    """-> k such that the candidate's row rn is k times the text's, or 1.0 when
    no single k explains every cell. Never raises; an unexplainable row simply
    is not normalized."""
    if rn not in cand.row_coefs or rn not in cand.row_limits:
        return 1.0
    at, ac = text.row_coefs[rn], cand.row_coefs[rn]
    if set(at) != set(ac):                     # different variable membership
        return 1.0

    ratios = []
    for v, a in at.items():
        b = ac[v]
        if abs(a) <= SCALE_TOL:
            if abs(b) > SCALE_TOL:
                return 1.0                     # 0 -> nonzero is not a scaling
            continue                           # 0 -> 0 constrains nothing
        ratios.append(b / a)

    bt, bc = text.row_limits[rn], cand.row_limits[rn]
    if abs(bt) > SCALE_TOL:
        ratios.append(bc / bt)
    elif abs(bc) > SCALE_TOL:
        return 1.0

    if not ratios:
        return 1.0
    k = ratios[0]
    if abs(k) <= SCALE_TOL:
        return 1.0
    for r in ratios[1:]:
        if abs(r - k) > 1e-7 * max(1.0, abs(k)):
            return 1.0                         # cells disagree: a real error
    # the operator must flip exactly when the scaling is negative
    flipped = text.row_operator[rn] != cand.row_operator[rn]
    if flipped != (k < 0):
        return 1.0
    return k


def scales_for(text, cand):
    return {rn: row_scale(text, cand, rn) for rn in text.row_names}


# --------------------------------------------------------------------------
# normalized checks: identical to mutations.check_a / check_b except that every
# pi is multiplied by its row's recovered scale factor
# --------------------------------------------------------------------------

def check_a_norm(text, sol, k):
    for name in text.var_names:
        c_j = text.obj_coef(name)
        total = 0.0
        for rn, a in text.terms_for(name):
            if rn in sol.pi:
                total += a * sol.pi[rn] * k.get(rn, 1.0)
        scale = REL_TOL * max(1.0, abs(c_j))
        v = sol.values[name]
        if abs(v) > ZERO_TOL:
            if abs(total - c_j) > scale:
                return True
        elif text.sense == "max":
            if total < c_j - scale:
                return True
        elif total > c_j + scale:
            return True
    return False


def check_b_norm(text, sol, k):
    total = sum(text.row_limits[rn] * sol.pi[rn] * k.get(rn, 1.0)
                for rn in text.row_names if rn in sol.pi)
    obj = sol.obj_value
    return abs(total - obj) > REL_TOL * max(1.0, abs(obj))


# --------------------------------------------------------------------------

def main():
    problems, _ = load_clean_problems()
    rows = []

    # ---- ARM 1: false positives on semantics-preserving restructurings -----
    ids = [p for p in RI.optimal_ids() if p in problems]
    arm1 = {}
    for pid in ids:
        prob = problems[pid]
        text = text_terms_of(prob)
        clean = solve(prob)
        if not clean.optimal:
            continue
        z = clean.obj_value
        for name, fn in RI.TRANSFORMS:
            if name.endswith("row0") and not prob.constraints:
                continue
            variant = fn(prob)
            sol = solve(variant)
            if not sol.optimal or abs(sol.obj_value - z) > REL_TOL * max(1.0, abs(z)):
                continue
            ct = text_terms_of(variant)
            k = scales_for(text, ct)
            fired = check_a_norm(text, sol, k) or check_b_norm(text, sol, k)
            d = arm1.setdefault(name, [0, 0])
            d[0] += 1
            d[1] += fired
    print("=" * 72)
    print("ARM 1 -- false positives on CORRECT restructured models")
    print("=" * 72)
    base = json.load(open(RI.OUT_JSON.replace(TODAY, "*"))) if False else None
    print(f"{'transformation':<20} {'n':>6} {'normalized FP':>15}   unnormalized")
    prev = {"negate_flip_all": 97.8, "negate_flip_row0": 64.2, "rescale_x2_all": 97.8,
            "rescale_x2_row0": 64.3, "rescale_x01_all": 97.8,
            "reorder_rows": 0.0, "reindex_vars": 0.0}
    for name, _ in RI.TRANSFORMS:
        if name not in arm1:
            continue
        n, f = arm1[name]
        rows.append({"arm": "restructure", "case": name, "n": n, "fired": f,
                     "pct": round(100 * f / n, 2)})
        print(f"{name:<20} {n:>6} {f:>6} ({100*f/n:5.1f}%)      "
              f"{prev.get(name, float('nan')):5.1f}%")

    # ---- ARM 2: detection on the evaluable primary mutants -----------------
    evaluable = set()
    with open("data/mutants_log.csv") as fh:
        for r in csv.DictReader(fh):
            if r["population"] == "primary" and r["bucket"] == "3 evaluable":
                evaluable.add(r["mutant_id"])

    n_tot = n_fired = 0
    per = {}
    for pid, _m in in_scope().items():
        prob = problems[pid]
        text = text_terms_of(prob)
        clean = solve(prob)
        if not clean.optimal:
            continue
        for et in TYPE_ORDER:
            for cand in enumerate_candidates(pid, prob, clean, et,
                                             both_directions=True):
                mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                if mid not in evaluable:
                    continue
                mut = copy.deepcopy(prob)
                cand.apply(mut)
                sol = solve(mut)
                if not sol.optimal:
                    continue
                ct = text_terms_of(mut)
                k = scales_for(text, ct)
                fired = check_a_norm(text, sol, k) or check_b_norm(text, sol, k)
                n_tot += 1
                n_fired += fired
                d = per.setdefault(cand.error_type, [0, 0])
                d[0] += 1
                d[1] += fired

    print("\n" + "=" * 72)
    print("ARM 2 -- detection on the evaluable primary mutants (Checks A or B)")
    print("=" * 72)
    print(f"normalized   {n_fired:>6}/{n_tot} = {100*n_fired/n_tot:.2f}%")
    print(f"unnormalized {14401:>6}/18525 = 77.74%   <- frozen figure")
    print(f"\n{'type':>5} {'n':>7} {'normalized':>12}")
    for t in sorted(per):
        n, f = per[t]
        rows.append({"arm": "detection", "case": f"type{t}", "n": n, "fired": f,
                     "pct": round(100 * f / n, 2)})
        print(f"{t:>5} {n:>7} {100*f/n:>11.2f}%")

    with open(f"row_normalization_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"row_normalization_{TODAY}.json", "w") as fh:
        json.dump({"stamp": TODAY,
                   "arm1": {k: {"n": v[0], "fired": v[1],
                                "pct": round(100 * v[1] / v[0], 2)}
                            for k, v in arm1.items()},
                   "arm2": {"n": n_tot, "fired": n_fired,
                            "pct": round(100 * n_fired / n_tot, 2),
                            "unnormalized_pct": 77.74},
                   "per_type": {t: {"n": v[0], "fired": v[1],
                                    "pct": round(100 * v[1] / v[0], 2)}
                                for t, v in per.items()}}, fh, indent=1)
    print(f"\nwrote row_normalization_{TODAY}.csv/.json")
    print("\nREADING: arm 1 near zero AND arm 2 near 77.74% means the fix "
          "restores invariance\nat no detection cost. Arm 2 falling means the "
          "normalization is suppressing real errors.")


if __name__ == "__main__":
    main()
