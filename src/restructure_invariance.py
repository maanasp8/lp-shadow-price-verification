"""Restructuring-invariance experiment -- 2026-08-24.

THE QUESTION
------------
Appendix D claims Checks A and B are invariant to valid restructuring of the
candidate model (its worked example: "GPUs - CPUs <= 0" vs "CPUs - GPUs >= 0").
That claim is asserted, not tested. This script tests it.

Every transform below is SEMANTICS-PRESERVING: the transformed candidate has
exactly the same feasible set and objective as the clean model, so a perfect
verifier must stay silent. Any fire here is a FALSE POSITIVE of the method on
a correct model -- the quantity the paper currently reports as zero.

The text side (TextTerms) is held at the CLEAN annotation throughout, exactly
as in the mutation harness: only the candidate is transformed. Row names are
preserved by every transform, so the row correspondence the checks key on
(pi by constraint name) is the identity map -- the best case for the checks.

TRANSFORMS
----------
  negate_flip_all   every row:  a -> -a, b -> -b, op flipped   (Appendix D's own class)
  negate_flip_row0  same, first row only
  rescale_x2_all    every row scaled by 2
  rescale_x2_row0   first row scaled by 2
  rescale_x01_all   every row scaled by 0.1
  reorder_rows      constraint list reversed          (control: name-keyed, must pass)
  reindex_vars      variable order reversed           (control: name-keyed, must pass)

SELF-TEST
---------
For every variant we assert the re-solved objective equals the clean objective
to the frozen tolerance. If that assertion fails the TRANSFORM is buggy, not
the checks, and the instance is excluded and counted separately.

Outputs: restructure_invariance_<date>.csv (one row per instance x transform)
         restructure_invariance_<date>.json (summary)
Reads:   results.csv (OPTIMAL ids), corpus jsonl. Writes nothing else.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json

from mutations import check_a, check_b, check_d, solve
from nl4opt_parser import ParsedProblem
from text_terms import load_clean_problems, text_terms_of
from checks_common import REL_TOL

TODAY = datetime.date.today().isoformat()
OUT_CSV = f"restructure_invariance_{TODAY}.csv"
OUT_JSON = f"restructure_invariance_{TODAY}.json"


# --------------------------------------------------------------------------
# transforms: ParsedProblem -> ParsedProblem (deep-copied, names preserved)
# --------------------------------------------------------------------------

def _scale_row(con, k: float) -> None:
    con.lhs = [(k * a, i) for a, i in con.lhs]
    con.rhs_terms = [(k * a, i) for a, i in con.rhs_terms]
    con.rhs_const = k * con.rhs_const


def _flip_row(con) -> None:
    _scale_row(con, -1.0)
    con.op = ">=" if con.op == "<=" else "<="


def negate_flip(prob: ParsedProblem, rows: str) -> ParsedProblem:
    p = copy.deepcopy(prob)
    for k, con in enumerate(p.constraints):
        if rows == "all" or k == 0:
            _flip_row(con)
    return p


def rescale(prob: ParsedProblem, k: float, rows: str) -> ParsedProblem:
    p = copy.deepcopy(prob)
    for i, con in enumerate(p.constraints):
        if rows == "all" or i == 0:
            _scale_row(con, k)
    return p


def reorder_rows(prob: ParsedProblem) -> ParsedProblem:
    p = copy.deepcopy(prob)
    p.constraints = list(reversed(p.constraints))
    return p


def reindex_vars(prob: ParsedProblem) -> ParsedProblem:
    """Reverse variable order; remap every index. Names travel with variables."""
    p = copy.deepcopy(prob)
    n = len(p.var_names)
    remap = {i: n - 1 - i for i in range(n)}
    p.var_names = list(reversed(p.var_names))
    p.obj_terms = [(c, remap[i]) for c, i in p.obj_terms]
    for con in p.constraints:
        con.lhs = [(c, remap[i]) for c, i in con.lhs]
        con.rhs_terms = [(c, remap[i]) for c, i in con.rhs_terms]
    return p


TRANSFORMS = [
    ("negate_flip_all", lambda p: negate_flip(p, "all")),
    ("negate_flip_row0", lambda p: negate_flip(p, "row0")),
    ("rescale_x2_all", lambda p: rescale(p, 2.0, "all")),
    ("rescale_x2_row0", lambda p: rescale(p, 2.0, "row0")),
    ("rescale_x01_all", lambda p: rescale(p, 0.1, "all")),
    ("reorder_rows", reorder_rows),          # control
    ("reindex_vars", reindex_vars),          # control
]


# --------------------------------------------------------------------------

def optimal_ids() -> list:
    with open("data/results.csv") as fh:
        return [r["problem_id"] for r in csv.DictReader(fh)
                if r["solve_status"] == "OPTIMAL"]


def main():
    problems, parse_failed = load_clean_problems()
    ids = [p for p in optimal_ids() if p in problems]
    print(f"OPTIMAL instances with a parse: {len(ids)} "
          f"({len(parse_failed)} corpus instances did not parse)")

    rows, transform_bugs = [], []
    for done, pid in enumerate(ids, 1):
        prob = problems[pid]
        text = text_terms_of(prob)          # CLEAN text side, held fixed
        clean = solve(prob)
        if not clean.optimal:
            transform_bugs.append((pid, "clean", clean.status))
            continue
        z = clean.obj_value

        skip_row0 = len(prob.constraints) == 0
        for name, fn in TRANSFORMS:
            if skip_row0 and name.endswith("row0"):
                continue
            variant = fn(prob)
            sol = solve(variant)

            # -- self-test: the transform must not change the problem --------
            if not sol.optimal or abs(sol.obj_value - z) > REL_TOL * max(1.0, abs(z)):
                transform_bugs.append((pid, name,
                                       sol.status if not sol.optimal
                                       else f"objval {sol.obj_value} != {z}"))
                continue

            a_f, a_d = check_a(text, sol)
            b_f, b_d = check_b(text, sol)
            d_f, d_d = check_d(text, sol)
            rows.append({
                "problem_id": pid, "transform": name,
                "n_rows": len(prob.constraints),
                "a_fired": a_f, "b_fired": b_f, "d_fired": d_f,
                "any_fired": a_f or b_f or d_f,
                "a_worst_residual": max((abs(x["residual"]) for x in a_d),
                                        default=0.0),
                "b_abs_diff": b_d["abs_diff"],
                "d_worst_violation": max((x["violation"] for x in d_d),
                                         default=0.0),
                "a_vars_fired": ";".join(x["variable"] for x in a_d if x["fired"]) or "-",
                "d_rows_fired": ";".join(x["row"] for x in d_d if x["fired"]) or "-",
            })
        if done % 200 == 0:
            print(f"  {done}/{len(ids)} instances done")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    print("FALSE-POSITIVE RATE ON SEMANTICS-PRESERVING RESTRUCTURINGS")
    print("(text side = clean annotation; every candidate is CORRECT)")
    print("=" * 74)
    print(f"{'transform':<18} {'n':>5} {'A fired':>12} {'B fired':>12} "
          f"{'C fired':>12} {'any':>12}")
    summary = {}
    by = {}
    for r in rows:
        by.setdefault(r["transform"], []).append(r)
    for name, _ in TRANSFORMS:
        g = by.get(name, [])
        if not g:
            continue
        n = len(g)
        na = sum(r["a_fired"] for r in g)
        nb = sum(r["b_fired"] for r in g)
        nd = sum(r["d_fired"] for r in g)
        ny = sum(r["any_fired"] for r in g)
        print(f"{name:<18} {n:>5} {na:>6} ({100*na/n:5.1f}%) "
              f"{nb:>6} ({100*nb/n:5.1f}%) {nd:>6} ({100*nd/n:5.1f}%) "
              f"{ny:>6} ({100*ny/n:5.1f}%)")
        summary[name] = {"n": n, "a_fired": na, "b_fired": nb,
                         "d_fired": nd, "any_fired": ny,
                         "any_pct": round(100 * ny / n, 2)}

    if transform_bugs:
        print(f"\ntransform self-test failures (excluded above): {len(transform_bugs)}")
        for t in transform_bugs[:10]:
            print("   ", t)

    with open(OUT_JSON, "w") as fh:
        json.dump({"stamp": TODAY, "n_instances": len(ids),
                   "self_test_failures": len(transform_bugs),
                   "summary": summary}, fh, indent=1)
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")
    print("\nREADING: controls (reorder_rows, reindex_vars) must be 0.0% "
          "everywhere.\nAny nonzero on negate_flip/rescale is a false positive "
          "of the checks on a\ncorrect model and bounds Appendix D's claim.")


if __name__ == "__main__":
    main()
