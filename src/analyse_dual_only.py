"""What do the checks see that solving the reference cannot? 2026-08-27.

solve_reference_baseline.py finds that a solve-the-reference oracle matches the
checks almost exactly on the evaluable population. The two disagree on 184
mutants of 18,525, and only 86 of those go the checks' way. Those 86 are the
entire advantage the checks hold over the obvious baseline, so they decide
whether the paper has an affirmative claim at all.

By construction they satisfy: the mutant's optimal objective value AND its
optimal solution both equal the reference's, yet a check reports a violation.
That should be impossible for a criterion reading only the primal. The
hypothesis is DUAL DEGENERACY: the mutation moves the dual solution while
leaving the primal optimum untouched, so a shadow-price identity sees a change
that no comparison of (z*, x*) can.

This script tests that hypothesis rather than assuming it, and reports which
check reacts and how far the duals moved.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
from collections import Counter

from checks_common import REL_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, solve
import mutations
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
X_TOL = 1e-6


def main():
    target = {r["id"] for r in csv.DictReader(open(f"solve_reference_{TODAY}.csv"))
              if r["arm"] == "evaluable" and r["checks"] == "True"
              and r["oracle"] == "False"}
    print(f"checks-only mutants to analyse: {len(target)}\n")

    problems, _ = load_clean_problems()
    rows, done = [], set()
    for pid, _m in in_scope().items():
        if not any(t.startswith(pid + "|") for t in target):
            continue
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        for et in TYPE_ORDER:
            for scope in ("narrow", "wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    if mid not in target or mid in done:
                        continue
                    done.add(mid)
                    mut = copy.deepcopy(prob)
                    cand.apply(mut)
                    sol = solve(mut)
                    if not sol.optimal:
                        continue
                    shared = [r for r in ref.pi if r in sol.pi]
                    dpi = max([abs(sol.pi[r] - ref.pi[r]) for r in shared] or [0.0])
                    scale = max([abs(ref.pi[r]) for r in shared] + [1.0])
                    rows.append({
                        "mutant_id": mid, "error_type": cand.error_type,
                        "a": check_a(text, sol)[0], "b": check_b(text, sol)[0],
                        "c": check_d(text, sol)[0],
                        "max_dual_shift": dpi,
                        "dual_moved": dpi > REL_TOL * scale,
                        "rows_ref": len(ref.pi), "rows_mut": len(sol.pi),
                        "row_set_changed": set(ref.pi) != set(sol.pi),
                    })

    n = len(rows)
    assert n == len(target), f"analysed {n} rows for {len(target)} mutants"
    dual = sum(r["dual_moved"] for r in rows)
    rowset = sum(r["row_set_changed"] for r in rows)
    print("=" * 66)
    print("WHAT THE CHECKS SEE THAT SOLVING DOES NOT")
    print("=" * 66)
    print(f"  analysed                        {n}")
    print(f"  dual solution moved             {dual}  ({100*dual/max(n,1):.1f}%)")
    print(f"  candidate's row SET changed     {rowset}  ({100*rowset/max(n,1):.1f}%)")
    print(f"\n  by error type: {dict(Counter(r['error_type'] for r in rows))}")
    print(f"  which check reacts: A={sum(r['a'] for r in rows)} "
          f"B={sum(r['b'] for r in rows)} C={sum(r['c'] for r in rows)}")
    q = sorted(r["max_dual_shift"] for r in rows)
    if q:
        print(f"  dual shift: median {q[len(q)//2]:.4g}, max {q[-1]:.4g}")

    with open(f"dual_only_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({"stamp": TODAY, "n": n, "dual_moved": dual,
               "row_set_changed": rowset,
               "by_type": dict(Counter(r["error_type"] for r in rows)),
               "by_check": {"a": sum(r["a"] for r in rows),
                            "b": sum(r["b"] for r in rows),
                            "c": sum(r["c"] for r in rows)}},
              open(f"dual_only_{TODAY}.json", "w"), indent=1)
    print(f"\nwrote dual_only_{TODAY}.csv/.json")


if __name__ == "__main__":
    main()
