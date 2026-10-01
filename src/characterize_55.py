"""What are the 55, concretely? And what does the certificate cost? 2026-08-28.

The v11 review asks three things this repository could answer but never had:

  (1) Of the 55 certified mutants, what is the error-type breakdown, and do they
      share a structural signature? "0.30% of a population" is not a description
      of a phenomenon; a reviewer cannot tell whether the class is interesting or
      pathological without knowing what is in it.

  (2) A minimal worked example. The core phenomenon -- a mutation that moves the
      dual optimal set while leaving (z*, x*) exactly where it was -- is stated
      abstractly everywhere in the paper. We take the smallest real member of the
      55 rather than inventing one, and print every number needed to check it by
      hand.

  (3) What does deciding membership cost, and does it scale? The certificate is
      the paper's contribution; its cost was never measured.

Outputs characterize_55_<date>.{json,md}.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
import time
from collections import Counter

from checks_common import REL_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, solve
import mutations
from reframe_measure import face_classes, pi_exists_over_face
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
TYPE_NAME = {
    "1": "constraint coefficient magnitude", "2": "objective coefficient magnitude",
    "3": "limit magnitude", "4a": "coefficient swap, within row",
    "4b": "coefficient swap, cross row", "5": "objective coefficient swap",
    "6": "coefficient omission", "7": "limit swap between rows",
    "8": "spurious constraint"}


def slack_of(prob, sol, text):
    """-> {row: tau_i (a_i x* - b_i)}, the signed slack at the model's optimum."""
    out = {}
    for rn in text.row_names:
        lhs = sum(text.row_coefs[rn].get(v, 0.0) * sol.values[v] for v in text.var_names)
        tau = 1.0 if text.row_operator[rn] == "<=" else -1.0
        out[rn] = tau * (lhs - text.row_limits[rn])
    return out


def main():
    genuine = {r["mutant_id"] for r in csv.DictReader(open("data/basis_test_2026-08-27.csv"))
               if r["dual_optimal_pi_exists"] == "False"}
    print(f"the certified 55: {len(genuine)}")

    problems, _ = load_clean_problems()
    rows, timings = [], []

    for pid, _m in in_scope().items():
        if not any(t.startswith(pid + "|") for t in genuine):
            continue
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        done = set()
        for et in TYPE_ORDER:
            for scope in ("narrow", "wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    if mid not in genuine or mid in done:
                        continue
                    done.add(mid)
                    mut = copy.deepcopy(prob)
                    cand.apply(mut)
                    sol = solve(mut)

                    # --- how many duals actually moved, and by how much -----
                    shared = [r for r in ref.pi if r in sol.pi]
                    moved = [r for r in shared
                             if abs(sol.pi[r] - ref.pi[r])
                             > REL_TOL * max(1.0, abs(ref.pi[r]))]
                    sl = slack_of(mut, sol, text_terms_of(mut))
                    n_binding = sum(1 for v in sl.values() if abs(v) <= 1e-7)
                    # a degenerate vertex: more active rows than variables
                    degenerate = n_binding > len(prob.var_names)

                    # --- cost of deciding membership -------------------------
                    t0 = time.perf_counter()
                    cls, uniq = face_classes(mut, sol.obj_value)
                    t1 = time.perf_counter()
                    pi_exists_over_face(mut, text, sol.obj_value, cls)
                    t2 = time.perf_counter()
                    timings.append({"mutant_id": mid, "n_vars": len(prob.var_names),
                                    "n_rows": len(text.row_names),
                                    "face_s": t1 - t0, "cert_s": t2 - t1})

                    rows.append({
                        "mutant_id": mid, "pid": pid, "error_type": cand.error_type,
                        "a": check_a(text, sol)[0], "b": check_b(text, sol)[0],
                        "c": check_d(text, sol)[0],
                        "n_vars": len(prob.var_names), "n_rows": len(text.row_names),
                        "n_binding_rows": n_binding, "degenerate_vertex": degenerate,
                        "n_duals_moved": len(moved), "n_duals_shared": len(shared),
                        "primal_unique": uniq, "z": sol.obj_value})

    n = len(rows)
    print(f"characterised {n}\n")

    by_type = Counter(r["error_type"] for r in rows)
    by_check = {"A only": sum(r["a"] and not r["b"] for r in rows),
                "B only": sum(r["b"] and not r["a"] for r in rows),
                "both": sum(r["a"] and r["b"] for r in rows),
                "C": sum(r["c"] for r in rows)}
    deg = sum(r["degenerate_vertex"] for r in rows)
    allmoved = Counter(r["n_duals_moved"] for r in rows)

    print("=" * 70)
    print("(1) WHAT ARE THE 55?")
    print("=" * 70)
    for t, k in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  type {t:2s} {TYPE_NAME[t]:34s} {k:3d}  ({100*k/n:.0f}%)")
    print(f"\n  which check reports: {by_check}")
    print(f"  at a degenerate vertex (active rows > variables): {deg}/{n}")
    print(f"  duals moved per mutant: {dict(sorted(allmoved.items()))}")
    print(f"  instance size: vars {Counter(r['n_vars'] for r in rows)}, "
          f"rows {Counter(r['n_rows'] for r in rows)}")

    print("\n" + "=" * 70)
    print("(3) WHAT DOES THE CERTIFICATE COST?")
    print("=" * 70)
    ft = sorted(t["face_s"] for t in timings)
    ct = sorted(t["cert_s"] for t in timings)
    tot = sorted(t["face_s"] + t["cert_s"] for t in timings)
    def q(a, p): return a[min(len(a) - 1, int(p * len(a)))]
    print(f"  face classification  median {1000*q(ft,.5):6.1f} ms   max {1000*ft[-1]:6.1f} ms")
    print(f"  feasibility program  median {1000*q(ct,.5):6.1f} ms   max {1000*ct[-1]:6.1f} ms")
    print(f"  per report, total    median {1000*q(tot,.5):6.1f} ms   max {1000*tot[-1]:6.1f} ms")
    print(f"  LPs solved per report: 2n+1 = {2*max(r['n_vars'] for r in rows)+1} at most here")

    out = {"stamp": TODAY, "n": n,
           "by_type": dict(by_type), "by_check": by_check,
           "degenerate_vertex": deg,
           "duals_moved_hist": {str(k): v for k, v in sorted(allmoved.items())},
           "n_vars": {str(k): v for k, v in Counter(r["n_vars"] for r in rows).items()},
           "n_rows": {str(k): v for k, v in Counter(r["n_rows"] for r in rows).items()},
           "cost_ms": {"face_median": 1000 * q(ft, .5), "cert_median": 1000 * q(ct, .5),
                       "total_median": 1000 * q(tot, .5), "total_max": 1000 * tot[-1]}}
    json.dump(out, open(f"characterize_55_{TODAY}.json", "w"), indent=1)
    with open(f"characterize_55_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote characterize_55_{TODAY}.json/.csv")

    # smallest members, for the worked example
    print("\nsmallest members (candidates for the worked example):")
    for r in sorted(rows, key=lambda r: (r["n_vars"], r["n_rows"]))[:6]:
        print(f"  {r['n_vars']}v {r['n_rows']}r  type {r['error_type']:2s}  "
              f"A={r['a']} B={r['b']}  {r['mutant_id'][:78]}")


if __name__ == "__main__":
    main()
