"""Coefficient-diff baseline -- 2026-08-24.

THE QUESTION
------------
Appendix D rejects the simplest verifier: diff the text's numbers against the
candidate's own parsed numbers, slot by slot. The paper never MEASURES that
baseline's detection rate on the mutant population. This script does.

The diff verifier: canonicalise both models (net coefficients, all-lhs form --
via text_terms_of, the same canonicalisation the checks use), then flag the
candidate iff any of these differ from the clean annotation beyond REL_TOL:
sense, an objective coefficient, the row-name set, or any row's operator,
limit, or coefficient. No solve is needed -- detection happens at parse time.

Mutants are RE-ENUMERATED with the same enumerators that produced the frozen
population (mutant_id = pid|type|direction|slot is deterministic), then joined
against mutants_log.csv so the detection rate is reported over EXACTLY the
evaluable primary population of the paper (the 18,525-row denominator of
checkD_full_population_2026-08-19.csv, modulo the 10-row log discrepancy,
which is reported when it appears).

Expected result: ~100 percent. The point is to measure and report it honestly
next to A/B/C's 91.87 percent, together with the restructuring experiment that
shows what the diff costs on correct-but-restructured models.

Reads:  mutants_log.csv, results.csv, corpus jsonl. Writes diff_baseline_<date>.*
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
from collections import defaultdict

from checks_common import REL_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import solve
import mutations
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
OUT_CSV = f"diff_baseline_{TODAY}.csv"
OUT_JSON = f"diff_baseline_{TODAY}.json"

TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])


def _ne(a: float, b: float) -> bool:
    return abs(a - b) > REL_TOL * max(1.0, abs(b))


def diff_detects(clean_t, mut_t) -> str:
    """-> '' if identical, else a short reason for the first difference found."""
    if clean_t.sense != mut_t.sense:
        return "sense"
    for v in clean_t.var_names:
        if _ne(mut_t.obj_coefs.get(v, 0.0), clean_t.obj_coefs[v]):
            return f"obj[{v}]"
    if set(clean_t.row_names) != set(mut_t.row_names):
        return "row-set"
    for rn in clean_t.row_names:
        if clean_t.row_operator[rn] != mut_t.row_operator[rn]:
            return f"op[{rn}]"
        if _ne(mut_t.row_limits[rn], clean_t.row_limits[rn]):
            return f"b[{rn}]"
        keys = set(clean_t.row_coefs[rn]) | set(mut_t.row_coefs[rn])
        for v in keys:
            if _ne(mut_t.row_coefs[rn].get(v, 0.0),
                   clean_t.row_coefs[rn].get(v, 0.0)):
                return f"a[{rn},{v}]"
    return ""


def evaluable_primary_ids() -> dict:
    """mutant_id -> (error_type, subclass) for the evaluable primary population."""
    out = {}
    with open("data/mutants_log.csv") as fh:
        for r in csv.DictReader(fh):
            if r["population"] == "primary" and r["bucket"] == "3 evaluable":
                out[r["mutant_id"]] = (r["error_type"], r["subclass"])
    return out


def main():
    evaluable = evaluable_primary_ids()
    print(f"evaluable primary mutants in log: {len(evaluable)}")

    problems, _ = load_clean_problems()
    instances = in_scope()

    rows, missing_join = [], 0
    seen = set()
    for done, (pid, meta) in enumerate(instances.items(), 1):
        prob = problems[pid]
        clean_t = text_terms_of(prob)
        clean = solve(prob)
        if not clean.optimal:
            raise AssertionError(f"{pid} recorded OPTIMAL, re-solves {clean.status}")
        for etype in TYPE_ORDER:
            for cand in enumerate_candidates(pid, prob, clean, etype,
                                             both_directions=True):
                mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                if mid not in evaluable:
                    continue                      # not in the paper's denominator
                seen.add(mid)
                mut = copy.deepcopy(prob)
                cand.apply(mut)
                reason = diff_detects(clean_t, text_terms_of(mut))
                et, sub = evaluable[mid]
                rows.append({"mutant_id": mid, "error_type": et,
                             "subclass": sub, "diff_detected": bool(reason),
                             "reason": reason or "-"})
        if done % 200 == 0:
            print(f"  {done}/{len(instances)} instances done")

    missing_join = len(evaluable) - len(seen)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print("COEFFICIENT-DIFF BASELINE over the evaluable primary population")
    print("=" * 70)
    n = len(rows)
    det = sum(r["diff_detected"] for r in rows)
    print(f"joined mutants          {n}  (log evaluable {len(evaluable)}, "
          f"unjoined {missing_join})")
    print(f"diff detects            {det}  ({100*det/n:.2f}%)   "
          f"[A|B|C detects 91.87%]")

    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["error_type"]][0] += 1
        per[r["error_type"]][1] += r["diff_detected"]
    print(f"\n{'type':>5} {'n':>7} {'detected':>9} {'%':>8}")
    per_type = {}
    for t in sorted(per):
        nn, dd = per[t]
        print(f"{t:>5} {nn:>7} {dd:>9} {100*dd/nn:>7.2f}%")
        per_type[t] = {"n": nn, "detected": dd, "pct": round(100 * dd / nn, 2)}

    misses = [r for r in rows if not r["diff_detected"]]
    if misses:
        print(f"\nmutants the diff does NOT see: {len(misses)} (first 10)")
        for r in misses[:10]:
            print("   ", r["mutant_id"])

    with open(OUT_JSON, "w") as fh:
        json.dump({"stamp": TODAY, "n": n, "detected": det,
                   "pct": round(100 * det / n, 2),
                   "unjoined": missing_join, "per_type": per_type}, fh, indent=1)
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")
    print("\nREADING: report this rate NEXT TO 91.87%, and next to the "
          "restructuring\nexperiment's false-positive rates -- the diff's "
          "weakness is correct models,\nnot mutants. If unjoined > 0, the "
          "enumeration drifted from the frozen run;\ninvestigate before using "
          "the number.")


if __name__ == "__main__":
    main()
