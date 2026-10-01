"""Three-way miss taxonomy over every row-touching type/subclass cell.

    venv/bin/python miss_taxonomy.py

§13 D split misses two ways: "binding in clean -> slack in mutant" (the genuine
miss §7 refuses to excuse) and "other silent". Investigating type 1's
single-resource cell turned up a third mechanism hiding inside "other silent",
and this re-runs the test across every type rather than the one cell.

CATEGORIES, in assignment order
-------------------------------
1. binding -> slack   any touched row binding in the clean model and slack in
                      the mutant. Assigned FIRST: §7 says this "does change the
                      answer and is a genuine miss, not excused", so a mutant
                      that qualifies is never reclassified into a softer bucket.
                      The overlap with category 2 is reported separately so the
                      precedence choice is visible rather than buried.
2. at-zero-inert      EVERY variable whose coefficient the mutation touched is
                      at zero in the mutant (|x| <= ZERO_TOL) AND the optimal
                      decision vector is unchanged from clean. Scaling or
                      zeroing a coefficient on a variable the optimum does not
                      use moves nothing; Check A's at-zero branch is an
                      inequality that still holds, so silence here is the check
                      being correct about a mutation that changed nothing.
3. genuinely other    everything else.

Category 2 is DEFINED ONLY for the coefficient types -- 1, 4a, 4b, 6. Types 3
and 7 mutate a limit and 8 invents a row, so no coefficient cell and no variable
is touched; the category is structurally undefined, not measured as zero. For
those types a supplementary "decision unchanged" count is reported instead,
which is well defined everywhere.

Types 2 and 5 touch no row at all and are excluded from this table, as in §13 D.

Every mutant is REBUILT and RE-SOLVED here; nothing is inferred from the log
beyond which mutants were misses.
"""

from __future__ import annotations

import copy
import re

import pandas as pd

from checks_common import ZERO_TOL
from compute_endpoint import PROVENANCE, detected_all
from mutation_harness import MUTANT_LOG, enumerate_candidates, in_scope
from mutations import NARROW, solve
from text_terms import load_clean_problems

# tolerance for "the optimal decision vector is unchanged"
DECISION_TOL = 1e-6

COEF_TYPES = ["1", "4a", "4b", "6"]       # category 2 is defined for these
OTHER_ROW_TYPES = ["3", "7", "8"]         # row-touching, but no coefficient cell
ROW_TYPES = ["1", "3", "4a", "4b", "6", "7", "8"]

# a[<row>,<var>]=  -- row names are c<i>_<type> and never contain a comma
CELL = re.compile(r"a\[([^,\]]+),(.+?)\]=")


def touched_vars(slot: str, var_names):
    """Variable names whose coefficient this slot touches. Validated against the
    instance's own variable list -- a name that does not validate is reported,
    never silently dropped."""
    found = [m.group(2) for m in CELL.finditer(slot)]
    bad = [v for v in found if v not in var_names]
    return found, bad


def main():
    d = pd.read_csv(MUTANT_LOG, dtype={"problem_id": str})
    full = d[d.provenance == PROVENANCE].copy()
    for c in ("check_a_fired", "check_b_fired"):
        full[c] = full[c].map({True: True, False: False, "True": True,
                               "False": False})
    ev = full[(full.population == "primary") & (full.bucket == "3 evaluable")]
    miss = ev[~detected_all(ev)].copy()
    miss = miss[miss.error_type.isin(ROW_TYPES)]

    def genuine(cb, mb):
        return any(c == "True" and m == "False"
                   for c, m in zip(str(cb).split(";"), str(mb).split(";")))

    miss["binding_to_slack"] = [genuine(a, b) for a, b
                                in zip(miss.clean_binding, miss.mutant_binding)]

    print("=" * 78)
    print("THREE-WAY MISS TAXONOMY -- every row-touching type/subclass cell")
    print("=" * 78)
    print(f"  source: {MUTANT_LOG}, provenance {PROVENANCE!r}, primary population,")
    print(f"          bucket 3 evaluable, applicable check silent")
    print(f"  row-touching misses to classify: {len(miss)}")
    print(f"  ZERO_TOL {ZERO_TOL:g}   decision-unchanged tol {DECISION_TOL:g}")
    print(f"  rebuilding and re-solving every one -- nothing inferred from the log")

    problems, _ = load_clean_problems()
    scope = in_scope()

    at_zero, decision_same, name_problems = {}, {}, []
    for (pid, etype), g in miss.groupby(["problem_id", "error_type"]):
        prob = problems[pid]
        clean = solve(prob)
        cands = {(c.direction, c.slot): c
                 for c in enumerate_candidates(pid, prob, clean, etype, True, NARROW)}
        for r in g.itertuples():
            cand = cands.get((r.direction, r.slot))
            if cand is None:
                name_problems.append((pid, etype, r.direction, r.slot, "no candidate"))
                continue
            mut = copy.deepcopy(prob)
            cand.apply(mut)
            sol = solve(mut)
            same = all(abs(sol.values[v] - clean.values[v])
                       <= DECISION_TOL * max(1.0, abs(clean.values[v]))
                       for v in prob.var_names)
            decision_same[r.mutant_id] = same
            if etype in COEF_TYPES:
                names, bad = touched_vars(r.slot, set(prob.var_names))
                if bad or not names:
                    name_problems.append((pid, etype, r.direction, r.slot,
                                          f"unparsed vars {bad or 'none found'}"))
                    continue
                at_zero[r.mutant_id] = all(abs(sol.values[v]) <= ZERO_TOL
                                           for v in names)

    if name_problems:
        print(f"\n  *** {len(name_problems)} slot(s) could not be resolved -- "
              f"NOT counted, listed here ***")
        for p in name_problems[:20]:
            print(f"      {p}")
    else:
        print("  every miss slot resolved to a rebuilt mutant and named variables: OK")

    miss["at_zero_touched"] = miss.mutant_id.map(at_zero)
    miss["decision_unchanged"] = miss.mutant_id.map(decision_same)
    miss["at_zero_inert"] = (miss.at_zero_touched.eq(True)
                             & miss.decision_unchanged.eq(True))

    # assignment order: binding->slack wins over at-zero-inert (§7)
    def cat(r):
        if r.binding_to_slack:
            return "binding->slack"
        if r.error_type in COEF_TYPES and r.at_zero_inert:
            return "at-zero-inert"
        return "genuinely other"

    miss["category"] = [cat(r) for r in miss.itertuples()]

    overlap = miss[miss.binding_to_slack & miss.at_zero_inert.eq(True)]
    print(f"\n  mutants qualifying as BOTH binding->slack and at-zero-inert: "
          f"{len(overlap)}")
    print(f"  -> assigned to binding->slack per §7 (not excused)")

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 78)
    print("MISS BREAKDOWN BY TYPE AND SUBCLASS")
    print("=" * 78)
    print(f"  {'type':<5}{'subclass':<9}{'misses':>8}{'binding->slack':>16}"
          f"{'at-zero-inert':>15}{'genuinely other':>17}")
    order = ["1", "3", "4a", "4b", "6", "7", "8"]
    for t in order:
        for sub in ("single", "multi"):
            g = miss[(miss.error_type == t) & (miss.subclass == sub)]
            if not len(g):
                continue
            b = int((g.category == "binding->slack").sum())
            z = int((g.category == "at-zero-inert").sum())
            o = int((g.category == "genuinely other").sum())
            zc = f"{z:>7} ({100*z/len(g):5.1f}%)" if t in COEF_TYPES else f"{'n/a':>15}"
            print(f"  {t:<5}{sub:<9}{len(g):>8}"
                  f"{b:>7} ({100*b/len(g):5.1f}%){zc}"
                  f"{o:>8} ({100*o/len(g):5.1f}%)")

    print("\n  pooled type 4 (4a + 4b), as the endpoint treats it:")
    for sub in ("single", "multi"):
        g = miss[miss.error_type.isin(["4a", "4b"]) & (miss.subclass == sub)]
        b = int((g.category == "binding->slack").sum())
        z = int((g.category == "at-zero-inert").sum())
        o = int((g.category == "genuinely other").sum())
        print(f"  {'4':<5}{sub:<9}{len(g):>8}{b:>7} ({100*b/len(g):5.1f}%)"
              f"{z:>7} ({100*z/len(g):5.1f}%){o:>8} ({100*o/len(g):5.1f}%)")

    print("\n  TOTALS over all row-touching misses:")
    for c in ("binding->slack", "at-zero-inert", "genuinely other"):
        n = int((miss.category == c).sum())
        print(f"    {c:<18} {n:>6}  ({100*n/len(miss):5.2f}%)")

    coef = miss[miss.error_type.isin(COEF_TYPES)]
    print(f"\n  restricted to the coefficient types (1, 4a, 4b, 6), where "
          f"at-zero-inert is defined:")
    for c in ("binding->slack", "at-zero-inert", "genuinely other"):
        n = int((coef.category == c).sum())
        print(f"    {c:<18} {n:>6}  ({100*n/len(coef):5.2f}%)")

    # ------------------------------------------- supplementary for 3, 7, 8
    print("\n" + "-" * 78)
    print("SUPPLEMENTARY -- types 3, 7, 8 have no coefficient cell, so")
    print("at-zero-inert is structurally undefined. 'Decision unchanged' is")
    print("well defined everywhere and is reported instead.")
    print("-" * 78)
    print(f"  {'type':<5}{'subclass':<9}{'misses':>8}{'decision unchanged':>21}")
    for t in order:
        for sub in ("single", "multi"):
            g = miss[(miss.error_type == t) & (miss.subclass == sub)]
            if not len(g):
                continue
            n = int(g.decision_unchanged.eq(True).sum())
            print(f"  {t:<5}{sub:<9}{len(g):>8}{n:>12} ({100*n/len(g):5.1f}%)")

    # ----------------------------------------- does it appear outside type 1
    print("\n" + "-" * 78)
    print("WHERE THE AT-ZERO MECHANISM APPEARS")
    print("-" * 78)
    for t in COEF_TYPES:
        for sub in ("single", "multi"):
            g = miss[(miss.error_type == t) & (miss.subclass == sub)]
            z = int((g.category == "at-zero-inert").sum())
            tot_other = int((g.category != "binding->slack").sum())
            share = 100 * z / tot_other if tot_other else float("nan")
            print(f"  type {t:<3} {sub:<7} at-zero-inert {z:>5} of "
                  f"{tot_other:>5} non-binding->slack misses ({share:5.1f}%)")

    miss.to_csv("data/miss_taxonomy.csv", index=False)
    print(f"\nwrote miss_taxonomy.csv ({len(miss)} rows)")


if __name__ == "__main__":
    main()
