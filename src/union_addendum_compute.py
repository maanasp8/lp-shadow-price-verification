#!/usr/bin/env python
"""
Post-hoc union computation for the 2026-08-15 addendum.

Computes A OR B OR D over the full evaluable primary population (n = 18,525),
which is NOT a pre-registered denominator. Everything here is read-only against
the frozen artefacts; Check D is re-run from scratch on the miss population that
`run_check_d.py` never covered.

Run once. No variant denominators are explored.
"""

import pandas as pd

from mutations import check_d, solve
from run_check_d import rebuild_and_check
from text_terms import TextTermsStore, load_clean_problems

PROV = "full-2026-08-02"


def main():
    log = pd.read_csv("data/mutants_log.csv", low_memory=False,
                      dtype={"problem_id": str})
    p = log[(log.provenance == PROV) & (log.population == "primary")
            & (log.bucket == "3 evaluable")].copy()
    p["a"] = p.check_a_fired == True          # noqa: E712
    p["b"] = p.check_b_fired == True          # noqa: E712
    n = len(p)

    ab = p["a"] | p["b"]
    misses = p[~ab]

    print("=" * 78)
    print("POST-HOC UNION — A OR B OR D over the full evaluable primary set")
    print("=" * 78)
    print(f"  denominator n                     {n}")
    print(f"  (every mutant here has at least one applicable check: "
          f"{int(((p.check_a_applicable == True) | (p.check_b_applicable == True)).sum())})")
    print()
    print("  (2) A+B ONLY, same denominator")
    print(f"      A fired                       {int(p.a.sum())}")
    print(f"      B fired                       {int(p.b.sum())}")
    print(f"      both                          {int((p.a & p.b).sum())}")
    print(f"      A OR B                        {int(ab.sum())}/{n} "
          f"= {100*ab.sum()/n:.2f} %")
    print(f"      neither (misses)              {len(misses)}")

    # ---- how much of the miss population has D ever been run on? ----------
    tax = pd.read_csv("data/miss_taxonomy.csv", dtype={"problem_id": str})
    tgt = pd.read_csv("data/checkD_target.csv")
    tested = set(tgt.mutant_id)
    d_fired_target = int((tgt.d_fired == True).sum())      # noqa: E712
    untested = misses[~misses.mutant_id.isin(tested)]

    print()
    print("  (4) WAS D EVALUATED ON ALL 18,525?  No.")
    print(f"      check_d_fired in mutants_log  "
          f"{dict(p.check_d_fired.value_counts())}")
    print(f"      D actually run on             {len(tested)} of {len(misses)} "
          f"misses (the binding->slack subset)")
    print(f"      of those, D fired             {d_fired_target}")
    print(f"      never tested by D             {len(untested)}")
    cat = tax.set_index("mutant_id")["category"].to_dict()
    breakdown = {}
    for mid in untested.mutant_id:
        breakdown[cat.get(mid, "rowless (objective-only)")] = \
            breakdown.get(cat.get(mid, "rowless (objective-only)"), 0) + 1
    for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"        {k:34} {v}")

    # ---- run D on every untested miss -------------------------------------
    print()
    print(f"  running D on all {len(untested)} untested misses "
          f"(rebuild + re-solve each) ...")
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    res = rebuild_and_check(untested, problems, store)
    res["category"] = res.mutant_id.map(lambda m: cat.get(m,
                                                          "rowless (objective-only)"))
    fired_untested = int((res.d_fired == True).sum())      # noqa: E712
    print(f"      resolved/re-solved            {len(res)}")
    print(f"      D FIRED on untested misses    {fired_untested}")
    for c, g in res.groupby("category"):
        print(f"        {c:34} {int((g.d_fired == True).sum())} / {len(g)}")

    # ---- (1) and (3) ------------------------------------------------------
    d_total = d_fired_target + fired_untested
    union = int(ab.sum()) + d_total
    print()
    print("  (3) MARGINAL CONTRIBUTION OF D  (caught by D, missed by A and B)")
    print(f"      from binding->slack           {d_fired_target}")
    print(f"      from every other miss         {fired_untested}")
    print(f"      TOTAL marginal D              {d_total}")
    print()
    print("  (1) UNION  A OR B OR D")
    print(f"      {union}/{n} = {100*union/n:.2f} %")

    # ---- (5) subclass -----------------------------------------------------
    print()
    print("  (5) UNION BY SUBCLASS")
    dmap = dict(zip(tgt.mutant_id, tgt.d_fired == True))    # noqa: E712
    dmap.update(dict(zip(res.mutant_id, res.d_fired == True)))  # noqa: E712
    p["d"] = p.mutant_id.map(lambda m: bool(dmap.get(m, False)))
    p["u"] = p.a | p.b | p.d
    rows = []
    for sub, g in p.groupby("subclass"):
        gab = (g.a | g.b)
        rows.append((sub, len(g), int(gab.sum()), int(g.u.sum()),
                     int((g.d & ~gab).sum())))
        print(f"      {sub:8} n={len(g):6}  A|B {int(gab.sum()):6} "
              f"({100*gab.sum()/len(g):5.2f} %)   A|B|D {int(g.u.sum()):6} "
              f"({100*g.u.sum()/len(g):5.2f} %)   marginal D {int((g.d & ~gab).sum())}")

    assert int(p.u.sum()) == union, "subclass split disagrees with the total"
    print("\n  consistency: subclass split sums to the reported union — OK")
    return dict(n=n, ab=int(ab.sum()), union=union, d_total=d_total,
                fired_untested=fired_untested, rows=rows)


if __name__ == "__main__":
    main()
