"""Check D, measured against its ACTUAL target population.

    venv/bin/python run_check_d.py

Check D is the primal-feasibility check specified in amendment §5: evaluate the
CLEAN annotation's constraints at the MUTANT's optimal x*. Implemented in
`mutations.check_d`.

WHAT IS MEASURED, AND WHY NOT THE FULL MISS POPULATION
------------------------------------------------------
§15 partitioned the 3655 row-touching misses into 2746 `binding -> slack`, 733
`at-zero-inert` and 176 `genuinely other`. D cannot recover the at-zero-inert
733 for the same structural reason A and B miss them: the mutant's primal
solution is IDENTICAL to the clean one, and the clean solution satisfies the
text's constraints by construction. Measuring D over all 4124 misses would
dilute its rate with mutants no primal-feasibility check could ever catch.

So D's rate is reported over the **2746 binding -> slack misses** -- the
population where the answer genuinely moved and A/B went silent.

FOUR RUNS
---------
  0  ZERO-FPR CONTROL   D on 1040 clean models. Must be 0: a clean optimum
                        satisfies its own constraints. If it fires, D is wrong.
  1  TARGET POPULATION  D on the 2746 binding -> slack misses.
  2  PREMISE CONTROL    D on the 733 at-zero-inert misses. Predicted 0, which
                        is the whole justification for excluding them from
                        run 1. Predicted, then checked -- not assumed.
  3  D-ONLY TYPES       D on the 5638 already-generated constraint-omission and
                        operator-flip mutants (§13 F), which have been sitting
                        at counts-only.

Every mutant is rebuilt and re-solved. Nothing is read back from the log except
which mutants belong to which population.
"""

from __future__ import annotations

import copy

import pandas as pd

from compute_endpoint import PROVENANCE
from mutation_harness import MUTANT_LOG, enumerate_candidates, in_scope
from mutations import NARROW, SECONDARY_ENUMERATORS, check_a, check_b, check_d, solve
from text_terms import TextTermsStore, load_clean_problems

SECONDARY_PROV = "full-2026-08-02-secondary"


def rebuild_and_check(rows, problems, store, secondary=False):
    """-> DataFrame with D's verdict per mutant. Rebuilds every one."""
    out = []
    for (pid, etype), g in rows.groupby(["problem_id", "error_type"]):
        prob = problems[pid]
        clean = solve(prob)
        text = store.get(pid)
        if secondary:
            cands = {(c.direction, c.slot): c
                     for c in SECONDARY_ENUMERATORS[etype](prob, clean, NARROW)}
        else:
            cands = {(c.direction, c.slot): c
                     for c in enumerate_candidates(pid, prob, clean, etype,
                                                   True, NARROW)}
        for r in g.itertuples():
            cand = cands.get((r.direction, r.slot))
            if cand is None:
                out.append({"mutant_id": r.mutant_id, "error_type": etype,
                            "subclass": r.subclass, "resolved": False})
                continue
            mut = copy.deepcopy(prob)
            cand.apply(mut)
            sol = solve(mut)
            if not sol.optimal:
                out.append({"mutant_id": r.mutant_id, "error_type": etype,
                            "subclass": r.subclass, "resolved": True,
                            "status": sol.status, "d_fired": None})
                continue
            d_fired, d_detail = check_d(text, sol)
            a_fired, _ = check_a(text, sol)
            b_fired, _ = check_b(text, sol)
            out.append({
                "mutant_id": r.mutant_id, "error_type": etype,
                "subclass": r.subclass, "resolved": True, "status": sol.status,
                "d_fired": d_fired, "a_fired": a_fired, "b_fired": b_fired,
                "n_rows_violated": sum(1 for x in d_detail if x["fired"]),
                "max_violation": max(x["violation"] for x in d_detail),
            })
    return pd.DataFrame(out)


def report(df, title, denom_note=""):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    bad = df[~df.resolved]
    if len(bad):
        print(f"  *** {len(bad)} mutant(s) could not be rebuilt -- NOT counted ***")
    ok = df[df.resolved & df.d_fired.notna()]
    refused = df[df.resolved & df.d_fired.isna()]
    if len(refused):
        print(f"  solver-refused on rebuild (excluded): {len(refused)}")
    n, k = len(ok), int(ok.d_fired.sum())
    print(f"  {denom_note}")
    print(f"  {'type':<6}{'subclass':<10}{'D fired':>10}{'n':>9}{'rate %':>10}")
    for t in sorted(ok.error_type.unique()):
        for sub in ("single", "multi"):
            g = ok[(ok.error_type == t) & (ok.subclass == sub)]
            if not len(g):
                continue
            print(f"  {t:<6}{sub:<10}{int(g.d_fired.sum()):>10}{len(g):>9}"
                  f"{100*g.d_fired.mean():>10.2f}")
    for sub in ("single", "multi"):
        g = ok[ok.subclass == sub]
        if len(g):
            print(f"  {'ALL':<6}{sub:<10}{int(g.d_fired.sum()):>10}{len(g):>9}"
                  f"{100*g.d_fired.mean():>10.2f}")
    print(f"\n  OVERALL: D fired on {k} of {n}  =  {100*k/n:.2f} %" if n else
          "\n  OVERALL: empty population")
    return n, k


def main():
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    scope = in_scope()

    print("=" * 78)
    print("CHECK D -- primal feasibility of the mutant optimum against the TEXT")
    print("=" * 78)
    print("  D fires when a CLEAN-annotation row is violated at the mutant's x*,")
    print("  by more than the frozen tolerance 1e-6 * max(1, |b_i|).")

    # ---- run 0: zero-FPR control -----------------------------------------
    print("\n" + "=" * 78)
    print("RUN 0 -- ZERO-FPR CONTROL: D on 1040 clean models")
    print("=" * 78)
    results = pd.read_csv("data/results.csv", dtype={"problem_id": str})
    optimal = sorted(results.loc[results.solve_status == "OPTIMAL", "problem_id"],
                     key=int)
    fired = []
    for pid in optimal:
        sol = solve(problems[pid])
        f, detail = check_d(store.get(pid), sol)
        if f:
            fired.append((pid, [x for x in detail if x["fired"]]))
    print(f"  instances {len(optimal)}   D FIRED {len(fired)}")
    if fired:
        for x in fired[:10]:
            print(f"    {x}")
        raise SystemExit("Check D fires on a CLEAN model -- D is wrong, stopping")
    print("  -> 0 false positives. A clean optimum satisfies its own constraints.")

    # ---- populations from §15 --------------------------------------------
    tax = pd.read_csv("data/miss_taxonomy.csv", dtype={"problem_id": str})
    target = tax[tax.category == "binding->slack"]
    inert = tax[tax.category == "at-zero-inert"]
    print(f"\n  §15 populations: binding->slack {len(target)}   "
          f"at-zero-inert {len(inert)}   genuinely other "
          f"{int((tax.category == 'genuinely other').sum())}")

    # ---- run 1: the target population ------------------------------------
    print("\n" + "=" * 78)
    print("RUN 1 -- D AGAINST ITS TARGET POPULATION")
    print("=" * 78)
    t1 = rebuild_and_check(target, problems, store)
    n1, k1 = report(t1, "the 2746 binding->slack misses: the answer genuinely "
                        "moved and A/B were silent",
                    "denominator = binding->slack misses only, per §15.6")

    # ---- run 2: premise control ------------------------------------------
    print("\n" + "=" * 78)
    print("RUN 2 -- PREMISE CONTROL: D on the at-zero-inert misses")
    print("=" * 78)
    print("  PREDICTED: 0. These mutants' primal solution is identical to the")
    print("  clean one, which satisfies the text by construction. Checked, not")
    print("  assumed -- this is the justification for excluding them from run 1.")
    t2 = rebuild_and_check(inert, problems, store)
    ok2 = t2[t2.resolved & t2.d_fired.notna()]
    print(f"\n  n {len(ok2)}   D FIRED {int(ok2.d_fired.sum())}")
    if int(ok2.d_fired.sum()) == 0:
        print("  -> prediction holds. D recovers none of them; excluding them")
        print("     from run 1's denominator is correct, not convenient.")
    else:
        print("  -> PREDICTION FAILED. D recovers some of them; §15.6's reasoning")
        print("     and run 1's denominator both need revisiting.")

    # ---- run 3: the D-only secondary types -------------------------------
    print("\n" + "=" * 78)
    print("RUN 3 -- THE TWO D-ONLY SECONDARY TYPES (§2), 5638 mutants")
    print("=" * 78)
    d = pd.read_csv(MUTANT_LOG, dtype={"problem_id": str})
    sec = d[(d.provenance == SECONDARY_PROV) & (d.bucket == "3 evaluable")]
    # bucket-3 evaluable only, so the denominator matches §13 F exactly and the
    # §7 bucket rules apply here as they do everywhere else
    print(f"  {len(d[d.provenance == SECONDARY_PROV])} generated, "
          f"{len(sec)} bucket-3 evaluable: "
          + str(sec.error_type.value_counts().to_dict()))
    t3 = rebuild_and_check(sec, problems, store, secondary=True)
    n3, k3 = report(t3, "constraint omission (CO) and operator flip (OF)",
                    "denominator = mutants that re-solve to optimality")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  run 0  clean-model false positives                    0 / {len(optimal)}")
    print(f"  run 1  D on binding->slack misses          {k1:>6} / {n1:<6} "
          f"= {100*k1/n1:.2f} %")
    print(f"  run 2  D on at-zero-inert misses (control) "
          f"{int(ok2.d_fired.sum()):>6} / {len(ok2):<6} "
          f"= {100*ok2.d_fired.mean():.2f} %")
    print(f"  run 3  D on the two D-only types           {k3:>6} / {n3:<6} "
          f"= {100*k3/n3:.2f} %")

    t1.to_csv("data/checkD_target.csv", index=False)
    t3.to_csv("data/checkD_secondary.csv", index=False)
    print("\nwrote checkD_target.csv, checkD_secondary.csv")


if __name__ == "__main__":
    main()
