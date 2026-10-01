#!/usr/bin/env python
"""Persist Check D's unpersisted runs, and measure D over the FULL primary
evaluable population.

Adds no analysis logic: it calls `run_check_d.rebuild_and_check` and
`mutations.check_d` unchanged. `run_check_d.py` is not modified.

Background: run_check_d.py writes only checkD_target.csv and
checkD_secondary.csv. Its run 0 (zero-FPR control on 1040 clean models) and
run 2 (at-zero-inert premise control) printed to stdout and were never saved, so
the soundness claim rested on a print statement.

    venv/bin/python persist_checkd_runs.py controls
    venv/bin/python persist_checkd_runs.py full

STATUS OF THE 'full' STAGE: post-hoc. n = 18,525 is NOT a pre-registered
denominator; the pre-registered Check D figures remain 2,531/2,746 = 92.17 %
against binding->slack misses and 2,762/4,672 = 59.12 % against its two
dedicated types. Naming note (revision 3 §R3.1): this check is "Check C" in
paper prose and "Check D" in all code and data, including every column here.
"""
import sys, time

import pandas as pd

from mutations import check_d, solve
from run_check_d import rebuild_and_check
from text_terms import TextTermsStore, load_clean_problems

PROV = "full-2026-08-02"
STAMP = "2026-08-19"


def controls(problems, store):
    # ---- run 0: zero-FPR control on the clean corpus ----------------------
    res = pd.read_csv("data/results.csv", dtype={"problem_id": str})
    optimal = sorted(res.loc[res.solve_status == "OPTIMAL", "problem_id"], key=int)
    rows = []
    t0 = time.perf_counter()
    for pid in optimal:
        sol = solve(problems[pid])
        fired, detail = check_d(store.get(pid), sol)
        rows.append({"problem_id": pid, "d_fired": bool(fired),
                     "n_rows_violated": sum(1 for x in detail if x["fired"]),
                     "max_violation": max([x.get("violation", 0.0) for x in detail],
                                          default=0.0)})
    df0 = pd.DataFrame(rows)
    df0.to_csv(f"checkD_clean_control_{STAMP}.csv", index=False)
    k0 = int(df0.d_fired.sum())
    print(f"RUN 0  zero-FPR control: {k0} / {len(df0)} clean models fired "
          f"= {100*k0/len(df0):.2f} % FPR   [{time.perf_counter()-t0:.1f}s]")
    print(f"       wrote checkD_clean_control_{STAMP}.csv")

    # ---- run 2: at-zero-inert premise control -----------------------------
    tax = pd.read_csv("data/miss_taxonomy.csv", dtype={"problem_id": str})
    inert = tax[tax.category == "at-zero-inert"]
    t0 = time.perf_counter()
    df2 = rebuild_and_check(inert, problems, store)
    df2.to_csv(f"checkD_at_zero_inert_{STAMP}.csv", index=False)
    ok = df2[df2.resolved & df2.d_fired.notna()]
    k2 = int(ok.d_fired.sum())
    print(f"RUN 2  at-zero-inert control: {k2} / {len(ok)} fired "
          f"(PREDICTED 0)   [{time.perf_counter()-t0:.1f}s]")
    print(f"       wrote checkD_at_zero_inert_{STAMP}.csv")
    return k0, len(df0), k2, len(ok)


def full(problems, store):
    log = pd.read_csv("data/mutants_log.csv", low_memory=False, dtype={"problem_id": str})
    p = log[(log.provenance == PROV) & (log.population == "primary")
            & (log.bucket == "3 evaluable")].copy()
    print(f"FULL   population n = {len(p)}  (rebuild + re-solve each)")
    t0 = time.perf_counter()
    df = rebuild_and_check(p, problems, store)
    df.to_csv(f"checkD_full_population_{STAMP}.csv", index=False)
    el = time.perf_counter() - t0

    p["a"] = p.check_a_fired == True          # noqa: E712
    p["b"] = p.check_b_fired == True          # noqa: E712
    dmap = dict(zip(df.mutant_id, df.d_fired == True))   # noqa: E712
    p["d"] = p.mutant_id.map(lambda m: bool(dmap.get(m, False)))

    print(f"       resolved {int(df.resolved.sum())} / {len(df)}   [{el:.1f}s]")
    print()
    print(f"  D fired overall            {int(p.d.sum())}/{len(p)} "
          f"= {100*p.d.sum()/len(p):.2f} %")
    for sub, g in p.groupby("subclass"):
        print(f"    {sub:8} {int(g.d.sum()):>6}/{len(g):<6} = {100*g.d.sum()/len(g):.2f} %")
    print()
    print("  by error type:")
    for t, g in p.groupby("error_type"):
        print(f"    type {t:<3} {int(g.d.sum()):>6}/{len(g):<6} = {100*g.d.sum()/len(g):6.2f} %")
    print()
    ab = p.a | p.b
    print(f"  cross-tab against A/B:")
    print(f"    A|B fired, D fired       {int((ab & p.d).sum())}")
    print(f"    A|B fired, D silent      {int((ab & ~p.d).sum())}")
    print(f"    A|B silent, D fired      {int((~ab & p.d).sum())}   <- marginal D")
    print(f"    A|B silent, D silent     {int((~ab & ~p.d).sum())}")
    print(f"    union A|B|D              {int((ab | p.d).sum())}/{len(p)} "
          f"= {100*(ab | p.d).sum()/len(p):.2f} %")
    print(f"       wrote checkD_full_population_{STAMP}.csv")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "controls"
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    if stage == "controls":
        controls(problems, store)
    elif stage == "full":
        full(problems, store)
    else:
        raise SystemExit("stage must be 'controls' or 'full'")
