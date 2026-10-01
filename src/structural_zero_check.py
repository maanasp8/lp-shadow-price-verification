"""STRUCTURAL-ZERO CONFIRMATION — CPT/OPT baseline vs the Check A/B primary population.

    venv/bin/python structural_zero_check.py

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is NOT "does Check A/B beat the baseline". That would be a category error:
CPT/OPT is a constraint/objective PRESENCE tester, and it was already scored
against the population it targets (constraint omission + operator flip,
`cpt_opt_baseline.py` step 3b).

This mirrors Li & Hai's own M8/M9 methodology. Their Table 2 does not average
detection uniformly across every mutation operator: it reports uniform objective
scaling (M8) and orbit-redundant rows (M9) as PREDICTED, CONFIRMED ZEROS, and
calls them "invisibility controls" rather than failures, because those error
classes sit structurally outside what their battery can see.

The equivalent control here: score the CPT/OPT baseline against types 1, 2, 4a,
4b, 5, 6, 7, 8 — magnitude edits, swaps, single-cell omission and spurious rows.
Presence testing asks only whether a slot RESPONDS to extreme perturbation, never
whether its value is CORRECT. A coefficient that is wrong but nonzero still
responds. The battery is therefore expected to be structurally blind here.

THE NULL IS NOT ZERO
--------------------
Stated before running. The baseline has a nonzero false-positive rate on CLEAN
models (5.29 % per instance, eligible-only, `cpt_opt_baseline.py` step 3a). Any
warnings on these mutants are expected to be those same false positives, not
detections. So the theory-required result is

    detection rate on mutants  ~=  false-positive rate on the same clean models

i.e. ZERO DISCRIMINATIVE POWER, not zero warnings. Both are computed and
compared on the identical instance set, so the claim is testable rather than
rhetorical.

Same four interpretive calls as `cpt_opt_baseline.py` (Calls 1-4), reused by
import — not restated, not varied.
"""

from __future__ import annotations

import copy

import pandas as pd

from cpt_opt_baseline import WARNING, run_battery
from mutation_harness import MUTANT_LOG, enumerate_candidates
from mutations import NARROW, solve
from text_terms import load_clean_problems

PROVENANCE = "full-2026-08-02"
TYPES = ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"]

LABEL = {"1": "constraint coefficient magnitude", "2": "objective coef magnitude",
         "3": "limit magnitude", "4a": "coefficient swap within-row",
         "4b": "coefficient swap cross-row", "5": "objective coefficient swap",
         "6": "coefficient omission", "7": "limit swap", "8": "spurious constraint"}


def main():
    problems, _ = load_clean_problems()

    m = pd.read_csv(MUTANT_LOG, dtype={"problem_id": str})
    pop = m[(m.provenance == PROVENANCE)
            & (m.population == "primary")
            & (m.bucket == "3 evaluable")].copy()

    print("=" * 78)
    print("STRUCTURAL-ZERO CONFIRMATION — CPT/OPT vs the Check A/B primary population")
    print("=" * 78)
    print("  mirrors Li & Hai's M8/M9 invisibility controls: a predicted,")
    print("  theory-required near-zero, confirmed empirically. NOT a capability")
    print("  comparison — the baseline was already scored on its own target class.")
    print(f"\n  population: mutants_log.csv, provenance=={PROVENANCE!r},")
    print(f"              population=='primary', bucket=='3 evaluable'")
    print(f"  n = {len(pop)}   distinct instances = {pop.problem_id.nunique()}")
    print("  " + str(pop.error_type.value_counts().sort_index().to_dict()))

    rows = []
    for n, (pid, g) in enumerate(pop.groupby("problem_id"), 1):
        if n % 200 == 0:
            print(f"    [{n}/{pop.problem_id.nunique()} instances, {len(rows)} mutants]")
        prob = problems[pid]
        base = solve(prob)
        cands = {}
        for etype in g.error_type.unique():
            for c in enumerate_candidates(pid, prob, base, etype, True, NARROW):
                cands[(etype, c.direction, c.slot)] = c
        for r in g.itertuples():
            cand = cands.get((r.error_type, r.direction, r.slot))
            if cand is None:
                rows.append({"mutant_id": r.mutant_id, "error_type": r.error_type,
                             "subclass": r.subclass, "resolved": False})
                continue
            mut = copy.deepcopy(prob)
            cand.apply(mut)
            mv = run_battery(mut, run_ineligible=False)
            if mv is None:
                rows.append({"mutant_id": r.mutant_id, "error_type": r.error_type,
                             "subclass": r.subclass, "resolved": True,
                             "solved": False})
                continue
            cpt = [t for t in mv.tests if t["test"] == "CPT" and t["eligible"]]
            opt = [t for t in mv.tests if t["test"] == "OPT" and t["eligible"]]
            rows.append({
                "mutant_id": r.mutant_id, "error_type": r.error_type,
                "subclass": r.subclass, "resolved": True, "solved": True,
                "n_cpt": len(cpt), "n_opt": len(opt),
                "cpt_warn": sum(1 for t in cpt if t["verdict"] == WARNING),
                "opt_warn": sum(1 for t in opt if t["verdict"] == WARNING),
                "detected_cpt": any(t["verdict"] == WARNING for t in cpt),
                "detected_opt": any(t["verdict"] == WARNING for t in opt),
                "detected": any(t["verdict"] == WARNING for t in cpt + opt),
                "check_a_fired": r.check_a_fired, "check_b_fired": r.check_b_fired,
            })
    d = pd.DataFrame(rows)
    ok = d[d.get("solved", pd.Series(dtype=bool)).eq(True)]
    unres = d[~d.resolved]
    unsolved = d[d.resolved & ~d.get("solved", pd.Series(dtype=bool)).eq(True)]

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  mutants scored        {len(ok)}")
    print(f"  not rebuildable       {len(unres)}")
    print(f"  refused on re-solve   {len(unsolved)}")

    print(f"\n  {'type':<5}{'':<34}{'n':>7}{'CPT w':>8}{'OPT w':>8}"
          f"{'any w':>8}{'rate %':>9}")
    for t in TYPES:
        g = ok[ok.error_type == t]
        if not len(g):
            continue
        print(f"  {t:<5}{LABEL[t]:<34}{len(g):>7}"
              f"{int(g.detected_cpt.sum()):>8}{int(g.detected_opt.sum()):>8}"
              f"{int(g.detected.sum()):>8}{100.0 * g.detected.mean():>9.2f}")
    print(f"  {'ALL':<5}{'':<34}{len(ok):>7}"
          f"{int(ok.detected_cpt.sum()):>8}{int(ok.detected_opt.sum()):>8}"
          f"{int(ok.detected.sum()):>8}{100.0 * ok.detected.mean():>9.2f}")

    print("\n  per-TEST counts (not per-mutant):")
    for lab, nc, wc in (("CPT", ok.n_cpt.sum(), ok.cpt_warn.sum()),
                        ("OPT", ok.n_opt.sum(), ok.opt_warn.sum())):
        print(f"    {lab}: {int(wc)} WARNING of {int(nc)} tests = "
              f"{100.0 * wc / nc:.2f} %")
    tot_n = ok.n_cpt.sum() + ok.n_opt.sum()
    tot_w = ok.cpt_warn.sum() + ok.opt_warn.sum()
    print(f"    BOTH: {int(tot_w)} of {int(tot_n)} = {100.0 * tot_w / tot_n:.2f} %")

    # ---- the actual test: mutant rate vs clean rate on the SAME instances ----
    print("\n" + "-" * 78)
    print("THE CONTROL — is any of that DISCRIMINATIVE?")
    print("-" * 78)
    clean = pd.read_csv("data/cpt_opt_clean_tests.csv", dtype={"problem_id": str})
    clean = clean[clean.eligible]
    inst = set(pop.problem_id)
    c_here = clean[clean.problem_id.isin(inst)]
    c_flagged = c_here[c_here.verdict == WARNING].problem_id.nunique()
    print(f"  instance set: the {len(inst)} instances contributing these mutants")
    print(f"  CLEAN models, per-instance FPR   "
          f"{c_flagged}/{len(inst)} = {100.0 * c_flagged / len(inst):.2f} %")
    print(f"  CLEAN models, per-test FPR       "
          f"{int((c_here.verdict == WARNING).sum())}/{len(c_here)} = "
          f"{100.0 * (c_here.verdict == WARNING).mean():.2f} %")
    print(f"  MUTANTS,      per-mutant rate    "
          f"{int(ok.detected.sum())}/{len(ok)} = {100.0 * ok.detected.mean():.2f} %")
    print(f"  MUTANTS,      per-test rate      "
          f"{int(tot_w)}/{int(tot_n)} = {100.0 * tot_w / tot_n:.2f} %")
    print("\n  If the mutant rate sits at or below the clean false-positive rate,")
    print("  the baseline has ZERO discriminative power on these error classes —")
    print("  the warnings are the same false positives, not detections.")

    print("\n  for orientation only, on the identical mutants (NOT a comparison")
    print("  this run is making — these numbers are from amendment §13):")
    for c in ("check_a_fired", "check_b_fired"):
        v = ok[c].map({True: True, "True": True, False: False, "False": False})
        print(f"    {c:<16} fired on {int(v.sum())} of {len(ok)} = "
              f"{100.0 * v.mean():.2f} %")

    d.to_csv("data/cpt_opt_structural_zero.csv", index=False)
    print("\nwrote cpt_opt_structural_zero.csv")


if __name__ == "__main__":
    main()
