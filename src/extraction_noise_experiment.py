"""Extraction-noise experiment -- operating characteristics of two verifiers
as a function of extraction quality.  2026-08-24.

THE QUESTION
------------
Both verifiers read the problem text.  The coefficient DIFF compares every
extracted cell against the candidate's own value.  Checks A/B/C compare the
extracted cells against the candidate's SOLUTION, which multiplies each cell by
a shadow price or a variable value -- annihilating every cell inactive at the
optimum (43.3% of rows carry pi == 0 on this corpus).

The diff therefore cannot miss a mutant (measured: 100.00%), but it also cannot
tolerate a bad extraction: every extraction error is a mismatch, hence a false
positive.  The checks discard information and should inherit the corresponding
robustness.  NOBODY HAS MEASURED THIS -- not us, not Li and Hai.  This script
measures it.

DESIGN (fixed before the first run)
-----------------------------------
Two arms, identical noise process, identical instances:

  ARM 1 (false positives).  Candidate = the CLEAN annotation, i.e. a CORRECT
      model.  Text side = the clean annotation corrupted at rate p.  Any fire
      is a false positive caused purely by extraction error.

  ARM 2 (detection).  Candidate = an evaluable primary MUTANT.  Text side = the
      clean annotation corrupted at rate p.  A fire is a detection.

The headline quantity is the USABLE MARGIN, detection minus false positives, at
each noise level.  A verifier with 100% detection and 100% false positives is
worthless; the margin is what survives.

NOISE MODEL
-----------
Calibrated to this project's own extraction pilot (Appendix C), which found
alignment accuracy 171/225 = 76.00% and, where alignment succeeded, numeric
accuracy 73/76 = 96.05%.  Alignment failure therefore dominates numeric error
by roughly 6:1, and the pilot records every alignment error as a VARIABLE
REFERENT failure.  Two corruption modes, mixed at that ratio by default:

  referent  -- swap which variable a row's coefficient attaches to (the pilot's
               dominant failure: the right number on the wrong variable)
  value     -- replace a cell by 10x or 0.1x it (the mutation harness's own
               x10 / div10 idiom, so magnitudes match the frozen population)

Each NUMERIC CELL of the text side -- every a_ij, every b_i, every c_j -- is
independently corrupted with probability p.  Row names, variable names and
operators are never touched, so the row correspondence both verifiers rely on
stays intact and the comparison isolates VALUE/REFERENT error alone.

SELF-TEST
---------
At p = 0.0 the harness must reproduce the frozen artefacts exactly:
    arm 1 false positives   0.00% for both verifiers
    arm 2 diff detection  100.00%
    arm 2 checks detection 91.87%
The run aborts if any of these misses, because a harness that cannot reproduce
the known numbers cannot be trusted on the unknown ones.

USAGE
-----
    venv/bin/python extraction_noise_experiment.py                 # full run
    venv/bin/python extraction_noise_experiment.py --quick         # smoke test
    venv/bin/python extraction_noise_experiment.py --reps 40
Outputs extraction_noise_<date>.csv (per level x replicate) and .json (summary).
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
import random

from diff_baseline import diff_detects
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, solve
import mutations
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
REFERENT_SHARE = 6.0 / 7.0          # pilot-calibrated 6:1 referent:value
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])


# ==========================================================================
# noise process
# ==========================================================================

def corrupt(text, p: float, rng: random.Random):
    """-> a copy of `text` with each numeric cell corrupted w.p. `p`.

    Never touches row names, variable names or operators: this experiment
    isolates value and referent error, not structural extraction failure.
    """
    if p <= 0.0:
        return text
    t = copy.deepcopy(text)

    def scale(v: float) -> float:
        return v * (10.0 if rng.random() < 0.5 else 0.1)

    # --- objective coefficients (value error only; c_j has no row referent) --
    for v in t.var_names:
        if rng.random() < p:
            t.obj_coefs[v] = scale(t.obj_coefs[v]) if t.obj_coefs[v] != 0.0 \
                else float(rng.choice([1.0, -1.0]))

    for rn in t.row_names:
        # --- limits (value error only) -------------------------------------
        if rng.random() < p:
            b = t.row_limits[rn]
            t.row_limits[rn] = scale(b) if b != 0.0 else float(rng.choice([1.0, 10.0]))

        # --- constraint coefficients ---------------------------------------
        cells = list(t.row_coefs[rn].keys())
        for v in cells:
            if rng.random() >= p:
                continue
            if len(cells) >= 2 and rng.random() < REFERENT_SHARE:
                # referent failure: right number, wrong variable
                w = rng.choice([c for c in cells if c != v])
                t.row_coefs[rn][v], t.row_coefs[rn][w] = \
                    t.row_coefs[rn][w], t.row_coefs[rn][v]
            else:
                a = t.row_coefs[rn][v]
                t.row_coefs[rn][v] = scale(a) if a != 0.0 \
                    else float(rng.choice([1.0, -1.0]))
    return t


# ==========================================================================
# the two verifiers, run on identical inputs
# ==========================================================================

def diff_fires(text, cand_tt) -> bool:
    return bool(diff_detects(text, cand_tt))


def checks_fire(text, sol) -> bool:
    return (check_a(text, sol)[0] or check_b(text, sol)[0]
            or check_d(text, sol)[0])


def wilson(k: int, n: int):
    """95% Wilson interval -- the same idiom Appendix C uses."""
    if n == 0:
        return (0.0, 0.0)
    z, phat = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (phat + z * z / (2 * n)) / d
    h = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


# ==========================================================================

def build_population(max_mutants=None, verbose=True):
    """-> (clean[], mutants[]) with every solve done ONCE and cached."""
    evaluable = set()
    with open("data/mutants_log.csv") as fh:
        for r in csv.DictReader(fh):
            if r["population"] == "primary" and r["bucket"] == "3 evaluable":
                evaluable.add(r["mutant_id"])

    problems, _ = load_clean_problems()
    instances = in_scope()
    clean, mutants = [], []

    for n, (pid, _meta) in enumerate(instances.items(), 1):
        prob = problems[pid]
        tt = text_terms_of(prob)
        sol = solve(prob)
        if not sol.optimal:
            raise AssertionError(f"{pid} recorded OPTIMAL, re-solves {sol.status}")
        clean.append({"pid": pid, "text": tt, "cand_tt": tt, "sol": sol})

        for etype in TYPE_ORDER:
            for cand in enumerate_candidates(pid, prob, sol, etype,
                                             both_directions=True):
                mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                if mid not in evaluable:
                    continue
                mut = copy.deepcopy(prob)
                cand.apply(mut)
                msol = solve(mut)
                if not msol.optimal:
                    continue
                mutants.append({"mid": mid, "text": tt,
                                "cand_tt": text_terms_of(mut), "sol": msol,
                                "error_type": cand.error_type})
        if verbose and n % 200 == 0:
            print(f"  built {n}/{len(instances)} instances "
                  f"({len(mutants)} mutants so far)")

    if max_mutants and len(mutants) > max_mutants:
        rng = random.Random(20260824)
        by = {}
        for m in mutants:
            by.setdefault(m["error_type"], []).append(m)
        keep, per = [], max(1, max_mutants // len(by))
        for t in sorted(by):                       # stratified by error type
            keep += rng.sample(by[t], min(per, len(by[t])))
        mutants = keep
    return clean, mutants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--max-mutants", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="2 reps, 2000 mutants -- smoke test, NOT a result")
    args = ap.parse_args()
    reps = 2 if args.quick else args.reps
    cap = 2000 if args.quick else args.max_mutants

    print("building population (each model solved once, then cached)...")
    clean, mutants = build_population(cap)
    print(f"  clean instances {len(clean)}   mutants {len(mutants)}\n")

    rows = []
    for p in LEVELS:
        n_rep = 1 if p == 0.0 else reps       # p=0 is deterministic
        for rep in range(n_rep):
            rng = random.Random(hash((20260824, round(p * 1000), rep)) & 0xFFFFFFFF)
            dfp = cfp = 0
            for c in clean:
                t = corrupt(c["text"], p, rng)
                dfp += diff_fires(t, c["cand_tt"])
                cfp += checks_fire(t, c["sol"])
            ddet = cdet = 0
            for m in mutants:
                t = corrupt(m["text"], p, rng)
                ddet += diff_fires(t, m["cand_tt"])
                cdet += checks_fire(t, m["sol"])
            rows.append({
                "noise_p": p, "rep": rep,
                "n_clean": len(clean), "n_mutants": len(mutants),
                "diff_fp": dfp, "checks_fp": cfp,
                "diff_det": ddet, "checks_det": cdet,
                "diff_fp_pct": 100 * dfp / len(clean),
                "checks_fp_pct": 100 * cfp / len(clean),
                "diff_det_pct": 100 * ddet / len(mutants),
                "checks_det_pct": 100 * cdet / len(mutants),
            })
        print(f"  p={p:.2f} done ({n_rep} rep(s))")

    out_csv = f"extraction_noise_{TODAY}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ self-test
    z = [r for r in rows if r["noise_p"] == 0.0][0]
    print("\n" + "=" * 76)
    print("SELF-TEST at p = 0.0 (must reproduce the frozen artefacts)")
    print("=" * 76)
    checks_full = args.max_mutants is None and not args.quick
    tests = [("diff false positives", z["diff_fp"], 0),
             ("checks false positives", z["checks_fp"], 0)]
    ok = True
    for name, got, want in tests:
        good = got == want
        ok &= good
        print(f"  {name:<26} {got:>8}  expect {want:<8} {'OK' if good else 'FAIL'}")
    print(f"  {'diff detection':<26} {z['diff_det_pct']:>7.2f}%  "
          f"expect  100.00%  {'OK' if abs(z['diff_det_pct']-100)<0.01 else 'CHECK'}")
    print(f"  {'checks detection':<26} {z['checks_det_pct']:>7.2f}%  "
          f"expect   91.87%"
          + ("  OK" if not checks_full or abs(z["checks_det_pct"] - 91.87) < 0.05
             else "  CHECK"))
    if not ok:
        raise SystemExit("\nSELF-TEST FAILED -- do not use these numbers.")
    if not checks_full:
        print("\n  NOTE: mutant population was capped/quick, so the 91.87% "
              "target does not apply.")

    # -------------------------------------------------------------- summary
    print("\n" + "=" * 76)
    print("OPERATING CHARACTERISTICS vs EXTRACTION NOISE")
    print("(candidate is CORRECT in the FP arm; noise is on the TEXT side only)")
    print("=" * 76)
    print(f"{'noise':>6} | {'diff FP':>9} {'diff det':>9} {'diff marg':>10} "
          f"| {'chk FP':>9} {'chk det':>9} {'chk marg':>10}")
    print("-" * 76)
    summary = {}
    for p in LEVELS:
        g = [r for r in rows if r["noise_p"] == p]
        avg = lambda k: sum(r[k] for r in g) / len(g)
        dfp, ddet = avg("diff_fp_pct"), avg("diff_det_pct")
        cfp, cdet = avg("checks_fp_pct"), avg("checks_det_pct")
        print(f"{p:>6.2f} | {dfp:>8.2f}% {ddet:>8.2f}% {ddet-dfp:>9.2f} "
              f"| {cfp:>8.2f}% {cdet:>8.2f}% {cdet-cfp:>9.2f}")
        summary[f"p={p}"] = {
            "reps": len(g),
            "diff": {"fp_pct": round(dfp, 3), "det_pct": round(ddet, 3),
                     "margin": round(ddet - dfp, 3),
                     "fp_ci95": [round(x, 2) for x in
                                 wilson(int(avg("diff_fp")), g[0]["n_clean"])]},
            "checks": {"fp_pct": round(cfp, 3), "det_pct": round(cdet, 3),
                       "margin": round(cdet - cfp, 3),
                       "fp_ci95": [round(x, 2) for x in
                                   wilson(int(avg("checks_fp")), g[0]["n_clean"])]},
        }

    with open(f"extraction_noise_{TODAY}.json", "w") as fh:
        json.dump({"stamp": TODAY, "levels": LEVELS, "reps": reps,
                   "referent_share": REFERENT_SHARE,
                   "n_clean": len(clean), "n_mutants": len(mutants),
                   "capped": bool(cap), "summary": summary}, fh, indent=1)
    print(f"\nwrote {out_csv} and extraction_noise_{TODAY}.json")
    print("\nREADING: 'margin' is detection minus false positives -- what the "
          "verifier is\nworth at that extraction quality. The claim under test "
          "is that the diff's\nmargin collapses faster than the checks', "
          "because the checks multiply every\nextracted cell by a shadow price "
          "and 43.3% of those are zero.")


if __name__ == "__main__":
    main()
