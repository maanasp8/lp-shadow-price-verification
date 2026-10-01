"""Amendment section 5 re-confirmation gate -- Checks A and B, REAL instances.

The applicability table in `preregistration_amendment_2026-08-02.md` section 5
was derived algebraically and confirmed on a single 3x3 toy instance. Section 5
requires: "Before Step 0 results are reported, every row must be re-confirmed on
at least one real corpus instance."

This script closes that gate for the A and B columns only. Check D does not
exist, so every D cell stays toy-only and is not touched here.

WHAT IT DOES NOT DO
-------------------
No mutant is persisted. Nothing is written. This is not the mutation harness and
does not start Step 1: each mutant is built in memory, solved, measured, and
discarded.

METHOD
------
Instances: three hand-verified fixture slots, whose ground truth is already
trusted (`handcheck_fixtures.json`).

For each instance and each of the eight primary types plus the sense-flip
footnote, the FIRST candidate that `mutations.py` enumerates is built (one
atomic operation, one re-solve, per amendment section 7), re-solved, and Checks
A and B are run on it.

Mutants come from `mutations.py`, the same engine `mutation_harness.py` uses, so
a mutant scored here and a mutant scored in Step 1 are built by identical code.
Revised 2026-08-02 (amendment section 12.3) -- the first version of this probe
carried its own constructors restricted to `linear` rows; the shared engine
addresses canonical structural cells across all row types, which is the
definition section 11's Step 0 counts are built on.

Every text-derived quantity -- a_ij, c_j, b_i, the sense -- comes from the CLEAN
annotation through `text_terms.py`. Only pi, the objective value and the
variable values come from the mutant solve.

READING THE RESULT
------------------
"Applicable" in section 5 means CAN fire, not MUST fire.

  * an **N/A** cell is contradicted if the check EVER fires   -> loud failure
  * an **applicable** cell is confirmed if the check fires on at least ONE
    real instance; silence on a particular instance is a rate observation
    (magnitude/slack), not a contradiction
"""

import copy
import json
import sys

from mutations import (DEFAULT_SCOPE, ENUMERATORS, ENUMERATORS_DIV, bucket_of,
                       check_a, check_b, is_noop, solve)
from text_terms import TextTermsStore, load_clean_problems, text_terms_of

# Fixture slots 1, 3 and 5. Chosen because their ground truth is hand-verified,
# and between them they cover: min sense with two binding >= rows (-451730029),
# a max with an at-zero variable and a zero-RHS xby row (-2061445103), and a max
# with two two-variable linear rows, one binding one slack (1727707716).
INSTANCES = ["-451730029", "-2061445103", "1727707716"]

TYPES = [("1",  "constraint coefficient magnitude x10"),
         ("2",  "objective coefficient magnitude x10"),
         ("3",  "limit magnitude x10"),
         ("4a", "coefficient swap - within row"),
         ("4b", "coefficient swap - cross row, same variable"),
         ("5",  "objective coefficient swap"),
         ("6",  "coefficient omission"),
         ("7",  "limit swap between rows"),
         ("8",  "spurious constraint"),
         ("SF", "sense flip (footnote)")]

# amendment section 5, A and B columns only
PREDICTED = {"1": ("applicable", "N/A"), "2": ("applicable", "N/A"),
             "3": ("N/A", "applicable"), "4a": ("applicable", "N/A"),
             "4b": ("applicable", "N/A"), "5": ("applicable", "N/A"),
             "6": ("applicable", "N/A"), "7": ("N/A", "applicable"),
             "8": ("applicable", "applicable"), "SF": ("applicable", "N/A")}


def run_one(text, prob, clean, tid, enum):
    cands = enum(prob, clean, DEFAULT_SCOPE)
    if not cands:
        return {"type": tid, "bucket": "n/a-no-candidate", "mutation": "-",
                "a": None, "b": None, "obj_change_pct": None}
    cand = cands[0]
    mut_prob = copy.deepcopy(prob)
    cand.apply(mut_prob)
    if is_noop(prob, mut_prob):
        return {"type": tid, "bucket": "0 no-op", "mutation": cand.slot,
                "a": None, "b": None, "obj_change_pct": None}
    sol = solve(mut_prob)
    row = {"type": tid, "mutation": cand.slot,
           "bucket": bucket_of(cand.touched_rows, clean, sol),
           "a": None, "b": None, "obj_change_pct": None}
    if not sol.optimal:
        return row
    a_fired, _ = check_a(text, sol)
    b_fired, _ = check_b(text, sol)
    row.update({"a": a_fired, "b": b_fired,
                "obj_change_pct": abs(sol.obj_value - clean.obj_value)
                                  / max(1.0, abs(clean.obj_value)) * 100.0})
    return row


def main():
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    fixtures = json.load(open("data/handcheck_fixtures.json"))
    by_pid = {v["problem_id"]: v for v in fixtures.values()}

    print("=" * 78)
    print("AMENDMENT SECTION 5 RE-CONFIRMATION -- Checks A and B on REAL instances")
    print("=" * 78)
    print("  mutants from mutations.py -- the same engine Step 1 uses")
    print("  no mutant persisted; nothing written; full population not generated")

    all_rows = {}
    for pid in INSTANCES:
        prob = problems[pid]
        text = store.get(pid)
        clean = solve(prob)

        fx = by_pid[pid]
        assert clean.optimal, f"{pid} clean solve is {clean.status}"
        bad = [(c["constraint_name"], c["pi"], clean.pi[c["constraint_name"]])
               for c in fx["constraints"]
               if abs(c["pi"] - clean.pi[c["constraint_name"]]) > 1e-9]
        assert not bad, f"{pid} clean pi disagrees with fixture: {bad}"
        assert abs(clean.obj_value - fx["objective_value"]) < 1e-9, \
            f"{pid} clean ObjVal disagrees with fixture"

        a0, _ = check_a(text, clean)
        b0, _ = check_b(text, clean)
        print("\n" + "=" * 78)
        print(f"INSTANCE {pid}   sense={text.sense}  vars={len(text.var_names)}  "
              f"rows={len(text.row_names)}  objval={clean.obj_value:.6g}")
        print("  clean solve reproduces fixture pi and ObjVal: True")
        print(f"  clean control -- Check A fired: {a0}   Check B fired: {b0}  "
              f"(both must be False)")
        assert not a0 and not b0, "a check fired on the CLEAN model"
        print("-" * 78)
        print(f"  {'type':<5} {'bucket':<17} {'A':<7} {'B':<7} {'dobj%':>9}  mutation")

        rows = []
        for tid, _label in TYPES:
            r = run_one(text, prob, clean, tid, ENUMERATORS[tid])
            rows.append(r)
            fa = "-" if r["a"] is None else ("FIRES" if r["a"] else "silent")
            fb = "-" if r["b"] is None else ("FIRES" if r["b"] else "silent")
            do = f"{r['obj_change_pct']:9.3f}" if r["obj_change_pct"] is not None \
                else "        -"
            print(f"  {tid:<5} {r['bucket']:<17} {fa:<7} {fb:<7} {do}  {r['mutation']}")
        for tid, enum in ENUMERATORS_DIV.items():
            r = run_one(text, prob, clean, tid, enum)
            fa = "-" if r["a"] is None else ("FIRES" if r["a"] else "silent")
            fb = "-" if r["b"] is None else ("FIRES" if r["b"] else "silent")
            do = f"{r['obj_change_pct']:9.3f}" if r["obj_change_pct"] is not None \
                else "        -"
            print(f"  {tid+'/10':<5} {r['bucket']:<17} {fa:<7} {fb:<7} {do}  "
                  f"{r['mutation']}  <-- SUPPLEMENTARY")
            assert not (r[{"1": "b", "2": "b", "3": "a"}[tid]]), \
                f"supplementary /10 fired an N/A cell for type {tid}"
        all_rows[pid] = rows

    print("\n" + "=" * 78)
    print("VERDICT PER SECTION-5 CELL  (A and B columns only; D untouched)")
    print("=" * 78)
    print(f"  {'type':<5} {'check':<6} {'predicted':<12} {'observed':<28} verdict")
    problems_found = []
    for tid, _label in TYPES:
        for ci, check in enumerate(("A", "B")):
            pred = PREDICTED[tid][ci]
            key = "a" if check == "A" else "b"
            obs = []
            for pid in INSTANCES:
                r = next(x for x in all_rows[pid] if x["type"] == tid)
                obs.append("F" if r[key] else ("s" if r[key] is False else "-"))
            fired_anywhere = "F" in obs
            if pred == "N/A":
                verdict = "CONTRADICTED" if fired_anywhere else "confirmed (never fires)"
            else:
                verdict = "confirmed (fires)" if fired_anywhere else \
                    "NOT CONFIRMED (silent on all 3)"
            if "confirmed (" not in verdict:
                problems_found.append((tid, check, pred, obs, verdict))
            per = "  ".join(f"{pid}:{o}" for pid, o in zip(INSTANCES, obs))
            print(f"  {tid:<5} {check:<6} {pred:<12} {per:<28} {verdict}")

    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TYPE-6 DEFECT DEMONSTRATION  (why the term list must come from the text)")
    print("=" * 78)
    print("  same type-6 mutant, same solve, two term lists:")
    print("    text     -- text_terms.py, fixed from the clean annotation")
    print("    filtered -- rebuilt from the MUTANT's own coefficients at |a|>1e-12,")
    print("                which is what constraints.csv would hold for that mutant")
    print(f"\n  {'instance':>13} {'status':<10} {'text':<8} {'filtered':<9} note")
    for pid in INSTANCES:
        prob = problems[pid]
        text = store.get(pid)
        clean = solve(prob)
        cands = ENUMERATORS["6"](prob, clean, DEFAULT_SCOPE)
        if not cands:
            print(f"  {pid:>13} {'-':<10} {'-':<8} {'-':<9} no candidate cell")
            continue
        mut_prob = copy.deepcopy(prob)
        cands[0].apply(mut_prob)
        sol = solve(mut_prob)
        if not sol.optimal:
            print(f"  {pid:>13} {sol.status:<10} {'-':<8} {'-':<9} "
                  f"bucket 1, never credited to A or B")
            continue
        a_text, _ = check_a(text, sol)
        a_filt, _ = check_a(_filtered_terms(mut_prob, text), sol)
        note = ("<-- defect: fixed loader detects it, old reading does not"
                if a_text and not a_filt else "")
        print(f"  {pid:>13} {sol.status:<10} "
              f"{('FIRES' if a_text else 'silent'):<8} "
              f"{('FIRES' if a_filt else 'silent'):<9} {note}")

    print("\n" + "-" * 78)
    if problems_found:
        print("  *** MISMATCHES -- NOT reconciled here, escalated ***")
        for p in problems_found:
            print(f"    type {p[0]} check {p[1]}: predicted {p[2]}, observed {p[3]} "
                  f"-> {p[4]}")
        sys.exit(2)
    print("  every A/B cell of the section-5 table is confirmed on real instances")


def _filtered_terms(mut_prob, clean):
    """The DEFECT, reproduced deliberately: a term list rebuilt from the
    mutant's own representation with zeros filtered at |a| > 1e-12 -- what
    `constraints.csv` would contain if that mutant had been persisted. Sense and
    objective coefficients stay text-derived so the comparison isolates the term
    list."""
    t = text_terms_of(mut_prob)
    t.row_coefs = {rn: {v: a for v, a in coefs.items() if abs(a) > 1e-12}
                   for rn, coefs in t.row_coefs.items()}
    t.sense = clean.sense
    t.obj_coefs = dict(clean.obj_coefs)
    return t


if __name__ == "__main__":
    main()
