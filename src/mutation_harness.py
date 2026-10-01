"""Step 1 mutation harness -- candidate enumeration and the per-mutant log.

    venv/bin/python mutation_harness.py --counts    enumerate only, generate nothing
    venv/bin/python mutation_harness.py --pilot     one small batch, end to end

NOT a full-population run. `--pilot` generates one type in one subclass, a
bounded number of mutants, and stops. The population-size decision (full vs
bounded sample per type) is a pre-registration decision and is NOT taken here.

SCOPE
-----
In scope = OPTIMAL instances with `n_usable` in {1, 2}: single-resource 383,
multi-resource 622 after removing the phantom instance `73102943`. Exclusion
lists are applied at candidate selection:

  1. zero coefficients  -- `_cells()` keeps only nonzero net cells
  2. zero RHS           -- `_nonzero_limit_rows()` for types 3 and 7 only
  3. phantom variables  -- `load_mutation_exclusions()`, whole instance

TEXT QUANTITIES
---------------
`TextTermsStore` builds each instance's `TextTerms` once and reuses it across
every mutant of that instance. The pilot asserts builds == distinct instances.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import os
import time
from typing import Dict, List

import pandas as pd

from checks_common import load_mutation_exclusions
from mutations import (DEFAULT_SCOPE, ENUMERATORS, ENUMERATORS_DIV, NARROW,
                       SCOPED_TYPES, WIDE, Candidate, Solved, binding,
                       bucket_of, check_a, check_b, is_noop, solve)
from text_terms import TextTermsStore, load_clean_problems

MUTANT_LOG = "data/mutants_log.csv"

LOG_COLS = [
    # identity
    "mutant_id", "generated_on", "provenance",
    # population coordinates -- §7 requires type, direction, subclass
    "problem_id", "subclass", "n_usable", "error_type", "direction", "slot",
    "touched_rows",
    # §12.9: "primary" = narrow scope, feeds the §6 macro-average.
    # "extension" = wide-minus-narrow, reported separately, NOT in the endpoint.
    "population",
    # outcome bucket, assigned before any check runs
    "bucket", "mutant_status",
    # binding status in clean and in mutant, per touched row
    "clean_binding", "mutant_binding",
    # objective movement
    "clean_objval", "mutant_objval", "obj_change_pct",
    # the checks, separately
    "check_a_fired", "check_b_fired", "check_d_fired",
    "check_a_n_vars_fired", "check_a_max_abs_residual",
    "check_b_sum", "check_b_target", "check_b_abs_diff",
    # applicability mask from amendment §5 -- N/A cells stay out of denominators
    "check_a_applicable", "check_b_applicable",
]

# amendment §5, A and B columns, re-confirmed on real instances 2026-08-02
APPLICABLE = {
    "1":  (True, False), "2": (True, False), "3": (False, True),
    "4a": (True, False), "4b": (True, False), "5": (True, False),
    "6":  (True, False), "7": (False, True), "8": (True, True),
    "SF": (True, False),
    # §2 secondary types: Check D only. Both N/A for A and B, so both are
    # excluded from every A/B denominator and can never be recorded as misses.
    "CO": (False, False), "OF": (False, False),
}

TYPE_ORDER = ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"]


# ==========================================================================
# scope
# ==========================================================================

def in_scope() -> Dict[str, Dict[str, object]]:
    """-> {problem_id: {"subclass": ..., "n_usable": ...}} in integer id order."""
    res = pd.read_csv("data/results.csv", dtype={"problem_id": str})
    opt = res[res.solve_status == "OPTIMAL"]
    excluded = load_mutation_exclusions()
    out = {}
    for pid, n in zip(opt.problem_id, opt.n_usable.astype(float).astype(int)):
        if pid in excluded or n not in (1, 2):
            continue
        out[pid] = {"subclass": "single" if n == 1 else "multi", "n_usable": n}
    return {p: out[p] for p in sorted(out, key=int)}


# ==========================================================================
# enumeration -- counting only, nothing generated
# ==========================================================================

def enumerate_candidates(pid: str, prob, clean: Solved, etype: str,
                         both_directions: bool = False,
                         scope: str = DEFAULT_SCOPE) -> List[Candidate]:
    cands = list(ENUMERATORS[etype](prob, clean, scope))
    if both_directions and etype in ENUMERATORS_DIV:
        cands += list(ENUMERATORS_DIV[etype](prob, clean, scope))
    return cands


def counts(both_directions: bool = False,
           scope: str = DEFAULT_SCOPE) -> pd.DataFrame:
    instances = in_scope()          # NOT `scope` -- that is the narrow/wide arg
    problems, _ = load_clean_problems()
    rows = []
    for pid, meta in instances.items():
        prob = problems[pid]
        clean = solve(prob)
        if not clean.optimal:
            raise AssertionError(f"{pid} is recorded OPTIMAL but re-solves "
                                 f"{clean.status}")
        for etype in TYPE_ORDER:
            n = len(enumerate_candidates(pid, prob, clean, etype,
                                         both_directions, scope))
            rows.append({"problem_id": pid, "subclass": meta["subclass"],
                         "error_type": etype, "n": n})
    return pd.DataFrame(rows)


# ==========================================================================
# one mutant, end to end
# ==========================================================================

def run_candidate(pid: str, meta: dict, prob, clean: Solved, cand: Candidate,
                  text, provenance: str, today: str,
                  population: str = "primary") -> dict:
    mut_prob = copy.deepcopy(prob)
    cand.apply(mut_prob)

    rec = {c: "" for c in LOG_COLS}
    rec.update({
        "mutant_id": f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}",
        "generated_on": today, "provenance": provenance,
        "problem_id": pid, "subclass": meta["subclass"],
        "n_usable": meta["n_usable"], "error_type": cand.error_type,
        "direction": cand.direction, "slot": cand.slot,
        "touched_rows": ";".join(cand.touched_rows) or "-",
        "population": population,
        "clean_objval": repr(clean.obj_value),
        "check_a_applicable": APPLICABLE[cand.error_type][0],
        "check_b_applicable": APPLICABLE[cand.error_type][1],
        # Check D does not exist. An explicit token, not a blank, so the
        # column never reads as a measurement that went missing.
        "check_d_fired": "not_built",
    })

    if is_noop(prob, mut_prob):
        rec.update({"bucket": "0 no-op", "mutant_status": "-"})
        return rec

    sol = solve(mut_prob)
    rec["mutant_status"] = sol.status
    rec["bucket"] = bucket_of(cand.touched_rows, clean, sol)
    rec["clean_binding"] = binding(clean, cand.touched_rows)
    rec["mutant_binding"] = binding(sol, cand.touched_rows)
    if not sol.optimal:
        return rec                              # bucket 1: never credited

    a_fired, a_detail = check_a(text, sol)
    b_fired, b_detail = check_b(text, sol)
    rec.update({
        "mutant_objval": repr(sol.obj_value),
        "obj_change_pct": abs(sol.obj_value - clean.obj_value)
                          / max(1.0, abs(clean.obj_value)) * 100.0,
        "check_a_fired": a_fired, "check_b_fired": b_fired,
        "check_a_n_vars_fired": sum(1 for d in a_detail if d["fired"]),
        "check_a_max_abs_residual": max(abs(d["residual"]) for d in a_detail),
        "check_b_sum": repr(b_detail["sum"]),
        "check_b_target": repr(b_detail["obj_value"]),
        "check_b_abs_diff": repr(b_detail["abs_diff"]),
    })
    return rec


def append_log(records: List[dict], path: str = MUTANT_LOG) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if not exists:
            w.writeheader()
        w.writerows(records)


# ==========================================================================
# entry points
# ==========================================================================

# The §11 Step 0 table, transcribed. Reproducing it from code that is IN the
# repo matters: the enumeration script that produced it is not.
STEP0 = {"1": (845 and 1861, 3219), "2": (795, 1253), "3": (845, 1557),
         "4a": (879, 1082), "4b": (1297, 1981), "5": (427, 576),
         "6": (1861, 3219), "7": (539, 1425), "8": (383, 622)}


def cmd_counts(args):
    df = counts(both_directions=args.both_directions)
    tab = (df.pivot_table(index="error_type", columns="subclass", values="n",
                          aggfunc="sum")
             .reindex(TYPE_ORDER).fillna(0).astype(int))
    scope = in_scope()
    print("=" * 78)
    print("CANDIDATE ENUMERATION -- counting only, no mutant generated")
    print("=" * 78)
    print(f"  scope: single {sum(1 for m in scope.values() if m['subclass']=='single')}"
          f"  multi {sum(1 for m in scope.values() if m['subclass']=='multi')}")
    print(f"  directions: "
          f"{'x10 and /10 counted separately' if args.both_directions else 'one candidate per cell (x10 arm only)'}")
    print("\n  {:<5} {:>8} {:>8} {:>8}   {:>8} {:>8}  {}".format(
        "type", "single", "multi", "total", "§11 sgl", "§11 mlt", "agreement"))
    grand = 0
    diffs = []
    for t in TYPE_ORDER:
        s, m = int(tab.loc[t, "single"]), int(tab.loc[t, "multi"])
        grand += s + m
        rs, rm = STEP0[t]
        ok = "exact" if (s, m) == (rs, rm) else "DIFFERS"
        if ok == "DIFFERS":
            diffs.append((t, s, m, rs, rm))
        print(f"  {t:<5} {s:>8} {m:>8} {s+m:>8}   {rs:>8} {rm:>8}  {ok}")
    print(f"\n  TOTAL candidate mutants: {grand}")
    print(f"  §11 table total:         {sum(a+b for a, b in STEP0.values())}")
    if diffs:
        print("\n  " + "-" * 74)
        print("  TYPES THAT DO NOT REPRODUCE §11 -- unit-of-enumeration questions,")
        print("  not arithmetic errors. Not resolved here.")
        for t, s, m, rs, rm in diffs:
            print(f"    type {t}: enumerated {s}/{m}, §11 recorded {rs}/{rm}")


def cmd_pilot(args):
    today = datetime.date.today().isoformat()
    scope = in_scope()
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)

    pool = {p: m for p, m in scope.items() if m["subclass"] == args.subclass}
    print("=" * 78)
    print(f"PILOT BATCH -- type {args.type}, {args.subclass}-resource, "
          f"limit {args.n} mutants")
    print("=" * 78)
    print(f"  instances in this subclass: {len(pool)}")
    print(f"  selection: deterministic -- instances in integer id order, "
          f"candidates in enumeration order, first {args.n}")
    if args.pin:
        print(f"  pinned first: {args.pin}  (carries the section-5 probe "
              f"observation into the log)")

    order = list(pool)
    if args.pin:
        order = [args.pin] + [p for p in order if p != args.pin]

    records, timings = [], []
    t_start = time.perf_counter()
    for pid in order:
        if len(records) >= args.n:
            break
        prob = problems[pid]
        clean = solve(prob)
        text = store.get(pid)
        for cand in enumerate_candidates(pid, prob, clean, args.type):
            if len(records) >= args.n:
                break
            t0 = time.perf_counter()
            rec = run_candidate(pid, pool[pid], prob, clean, cand, text,
                                provenance=args.provenance, today=today)
            timings.append((time.perf_counter() - t0) * 1e6)
            records.append(rec)
    t_total = time.perf_counter() - t_start

    distinct = len({r["problem_id"] for r in records})
    assert store.builds == distinct, \
        f"TextTerms built {store.builds} times for {distinct} instances"
    print(f"\n  TextTerms builds {store.builds} for {distinct} distinct "
          f"instances, {len(records)} mutants -> once per instance: OK")

    print("\n" + "-" * 78)
    print(f"  {'problem_id':>13} {'slot':<22} {'bucket':<17} {'A':<7} {'B':<7} "
          f"{'dobj%':>9} {'us':>7}")
    for rec, us in zip(records, timings):
        fa = "-" if rec["check_a_fired"] == "" else ("FIRES" if rec["check_a_fired"] else "silent")
        fb = "-" if rec["check_b_fired"] == "" else ("FIRES" if rec["check_b_fired"] else "silent")
        do = f"{rec['obj_change_pct']:9.3f}" if rec["obj_change_pct"] != "" else "        -"
        print(f"  {rec['problem_id']:>13} {rec['slot']:<22} {rec['bucket']:<17} "
              f"{fa:<7} {fb:<7} {do} {us:7.0f}")

    timings.sort()
    n = len(timings)
    print("\n" + "-" * 78)
    print("RUNTIME (generate + solve + bucket + check A/B + build log record)")
    print("-" * 78)
    print(f"  mutants                 {n}")
    print(f"  wall total              {t_total*1e3:.1f} ms")
    print(f"  per mutant  min/p50/max {timings[0]:.0f} / "
          f"{timings[n//2]:.0f} / {timings[-1]:.0f} us")
    print(f"  mean                    {sum(timings)/n:.0f} us")

    df = pd.DataFrame(records)
    print("\n  buckets: " + str(df.bucket.value_counts().to_dict()))
    ev = df[df.bucket == "3 evaluable"]
    if len(ev):
        applicable = APPLICABLE[args.type]
        which = "check_a_fired" if applicable[0] else "check_b_fired"
        print(f"  evaluable {len(ev)}; within-applicability check is "
              f"{which.replace('_fired','')} -> fired "
              f"{int(ev[which].astype(str).eq('True').sum())}/{len(ev)}")

    if args.write:
        append_log(records)
        print(f"\n  appended {len(records)} rows to {MUTANT_LOG}")
    else:
        print(f"\n  --write not given; nothing written")


def cmd_full(args):
    """Generate the FULL population ratified in amendment §12.7 / §12.10.

    Primary = narrow scope, both directions on types 1/2/3, type 8 one per
    instance. Extension = the wide-minus-narrow cells of types 1/4a/4b/6,
    labelled and excluded from the §6 macro-average.

    Integrity check run over every mutant, not a sample: a check that §5 marks
    N/A for a type must never fire on that type. One firing N/A cell aborts the
    run before anything is written.
    """
    today = datetime.date.today().isoformat()
    instances = in_scope()
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)

    print("=" * 78)
    print("FULL POPULATION GENERATION -- amendment §12.7 (a)(b)(d), §12.9, §12.10")
    print("=" * 78)
    print(f"  instances in scope {len(instances)}   provenance {args.provenance}")
    print("  primary = narrow scope; extension = wide-minus-narrow on types 1/4a/4b/6")

    records, violations = [], []
    t0 = time.perf_counter()
    for n, (pid, meta) in enumerate(instances.items(), 1):
        if n % 200 == 0:
            print(f"    [{n}/{len(instances)}] {len(records)} mutants")
        prob = problems[pid]
        clean = solve(prob)
        text = store.get(pid)
        for etype in TYPE_ORDER:
            narrow = enumerate_candidates(pid, prob, clean, etype, True, NARROW)
            batch = [(c, "primary") for c in narrow]
            if etype in SCOPED_TYPES:
                seen = {(c.error_type, c.direction, c.slot) for c in narrow}
                batch += [(c, "extension")
                          for c in enumerate_candidates(pid, prob, clean, etype,
                                                        True, WIDE)
                          if (c.error_type, c.direction, c.slot) not in seen]
            for cand, population in batch:
                rec = run_candidate(pid, meta, prob, clean, cand, text,
                                    args.provenance, today, population)
                records.append(rec)
                for check, applicable in (("a", APPLICABLE[etype][0]),
                                          ("b", APPLICABLE[etype][1])):
                    if not applicable and rec[f"check_{check}_fired"] is True:
                        violations.append((pid, etype, check, cand.slot))
    elapsed = time.perf_counter() - t0

    print(f"\n  generated {len(records)} mutants in {elapsed:.1f} s "
          f"({elapsed/len(records)*1e6:.0f} us each)")
    print(f"  TextTerms builds {store.builds} for {len(instances)} instances")

    print("\n" + "-" * 78)
    print("INTEGRITY -- §5 applicability, checked on EVERY mutant")
    print("-" * 78)
    print(f"  N/A cells that fired: {len(violations)}")
    if violations:
        for v in violations[:20]:
            print(f"    {v}")
        raise SystemExit("N/A cell fired -- §5 contradicted; nothing written")
    print("  -> no check fired on a type its §5 mask marks N/A")

    df = pd.DataFrame(records)
    print("\n" + "-" * 78)
    print("POPULATION")
    print("-" * 78)
    print(df.groupby(["population", "error_type"]).size().unstack(fill_value=0).to_string())
    print("\n  buckets: " + str(df.bucket.value_counts().to_dict()))

    if args.write:
        append_log(records)
        print(f"\n  appended {len(records)} rows to {MUTANT_LOG}")
    else:
        print("\n  --write not given; nothing written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("--counts".strip("-"), aliases=["counts"])
    c.add_argument("--both-directions", action="store_true")
    c.set_defaults(func=cmd_counts)
    p = sub.add_parser("pilot")
    p.add_argument("--type", default="3")
    p.add_argument("--subclass", default="single", choices=["single", "multi"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--pin", default=None)
    p.add_argument("--provenance", default="pilot-2026-08-02")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_pilot)
    f = sub.add_parser("full")
    f.add_argument("--provenance", default="full-2026-08-02")
    f.add_argument("--write", action="store_true")
    f.set_defaults(func=cmd_full)
    a = ap.parse_args()
    a.func(a)
