"""The §6 primary endpoint -- the pre-registered decision-gate computation.

    venv/bin/python compute_endpoint.py [--write-secondary]

Reads `mutants_log.csv`, filtered to `provenance == "full-2026-08-02"`.
The 18 pilot rows are excluded; they are duplicates of mutants inside the full
run and are retained only for provenance.

DEFINITIONS APPLIED, all from the amendment
-------------------------------------------
denominator   bucket-3 EVALUABLE mutants of that type, within that check's §5
              applicability set. Bucket 0 (no-op), 1 (solver refused) and 2
              (inert) never enter a numerator or a denominator. §7: solver
              refusal is a detection mechanism but is NOT these checks, and
              crediting it would inflate them.
numerator     mutants of that denominator on which the applicable check fired.
which check   §5 mask: types 1, 2, 4, 5, 6 -> Check A only. Types 3, 7 ->
              Check B only. Type 8 -> both applicable.
population    PRIMARY only (§12.9 narrow scope). The 7953-mutant wide-scope
              extension is reported separately in part E and never enters the
              macro-average.

TWO READINGS THE AMENDMENT DOES NOT SPELL OUT, RESOLVED HERE AND FLAGGED
-----------------------------------------------------------------------
1. **Type 4 is ONE type, not two.** §1's table lists eight types, of which #4
   is "coefficient swap" with variants "within-row, cross-row". §11 enumerated
   them as 4a and 4b for counting convenience. The macro-average in §6 is over
   the EIGHT types of §1, so 4a and 4b are POOLED into type 4 for the endpoint,
   exactly as §1 treats ×10/÷10 -- "a recorded dimension, not a separate type.
   Rates are reported by direction within each type." 4a and 4b are reported
   separately as variants.
2. **Type 8 has two applicable checks.** A mutant is counted as detected when
   EITHER applicable check fires, since the method is the pair of checks. The
   A-only and B-only rates are also reported.

Neither reading is a free parameter chosen after seeing results: 1 follows
§1's own type list, 2 follows from the method being A and B together. Both are
stated so a reviewer can apply the other reading if they disagree -- the
alternative numbers are printed alongside.
"""

from __future__ import annotations

import argparse
import copy
import datetime

import pandas as pd

from mutation_harness import (APPLICABLE, LOG_COLS, MUTANT_LOG, append_log,
                              enumerate_candidates, in_scope, run_candidate)
from mutations import NARROW, SECONDARY_ENUMERATORS, WIDE, bucket_of, solve
from text_terms import TextTermsStore, load_clean_problems

PROVENANCE = "full-2026-08-02"

# §6 decision table, unchanged from 26 July
BANDS = [(50.0, "build confidently"),
         (30.0, "build with complementarity framing"),
         (15.0, "build only if the miss breakdown is clean"),
         (0.0, "STOP and reassess")]

# the eight types of §1; 4a and 4b pool into 4
ENDPOINT_TYPES = ["1", "2", "3", "4", "5", "6", "7", "8"]
POOLS = {"4": ["4a", "4b"]}


def band(rate: float) -> str:
    for threshold, action in BANDS:
        if rate >= threshold:
            return action
    return BANDS[-1][1]


def fired_col(etype: str) -> str:
    """Which column carries 'detected' for this type, per the §5 mask."""
    a, b = APPLICABLE[etype]
    if a and b:
        return "either"
    return "check_a_fired" if a else "check_b_fired"


def detected(df: pd.DataFrame, etype: str) -> pd.Series:
    col = fired_col(etype)
    if col == "either":
        return df["check_a_fired"].eq(True) | df["check_b_fired"].eq(True)
    return df[col].eq(True)


def rate_table(ev: pd.DataFrame, types, label_map=None) -> pd.DataFrame:
    rows = []
    for t in types:
        members = POOLS.get(t, [t])
        sub = ev[ev.error_type.isin(members)]
        for subclass in ("single", "multi"):
            g = sub[sub.subclass == subclass]
            n = len(g)
            k = int(detected(g, members[0]).sum()) if n else 0
            rows.append({"error_type": t, "subclass": subclass, "n": n,
                         "fired": k, "rate": (100.0 * k / n) if n else float("nan"),
                         "check": {"check_a_fired": "A", "check_b_fired": "B",
                                   "either": "A or B"}[fired_col(members[0])]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-secondary", action="store_true",
                    help="append the sense-flip and D-only populations to the log")
    args = ap.parse_args()
    today = datetime.date.today().isoformat()

    d = pd.read_csv(MUTANT_LOG, dtype={"problem_id": str})
    full = d[d.provenance == PROVENANCE].copy()
    for c in ("check_a_fired", "check_b_fired"):
        full[c] = full[c].map({True: True, False: False, "True": True,
                               "False": False})

    print("=" * 78)
    print("§6 PRIMARY ENDPOINT -- pre-registered decision-gate computation")
    print("=" * 78)
    print(f"  computed {today}")
    print(f"  log rows {len(d)}   filtered to provenance=={PROVENANCE!r}: {len(full)}")
    print(f"  excluded: {len(d) - len(full)} pilot rows (duplicates, provenance only)")

    prim = full[full.population == "primary"]
    ext = full[full.population == "extension"]
    print(f"  PRIMARY population {len(prim)}   secondary extension {len(ext)} "
          f"(part E only, never in the macro-average)")

    ev = prim[prim.bucket == "3 evaluable"]
    print(f"  bucket-3 evaluable within primary: {len(ev)}")
    print("  buckets excluded from every rate: "
          + str(prim[prim.bucket != "3 evaluable"].bucket.value_counts().to_dict()))

    # ---------------------------------------------------------------- A ----
    print("\n" + "=" * 78)
    print("A. PER-TYPE DETECTION RATE  (denominator = evaluable within applicability)")
    print("=" * 78)
    variants = rate_table(ev, ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
    print(f"  {'type':<6}{'check':<8}{'subclass':<10}{'fired':>8}{'evaluable':>11}"
          f"{'rate %':>10}")
    for r in variants.itertuples():
        print(f"  {r.error_type:<6}{r.check:<8}{r.subclass:<10}{r.fired:>8}"
              f"{r.n:>11}{r.rate:>10.2f}")

    print("\n  --- 4a and 4b POOLED into type 4 (the §1 type list has eight types) ---")
    endpoint = rate_table(ev, ENDPOINT_TYPES)
    print(f"  {'type':<6}{'check':<8}{'subclass':<10}{'fired':>8}{'evaluable':>11}"
          f"{'rate %':>10}")
    for r in endpoint.itertuples():
        print(f"  {r.error_type:<6}{r.check:<8}{r.subclass:<10}{r.fired:>8}"
              f"{r.n:>11}{r.rate:>10.2f}")

    # type 8, both readings
    t8 = ev[ev.error_type == "8"]
    print("\n  type 8 detail (two applicable checks):")
    for subclass in ("single", "multi"):
        g = t8[t8.subclass == subclass]
        n = len(g)
        print(f"    {subclass:<8} n={n:<5} A {int(g.check_a_fired.sum()):>4} "
              f"({100*g.check_a_fired.sum()/n:6.2f}%)   "
              f"B {int(g.check_b_fired.sum()):>4} ({100*g.check_b_fired.sum()/n:6.2f}%)   "
              f"A or B {int((g.check_a_fired|g.check_b_fired).sum()):>4} "
              f"({100*(g.check_a_fired|g.check_b_fired).sum()/n:6.2f}%)")

    # ---------------------------------------------------------------- B ----
    print("\n" + "=" * 78)
    print("B. MACRO-AVERAGE  (equal weight per type, 8 types, per subclass)")
    print("=" * 78)
    macro = {}
    for subclass in ("single", "multi"):
        rates = endpoint[endpoint.subclass == subclass].set_index("error_type")["rate"]
        macro[subclass] = float(rates.mean())
        print(f"  {subclass}-resource: mean of "
              + ", ".join(f"{t}={rates[t]:.2f}" for t in ENDPOINT_TYPES))
        print(f"    -> macro-average {macro[subclass]:.2f} %")
    lower = min(macro, key=macro.get)
    print(f"\n  §6: 'where the two disagree, the action taken is the one indicated")
    print(f"      by the LOWER of the two.'")
    print(f"  LOWER = {lower}-resource at {macro[lower]:.2f} %")

    # pooled (micro) rate, reported as secondary per §6
    micro = {}
    for subclass in ("single", "multi"):
        g = ev[ev.subclass == subclass]
        k = sum(int(detected(g[g.error_type.isin(POOLS.get(t, [t]))],
                             POOLS.get(t, [t])[0]).sum()) for t in ENDPOINT_TYPES)
        micro[subclass] = 100.0 * k / len(g)
    print(f"  secondary, pooled (micro) rate: single {micro['single']:.2f} %, "
          f"multi {micro['multi']:.2f} %")

    # ---------------------------------------------------------------- C ----
    print("\n" + "=" * 78)
    print("C. §6 DECISION TABLE APPLIED")
    print("=" * 78)
    for threshold, action in BANDS:
        mark = "  <-- " if band(macro[lower]) == action else "      "
        rng = {50.0: ">= 50%", 30.0: "30-50%", 15.0: "15-30%", 0.0: "< 15%"}[threshold]
        print(f"  {rng:<8} {action:<45}{mark}")
    print(f"\n  macro-average (lower subclass) = {macro[lower]:.2f} %")
    print(f"  ACTION INDICATED: {band(macro[lower])}")

    # misses and the override
    miss = ev[~detected_all(ev)]
    if len(miss):
        small = miss[pd.to_numeric(miss.obj_change_pct, errors="coerce") < 1.0]
        frac = 100.0 * len(small) / len(miss)
        print(f"\n  §6 OVERRIDE -- 'if MOST misses changed the objective by < 1%,")
        print(f"                  build regardless'")
        print(f"    misses (evaluable, applicable check silent)   {len(miss)}")
        print(f"    of which objective change < 1%                {len(small)} "
              f"({frac:.2f} %)")
        print(f"    override applies (>50% of misses): {frac > 50.0}")
    else:
        print("\n  no misses; override moot")

    # ---------------------------------------------------------------- D ----
    print("\n" + "=" * 78)
    print("D. MISS BREAKDOWN  (evaluable, applicable check did NOT fire)")
    print("=" * 78)
    print("  §7: 'binding-in-clean -> slack-in-mutant DOES change the answer and is")
    print("       a GENUINE MISS, not excused.'")
    m = miss.copy()
    m["genuine"] = [
        any(c == "True" and mu == "False"
            for c, mu in zip(str(cb).split(";"), str(mb).split(";")))
        for cb, mb in zip(m.clean_binding, m.mutant_binding)]
    print(f"\n  {'type':<6}{'subclass':<10}{'misses':>8}{'binding->slack':>16}"
          f"{'other silent':>14}")
    for t in ENDPOINT_TYPES:
        for subclass in ("single", "multi"):
            g = m[(m.error_type.isin(POOLS.get(t, [t]))) & (m.subclass == subclass)]
            if not len(g):
                continue
            print(f"  {t:<6}{subclass:<10}{len(g):>8}{int(g.genuine.sum()):>16}"
                  f"{int((~g.genuine).sum()):>14}")
    print(f"\n  TOTAL misses {len(m)}   binding->slack {int(m.genuine.sum())} "
          f"({100*m.genuine.mean():.2f}%)   other silent "
          f"{int((~m.genuine).sum())} ({100*(~m.genuine).mean():.2f}%)")
    oth = m[~m.genuine]
    if len(oth):
        print("\n  'other silent' by clean->mutant binding pattern:")
        pat = (oth.clean_binding.astype(str) + " -> " + oth.mutant_binding.astype(str))
        for k, v in pat.value_counts().head(8).items():
            print(f"    {k:<40} {v}")

    # ---------------------------------------------------------------- E ----
    print("\n" + "=" * 78)
    print("E. SECONDARY -- NOT IN THE MACRO-AVERAGE")
    print("=" * 78)
    print("\n  E1. Wide-scope extension (types 1/4a/4b/6), §12.9")
    eev = ext[ext.bucket == "3 evaluable"]
    etab = rate_table(eev, ["1", "4a", "4b", "6"])
    print(f"  {'type':<6}{'check':<8}{'subclass':<10}{'fired':>8}{'evaluable':>11}"
          f"{'rate %':>10}")
    for r in etab.itertuples():
        print(f"  {r.error_type:<6}{r.check:<8}{r.subclass:<10}{r.fired:>8}"
              f"{r.n:>11}{r.rate:>10.2f}")

    print("\n  E2. Sense flip (§3 footnote) -- fires through Check A's at-zero branch")
    sf = run_extra(["SF"], args.write_secondary, today, "full-2026-08-02-sf")
    show_extra(sf, "SF", rate=True)

    # ---------------------------------------------------------------- F ----
    print("\n" + "=" * 78)
    print("F. §2 SECONDARY TYPES -- COUNTS ONLY, NO DETECTION RATE")
    print("=" * 78)
    print("  Check D does not exist. §5 marks both N/A for A and B, so a rate")
    print("  against A/B would be a guaranteed zero that says nothing about the")
    print("  mutation. §2: if D is not built these move to §8 as demonstrated")
    print("  impossibility results and are NOT reported as detection classes with")
    print("  zero rates. Counts only, pending the open Check D decision.")
    co_of = run_extra(["CO", "OF"], args.write_secondary, today,
                      "full-2026-08-02-secondary", secondary=True)
    for t in ("CO", "OF"):
        show_extra(co_of, t, rate=False)
    leak = co_of[(co_of.check_a_fired == True) | (co_of.check_b_fired == True)]
    print(f"\n    §5 says both are N/A for A and B. Mutants on which A or B fired "
          f"anyway: {len(leak)}")
    print("    (a nonzero count would contradict §5; it is checked, not assumed)")

    print("\n" + "=" * 78)
    print(f"HEADLINE: macro-average {macro['single']:.2f} % single / "
          f"{macro['multi']:.2f} % multi; §6 acts on the lower "
          f"({macro[lower]:.2f} %) -> {band(macro[lower])}")
    print("=" * 78)


def detected_all(ev: pd.DataFrame) -> pd.Series:
    """Row-wise 'the applicable check fired', respecting each row's own type."""
    out = pd.Series(False, index=ev.index)
    for t in ev.error_type.unique():
        mask = ev.error_type == t
        out.loc[mask] = detected(ev[mask], t).values
    return out


def run_extra(types, write, today, provenance, secondary=False):
    """Generate a population that the primary run did not cover."""
    from mutations import ENUMERATORS
    instances = in_scope()
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    recs = []
    for pid, meta in instances.items():
        prob = problems[pid]
        clean = solve(prob)
        text = store.get(pid)
        for t in types:
            enum = (SECONDARY_ENUMERATORS if secondary else ENUMERATORS)[t]
            for cand in enum(prob, clean, NARROW):
                recs.append(run_candidate(pid, meta, prob, clean, cand, text,
                                          provenance, today, "secondary"))
    df = pd.DataFrame(recs)
    if write:
        append_log(recs)
        print(f"    [appended {len(recs)} rows, provenance {provenance}]")
    return df


def show_extra(df, t, rate):
    g = df[df.error_type == t]
    name = {"SF": "sense flip", "CO": "constraint omission",
            "OF": "operator flip"}[t]
    print(f"\n    {name}: candidates {len(g)}   buckets "
          + str(g.bucket.value_counts().to_dict()))
    ev = g[g.bucket == "3 evaluable"]
    for subclass in ("single", "multi"):
        s = ev[ev.subclass == subclass]
        if rate and len(s):
            k = int(s.check_a_fired.eq(True).sum())
            print(f"      {subclass:<8} evaluable {len(s):>5}   Check A fired "
                  f"{k:>5}  ({100.0*k/len(s):.2f} %)")
        else:
            print(f"      {subclass:<8} candidates "
                  f"{len(g[g.subclass==subclass]):>5}   evaluable {len(s):>5}   "
                  f"detection rate: NOT COMPUTED (Check D not built)")


if __name__ == "__main__":
    main()
