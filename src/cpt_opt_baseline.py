"""BASELINE — ReLoop threshold tester (CPT + OPT), reimplemented on this corpus.

    venv/bin/python cpt_opt_baseline.py

Reimplementation of the behavioural-testing layer of ReLoop (Lian et al. 2026,
arXiv:2602.15983, §3.3.2, "L2: Behavioral Testing (CPT + OPT)"), measured on
this project's own 1040-model corpus. This is a BASELINE to compare Check A /
Check B / Check D against. It is deliberately faithful to the published spec —
not improved, not reinterpreted.

NOTHING IN THIS FILE MODIFIES CHECK A, CHECK B, CHECK D OR THE MUTATION
HARNESS. It imports the existing model-building and solve path and adds one
new measurement.

--------------------------------------------------------------------------
CONSTANT PROVENANCE — primary source, verified 2026-08-04
--------------------------------------------------------------------------
Every constant below is quoted from ReLoop §3.3.2, arXiv:2602.15983v2 (revised
29 Apr 2026), fetched from arxiv.org/html/2602.15983v2 on 2026-08-04:

  "Constraint Presence Testing (CPT)... applies extreme perturbation to the
   governing parameter (capacity x0.001, demand x100, other x0.01). Objective
   Presence Testing (OPT) targets missing cost/revenue terms with analogous
   perturbation (cost x0.001, revenue x100, other x0.01)."

  "r < tau_l=5%: Warning... tau_l<=r<=tau_h=30%: Info... r>tau_h or
   infeasibility: Pass."

This supersedes amendment §17.5, which recorded the OPT multiplier and tau_h as
NOT confirmed and the baseline as "do not build yet". Both are now resolved:
the constants are verified against the primary source, at the same evidentiary
tier as the project's other novelty-gate citations — NOT the relayed tier that
§17.0 guards against.

--------------------------------------------------------------------------
FOUR INTERPRETIVE CALLS — stated and ratified 2026-08-04, not silent
--------------------------------------------------------------------------
ReLoop's CPT/OPT choose a multiplier from an LLM semantic tagging of each slot
(capacity / demand / cost / revenue). This corpus has no such tagging step, so
the role is derived structurally. That substitution is a DECISION, logged here:

  CALL 0 — role substitution
    CPT: row operator "<="  -> capacity rule (RHS x 0.001)
         row operator ">="  -> demand   rule (RHS x 100)
         row operator "="   -> other    rule (RHS x 0.01)   [expected unused]
    OPT: model sense max    -> revenue  rule (coefficient x 100)
         model sense min    -> cost     rule (coefficient x 0.001)
    Corpus check: 1734 "<=" and 1201 ">=" rows across the 1040 OPTIMAL
    instances, ZERO "=" rows, so the "other" bucket should never be reached.
    The run asserts this and reports loudly if a "=" row appears.

  CALL 1 — zero-RHS rows and zero objective coefficients are INELIGIBLE.
    0 x 0.001 = 0 and 0 x 100 = 0: the perturbation is a no-op, so r == 0
    exactly and the verdict is WARNING with certainty. Those would be false
    positives manufactured by arithmetic, not produced by the tester's logic.
    This is FAITHFUL, not a repair: ReLoop tags capacity/demand slots
    semantically, and a structural row like `x - 2y <= 0` carries no capacity
    slot to tag, so ReLoop would never emit a test for it.
    The Call-1-INCLUSIVE number is measured and reported as a sensitivity
    check — the ineligible tests are actually run, not assumed.

  CALL 2 — CPT iterates the rows PRESENT IN THE MODEL UNDER TEST, never the
    text's rows. ReLoop is a self-verification loop with no text-side row
    list; handing it one would hand it Check D's information. Consequence,
    stated in advance: for a constraint-omission mutant the deleted row is not
    there to perturb, so CPT can only see the omission through its effect on
    the surviving rows.

  CALL 3 — a mutant is DETECTED if any CPT or OPT test on it returns WARNING.
    Matches how Check A is scored (fires if any variable row fires). INFO
    scores neither way, per the spec.

  CALL 4 — z* is the objective of the MODEL UNDER TEST (the mutant), never the
    clean model. Self-verification has access to one model only.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gurobipy import GRB

from mutations import SECONDARY_ENUMERATORS, NARROW, solve
from text_terms import load_clean_problems

# ---- ReLoop §3.3.2 constants, verified against the primary source ----------
TAU_L = 0.05
TAU_H = 0.30

CPT_CAPACITY = 0.001
CPT_DEMAND = 100.0
CPT_OTHER = 0.01

OPT_COST = 0.001
OPT_REVENUE = 100.0
OPT_OTHER = 0.01

WARNING, INFO, PASS = "WARNING", "INFO", "PASS"

SECONDARY_LOG = "data/checkD_secondary.csv"


# ==========================================================================
# the rule
# ==========================================================================

def sensitivity(z_star: float, z_prime: Optional[float]) -> Tuple[str, Optional[float]]:
    """ReLoop's verdict rule. -> (verdict, r or None if infeasible/unbounded).

        r = |z' - z*| / |z*|, falling back to |z' - z*| when z* == 0
        r < TAU_L            -> WARNING   (likely missing; the detection signal)
        TAU_L <= r <= TAU_H  -> INFO      (uncertain; scores neither way)
        r > TAU_H or infeasible/unbounded -> PASS  (confirmed present)
    """
    if z_prime is None:
        return PASS, None
    d = abs(z_prime - z_star)
    r = d / abs(z_star) if z_star != 0.0 else d
    if r < TAU_L:
        return WARNING, r
    if r <= TAU_H:
        return INFO, r
    return PASS, r


def cpt_multiplier(operator: str) -> Tuple[float, str]:
    if operator == "<=":
        return CPT_CAPACITY, "capacity"
    if operator == ">=":
        return CPT_DEMAND, "demand"
    return CPT_OTHER, "other"


def opt_multiplier(sense: str) -> Tuple[float, str]:
    return (OPT_REVENUE, "revenue") if sense == "max" else (OPT_COST, "cost")


# ==========================================================================
# resolve_fn — reuses the existing pipeline, does not duplicate it
# ==========================================================================

def resolve_fn(prob, override: Tuple[str, str, float]) -> Optional[float]:
    """Apply ONE override to a ParsedProblem, re-solve, return z' or None.

    override = ("rhs", constraint_name, new_rhs)
             | ("obj", variable_name,   new_coefficient)

    The model is rebuilt through `mutations.solve` -> `nl4opt_parser.build_model`,
    so Presolve=0 / DualReductions=0 / OutputFlag=0 and the bounds-as-rows
    convention are inherited from the existing pipeline rather than restated.

    The override is applied in place and restored before returning, so the
    caller's ParsedProblem is unchanged. `verify_restore` below re-solves the
    base model to prove the restore actually worked.
    """
    kind, target, value = override
    if kind == "rhs":
        k = next(i for i, c in enumerate(prob.constraints) if c.name == target)
        saved = prob.constraints[k].rhs_const
        prob.constraints[k].rhs_const = value
        try:
            sol = solve(prob)
        finally:
            prob.constraints[k].rhs_const = saved
    elif kind == "obj":
        j = prob.var_names.index(target)
        t = next(i for i, (c, idx) in enumerate(prob.obj_terms) if idx == j)
        saved = prob.obj_terms[t]
        prob.obj_terms[t] = (value, j)
        try:
            sol = solve(prob)
        finally:
            prob.obj_terms[t] = saved
    else:
        raise ValueError(f"unknown override kind {kind!r}")
    return sol.obj_value if sol.optimal else None


# ==========================================================================
# eligibility (CALL 1)
# ==========================================================================

def cpt_rows(prob) -> List[Tuple[str, float, str, bool]]:
    """-> [(row name, rhs, operator, eligible)] for every row of the model."""
    return [(c.name, c.rhs_const, c.op, c.rhs_const != 0.0) for c in prob.constraints]


def opt_cells(prob) -> List[Tuple[str, float, bool]]:
    """-> [(variable name, objective coefficient, eligible)] for every variable
    that carries an objective term."""
    coefs: Dict[int, float] = {}
    for c, i in prob.obj_terms:
        coefs[i] = coefs.get(i, 0.0) + c
    return [(prob.var_names[i], a, a != 0.0) for i, a in sorted(coefs.items())]


# ==========================================================================
# one model under test
# ==========================================================================

@dataclass
class ModelVerdict:
    z_star: float
    tests: List[dict]

    def warned(self, eligible_only: bool = True) -> bool:
        return any(t["verdict"] == WARNING
                   for t in self.tests if t["eligible"] or not eligible_only)


def run_battery(prob, run_ineligible: bool = True) -> Optional[ModelVerdict]:
    """CPT over every row + OPT over every objective cell of ONE model.

    `run_ineligible=True` actually executes the Call-1-excluded tests as well,
    tagged `eligible=False`, so the inclusive sensitivity number is MEASURED
    rather than assumed to be all-WARNING.
    """
    base = solve(prob)
    if not base.optimal:
        return None
    z_star = base.obj_value
    tests: List[dict] = []

    for name, rhs, op, eligible in cpt_rows(prob):
        if not eligible and not run_ineligible:
            continue
        mult, role = cpt_multiplier(op)
        z_prime = resolve_fn(prob, ("rhs", name, rhs * mult))
        verdict, r = sensitivity(z_star, z_prime)
        tests.append({"test": "CPT", "target": name, "role": role, "op": op,
                      "base_value": rhs, "new_value": rhs * mult,
                      "z_prime": z_prime, "r": r, "verdict": verdict,
                      "eligible": eligible})

    sense = "max" if prob.sense == GRB.MAXIMIZE else "min"
    mult, role = opt_multiplier(sense)
    for name, coef, eligible in opt_cells(prob):
        if not eligible and not run_ineligible:
            continue
        z_prime = resolve_fn(prob, ("obj", name, coef * mult))
        verdict, r = sensitivity(z_star, z_prime)
        tests.append({"test": "OPT", "target": name, "role": role, "op": sense,
                      "base_value": coef, "new_value": coef * mult,
                      "z_prime": z_prime, "r": r, "verdict": verdict,
                      "eligible": eligible})

    return ModelVerdict(z_star, tests)


def verify_restore(prob, z_star: float) -> bool:
    """Re-solve the base model after a battery to prove every override was
    restored. A silent restore bug would corrupt every later test."""
    s = solve(prob)
    return s.optimal and abs(s.obj_value - z_star) <= 1e-9 * max(1.0, abs(z_star))


# ==========================================================================
# STEP 4 — cheap FPR proxy, pure pandas, no solver calls
# ==========================================================================

def cheap_fpr_proxy(constraints_df: pd.DataFrame,
                    optimal_ids: set) -> dict:
    """How many of the 1040 clean instances have at least one NON-BINDING
    CPT-eligible row?

    Per ReLoop Prop. 10's mechanism (as characterised by Li & Hai): a false
    positive fires whenever a capacity is slack because a sibling constraint
    binds first — perturbing a slack row cannot move the optimum, so r == 0 and
    the verdict is WARNING. That is STRUCTURAL, not a threshold-tuning problem.
    Counting slack CPT-eligible rows therefore predicts the FPR before any
    perturbation code runs.

    Column names are the real ones from `constraints.csv`: `rhs_limit` for the
    canonical RHS, `binding` (bool) for |Slack| < 1e-6.
    """
    c = constraints_df[constraints_df.problem_id.isin(optimal_ids)].copy()
    c["cpt_eligible"] = c["rhs_limit"] != 0.0
    elig = c[c.cpt_eligible]
    slack_elig = elig[~elig["binding"]]
    per_inst = slack_elig.groupby("problem_id").size()
    return {
        "instances": len(optimal_ids),
        "rows_total": len(c),
        "rows_cpt_eligible": len(elig),
        "rows_cpt_ineligible_zero_rhs": int((~c.cpt_eligible).sum()),
        "eligible_rows_non_binding": len(slack_elig),
        "eligible_rows_binding": int(elig["binding"].sum()),
        "instances_with_ge1_slack_eligible_row": int(len(per_inst)),
        "predicted_instance_fpr_pct": 100.0 * len(per_inst) / len(optimal_ids),
        "predicted_cpt_test_fpr_pct": 100.0 * len(slack_elig) / len(elig),
    }


# ==========================================================================
# run
# ==========================================================================

def _hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    problems, _ = load_clean_problems()
    results = pd.read_csv("data/results.csv", dtype={"problem_id": str})
    optimal = sorted(results.loc[results.solve_status == "OPTIMAL", "problem_id"],
                     key=int)
    optimal_set = set(optimal)

    _hdr("BASELINE — ReLoop CPT + OPT (Lian et al. 2026, §3.3.2), this corpus")
    print(f"  TAU_L {TAU_L}   TAU_H {TAU_H}")
    print(f"  CPT  capacity x{CPT_CAPACITY}  demand x{CPT_DEMAND}  other x{CPT_OTHER}")
    print(f"  OPT  cost     x{OPT_COST}  revenue x{OPT_REVENUE}  other x{OPT_OTHER}")
    print("  constants verified against arXiv:2602.15983v2 §3.3.2 on 2026-08-04")
    print(f"  clean OPTIMAL instances: {len(optimal)}")

    # ---------------------------------------------------------- STEP 4 ----
    _hdr("STEP 4 — cheap_fpr_proxy (pandas only, zero solver calls)")
    cons = pd.read_csv("data/constraints.csv", dtype={"problem_id": str})
    px = cheap_fpr_proxy(cons, optimal_set)
    for k, v in px.items():
        print(f"  {k:<38} {v}")
    print("\n  PREDICTION, before any perturbation is solved:")
    print(f"    per-CPT-test FPR ~ {px['predicted_cpt_test_fpr_pct']:.2f} %")
    print(f"    per-instance FPR ~ {px['predicted_instance_fpr_pct']:.2f} %")
    print("    (lower bound on the instance figure: OPT can only add warnings)")

    # ---------------------------------------------------------- STEP 3a ---
    _hdr("STEP 3(a) — FPR on the full clean 1040-instance corpus")
    print("  every CPT-eligible row and every OPT-eligible variable, all 1040")
    print("  Call-1-ineligible tests ARE executed too, tagged, and reported")
    print("  separately as the inclusive sensitivity number.")

    rows = []
    eq_rows = []
    restore_failures = []
    for n, pid in enumerate(optimal, 1):
        if n % 250 == 0:
            print(f"    [{n}/{len(optimal)}]")
        prob = problems[pid]
        for c in prob.constraints:
            if c.op not in ("<=", ">="):
                eq_rows.append((pid, c.name, c.op))
        mv = run_battery(prob, run_ineligible=True)
        if mv is None:
            raise AssertionError(f"{pid} is OPTIMAL in results.csv but re-solves badly")
        if not verify_restore(prob, mv.z_star):
            restore_failures.append(pid)
        for t in mv.tests:
            rows.append({"problem_id": pid, **t})
    clean = pd.DataFrame(rows)

    print(f"\n  '=' rows encountered (CPT 'other' bucket): {len(eq_rows)}")
    if eq_rows:
        print("    *** UNEXPECTED — the corpus was asserted to have none ***")
        for e in eq_rows[:10]:
            print(f"      {e}")
    print(f"  restore-verification failures: {len(restore_failures)}")
    if restore_failures:
        raise SystemExit(f"override not restored on {restore_failures[:5]} — stopping")

    def report_fpr(df, label):
        n = len(df)
        w = int((df.verdict == WARNING).sum())
        i = int((df.verdict == INFO).sum())
        p = int((df.verdict == PASS).sum())
        print(f"\n  {label}")
        print(f"    tests            {n}")
        print(f"    WARNING (FP)     {w}")
        print(f"    INFO             {i}")
        print(f"    PASS             {p}")
        print(f"    check n = W+I+P  {w + i + p == n}")
        print(f"    FPR (per test)   {w}/{n} = {100.0 * w / n:.2f} %")
        return n, w

    elig = clean[clean.eligible]
    _hdr("STEP 3(a) RESULTS")
    n_e, w_e = report_fpr(elig, "PRIMARY — Call-1 eligible tests only")
    for t in ("CPT", "OPT"):
        g = elig[elig.test == t]
        print(f"      {t}: {int((g.verdict == WARNING).sum())}/{len(g)} = "
              f"{100.0 * (g.verdict == WARNING).mean():.2f} %")
    n_i, w_i = report_fpr(clean, "SENSITIVITY — Call-1 INCLUSIVE (all tests)")
    inelig = clean[~clean.eligible]
    print(f"      ineligible tests {len(inelig)}, of which WARNING "
          f"{int((inelig.verdict == WARNING).sum())} "
          f"({100.0 * (inelig.verdict == WARNING).mean():.2f} % — measured, not assumed)")

    for label, df in (("eligible-only", elig), ("inclusive", clean)):
        flagged = df[df.verdict == WARNING].problem_id.nunique()
        print(f"\n  per-INSTANCE FPR, {label}: {flagged}/{len(optimal)} = "
              f"{100.0 * flagged / len(optimal):.2f} %  "
              f"(an instance is a false positive if ANY test warns)")

    clean.to_csv("data/cpt_opt_clean_tests.csv", index=False)

    # ---------------------------------------------------------- STEP 3b ---
    _hdr("STEP 3(b) — detection against Check D's target population")
    print(f"  source: {SECONDARY_LOG} — constraint-omission and operator-flip")
    print("  mutants only. NOT the full macro-average: CPT's stated logic")
    print("  ('insensitivity implies a missing constraint') targets this class.")

    sec = pd.read_csv(SECONDARY_LOG, dtype={"problem_id": str})
    parts = sec.mutant_id.str.split("|", n=3, expand=True)
    sec["problem_id"] = parts[0]
    sec["etype"] = parts[1]
    sec["direction"] = parts[2]
    sec["slot"] = parts[3]
    assert (sec.etype == sec.error_type).all(), "mutant_id etype disagrees with column"
    print(f"  mutants: {len(sec)}  {sec.error_type.value_counts().to_dict()}")

    det = []
    unresolved = []
    for n, ((pid, etype), g) in enumerate(sec.groupby(["problem_id", "error_type"]), 1):
        if n % 250 == 0:
            print(f"    [{n} instance-groups]")
        prob = problems[pid]
        base = solve(prob)
        cands = {(c.direction, c.slot): c
                 for c in SECONDARY_ENUMERATORS[etype](prob, base, NARROW)}
        for r in g.itertuples():
            cand = cands.get((r.direction, r.slot))
            if cand is None:
                unresolved.append(r.mutant_id)
                continue
            mut = copy.deepcopy(prob)
            cand.apply(mut)
            mv = run_battery(mut, run_ineligible=False)
            if mv is None:
                det.append({"mutant_id": r.mutant_id, "error_type": etype,
                            "subclass": r.subclass, "solved": False,
                            "detected": None, "n_tests": 0, "d_fired": r.d_fired})
                continue
            det.append({"mutant_id": r.mutant_id, "error_type": etype,
                        "subclass": r.subclass, "solved": True,
                        "detected": mv.warned(eligible_only=True),
                        "n_tests": sum(1 for t in mv.tests if t["eligible"]),
                        "n_warn": sum(1 for t in mv.tests
                                      if t["eligible"] and t["verdict"] == WARNING),
                        "d_fired": r.d_fired})
    dd = pd.DataFrame(det)
    if unresolved:
        print(f"\n  *** {len(unresolved)} mutant(s) not rebuildable — NOT counted ***")

    _hdr("STEP 3(b) RESULTS")
    ok = dd[dd.solved]
    refused = dd[~dd.solved]
    print(f"  rebuilt and solved   {len(ok)}")
    print(f"  refused on re-solve  {len(refused)} (excluded from rates)")
    print(f"\n  {'type':<6}{'subclass':<10}{'detected':>10}{'n':>8}{'rate %':>10}"
          f"{'  |  Check D':>14}")
    for t in ("CO", "OF"):
        for sub in ("single", "multi"):
            g = ok[(ok.error_type == t) & (ok.subclass == sub)]
            if not len(g):
                continue
            d_rate = 100.0 * g.d_fired.mean()
            print(f"  {t:<6}{sub:<10}{int(g.detected.sum()):>10}{len(g):>8}"
                  f"{100.0 * g.detected.mean():>10.2f}{d_rate:>13.2f} %")
        g = ok[ok.error_type == t]
        print(f"  {t:<6}{'ALL':<10}{int(g.detected.sum()):>10}{len(g):>8}"
              f"{100.0 * g.detected.mean():>10.2f}{100.0 * g.d_fired.mean():>13.2f} %")
    print(f"\n  {'BOTH':<6}{'ALL':<10}{int(ok.detected.sum()):>10}{len(ok):>8}"
          f"{100.0 * ok.detected.mean():>10.2f}{100.0 * ok.d_fired.mean():>13.2f} %")

    print("\n  agreement with Check D on the same mutants:")
    both = int((ok.detected & ok.d_fired).sum())
    base_only = int((ok.detected & ~ok.d_fired).sum())
    d_only = int((~ok.detected & ok.d_fired).sum())
    neither = int((~ok.detected & ~ok.d_fired).sum())
    print(f"    both fired            {both}")
    print(f"    baseline only         {base_only}")
    print(f"    Check D only          {d_only}")
    print(f"    neither               {neither}")
    print(f"    check sums to n       {both + base_only + d_only + neither == len(ok)}")

    dd.to_csv("data/cpt_opt_secondary_detection.csv", index=False)
    print("\nwrote cpt_opt_clean_tests.csv, cpt_opt_secondary_detection.csv")


if __name__ == "__main__":
    main()
