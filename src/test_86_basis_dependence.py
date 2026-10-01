"""Are the 86 detections, or solver artifacts? 2026-08-27.

THE OBJECTION
-------------
analyse_dual_only.py found 86 mutants whose optimal value and optimal solution
both equal the reference's, yet a check reports. It attributed this to the dual
moving while the primal did not, and called it a capability.

At a degenerate optimum the dual solution is NOT UNIQUE. Which pi the solver
returns depends on which optimal basis it lands in. Disabling presolve and dual
reductions makes that reproducible for a fixed input; it does not make it
unique. So "the dual moved" may mean only that the solver picked a different
vertex of an unchanged dual polytope -- a solver artifact, not a property of the
model. The paper cannot call these detections without ruling that out, and it
elsewhere dismisses the reference solve's 75 answer-neutral flags as exactly
this kind of artifact.

THE TEST
--------
Stage 1, basis dependence. Re-solve each of the 86 under primal simplex, dual
simplex and barrier. These land in different optimal bases when one exists. If
ANY method returns a pi on which the checks are silent, then a dual-optimal pi
satisfying the text's identities EXISTS and the report is an artifact of basis
selection.

Stage 2, existence. For the survivors, ask the question directly: does the
mutant's dual optimal set contain any pi satisfying the text-side identities?
That is a feasibility LP in pi, combining
    dual feasibility and complementary slackness w.r.t. the MUTANT (which
      characterises the dual optimal set given x*), with
    Check A and Check B written against the TEXT's coefficients.
Infeasible means no dual-optimal pi satisfies the checks, so the report is a
genuine property of the model. Feasible means the check is reading solver state.

The same stage-1 test is applied to the reference solve's answer-neutral flags,
because calling one set artifacts and the other capabilities without testing
both is the asymmetry the review identified.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json

import gurobipy as gp
from gurobipy import GRB

from checks_common import REL_TOL, ZERO_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, env, solve
import mutations
from nl4opt_parser import build_model
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
METHODS = {0: "primal simplex", 1: "dual simplex", 2: "barrier"}


def solve_with(prob, method: int):
    """Solve under a named LP method. -> (obj, pi dict, x dict) or None."""
    m = build_model(prob, env=env())
    m.setParam("Method", method)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return None
    out = (m.ObjVal,
           {r.ConstrName: r.Pi for r in m.getConstrs()},
           {nm: gv.X for gv, nm in zip(m.getVars(), prob.var_names)})
    m.dispose()
    return out


class Sol:
    """Minimal stand-in for mutations.Solved so the real checks can be reused."""
    def __init__(self, obj, pi, values):
        self.status, self.obj_value, self.pi, self.values = "OPTIMAL", obj, pi, values
        self.slack = {}
    optimal = True


def dual_optimal_pi_satisfying_checks(prob, text, x_star, z_star):
    """Stage 2. -> True if SOME dual-optimal pi satisfies Checks A and B.

    Variables are the raw duals pi_i as the checks read them. Dual optimality is
    imposed through the standard-form reduction of Proposition A.1: with
    tau_i = +1 for <= and -1 for >=, and sigma = +1 for max, pi_i = sigma*tau_i*y_i
    with y_i >= 0, and the candidate is max/min over {x >= 0 : A~x <= b~}.
    """
    ct = text_terms_of(prob)
    sigma = 1.0 if ct.sense == "max" else -1.0
    rows, vs = ct.row_names, ct.var_names
    m = gp.Model(env=env())
    m.setParam("OutputFlag", 0)
    y = {r: m.addVar(lb=0.0, name=f"y_{i}") for i, r in enumerate(rows)}
    m.update()

    def pi(r):
        tau = 1.0 if ct.row_operator[r] == "<=" else -1.0
        return sigma * tau * y[r]

    # --- dual optimality w.r.t. the MUTANT ---------------------------------
    for r in rows:
        lhs = sum(ct.row_coefs[r].get(v, 0.0) * x_star[v] for v in vs)
        b = ct.row_limits[r]
        tau = 1.0 if ct.row_operator[r] == "<=" else -1.0
        if tau * (lhs - b) < -1e-7 * max(1.0, abs(b)):      # row slack at x*
            m.addConstr(y[r] == 0.0)
    for v in vs:
        red = gp.quicksum((1.0 if ct.row_operator[r] == "<=" else -1.0)
                          * ct.row_coefs[r].get(v, 0.0) * y[r] for r in rows)
        cj = sigma * ct.obj_coefs[v]
        if abs(x_star[v]) > ZERO_TOL:
            m.addConstr(red == cj)                          # comp. slackness
        else:
            m.addConstr(red >= cj)                          # dual feasibility

    # --- the TEXT-side identities the checks test --------------------------
    tv = text
    for v in tv.var_names:
        s_j = gp.quicksum(a * pi(r) for r, a in tv.terms_for(v) if r in y)
        cj = tv.obj_coef(v)
        if abs(x_star.get(v, 0.0)) > ZERO_TOL:
            m.addConstr(s_j == cj)
        elif tv.sense == "max":
            m.addConstr(s_j >= cj)
        else:
            m.addConstr(s_j <= cj)
    m.addConstr(gp.quicksum(tv.row_limits[r] * pi(r) for r in tv.row_names
                            if r in y) == z_star)

    m.setObjective(0.0)
    m.optimize()
    feasible = m.Status == GRB.OPTIMAL
    m.dispose()
    return feasible


def main():
    target = {r["id"] for r in csv.DictReader(open(f"solve_reference_{TODAY}.csv"))
              if r["arm"] == "evaluable" and r["checks"] == "True"
              and r["oracle"] == "False"}
    print(f"testing {len(target)} checks-only mutants\n")

    problems, _ = load_clean_problems()
    rows, done = [], set()
    for pid, _m in in_scope().items():
        if not any(t.startswith(pid + "|") for t in target):
            continue
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        for et in TYPE_ORDER:
            for scope in ("narrow", "wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    if mid not in target or mid in done:
                        continue
                    done.add(mid)
                    mut = copy.deepcopy(prob)
                    cand.apply(mut)

                    # ---- stage 1: does any LP method silence the checks? ----
                    silenced_by = []
                    base = None
                    for meth in METHODS:
                        got = solve_with(mut, meth)
                        if got is None:
                            continue
                        obj, pi, xs = got
                        s = Sol(obj, pi, xs)
                        if base is None:
                            base = s
                        fired = (check_a(text, s)[0] or check_b(text, s)[0]
                                 or check_d(text, s)[0])
                        if not fired:
                            silenced_by.append(METHODS[meth])

                    # ---- stage 2: existence over the whole dual optimal set --
                    exists = None
                    if base is not None and not silenced_by:
                        try:
                            exists = dual_optimal_pi_satisfying_checks(
                                mut, text, base.values, base.obj_value)
                        except Exception as e:                    # noqa: BLE001
                            exists = f"error: {type(e).__name__}"
                    rows.append({"mutant_id": mid, "error_type": cand.error_type,
                                 "silenced_by": ";".join(silenced_by) or "-",
                                 "basis_dependent": bool(silenced_by),
                                 "dual_optimal_pi_exists": exists})

    n = len(rows)
    bd = sum(r["basis_dependent"] for r in rows)
    ex = sum(r["dual_optimal_pi_exists"] is True for r in rows)
    gen = sum(r["dual_optimal_pi_exists"] is False for r in rows)
    err = sum(isinstance(r["dual_optimal_pi_exists"], str) for r in rows)
    print("=" * 70)
    print("ARE THE 86 DETECTIONS OR SOLVER ARTIFACTS?")
    print("=" * 70)
    print(f"  tested                                          {n}")
    print(f"  silenced by some LP method (BASIS ARTIFACT)     {bd}")
    print(f"  a dual-optimal pi satisfying the checks EXISTS   {ex}   <- artifact")
    print(f"  NO such pi exists (GENUINE DETECTION)           {gen}")
    if err:
        print(f"  feasibility model errored                       {err}")
    print(f"\n  genuine detections as a share of 18,525: "
          f"{100*gen/18525:.3f}%")

    with open(f"basis_test_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({"stamp": TODAY, "n": n, "basis_dependent": bd,
               "artifact_pi_exists": ex, "genuine": gen, "errors": err},
              open(f"basis_test_{TODAY}.json", "w"), indent=1)
    print(f"\nwrote basis_test_{TODAY}.csv/.json")


if __name__ == "__main__":
    main()
