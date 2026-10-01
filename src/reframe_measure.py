"""The measurements the v9 review demanded. 2026-08-28.

Four things, none of which existed as a persisted artefact before this script.

(1) THE NORMALIZED CHECKS AGAINST THE REFERENCE, ON EVERY STRATUM OF TABLE 1
    row_normalization.py measured the normalization on the evaluable population
    (Checks A or B) and on the restructurings. It never measured it on the
    5,507 answer-neutral control, so the body compared the UNNORMALIZED checks
    against the reference solve and concluded the reference dominates. The
    normalization is our own contribution and costs nothing in detection, so
    that is the wrong variant to compare. This arm measures the normalized
    union A|B|C on all three strata beside the reference.

(2) A PRIMAL-FACE-ROBUST CERTIFICATE FOR THE 86  (review question 2)
    test_86_basis_dependence.py quantifies over the dual optimal set with x*
    held at the value one solver returned. Check A's equality branch is gated
    on x*_j > 0, so a different primal optimum can change which identities are
    equalities. If the mutant has alternative primal optima the certificate is
    conditional on a vertex, which is the objection it was built to answer.

    Fix: classify each variable over the mutant's whole primal optimal face --
    always zero, always positive, or varying -- and impose the EQUALITY branch
    only where x_j > 0 is forced. Where it varies, impose the inequality alone,
    which every branch implies. The result is a relaxation, so infeasibility
    certifies that no (primal optimum, dual-optimal pi) pair silences the
    checks. Dual optimality is imposed x*-free, as dual feasibility plus strong
    duality, which is the dual optimal set exactly.

(3) THE SAME CERTIFICATE FOR THE REFERENCE'S 75  (review question 3)
    The body states "43 of them are silenced by some LP method and 32 are not."
    NO ARTEFACT IN THIS REPOSITORY PRODUCES THOSE NUMBERS. The docstring of
    test_86_basis_dependence.py promises the arm; its main() never ran it. This
    script runs stage 1 for real, and adds the certificate that is genuinely
    analogous to ours: the reference flags a mutant when its (z*, x*) differs,
    so the question is whether the mutant's optimal FACE and the text model's
    optimal face intersect. If they do, a solver could have returned the same
    x from both and the report is a vertex artifact. If they are disjoint, no
    choice of optimal basis silences it. That is a feasibility LP in x.

(4) WHY THE REFERENCE REPORTS AT ALL ON A POPULATION WHOSE ANSWER IS UNCHANGED
    (review question 4). Recorded as the obj_moved/sol_moved split of the 75.

Outputs reframe_<date>.{csv,json}.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
from collections import Counter

import gurobipy as gp
from gurobipy import GRB

from checks_common import REL_TOL, ZERO_TOL
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, env, solve
import mutations
from nl4opt_parser import build_model
from row_normalization import check_a_norm, check_b_norm, scales_for
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
TYPE_ORDER = getattr(mutations, "TYPE_ORDER",
                     ["1", "2", "3", "4a", "4b", "5", "6", "7", "8"])
METHODS = {0: "primal simplex", 1: "dual simplex", 2: "barrier"}
X_TOL = 1e-6
FACE_TOL = 1e-7


class Sol:
    """Stand-in for mutations.Solved so the real checks can be reused."""
    def __init__(self, obj, pi, values):
        self.status, self.obj_value, self.pi, self.values = "OPTIMAL", obj, pi, values
        self.slack = {}
    optimal = True


def solve_with(prob, method: int):
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


def differs(ref, cand, names):
    dz = abs(cand.obj_value - ref.obj_value) > REL_TOL * max(1.0, abs(ref.obj_value))
    scale = max([abs(ref.values[v]) for v in names] + [1.0])
    dx = any(abs(cand.values[v] - ref.values[v]) > X_TOL * scale for v in names)
    return dz, dx


# ==========================================================================
# (2)  the primal optimal face, and a certificate that quantifies over it
# ==========================================================================

def face_classes(prob, z_star):
    """Classify each variable over the primal optimal face of `prob`.

    -> dict var -> "zero" | "pos" | "varies", or None if the face LP fails.
    "zero"   x_j = 0 at every optimum      -> the check takes the inequality branch
    "pos"    x_j > 0 at every optimum      -> the check takes the equality branch
    "varies" both occur on the face        -> branch is not determined by the model
    """
    out, uniq = {}, True
    m = build_model(prob, env=env())
    obj = m.getObjective()
    m.addConstr(obj == z_star, name="_face")
    m.update()
    gvars = list(m.getVars())
    for gv, nm in zip(gvars, prob.var_names):
        m.setObjective(gv, GRB.MAXIMIZE)
        m.optimize()
        if m.Status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
            hi = float("inf")            # face unbounded in x_j; x_j > 0 attainable
        elif m.Status != GRB.OPTIMAL:
            m.dispose()
            return None, None
        else:
            hi = m.ObjVal
        m.setObjective(gv, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            m.dispose()
            return None, None
        lo = m.ObjVal
        if hi - lo > FACE_TOL * max(1.0, abs(hi)):
            uniq = False
        if hi <= ZERO_TOL:
            out[nm] = "zero"
        elif lo > ZERO_TOL:
            out[nm] = "pos"
        else:
            out[nm] = "varies"
            uniq = False
    m.dispose()
    return out, uniq


def pi_exists_over_face(mut, text, z_star, cls):
    """True if SOME dual-optimal pi of `mut` silences the checks, for SOME
    primal optimum of `mut`. A relaxation: where the branch varies over the
    face we impose only the inequality, which both branches imply. Infeasible
    therefore certifies the report on every optimal basis, primal and dual.

    Dual optimality is imposed WITHOUT reference to x*: dual feasibility plus
    strong duality is the dual optimal set exactly.
    """
    ct = text_terms_of(mut)
    sigma = 1.0 if ct.sense == "max" else -1.0
    rows, vs = ct.row_names, ct.var_names
    m = gp.Model(env=env())
    m.setParam("OutputFlag", 0)
    y = {r: m.addVar(lb=0.0, name=f"y_{i}") for i, r in enumerate(rows)}
    m.update()

    def tau(r):
        return 1.0 if ct.row_operator[r] == "<=" else -1.0

    def pi(r):
        return sigma * tau(r) * y[r]

    # --- the mutant's dual optimal set, x*-free --------------------------
    for v in vs:                                          # dual feasibility
        red = gp.quicksum(tau(r) * ct.row_coefs[r].get(v, 0.0) * y[r] for r in rows)
        m.addConstr(red >= sigma * ct.obj_coefs[v])
    m.addConstr(gp.quicksum(ct.row_limits[r] * pi(r)      # strong duality
                            for r in rows) == z_star)

    # --- the text-side identities, gated over the whole face -------------
    for v in text.var_names:
        s_j = gp.quicksum(a * pi(r) for r, a in text.terms_for(v) if r in y)
        cj = text.obj_coef(v)
        if cls.get(v) == "pos":
            m.addConstr(s_j == cj)
        elif text.sense == "max":
            m.addConstr(s_j >= cj)
        else:
            m.addConstr(s_j <= cj)
    m.addConstr(gp.quicksum(text.row_limits[r] * pi(r)
                            for r in text.row_names if r in y) == z_star)

    m.setObjective(0.0)
    m.optimize()
    feasible = m.Status == GRB.OPTIMAL
    m.dispose()
    return feasible


# ==========================================================================
# (3)  the certificate that is actually analogous, for the reference
# ==========================================================================

def optimal_faces_intersect(text_prob, mut, z_text, z_mut):
    """True if some x is optimal for BOTH the text's model and the mutant.

    The reference flags a candidate when its returned (z*, x*) differs from the
    text model's. If the two optimal faces intersect, a solver could return the
    same x from both, so the report reflects vertex selection rather than any
    difference the model forces. Disjoint faces mean no choice of optimal basis
    silences it -- the exact analogue of our infeasibility certificate.
    """
    m = gp.Model(env=env())
    m.setParam("OutputFlag", 0)
    m.setParam("Presolve", 0)
    m.setParam("DualReductions", 0)
    n = len(text_prob.var_names)
    x = [m.addVar(lb=0.0, ub=GRB.INFINITY, name=f"x{i}") for i in range(n)]
    m.update()

    def add_rows(prob):
        for con in prob.constraints:
            lhs = gp.quicksum(c * x[i] for c, i in con.lhs)
            rhs = gp.quicksum(c * x[i] for c, i in con.rhs_terms) + con.rhs_const
            m.addConstr(lhs <= rhs) if con.op == "<=" else m.addConstr(lhs >= rhs)

    add_rows(text_prob)
    add_rows(mut)
    m.addConstr(gp.quicksum(c * x[i] for c, i in text_prob.obj_terms) == z_text)
    m.addConstr(gp.quicksum(c * x[i] for c, i in mut.obj_terms) == z_mut)
    m.setObjective(0.0)
    m.optimize()
    ok = m.Status == GRB.OPTIMAL
    m.dispose()
    return ok


# ==========================================================================

def main():
    problems, _ = load_clean_problems()

    evaluable, inert = set(), set()
    seen = set()
    with open("data/mutants_log.csv") as fh:
        for r in csv.DictReader(fh):
            if r["mutant_id"] in seen:
                continue
            seen.add(r["mutant_id"])
            if r["population"] == "primary" and r["bucket"] == "3 evaluable":
                evaluable.add(r["mutant_id"])
            elif (r["bucket"] == "2 inert"
                  and r["population"] in ("primary", "extension")):
                inert.add(r["mutant_id"])
    print(f"evaluable {len(evaluable)}  inert {len(inert)}")

    checks_only, ref_only, checks_inert = set(), set(), set()
    ref_only_ev = set()
    for r in csv.DictReader(open("data/solve_reference_2026-08-27.csv")):
        if r["arm"] == "evaluable" and r["checks"] == "True" and r["oracle"] == "False":
            checks_only.add(r["id"])
        elif r["arm"] == "evaluable" and r["checks"] == "False" and r["oracle"] == "True":
            ref_only_ev.add(r["id"])
        if r["arm"] == "inert":
            if r["oracle"] == "True":
                ref_only.add(r["id"])
            if r["checks"] == "True":
                checks_inert.add(r["id"])
    print(f"checks-only (the 86) {len(checks_only)}  "
          f"reference-only on inert (the 75) {len(ref_only)}\n")

    genuine_55 = {r["mutant_id"] for r in csv.DictReader(open("data/basis_test_2026-08-27.csv"))
                  if r["dual_optimal_pi_exists"] == "False"}
    print(f"the 55 certified genuine under the fixed-x* test: {len(genuine_55)}\n")

    rows, out = [], {}
    strat = {k: Counter() for k in ("moved", "still", "inert")}
    face, refcert, ourcert, refdet = [], [], [], []
    todo_left = set(evaluable) | set(inert)

    for pid, _m in in_scope().items():
        prob = problems[pid]
        text = text_terms_of(prob)
        ref = solve(prob)
        if not ref.optimal:
            continue
        for et in TYPE_ORDER:
            for scope in ("narrow", "wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    mid = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    if mid not in todo_left:
                        continue
                    todo_left.discard(mid)
                    arm0 = "evaluable" if mid in evaluable else "inert"
                    mut = copy.deepcopy(prob)
                    cand.apply(mut)
                    sol = solve(mut)
                    if not sol.optimal:
                        continue
                    dz, dx = differs(ref, sol, prob.var_names)
                    ct = text_terms_of(mut)
                    k = scales_for(text, ct)
                    a, b = check_a(text, sol)[0], check_b(text, sol)[0]
                    c = check_d(text, sol)[0]
                    an = check_a_norm(text, sol, k)
                    bn = check_b_norm(text, sol, k)
                    plain, norm = (a or b or c), (an or bn or c)

                    stratum = "inert" if arm0 == "inert" else ("moved" if dz else "still")
                    s = strat[stratum]
                    s["n"] += 1
                    s["checks"] += plain
                    s["checks_norm"] += norm
                    s["reference"] += (dz or dx)
                    rows.append({"arm": stratum, "id": mid, "checks": plain,
                                 "checks_norm": norm, "reference": dz or dx,
                                 "obj_moved": dz, "sol_moved": dx})

                    # ---- (2) face-robust certificate for the 86 -------------
                    if mid in checks_only:
                        cls, uniq = face_classes(mut, sol.obj_value)
                        if cls is None:
                            face.append({"mutant_id": mid, "primal_unique": None,
                                         "pi_exists_over_face": "error"})
                        else:
                            ex = pi_exists_over_face(mut, text, sol.obj_value, cls)
                            face.append({
                                "mutant_id": mid, "primal_unique": uniq,
                                "n_varies": sum(v == "varies" for v in cls.values()),
                                "pi_exists_over_face": ex,
                                "genuine_fixed_x": mid in genuine_55})

                    # ---- (5) the reference's OWN unique detections ---------
                    # The mirror of our 86. If the reference's edge on coverage
                    # is itself vertex selection, it is not an edge.
                    if mid in ref_only_ev:
                        inter5 = optimal_faces_intersect(prob, mut, ref.obj_value,
                                                         sol.obj_value)
                        refdet.append({"mutant_id": mid, "faces_intersect": inter5,
                                       "obj_moved": dz, "sol_moved": dx})

                    # ---- (3b) the SAME certificate, on our own 29 ----------
                    # The checks' answer-neutral reports must face the test we
                    # apply to the reference's, or the asymmetry is only moved.
                    if mid in checks_inert:
                        sil = []
                        for meth in METHODS:
                            got = solve_with(mut, meth)
                            if got is None:
                                continue
                            s3 = Sol(*got)
                            if not (check_a(text, s3)[0] or check_b(text, s3)[0]
                                    or check_d(text, s3)[0]):
                                sil.append(METHODS[meth])
                        cls3, uq3 = face_classes(mut, sol.obj_value)
                        ex3 = (pi_exists_over_face(mut, text, sol.obj_value, cls3)
                               if cls3 is not None else "error")
                        ourcert.append({"mutant_id": mid,
                                        "silenced_by": ";".join(sil) or "-",
                                        "basis_dependent": bool(sil),
                                        "pi_exists_over_face": ex3,
                                        "primal_unique": uq3})

                    # ---- (3) the reference's 75 ----------------------------
                    if mid in ref_only:
                        silenced = []
                        for meth in METHODS:
                            got = solve_with(mut, meth)
                            if got is None:
                                continue
                            s2 = Sol(*got)
                            d2, x2 = differs(ref, s2, prob.var_names)
                            if not (d2 or x2):
                                silenced.append(METHODS[meth])
                        inter = optimal_faces_intersect(prob, mut, ref.obj_value,
                                                        sol.obj_value)
                        refcert.append({"mutant_id": mid,
                                        "silenced_by": ";".join(silenced) or "-",
                                        "basis_dependent": bool(silenced),
                                        "faces_intersect": inter,
                                        "obj_moved": dz, "sol_moved": dx})

    # -------------------------------------------------------------- report
    print("=" * 78)
    print("(1) NORMALIZED CHECKS vs REFERENCE, EVERY STRATUM OF TABLE 1")
    print("=" * 78)
    print(f"{'stratum':<28} {'n':>7} {'checks':>10} {'normalized':>12} {'reference':>11}")
    for kk, label in (("moved", "optimum moved"),
                      ("still", "objective unchanged, eval"),
                      ("inert", "no change to the answer")):
        s = strat[kk]
        print(f"{label:<28} {s['n']:>7} {s['checks']:>10} {s['checks_norm']:>12} "
              f"{s['reference']:>11}")
        out[kk] = {"n": s["n"], "checks": s["checks"],
                   "checks_norm": s["checks_norm"], "reference": s["reference"]}

    print("\n" + "=" * 78)
    print("(2) DOES THE CERTIFICATE SURVIVE THE PRIMAL OPTIMAL FACE?")
    print("=" * 78)
    fu = sum(r.get("primal_unique") is True for r in face)
    fg = sum(r.get("pi_exists_over_face") is False for r in face)
    fe = sum(r.get("pi_exists_over_face") is True for r in face)
    g55 = [r for r in face if r.get("genuine_fixed_x")]
    g55_hold = sum(r.get("pi_exists_over_face") is False for r in g55)
    print(f"  tested                                     {len(face)}")
    print(f"  primal optimum unique                      {fu}")
    print(f"  no dual-optimal pi over the WHOLE face     {fg}   <- genuine")
    print(f"  some pi silences on some optimum           {fe}")
    print(f"  of the 55 certified with x* fixed, hold    {g55_hold} / {len(g55)}")
    out["face"] = {"n": len(face), "primal_unique": fu, "genuine": fg,
                   "artifact": fe, "held_of_55": g55_hold, "n_55": len(g55)}

    print("\n" + "=" * 78)
    print("(3) THE SAME QUESTION, ASKED OF THE REFERENCE'S 75")
    print("=" * 78)
    rb = sum(r["basis_dependent"] for r in refcert)
    ri = sum(r["faces_intersect"] is True for r in refcert)
    rg = sum(r["faces_intersect"] is False for r in refcert)
    print(f"  tested                                     {len(refcert)}")
    print(f"  silenced by some LP method                 {rb}")
    print(f"  optimal faces INTERSECT (vertex artifact)  {ri}")
    print(f"  optimal faces disjoint (genuine)           {rg}")
    print(f"\n(4) of the {len(refcert)}: objective moved on "
          f"{sum(r['obj_moved'] for r in refcert)}, "
          f"solution moved on {sum(r['sol_moved'] for r in refcert)}")
    ob = sum(r["basis_dependent"] for r in ourcert)
    og = sum(r["pi_exists_over_face"] is False for r in ourcert)
    oe = sum(r["pi_exists_over_face"] is True for r in ourcert)
    print("\n" + "=" * 78)
    print("(3b) THE SAME QUESTION, ASKED OF THE CHECKS' OWN 29")
    print("=" * 78)
    print(f"  tested                                     {len(ourcert)}")
    print(f"  silenced by some LP method                 {ob}")
    print(f"  some dual-optimal pi silences (artifact)   {oe}")
    print(f"  no such pi over the whole face (forced)    {og}")
    out["checks_29"] = {"n": len(ourcert), "basis_dependent": ob,
                        "artifact": oe, "forced": og}

    out["reference_75"] = {"n": len(refcert), "basis_dependent": rb,
                           "faces_intersect": ri, "genuine": rg,
                           "obj_moved": sum(r["obj_moved"] for r in refcert),
                           "sol_moved": sum(r["sol_moved"] for r in refcert)}

    di = sum(r["faces_intersect"] is True for r in refdet)
    dg = sum(r["faces_intersect"] is False for r in refdet)
    print("\n" + "=" * 78)
    print("(5) THE REFERENCE'S OWN UNIQUE DETECTIONS, UNDER THE SAME TEST")
    print("=" * 78)
    print(f"  tested                                     {len(refdet)}")
    print(f"  optimal faces INTERSECT (vertex artifact)  {di}")
    print(f"  optimal faces disjoint (genuine)           {dg}")
    print(f"  of these, objective moved on               "
          f"{sum(r['obj_moved'] for r in refdet)}")
    out["reference_unique_det"] = {"n": len(refdet), "faces_intersect": di,
                                   "genuine": dg,
                                   "obj_moved": sum(r["obj_moved"] for r in refdet)}

    out["stamp"] = TODAY
    with open(f"reframe_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(f"reframe_face_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(face[0].keys()))
        w.writeheader(); w.writerows(face)
    with open(f"reframe_refdet_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(refdet[0].keys()))
        w.writeheader(); w.writerows(refdet)
    with open(f"reframe_ourcert_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ourcert[0].keys()))
        w.writeheader(); w.writerows(ourcert)
    with open(f"reframe_refcert_{TODAY}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(refcert[0].keys()))
        w.writeheader(); w.writerows(refcert)
    json.dump(out, open(f"reframe_{TODAY}.json", "w"), indent=1)
    print(f"\nwrote reframe_{TODAY}.csv/.json, reframe_face_, reframe_refcert_")


if __name__ == "__main__":
    main()
