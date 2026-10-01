"""Does the primal-invisible class thin out at scale? 2026-09-03.

THE QUESTION. A reviewer objected that the paper covers "a narrow error
class in small continuous LPs". NL4OPT cannot answer that: every instance has
2-4 variables, 1-6 rows and at most two usable resource rows, and n_rows barely
varies independently of n_vars. Measured within that range the certified rate is
flat to declining (0.84% at 2 variables, 0.44% at 3, 0.00% at 4), but the cells
are small and the two effects are confounded.

The paper's Limitations makes a specific mechanistic claim, citing Bertsimas and
Tsitsiklis: degeneracy "only becomes more likely as rows are added WITHOUT
variables". That is the prediction this script tests, by doing the one thing
NL4OPT cannot: hold the variable count fixed and sweep the row count.

WHY SYNTHETIC INSTANCES ARE THE RIGHT INSTRUMENT. Whether a value swap leaves
(z*, x*) fixed while displacing the dual optimal set is a question about LP
geometry, not about natural language. It needs no problem text, no annotation
and no generator. Those are needed only to ask how often a MODEL WRITER commits
the error, which is a different question and is bottlenecked by annotated
corpora. Synthesis lets the geometry question be asked at sizes no annotated
corpus reaches.

WHAT IS MEASURED. For each (n_vars, n_rows) cell, resource-allocation LPs are
drawn, every value swap of the four types that produce the certified class is
applied, and each mutant is scored exactly as the paper scores its own:

    certified primal-invisible  <=>  the mutant solves optimal
                                AND  z* equals the clean model's
                                AND  x* equals the clean model's
                                AND  a check reports
                                AND  no dual-optimal pi over the whole optimal
                                     face satisfies the text's identities

The last line is reframe_measure.pi_exists_over_face, the paper's certificate,
used unmodified.

    venv/bin/python src/scale_sweep.py --quick
    venv/bin/python src/scale_sweep.py --vars 6 --rows 3,6,9,12,18,24 --instances 60

Writes data/scale_sweep_<date>.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks_common import REL_TOL, ZERO_TOL
from mutations import check_a, check_b, check_d, solve
from nl4opt_parser import ParsedConstraint, ParsedProblem
from reframe_measure import face_classes, pi_exists_over_face
from text_terms import text_terms_of

TODAY = datetime.date.today().isoformat()


# --------------------------------------------------------------------------
# instance synthesis
# --------------------------------------------------------------------------

def make_instance(n_vars, n_rows, rng, pid):
    """A resource-allocation LP: max c'x subject to Ax <= b, x >= 0.

    Shaped to look like the corpus rather than like a random matrix. Every
    coefficient is a small positive integer, as NL4OPT's are, because the
    phenomenon is about which values sit in which slots and integer values make
    a swap between two slots meaningful. Limits are drawn so that some rows bind
    and some do not: a swap on a row that never binds is exactly the case that
    can leave the primal optimum untouched, and an instance where every row
    binds would suppress the class by construction.
    """
    from gurobipy import GRB
    names = [f"x{j}" for j in range(n_vars)]
    obj = [(float(rng.randint(2, 20)), j) for j in range(n_vars)]

    cons = []
    for i in range(n_rows):
        # each row touches a random subset of the variables, at least two
        k = rng.randint(2, max(2, min(n_vars, 5)))
        support = rng.sample(range(n_vars), k)
        lhs = [(float(rng.randint(1, 9)), j) for j in sorted(support)]
        # a limit scaled so roughly half the rows are slack at the optimum
        base = sum(a for a, _ in lhs) * rng.uniform(1.0, 6.0)
        cons.append(ParsedConstraint(
            name=f"c{i}_linear", ctype="linear", lhs=lhs, op="<=",
            rhs_terms=[], rhs_const=float(round(base))))
    return ParsedProblem(pid=pid, split="synthetic", document="",
                         var_names=names, sense=GRB.MAXIMIZE,
                         obj_terms=obj, constraints=cons)


def make_instance_blend(n_vars, n_rows, rng, pid):
    """A blending/diet LP: min c'x subject to requirement, capacity and ratio
    rows, x >= 0. THE SECOND FAMILY, and it differs from make_instance on every
    axis that could be driving the result:

      sense       minimize, not maximize
      operators   a mix of >= and <=, not all <=
      signs       ratio rows carry a negative coefficient
      structure   rows are typed (requirement / capacity / ratio) rather than
                  a random support over a single row kind

    Ratio rows are included because NL4OPT actually has them (c*_ratio, of the
    form 0.7a - 0.3b >= 0) and because the one answer-neutral case the real
    generator produced was on one.

    FEASIBILITY IS BY CONSTRUCTION. A random interior point x0 is drawn first
    and every limit is set relative to a_i . x0, so x0 is feasible in every
    instance. Requirement rows are drawn tight-ish and capacity rows loose, so
    each instance carries a mix of binding and slack rows. Without this, a
    random min-cost LP with >= rows is infeasible most of the time and the
    surviving sample is biased toward whatever happens to be satisfiable.
    """
    from gurobipy import GRB
    names = [f"x{j}" for j in range(n_vars)]
    obj = [(float(rng.randint(2, 20)), j) for j in range(n_vars)]
    x0 = [rng.uniform(1.0, 10.0) for _ in range(n_vars)]

    cons = []
    for i in range(n_rows):
        u = rng.random()
        if u < 0.20 and n_vars >= 2:
            # ratio row: p*x_j - q*x_k >= 0, NL4OPT's c*_ratio shape
            j, k = rng.sample(range(n_vars), 2)
            p = float(rng.randint(1, 9))
            q = float(rng.randint(1, 9))
            lhs = sorted([(p, j), (-q, k)], key=lambda t: t[1])
            act = p * x0[j] - q * x0[k]
            cons.append(ParsedConstraint(
                name=f"c{i}_ratio", ctype="linear", lhs=lhs, op=">=",
                rhs_terms=[], rhs_const=float(round(min(0.0, act)))))
            continue
        kk = rng.randint(2, max(2, min(n_vars, 5)))
        support = sorted(rng.sample(range(n_vars), kk))
        lhs = [(float(rng.randint(1, 9)), j) for j in support]
        act = sum(a * x0[j] for a, j in lhs)
        if u < 0.70:
            # requirement row, drawn close to active so many of them bind
            b = act * rng.uniform(0.55, 1.0)
            cons.append(ParsedConstraint(
                name=f"c{i}_req", ctype="linear", lhs=lhs, op=">=",
                rhs_terms=[], rhs_const=float(round(b))))
        else:
            # capacity row, drawn loose so many of them are slack
            b = act * rng.uniform(1.0, 2.5)
            cons.append(ParsedConstraint(
                name=f"c{i}_cap", ctype="linear", lhs=lhs, op="<=",
                rhs_terms=[], rhs_const=float(round(b))))
    return ParsedProblem(pid=pid, split="synthetic", document="",
                         var_names=names, sense=GRB.MINIMIZE,
                         obj_terms=obj, constraints=cons)


FAMILIES = {"resource": make_instance, "blend": make_instance_blend}


# --------------------------------------------------------------------------
# the four swap types that produce the certified class
# --------------------------------------------------------------------------

def swap_candidates(prob, rng, cap):
    """-> list of (type, description, mutate_fn).

    Mirrors characterize_55's taxonomy: 4a coefficient swap within a row, 4b
    coefficient swap across rows, 5 objective coefficient swap, 7 limit swap
    between rows. Those four produce all 55 certified errors and nothing else
    does, so nothing else is enumerated here.
    """
    out = []

    # 4a: two coefficients inside one row exchange values
    for ci, c in enumerate(prob.constraints):
        for p, q in itertools.combinations(range(len(c.lhs)), 2):
            if abs(c.lhs[p][0] - c.lhs[q][0]) <= ZERO_TOL:
                continue
            out.append(("4a", f"row{ci}:{p}<->{q}", ("row_in", ci, p, q)))

    # 4b: one variable's coefficient in two different rows exchange values
    for (ci, cj) in itertools.combinations(range(len(prob.constraints)), 2):
        a, b = prob.constraints[ci], prob.constraints[cj]
        ai = {j: k for k, (_, j) in enumerate(a.lhs)}
        bi = {j: k for k, (_, j) in enumerate(b.lhs)}
        for j in set(ai) & set(bi):
            if abs(a.lhs[ai[j]][0] - b.lhs[bi[j]][0]) <= ZERO_TOL:
                continue
            out.append(("4b", f"r{ci}/r{cj}:v{j}", ("row_x", ci, ai[j], cj, bi[j])))

    # 5: two objective coefficients exchange values
    for p, q in itertools.combinations(range(len(prob.obj_terms)), 2):
        if abs(prob.obj_terms[p][0] - prob.obj_terms[q][0]) <= ZERO_TOL:
            continue
        out.append(("5", f"obj:{p}<->{q}", ("obj", p, q)))

    # 7: two row limits exchange values
    for ci, cj in itertools.combinations(range(len(prob.constraints)), 2):
        if abs(prob.constraints[ci].rhs_const
               - prob.constraints[cj].rhs_const) <= ZERO_TOL:
            continue
        out.append(("7", f"rhs:{ci}<->{cj}", ("rhs", ci, cj)))

    if cap and len(out) > cap:
        out = rng.sample(out, cap)
    return out


def apply_swap(prob, op):
    """Apply one swap in place. `prob` must already be a deep copy."""
    kind = op[0]
    if kind == "row_in":
        _, ci, p, q = op
        c = prob.constraints[ci]
        (ap, jp), (aq, jq) = c.lhs[p], c.lhs[q]
        c.lhs[p], c.lhs[q] = (aq, jp), (ap, jq)
    elif kind == "row_x":
        _, ci, pi_, cj, pj = op
        a, b = prob.constraints[ci], prob.constraints[cj]
        (aa, ja), (bb, jb) = a.lhs[pi_], b.lhs[pj]
        a.lhs[pi_], b.lhs[pj] = (bb, ja), (aa, jb)
    elif kind == "obj":
        _, p, q = op
        (ap, jp), (aq, jq) = prob.obj_terms[p], prob.obj_terms[q]
        prob.obj_terms[p], prob.obj_terms[q] = (aq, jp), (ap, jq)
    elif kind == "rhs":
        _, ci, cj = op
        a, b = prob.constraints[ci], prob.constraints[cj]
        a.rhs_const, b.rhs_const = b.rhs_const, a.rhs_const
    else:
        raise ValueError(kind)


# --------------------------------------------------------------------------

def binding_rows(prob, sol, tol=1e-7):
    """-> {row name: True if the row is tight at sol's optimum}."""
    t = text_terms_of(prob)
    out = {}
    for rn in t.row_names:
        lhs = sum(t.row_coefs[rn].get(v, 0.0) * sol.values[v]
                  for v in t.var_names)
        b = t.row_limits[rn]
        out[rn] = abs(lhs - b) <= tol * max(1.0, abs(b))
    return out


def touched_rows(prob, op):
    """Which row names a swap damages. An objective swap touches none, which is
    why the paper's log records '-' for its binding columns and why an objective
    swap can never be inert."""
    kind = op[0]
    if kind == "row_in":
        return [prob.constraints[op[1]].name]
    if kind == "row_x":
        return [prob.constraints[op[1]].name, prob.constraints[op[3]].name]
    if kind == "rhs":
        return [prob.constraints[op[1]].name, prob.constraints[op[2]].name]
    return []


def is_inert(prob, mut, ref, sol, op):
    """The paper's bucket 2: every touched row is non-binding in BOTH solves, so
    the text changes and the optimum does not. Excluded from every published
    rate, so excluded here. Over the whole study this is 751 of the 6,934
    swap-type mutants that solve, which is what separates the 0.793% figure from
    the correct 0.890%."""
    rows = touched_rows(prob, op)
    if not rows:
        return False                      # objective swap: never inert
    cb, mb = binding_rows(prob, ref), binding_rows(mut, sol)
    return not any(cb.get(r, False) or mb.get(r, False) for r in rows)


def same_point(u, v):
    if set(u) != set(v):
        return False
    return all(abs(u[k] - v[k]) <= REL_TOL * max(1.0, abs(u[k]), abs(v[k]))
               for k in u)


def score_cell(n_vars, n_rows, n_inst, seed, cap, verbose=False, family="resource"):
    import copy
    rng = random.Random(seed)
    mk = FAMILIES[family]
    stat = Counter()
    by_type = defaultdict(Counter)

    for t in range(n_inst):
        pid = f"syn_{n_vars}x{n_rows}_{t}"
        prob = mk(n_vars, n_rows, rng, pid)
        ref = solve(prob)
        if not ref.optimal:
            stat["instance_not_optimal"] += 1
            continue
        stat["instances"] += 1
        text = text_terms_of(prob)

        for etype, desc, op in swap_candidates(prob, rng, cap):
            mut = copy.deepcopy(prob)
            apply_swap(mut, op)
            sol = solve(mut)
            stat["generated"] += 1
            if not sol.optimal:
                stat["not_optimal"] += 1
                continue
            if is_inert(prob, mut, ref, sol, op):
                stat["inert"] += 1
                continue
            stat["swaps"] += 1                       # bucket 3, evaluable
            by_type[etype]["swaps"] += 1
            z_same = (abs(sol.obj_value - ref.obj_value)
                      <= REL_TOL * max(1.0, abs(ref.obj_value)))
            if not z_same:
                stat["objective_moved"] += 1
                continue
            stat["objective_unchanged"] += 1
            by_type[etype]["objective_unchanged"] += 1
            if not same_point(ref.values, sol.values):
                stat["solution_moved"] += 1
                continue
            stat["answer_neutral"] += 1
            by_type[etype]["answer_neutral"] += 1

            fired = (check_a(text, sol)[0] or check_b(text, sol)[0]
                     or check_d(text, sol)[0])
            if not fired:
                stat["no_report"] += 1
                continue
            stat["reported"] += 1
            by_type[etype]["reported"] += 1

            cls, _u = face_classes(mut, sol.obj_value)
            if cls is None:
                stat["face_failed"] += 1
                continue
            try:
                if not pi_exists_over_face(mut, text, sol.obj_value, cls):
                    stat["certified"] += 1
                    by_type[etype]["certified"] += 1
                else:
                    stat["basis_artifact"] += 1
            except Exception:                                    # noqa: BLE001
                stat["cert_error"] += 1
        if verbose and (t + 1) % 10 == 0:
            print(f"      {t+1}/{n_inst} instances, {stat['swaps']} swaps, "
                  f"{stat['certified']} certified", flush=True)
    return stat, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", type=int, default=6,
                    help="variable count, HELD FIXED across the sweep")
    ap.add_argument("--rows", default="3,6,9,12,18,24",
                    help="comma-separated row counts to sweep")
    ap.add_argument("--instances", type=int, default=40)
    ap.add_argument("--cap", type=int, default=400,
                    help="max swaps enumerated per instance (0 = all)")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--family", default="resource", choices=sorted(FAMILIES),
                    help="instance family; 'blend' is the structural control")
    ap.add_argument("--quick", action="store_true",
                    help="small run to check the pipeline end to end")
    args = ap.parse_args()
    if args.quick:
        args.rows, args.instances, args.cap = "3,6,12", 8, 120

    rows = [int(x) for x in args.rows.split(",")]
    print(f"sweep: family={args.family}, n_vars fixed at {args.vars}, "
          f"n_rows in {rows}, {args.instances} instances per cell, "
          f"cap {args.cap}\n")

    out, cells = [], {}
    for nr in rows:
        print(f"  n_rows={nr} ...", flush=True)
        stat, by_type = score_cell(args.vars, nr, args.instances,
                                   args.seed + nr, args.cap, verbose=True,
                                   family=args.family)
        sw = stat["swaps"]
        an = stat["answer_neutral"]
        ce = stat["certified"]
        row = {
            "n_vars": args.vars, "n_rows": nr,
            "instances": stat["instances"],
            "generated": stat["generated"], "inert": stat["inert"],
            "swaps": sw,
            "objective_unchanged": stat["objective_unchanged"],
            "answer_neutral": an, "reported": stat["reported"],
            "certified": ce, "basis_artifact": stat["basis_artifact"],
            "rate_answer_neutral": (an / sw if sw else 0.0),
            "rate_certified": (ce / sw if sw else 0.0),
        }
        out.append(row)
        cells[nr] = {"stat": dict(stat),
                     "by_type": {k: dict(v) for k, v in by_type.items()}}
        print(f"    swaps {sw}, answer-neutral {an}, reported {stat['reported']}, "
              f"CERTIFIED {ce}  ({100*row['rate_certified']:.3f}%)\n", flush=True)

    print("=" * 74)
    print(f"CERTIFIED PRIMAL-INVISIBLE RATE, n_vars FIXED AT {args.vars}")
    print("=" * 74)
    print("  n_rows  rows/var   swaps   answer-neutral   certified    rate")
    for r in out:
        print("  %5d   %7.2f  %7d   %12d   %9d   %6.3f%%"
              % (r["n_rows"], r["n_rows"] / r["n_vars"], r["swaps"],
                 r["answer_neutral"], r["certified"],
                 100 * r["rate_certified"]))
    print()
    print("  NL4OPT for comparison: 6,183 EVALUABLE swap mutants, 55 certified, 0.890%")
    print("  (2-4 variables, 1-6 rows, at most two usable resource rows)")

    os.makedirs("data", exist_ok=True)
    fam = "" if args.family == "resource" else f"_{args.family}"
    stem = f"data/scale_sweep_v{args.vars}{fam}_{TODAY}"
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    json.dump({"stamp": TODAY, "family": args.family,
               "n_vars": args.vars, "rows": rows,
               "instances_per_cell": args.instances, "cap": args.cap,
               "seed": args.seed, "cells": cells, "summary": out},
              open(stem + ".json", "w"), indent=1)
    print(f"\nwrote {stem}.csv/.json")


if __name__ == "__main__":
    main()
