"""Part 1: Check C eligibility, corpus-wide. Part 2: raw dump for 5 instances.

shared rows = linear, sum, ratio, xby, xy   (declaration type, per spec)
bound  rows = upperbound, lowerbound
USABLE row  = |Pi| > 1e-6

Coefficients a_ij are read from the gurobi row in canonical form (every
variable term on the LHS), so signs reflect the row as solved.
"""

import collections

import gurobipy as gp
from gurobipy import GRB

from nl4opt_parser import ParseError, build_model, load_corpus, parse_instance, render_algebra

TOL = 1e-6
ZERO = 1e-12
DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]
SHARED_TYPES = {"linear", "sum", "ratio", "xby", "xy"}
BOUND_TYPES = {"upperbound", "lowerbound"}

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

k_hist = collections.Counter()
k1_positive = 0
k1_positive_shared = 0
k1_positive_bound = 0
eligible_instances = set()
neg_coef_pairs = 0
n_pairs = 0
n_solved = 0
catalog = []          # for part 2 selection

for split, pid, inst in load_corpus(DATA):
    try:
        prob = parse_instance(pid, inst, split)
    except ParseError:
        continue
    m = build_model(prob, env=env)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        m.dispose()
        continue
    n_solved += 1

    gvars = m.getVars()
    rows = m.getConstrs()
    ctype = {c.name: c.ctype for c in prob.constraints}

    usable_rows = [r for r in rows if abs(r.Pi) > TOL]
    n_usable_shared = sum(1 for r in usable_rows if ctype[r.ConstrName] in SHARED_TYPES)
    n_degenerate = sum(1 for r in rows if abs(r.Slack) < TOL and abs(r.Pi) < TOL)

    for j, gv in enumerate(gvars):
        n_pairs += 1
        contributing = []
        has_negative = False
        for r in usable_rows:
            a = m.getCoeff(r, gv)
            if abs(a) > ZERO:
                contributing.append((r, a))
                if a < 0:
                    has_negative = True
        k = len(contributing)
        k_hist[k] += 1
        if has_negative:
            neg_coef_pairs += 1
        if k == 1:
            row, a = contributing[0]
            if a > 0:
                k1_positive += 1
                eligible_instances.add(pid)
                if ctype[row.ConstrName] in SHARED_TYPES:
                    k1_positive_shared += 1
                else:
                    k1_positive_bound += 1

    catalog.append({"pid": pid, "split": split, "sense": prob.sense,
                    "n_usable_shared": n_usable_shared, "n_degenerate": n_degenerate})
    m.dispose()

print("=" * 72)
print("PART 1: CHECK C ELIGIBILITY, CORPUS-WIDE")
print("=" * 72)
print(f"solved instances                                 {n_solved}")
print(f"(instance, product) pairs                        {n_pairs}")
print()
print("distribution of k_j:")
for k in (0, 1, 2):
    print(f"  k_j = {k}                                      {k_hist[k]}")
print(f"  k_j = 3+                                     {sum(v for k, v in k_hist.items() if k >= 3)}")
print()
print(f"k_j = 1 pairs                                    {k_hist[1]}")
print(f"  with POSITIVE coefficient in that row          {k1_positive}")
print(f"    on a SHARED row                              {k1_positive_shared}")
print(f"    on a BOUND row                               {k1_positive_bound}")
print(f"  with negative coefficient in that row          {k_hist[1] - k1_positive}")
print()
print(f"distinct instances with >=1 eligible product     {len(eligible_instances)}")
print(f"  as fraction of solved                          "
      f"{len(eligible_instances) / n_solved * 100:.1f}%")
print()
print(f"pairs with a NEGATIVE coefficient in any row")
print(f"  carrying a nonzero dual                        {neg_coef_pairs}")

# --------------------------------------------------------------------------
# PART 2
# --------------------------------------------------------------------------
clean = [c for c in catalog if c["n_degenerate"] == 0]
by = lambda n, mn: sorted([c for c in clean if c["n_usable_shared"] == n
                           and (c["sense"] == GRB.MINIMIZE) == mn],
                          key=lambda c: c["pid"])

pick = []
for n in (1, 2):
    mins, maxs = by(n, True), by(n, False)
    pick.append(mins[0]["pid"])          # guarantee minimisations appear
    pick.append(maxs[0]["pid"])
extra = [c for c in sorted(clean, key=lambda c: c["pid"])
         if c["sense"] == GRB.MINIMIZE and c["pid"] not in pick]
pick.append(extra[0]["pid"])

corpus = {pid: (s, i) for s, pid, i in load_corpus(DATA)}

print("\n" + "=" * 72)
print("PART 2: RAW DUMP")
print("=" * 72)
for pid in pick:
    split, inst = corpus[pid]
    prob = parse_instance(pid, inst, split)
    m = build_model(prob, env=env)
    m.optimize()
    gvars = m.getVars()
    ctype = {c.name: c.ctype for c in prob.constraints}

    print("\n" + "-" * 72)
    print(f"instance   {pid}")
    print(f"direction  {'min' if prob.sense == GRB.MINIMIZE else 'max'}")
    print(f"objective  {m.ObjVal:.10g}")
    print("-" * 72)
    print("\ntext:")
    print(f"  {prob.document}")
    print("\nalgebra:")
    for line in render_algebra(prob).splitlines():
        print(f"  {line}")

    print("\nvariables:")
    print(f"  {'name':<24}{'value':>14}{'LB':>8}{'UB':>8}{'RC':>14}")
    for gv, nm in zip(gvars, prob.var_names):
        print(f"  {nm:<24}{gv.X:>14.10g}{gv.LB:>8g}{gv.UB:>8g}{gv.RC:>14.10g}")

    print("\nrows:")
    for r in m.getConstrs():
        terms = []
        for gv, nm in zip(gvars, prob.var_names):
            a = m.getCoeff(r, gv)
            if abs(a) > ZERO:
                terms.append(f"{a:g}*{nm}")
        print(f"  {r.ConstrName:<16} {ctype[r.ConstrName]:<11} "
              f"{' + '.join(terms)}  {r.Sense}  {r.RHS:g}")
        print(f"  {'':<16} {'':<11} Pi={r.Pi:.10g}  Slack={r.Slack:.10g}")
    m.dispose()
