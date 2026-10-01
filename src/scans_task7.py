"""TASK 7 -- two read-only scans. Reports to stdout. Writes nothing.

7a  zero-RHS scan over constraints.csv
7b  variable-bound assertion: no value may hide in a Gurobi variable bound
"""

import collections
import csv

import gurobipy as gp
from gurobipy import GRB

from nl4opt_parser import ParseError, build_model, load_corpus, parse_instance

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]
FIXTURE_IDS = ["-451730029", "-503693411", "-2061445103", "294897151", "1727707716"]

# ==========================================================================
# 7a. ZERO-RHS SCAN
# ==========================================================================
print("=" * 74)
print("7a. ZERO-RHS SCAN  (constraints.csv, rhs_limit == 0)")
print("=" * 74)
by_ctype = collections.Counter()
tot_by_ctype = collections.Counter()
instances = set()
n_zero = n_rows = 0
for r in csv.DictReader(open("data/constraints.csv")):
    n_rows += 1
    tot_by_ctype[r["ctype"]] += 1
    if float(r["rhs_limit"]) == 0.0:
        n_zero += 1
        by_ctype[r["ctype"]] += 1
        instances.add(r["problem_id"])

print(f"  rows scanned                  {n_rows}")
print(f"  rows with rhs_limit == 0      {n_zero}")
print(f"  distinct instances affected   {len(instances)}")
print()
print(f"  {'ctype':<12}{'rows':>8}{'rhs==0':>9}{'pct':>8}")
for t in sorted(tot_by_ctype, key=lambda t: -by_ctype[t]):
    pct = by_ctype[t] / tot_by_ctype[t] * 100 if tot_by_ctype[t] else 0
    print(f"  {t:<12}{tot_by_ctype[t]:>8}{by_ctype[t]:>9}{pct:>7.1f}%")
print("\n  (report only -- no exclusion list modified)")

# ==========================================================================
# 7b. VARIABLE BOUND ASSERTION
# ==========================================================================
print("\n" + "=" * 74)
print("7b. VARIABLE BOUND ASSERTION  (LB must be 0, UB must be inf)")
print("=" * 74)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()


def check(pids=None):
    violations = []
    n_vars = n_inst = 0
    for split, pid, inst in load_corpus(DATA):
        if pids is not None and pid not in pids:
            continue
        try:
            prob = parse_instance(pid, inst, split)
        except ParseError:
            continue
        m = build_model(prob, env=env)      # bounds are set at addVar time
        n_inst += 1
        for gv, nm in zip(m.getVars(), prob.var_names):
            n_vars += 1
            # Gurobi reports an unbounded UB as float('inf') while GRB.INFINITY
            # is 1e100, so `!=` false-positives. Finite means UB < GRB.INFINITY.
            if gv.LB != 0.0 or gv.UB < GRB.INFINITY:
                violations.append((pid, nm, gv.LB, gv.UB))
        m.dispose()
    return n_inst, n_vars, violations


n_i, n_v, viol = check(set(FIXTURE_IDS))
print(f"  5 FIXTURES: instances={n_i}  variables={n_v}  violations={len(viol)}")
if viol:
    for pid, nm, lb, ub in viol:
        print(f"    VIOLATION  {pid}  {nm}  LB={lb!r}  UB={ub!r}")
else:
    print("    PASS -- all LB == 0 and all UB == inf")

n_i, n_v, viol = check(None)
print(f"\n  WHOLE CORPUS: instances={n_i}  variables={n_v}  violations={len(viol)}")
if viol:
    print(f"    violating variables: {len(viol)}")
    print(f"    violating instances: {len({v[0] for v in viol})}")
    for pid, nm, lb, ub in viol[:25]:
        print(f"    VIOLATION  {pid}  {nm}  LB={lb!r}  UB={ub!r}")
else:
    print("    PASS -- all LB == 0 and all UB == inf; "
          "no value can hide in a variable bound")
