"""Re-solve pass: persist raw per-constraint and per-objective facts.

Writes results_resolve.csv (NEW file -- results.csv is never touched),
constraints.csv and objectives.csv, then:
  (1) diffs results_resolve.csv against results.csv column by column;
  (2) recomputes the degeneracy count from constraints.csv and compares it to
      results.csv.n_degenerate.

Raw solver and parser output only. The sole derived columns are `n_vars_in_row`
and `binding`; no Check A/B/D, no usability verdicts.

Tolerance: 1e-6 throughout, matching every earlier phase.
  binding    := |Slack| < 1e-6
  degenerate := binding AND |Pi| < 1e-6
"""

import collections
import csv
import json
import sys

import gurobipy as gp
from gurobipy import GRB

from nl4opt_measure import classify_constraints, roll_up, row_scope, suspicious_values, SHARED
from nl4opt_parser import (ParseError, build_model, load_corpus, parse_instance,
                           ParsedConstraint)
from nl4opt_measure import var_indices

TOL = 1e-6
ZERO = 1e-12
DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]

STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED", GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.NUMERIC: "NUMERIC",
}

RESULT_COLS = ["problem_id", "split", "solve_status", "n_constraints", "n_shared",
               "n_usable", "n_usable_shared", "n_usable_bound", "n_degenerate",
               "n_inactive", "n_unclassified", "qualifies", "suspicious"]
CONSTRAINT_COLS = ["problem_id", "constraint_name", "ctype", "n_vars_in_row",
                   "operator", "rhs_limit", "pi", "slack", "binding", "coefficients"]
OBJECTIVE_COLS = ["problem_id", "sense", "obj_value", "coefficients"]

BLANK = {c: "" for c in RESULT_COLS[3:]}

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

results, constraints, objectives = [], [], []
coef_mismatch = []          # integrity: parsed var count vs nonzero canonical count

corpus = load_corpus(DATA)
total = len(corpus)
print(f"loaded {total} instances")

for n, (split, pid, inst) in enumerate(corpus, 1):
    if n % 25 == 0 or n == total:
        print(f"  [{n}/{total}] processed")

    try:
        prob = parse_instance(pid, inst, split)
    except Exception:                                   # noqa: BLE001
        results.append({"problem_id": pid, "split": split,
                        "solve_status": "PARSE_FAILED", **BLANK})
        continue

    n_shared = sum(1 for c in prob.constraints if row_scope(c) == SHARED)
    offenders = suspicious_values(prob)

    m = build_model(prob, env=env)     # Presolve=0, DualReductions=0 set inside
    m.optimize()
    status = STATUS_NAMES.get(m.Status, f"STATUS_{m.Status}")

    if status != "OPTIMAL":
        results.append({"problem_id": pid, "split": split, "solve_status": status,
                        "n_constraints": len(prob.constraints), "n_shared": n_shared,
                        "n_usable": "", "n_usable_shared": "", "n_usable_bound": "",
                        "n_degenerate": "", "n_inactive": "", "n_unclassified": "",
                        "qualifies": "", "suspicious": bool(offenders)})
        m.dispose()
        continue

    gvars = m.getVars()
    parsed_by_name = {c.name: c for c in prob.constraints}

    # ---- raw per-row facts ----
    for r in m.getConstrs():
        con: ParsedConstraint = parsed_by_name[r.ConstrName]
        coefs = {}
        for gv, nm in zip(gvars, prob.var_names):
            a = m.getCoeff(r, gv)
            if abs(a) > ZERO:
                coefs[nm] = a
        # n_vars_in_row uses the SAME function that produced results.csv.n_shared
        n_in_row = len(var_indices(con))
        if n_in_row != len(coefs):
            coef_mismatch.append((pid, r.ConstrName, n_in_row, len(coefs)))
        constraints.append({
            "problem_id": pid, "constraint_name": r.ConstrName, "ctype": con.ctype,
            "n_vars_in_row": n_in_row,
            "operator": {"<": "<=", ">": ">=", "=": "=="}.get(r.Sense, r.Sense),
            "rhs_limit": repr(r.RHS), "pi": repr(r.Pi), "slack": repr(r.Slack),
            "binding": abs(r.Slack) < TOL,
            "coefficients": json.dumps(coefs),
        })

    obj_coefs = {nm: gv.Obj for gv, nm in zip(gvars, prob.var_names)}
    objectives.append({
        "problem_id": pid,
        "sense": "min" if prob.sense == GRB.MINIMIZE else "max",
        "obj_value": repr(m.ObjVal),
        "coefficients": json.dumps(obj_coefs),
    })

    cons = classify_constraints(prob, m)
    r_ = roll_up(prob, status, cons)
    results.append({"problem_id": pid, "split": split, "solve_status": status,
                    "n_constraints": r_.n_constraints, "n_shared": n_shared,
                    "n_usable": r_.n_usable, "n_usable_shared": r_.n_usable_shared,
                    "n_usable_bound": r_.n_usable_bound,
                    "n_degenerate": r_.n_degenerate, "n_inactive": r_.n_inactive,
                    "n_unclassified": r_.n_unclassified,
                    "qualifies": r_.qualifies, "suspicious": bool(offenders)})
    m.dispose()


def write(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


write("data/results_resolve.csv", results, RESULT_COLS)
write("data/constraints.csv", constraints, CONSTRAINT_COLS)
write("data/objectives.csv", objectives, OBJECTIVE_COLS)
print(f"\nwrote results_resolve.csv ({len(results)} rows), "
      f"constraints.csv ({len(constraints)} rows), "
      f"objectives.csv ({len(objectives)} rows)")

# ==========================================================================
# (1) verify the reproduction claim
# ==========================================================================
print("\n" + "=" * 72)
print("(1) DIFF: results_resolve.csv vs results.csv, column by column")
print("=" * 72)
old = {r["problem_id"]: r for r in csv.DictReader(open("data/results.csv"))}
new = {r["problem_id"]: r for r in csv.DictReader(open("data/results_resolve.csv"))}

print(f"  rows: results.csv={len(old)}  results_resolve.csv={len(new)}")
only_old, only_new = set(old) - set(new), set(new) - set(old)
print(f"  ids only in results.csv: {len(only_old)}   only in resolve: {len(only_new)}")

diffs = []
per_col = collections.Counter()
for pid in old.keys() & new.keys():
    for col in RESULT_COLS:
        if old[pid][col] != new[pid][col]:
            diffs.append((pid, col, old[pid][col], new[pid][col]))
            per_col[col] += 1

print(f"  cells compared: {len(old.keys() & new.keys()) * len(RESULT_COLS)}")
print(f"  differing cells: {len(diffs)}")
if diffs:
    print("\n  MISMATCH -- stopping. Differing rows:")
    for pid, col, o, nv in diffs[:50]:
        print(f"    {pid:>13}  {col:<18} results.csv={o!r}  resolve={nv!r}")
    print(f"\n  per-column counts: {dict(per_col)}")
    sys.exit(1)
print("  -> identical on every column of every row")

# ==========================================================================
# (2) degeneracy cross-check, recomputed from constraints.csv
# ==========================================================================
print("\n" + "=" * 72)
print("(2) DEGENERACY CROSS-CHECK from constraints.csv")
print("=" * 72)
print("  tolerance: binding := |Slack| < 1e-6 ; degenerate := binding AND |Pi| < 1e-6")

recomputed = collections.Counter()
n_rows_seen = 0
missing_binding = []
for row in csv.DictReader(open("data/constraints.csv")):
    n_rows_seen += 1
    if row["binding"] not in ("True", "False"):
        missing_binding.append((row["problem_id"], row["constraint_name"], row["binding"]))
        continue
    if row["binding"] == "True" and abs(float(row["pi"])) < TOL:
        recomputed[row["problem_id"]] += 1

recorded = {p: int(r["n_degenerate"]) for p, r in old.items()
            if r["solve_status"] == "OPTIMAL"}

mismatch = [(p, recorded[p], recomputed.get(p, 0))
            for p in recorded if recorded[p] != recomputed.get(p, 0)]

print(f"  constraint rows read           {n_rows_seen}")
print(f"  recomputed degenerate rows     {sum(recomputed.values())}")
print(f"  recomputed instances w/ >0     {len(recomputed)}")
print(f"  results.csv degenerate rows    {sum(recorded.values())}")
print(f"  results.csv instances w/ >0    {sum(1 for v in recorded.values() if v > 0)}")
print(f"  per-instance mismatches        {len(mismatch)}")
if mismatch:
    print("\n  MISMATCH -- stopping. Differing instances:")
    for p, rec, got in mismatch[:50]:
        print(f"    {p:>13}  results.csv={rec}  recomputed={got}")
    sys.exit(1)
expected = (sum(recomputed.values()) == 91 and len(recomputed) == 81)
print(f"  matches results.csv exactly    True")
print(f"  equals 91 rows across 81 instances  {expected}")

# ==========================================================================
# integrity + binding coverage
# ==========================================================================
print("\n" + "=" * 72)
print("INTEGRITY")
print("=" * 72)
n_opt = sum(1 for r in results if r["solve_status"] == "OPTIMAL")
print(f"  OPTIMAL instances                        {n_opt}")
print(f"  constraint rows written                  {len(constraints)}")
print(f"  rows with binding present (True/False)   "
      f"{sum(1 for c in constraints if isinstance(c['binding'], bool))}")
print(f"  rows with binding missing/blank          {len(missing_binding)}")
print(f"  n_vars_in_row != nonzero canonical coefs {len(coef_mismatch)}")
for row in coef_mismatch[:10]:
    print(f"    {row}")
