"""A: corpus-wide zero-coefficient scan.  B: add n_vars_nonzero to constraints.csv.

The `coefficients` column in constraints.csv holds only NONZERO entries (they
were filtered at |a| > 1e-12 on write), so zeros cannot be counted from it. The
scan therefore recomputes each row's canonical coefficient per DECLARED variable
straight from the parsed declarations:

    a_ij = sum(lhs coefs for j) - sum(rhs_terms coefs for j)   for j in var_indices(con)

Scope is the 1040 OPTIMAL instances / 2935 rows that constraints.csv covers.
Counts only -- no mutation work.
"""

import collections
import csv
import json

from nl4opt_measure import var_indices
from nl4opt_parser import ParseError, load_corpus, parse_instance

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]
optimal = {r["problem_id"] for r in csv.DictReader(open("data/results.csv"))
           if r["solve_status"] == "OPTIMAL"}


def canonical(con, j):
    return (sum(c for c, i in con.lhs if i == j)
            - sum(c for c, i in con.rhs_terms if i == j))


zero_coefs = collections.Counter()      # ctype -> count of zero coefficients
rows_with_zero = collections.Counter()  # ctype -> count of rows holding >=1 zero
total_coefs = collections.Counter()     # ctype -> total declared coefficients
rows_by_type = collections.Counter()
instances_hit = set()
n_rows = 0
obj_zero_coefs = 0
obj_instances_hit = set()
examples = []

for split, pid, inst in load_corpus(DATA):
    if pid not in optimal:
        continue
    try:
        prob = parse_instance(pid, inst, split)
    except ParseError:
        continue
    for con in prob.constraints:
        n_rows += 1
        rows_by_type[con.ctype] += 1
        zeros_here = 0
        for j in var_indices(con):
            total_coefs[con.ctype] += 1
            if canonical(con, j) == 0.0:
                zeros_here += 1
                zero_coefs[con.ctype] += 1
        if zeros_here:
            rows_with_zero[con.ctype] += 1
            instances_hit.add(pid)
            if len(examples) < 10:
                examples.append((pid, con.name, con.ctype,
                                 {prob.var_names[j]: canonical(con, j)
                                  for j in sorted(var_indices(con))}))
    for c, j in prob.obj_terms:
        if c == 0.0:
            obj_zero_coefs += 1
            obj_instances_hit.add(pid)

print("=" * 74)
print("A. ZERO-COEFFICIENT SCAN (canonical coefficient == 0.0 exactly)")
print("=" * 74)
print(f"  instances scanned (OPTIMAL)      {len(optimal)}")
print(f"  constraint rows scanned          {n_rows}")
print(f"  declared coefficients scanned    {sum(total_coefs.values())}")
print()
print(f"  coefficients exactly 0.0         {sum(zero_coefs.values())}")
print(f"  rows containing >=1 zero         {sum(rows_with_zero.values())}")
print(f"  instances affected               {len(instances_hit)}")
print()
print(f"  {'ctype':<12}{'rows':>8}{'coefs':>8}{'zero coefs':>12}{'rows w/ zero':>14}")
for t in sorted(rows_by_type, key=lambda t: -rows_by_type[t]):
    print(f"  {t:<12}{rows_by_type[t]:>8}{total_coefs[t]:>8}"
          f"{zero_coefs[t]:>12}{rows_with_zero[t]:>14}")
print()
print("  affected rows:")
for pid, name, ctype, coefs in examples:
    print(f"    {pid:>13}  {name:<14} {ctype:<10} {coefs}")
print()
print(f"  objective coefficients exactly 0.0   {obj_zero_coefs} "
      f"(in {len(obj_instances_hit)} instances)")

# -------------------------------------------------------------------------
# B. add n_vars_nonzero -- pure file transform, no re-solve, nothing recomputed
# -------------------------------------------------------------------------
rows = list(csv.DictReader(open("data/constraints.csv")))
cols = list(rows[0].keys())
cols.insert(cols.index("n_vars_in_row") + 1, "n_vars_nonzero")
for r in rows:
    r["n_vars_nonzero"] = len(json.loads(r["coefficients"]))
with open("data/constraints.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

diff = [r for r in rows if r["n_vars_nonzero"] != int(r["n_vars_in_row"])]
print("\n" + "=" * 74)
print("B. n_vars_nonzero ADDED to constraints.csv")
print("=" * 74)
print(f"  rows rewritten                   {len(rows)}")
print(f"  columns                          {cols}")
print(f"  rows where n_vars_nonzero != n_vars_in_row   {len(diff)}")
for r in diff:
    print(f"    {r['problem_id']:>13}  {r['constraint_name']:<12} "
          f"declared={r['n_vars_in_row']} nonzero={r['n_vars_nonzero']}  "
          f"{r['coefficients']}")
