"""Cross-validate text_terms.py against the persisted clean artifacts.

The loader re-derives every row's canonical coefficient vector and RHS from the
annotation. Those must agree, on the clean corpus, with what gurobipy actually
built and `resolve_pass.py` persisted -- otherwise the loader is not a faithful
stand-in for `constraints.csv` and no Check A result computed with it can be
compared to the ones already reported.

Checks, all over the 1040 OPTIMAL instances:

  1. same set of row names per instance
  2. same rhs_limit per row
  3. same operator per row
  4. every NONZERO loader coefficient equals the persisted one (exact float)
  5. every persisted coefficient is present in the loader's list
  6. loader entries absent from the persisted list are exactly the ones with
     |a| <= 1e-12 -- i.e. the zeros resolve_pass.py filtered on write
  7. loader objective coefficients equal objectives.csv (which is NOT filtered)
"""

import json
import sys

import pandas as pd

from text_terms import load_text_terms

ZERO = 1e-12          # the filter resolve_pass.py applied when writing

print("=" * 78)
print("VALIDATE text_terms.py  vs  constraints.csv / objectives.csv  (clean corpus)")
print("=" * 78)

results = pd.read_csv("data/results.csv", dtype={"problem_id": str})
optimal = set(results.loc[results.solve_status == "OPTIMAL", "problem_id"])
cons = pd.read_csv("data/constraints.csv", dtype={"problem_id": str})
objs = pd.read_csv("data/objectives.csv", dtype={"problem_id": str})
cons = cons[cons.problem_id.isin(optimal)].copy()
objs = objs[objs.problem_id.isin(optimal)].copy()
cons["coefs"] = cons["coefficients"].map(json.loads)
objs["coefs"] = objs["coefficients"].map(json.loads)
print(f"  OPTIMAL instances {len(optimal)}   persisted rows {len(cons)}")

terms, failed = load_text_terms()
print(f"  parse failures    {len(failed)}   "
      f"any OPTIMAL among them: {bool(set(failed) & optimal)}")

missing = optimal - set(terms)
if missing:
    raise SystemExit(f"FAIL: {len(missing)} OPTIMAL instances have no text terms: "
                     f"{sorted(missing)[:10]}")

bad_names, bad_limit, bad_op, bad_coef, bad_missing, bad_extra = [], [], [], [], [], []
zeros_explained = []          # loader entries the CSV dropped, with |a| <= ZERO

for pid, g in cons.groupby("problem_id"):
    t = terms[pid]
    csv_names = list(g.constraint_name)
    if sorted(csv_names) != sorted(t.row_names):
        bad_names.append((pid, csv_names, t.row_names))
        continue
    for r in g.itertuples():
        rn = r.constraint_name
        if float(r.rhs_limit) != t.row_limits[rn]:
            bad_limit.append((pid, rn, float(r.rhs_limit), t.row_limits[rn]))
        if r.operator != t.row_operator[rn]:
            bad_op.append((pid, rn, r.operator, t.row_operator[rn]))
        mine, theirs = t.row_coefs[rn], r.coefs
        for v, a in mine.items():
            if v in theirs:
                if a != theirs[v]:
                    bad_coef.append((pid, rn, v, a, theirs[v]))
            elif abs(a) <= ZERO:
                zeros_explained.append((pid, rn, v, a))
            else:
                bad_missing.append((pid, rn, v, a))      # nonzero, CSV lacks it
        for v in theirs:
            if v not in mine:
                bad_extra.append((pid, rn, v, theirs[v]))  # CSV has, loader lacks

bad_obj = []
for r in objs.itertuples():
    t = terms[r.problem_id]
    if set(r.coefs) != set(t.obj_coefs):
        bad_obj.append((r.problem_id, "key set", sorted(r.coefs), sorted(t.obj_coefs)))
        continue
    for v, c in r.coefs.items():
        if float(c) != t.obj_coefs[v]:
            bad_obj.append((r.problem_id, v, float(c), t.obj_coefs[v]))

print("\n" + "-" * 78)
print("RESULT")
print("-" * 78)
rows = [
    ("row-name set mismatches", bad_names),
    ("rhs_limit mismatches", bad_limit),
    ("operator mismatches", bad_op),
    ("coefficient value mismatches", bad_coef),
    ("nonzero loader coef missing from CSV", bad_missing),
    ("CSV coef missing from loader", bad_extra),
    ("objective coefficient mismatches", bad_obj),
]
for label, lst in rows:
    print(f"  {label:<42} {len(lst)}")
    for item in lst[:10]:
        print(f"      {item}")

print(f"\n  loader terms the CSV dropped as zeros      {len(zeros_explained)}")
for item in zeros_explained:
    print(f"      {item[0]}  {item[1]}  {item[2]!r}  a={item[3]!r}")
print("\n  (these are the type-6 blind spot: present in the text's term list,")
print("   absent from constraints.csv, indistinguishable there from 'not in row')")

fail = sum(len(l) for _, l in rows)
print(f"\n  TOTAL DISAGREEMENTS  {fail}")
if fail:
    sys.exit(1)
print("  -> loader reproduces the persisted clean model exactly")
