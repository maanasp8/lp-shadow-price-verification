"""Select slot 6: a MIN instance with a binding single-variable bound row.

Deterministic filter + sort, run once. No hand-picking: ties break on the lowest
problem_id under INTEGER ordering (ids are signed integers stored as strings).

Read-only w.r.t. every .csv. Appends slot_6 to handcheck_fixtures.json and leaves
slots 1-5 untouched.
"""

import csv
import json
from collections import defaultdict

TOL = 1e-6

results = {r["problem_id"]: r for r in csv.DictReader(open("data/results.csv"))}
objectives = {r["problem_id"]: r for r in csv.DictReader(open("data/objectives.csv"))}
suspicious = {r["problem_id"] for r in csv.DictReader(open("data/suspicious.csv"))}

cons = defaultdict(list)
for r in csv.DictReader(open("data/constraints.csv")):
    r["pi"] = float(r["pi"]); r["slack"] = float(r["slack"])
    r["rhs_limit"] = float(r["rhs_limit"])
    r["binding"] = r["binding"] == "True"
    r["n_vars_in_row"] = int(r["n_vars_in_row"])
    r["n_vars_nonzero"] = int(r["n_vars_nonzero"])
    r["coefs"] = json.loads(r["coefficients"])
    cons[r["problem_id"]].append(r)

vars_ = defaultdict(list)
for r in csv.DictReader(open("data/variables.csv")):
    r["value"] = float(r["value"])
    vars_[r["problem_id"]].append(r)

print("=" * 74)
print("SLOT 6 SELECTION FUNNEL")
print("=" * 74)
print(f"  instances in results.csv                     {len(results)}")

# ---- step 1: OPTIMAL, not suspicious, not degenerate ----------------------
s1, excl = [], defaultdict(int)
for pid, r in results.items():
    if r["solve_status"] != "OPTIMAL":
        excl["not OPTIMAL"] += 1
    elif pid in suspicious:
        excl["in suspicious.csv"] += 1
    elif int(r["n_degenerate"]) > 0:
        excl["has degenerate row(s)"] += 1
    else:
        s1.append(pid)
print(f"  step 1  OPTIMAL & clean & not suspicious     {len(s1)}")
for k, v in excl.items():
    print(f"            excluded by {k:<24} {v}")

# ---- step 2: sense == MIN -------------------------------------------------
s2 = [p for p in s1 if objectives[p]["sense"] == "min"]
print(f"  step 2  sense == MIN                         {len(s2)}"
      f"   (removed {len(s1) - len(s2)})")


# ---- step 3: binding single-variable bound row with a nonzero dual --------
def step3_rows(pid):
    return [c for c in cons[pid]
            if c["ctype"] in ("lowerbound", "upperbound")
            and c["n_vars_in_row"] == 1
            and c["binding"] is True
            and abs(c["pi"]) > TOL]


s3 = [p for p in s2 if step3_rows(p)]
kinds = defaultdict(int)
for p in s3:
    for c in step3_rows(p):
        kinds[c["ctype"]] += 1
print(f"  step 3  binding 1-var bound row, |pi|>1e-6   {len(s3)}"
      f"   (removed {len(s2) - len(s3)})")
print(f"            matching rows by ctype: {dict(kinds)}")


# ---- step 4 (PREFERRED): another variable at value 0, not the pinned one ---
def step4_ok(pid):
    pinned = {v for c in step3_rows(pid) for v in c["coefs"]}
    return [v for v in vars_[pid] if v["value"] == 0.0 and v["var_name"] not in pinned]


s4 = [p for p in s3 if step4_ok(p)]
print(f"  step 4  WITH preference (other var at 0)     {len(s4)}")
print(f"          WITHOUT preference (= step 3)        {len(s3)}")

if not s3:
    print("\n  ZERO CANDIDATES after the exact AND. Stopping, not loosening.")
    raise SystemExit(1)

used_pref = bool(s4)
pool = s4 if used_pref else s3
print(f"  preference satisfiable: {used_pref}"
      + ("" if used_pref else "  -> FELL BACK to the unfiltered step-3 pool"))

existing = {json.load(open("data/handcheck_fixtures.json"))[s]["problem_id"]
            for s in json.load(open("data/handcheck_fixtures.json"))}
overlap = [p for p in pool if p in existing]
print(f"  overlap with existing slots 1-5: {overlap or 'none'}")

# ---- step 5: lowest problem_id (integer ordering) -------------------------
pool_sorted = sorted(pool, key=lambda p: int(p))
chosen = pool_sorted[0]
print(f"  step 5  lowest problem_id (integer order)    {chosen}")
print(f"          pool (first 10, sorted): {pool_sorted[:10]}")

# ==========================================================================
# build the slot_6 entry, exact schema of slots 1-5
# ==========================================================================
from nl4opt_parser import load_corpus  # noqa: E402

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]
annot = {pid: inst for _, pid, inst in load_corpus(DATA)}

inst = annot[chosen]
ocoefs = json.loads(objectives[chosen]["coefficients"])
entry = {
    "problem_id": chosen,
    "split": results[chosen]["split"],
    "sense": objectives[chosen]["sense"],
    "objective_value": float(objectives[chosen]["obj_value"]),
    "n_usable": int(results[chosen]["n_usable"]),
    "document": inst["document"],
    "obj_declaration": inst["obj_declaration"],
    "const_declarations": inst["const_declarations"],
    "objective_coefficients": ocoefs,
    "constraints": [{k: c[k] for k in
                     ("constraint_name", "ctype", "n_vars_in_row",
                      "n_vars_nonzero", "operator", "rhs_limit", "pi",
                      "slack", "binding")} | {"coefficients": c["coefs"]}
                    for c in cons[chosen]],
    "variables": [{"var_name": v["var_name"], "value": v["value"],
                   "label": v["label"], "restricted_by": v["restricted_by"],
                   "objective_coefficient": ocoefs[v["var_name"]]}
                  for v in vars_[chosen]],
}

fx = json.load(open("data/handcheck_fixtures.json"))
assert "slot_6" not in fx, "slot_6 already exists -- refusing to overwrite"
fx["slot_6"] = entry
with open("data/handcheck_fixtures.json", "w") as fh:
    json.dump(fx, fh, indent=2)

print("\n" + "=" * 74)
print("SLOT 6 ADDED")
print("=" * 74)
print(f"  problem_id  {chosen}   split={entry['split']}   sense={entry['sense']}")
print(f"  obj_value   {entry['objective_value']!r}   n_usable={entry['n_usable']}")
print(f"  rows        {len(entry['constraints'])}   vars={len(entry['variables'])}")
print("  step-3 matching row(s):")
for c in step3_rows(chosen):
    print(f"    {c['constraint_name']:<15} ctype={c['ctype']:<11} "
          f"n_vars_in_row={c['n_vars_in_row']} binding={c['binding']} "
          f"pi={c['pi']!r} coefs={c['coefs']}")
if used_pref:
    print("  step-4 other variable(s) at value 0:")
    for v in step4_ok(chosen):
        print(f"    {v['var_name']:<22} value={v['value']!r} label={v['label']} "
              f"restricted_by={v['restricted_by']!r}")
