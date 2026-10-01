"""TASK 8 -- regenerate audit. Read-only; reports to stdout.

  - blank count per slot vs the previous 14/19/14/18/14
  - LEAK AUDIT above the DO-NOT-READ footer
  - FIDELITY DIFF: every numeric printed must match constraints.csv /
    variables.csv / objectives.csv exactly at full precision
"""

import csv
import json
import os
import re
from collections import defaultdict

D = "handcheck_worksheets"
BANNER = "---- DO NOT READ UNTIL WORKSHEET IS FILLED ----"
PREVIOUS = {"1": 14, "2": 19, "3": 14, "4": 18, "5": 14}

fixtures = json.load(open("data/handcheck_fixtures.json"))
slot_of = {fixtures[s]["problem_id"]: s[-1] for s in fixtures}

cons_csv = defaultdict(list)
for r in csv.DictReader(open("data/constraints.csv")):
    cons_csv[r["problem_id"]].append(r)
vars_csv = defaultdict(list)
for r in csv.DictReader(open("data/variables.csv")):
    vars_csv[r["problem_id"]].append(r)
obj_csv = {r["problem_id"]: r for r in csv.DictReader(open("data/objectives.csv"))}

# -------------------------------------------------------------------------
print("=" * 74)
print("BLANK COUNTS")
print("=" * 74)
print(f"  {'slot':<8}{'previous':>10}{'now':>8}{'delta':>8}")
for n in "12345":
    t = open(os.path.join(D, f"slot_{n}.txt")).read()
    now = t.count("__________")
    print(f"  slot {n:<3}{PREVIOUS[n]:>10}{now:>8}{now - PREVIOUS[n]:>+8}")
pred = open(os.path.join(D, "predictions.txt")).read()
print(f"  predictions.txt blanks: {pred.count('__________')}")

# -------------------------------------------------------------------------
print("\n" + "=" * 74)
print("LEAK AUDIT  (everything above the DO-NOT-READ footer)")
print("=" * 74)
ANSWER_KEYS = ("product =", "SUM OF PRODUCTS", "DO THEY MATCH",
               "DOES IT HOLD", "EXPECTED RELATION")
total_leaks = 0
for n in "12345":
    t = open(os.path.join(D, f"slot_{n}.txt")).read()
    assert BANNER in t, f"slot {n} missing footer banner"
    above, below = t.split(BANNER, 1)
    leaks = []
    for ln in above.splitlines():
        if any(k in ln for k in ANSWER_KEYS) and not ln.rstrip().endswith("__________"):
            leaks.append(("unfilled-blank-violated", ln.strip()))
    # purpose string must not appear above the banner
    for slot, purpose in [(s, fixtures[s]) for s in fixtures]:
        pass
    purposes = {
        "1": "minimization with a binding GE row",
        "2": "binding upperbound-as-row",
        "3": "variable at zero or on a binding floor",
        "4": "multi-resource with a negative coefficient",
        "5": "single usable resource, FREE variable",
    }
    if purposes[n] in above:
        leaks.append(("purpose-above-banner", purposes[n]))
    if "purpose" in above.lower():
        leaks.append(("word-purpose-above-banner", ""))
    # the footer must contain the purpose
    ok_footer = purposes[n] in below
    print(f"  slot {n}: leaks={len(leaks)}  purpose_in_footer={ok_footer}")
    for kind, ln in leaks:
        print(f"    LEAK [{kind}] {ln}")
    total_leaks += len(leaks)
p_leaks = [k for k in ("coeff", "Pi =", "pi ", "limit", "objective coefficient", "purpose")
           if k in pred]
print(f"  predictions.txt forbidden-content hits: {p_leaks or 'none'}")
print(f"\n  TOTAL LEAKS: {total_leaks}")

# -------------------------------------------------------------------------
print("\n" + "=" * 74)
print("FIDELITY DIFF  (worksheet text vs constraints/variables/objectives CSV)")
print("=" * 74)
mismatches = []
checked = 0
for n in "12345":
    path = os.path.join(D, f"slot_{n}.txt")
    t = open(path).read()
    pid = [p for p, s in slot_of.items() if s == n][0]

    def want(label, literal):
        global checked
        checked += 1
        if literal not in t:
            mismatches.append((n, pid, label, literal))

    for c in cons_csv[pid]:
        nm = c["constraint_name"]
        want(f"{nm}.rhs_limit", f"rhs_limit      : {float(c['rhs_limit'])!r}")
        want(f"{nm}.pi", f"pi             : {float(c['pi'])!r}")
        want(f"{nm}.slack", f"slack          : {float(c['slack'])!r}")
        want(f"{nm}.binding", f"binding        : {c['binding']}")
        want(f"{nm}.n_vars_in_row", f"n_vars_in_row  : {c['n_vars_in_row']}")
        want(f"{nm}.n_vars_nonzero", f"n_vars_nonzero : {c['n_vars_nonzero']}")
        for vn, co in json.loads(c["coefficients"]).items():
            checked += 1
            if not re.search(rf"coeff\[{re.escape(vn)}(?: \([^)]*\))?\] = "
                             rf"{re.escape(repr(co))}\b", t):
                mismatches.append((n, pid, f"{nm}.coeff[{vn}]", repr(co)))

    ocoefs = json.loads(obj_csv[pid]["coefficients"])
    want("obj_value", f"optimal objective value : {float(obj_csv[pid]['obj_value'])!r}")
    for v in vars_csv[pid]:
        want(f"var {v['var_name']}.value",
             f"optimal value         : {float(v['value'])!r}")
        want(f"var {v['var_name']}.status", f"status                : {v['label']}")
        want(f"var {v['var_name']}.objcoef",
             f"objective coefficient : {ocoefs[v['var_name']]!r}")

print(f"  literals checked : {checked}")
print(f"  mismatches       : {len(mismatches)}")
if mismatches:
    print("\n  *** FIDELITY MISMATCH ***")
    for n, pid, label, lit in mismatches:
        print(f"    slot {n} {pid} {label}: expected literal not found -> {lit!r}")
else:
    print("  PASS -- every numeric printed matches the CSVs at full precision")

# -------------------------------------------------------------------------
print("\n" + "=" * 74)
print("GLOSSARY CLAIM CHECK  ('a non-binding row has pi = 0')")
print("=" * 74)
bad = [(r["problem_id"], r["constraint_name"], r["pi"], r["slack"])
       for rows in cons_csv.values() for r in rows
       if r["binding"] == "False" and abs(float(r["pi"])) > 1e-6]
print(f"  non-binding rows with |pi| > 1e-6 across corpus: {len(bad)}")
for b in bad[:10]:
    print(f"    {b}")
