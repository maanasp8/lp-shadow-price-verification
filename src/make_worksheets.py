"""Generate hand-check worksheets from handcheck_fixtures.json.

STEP 0 gate on slot 1, then one plain-text worksheet per slot plus a separate
prediction sheet.

Computes nothing. Every field requiring arithmetic stays underscores. Values are
read from the fixtures file or the source annotation and printed via repr().

Read-only with respect to every .csv and handcheck_fixtures.json.
"""

import json
import os
import sys

from nl4opt_parser import load_corpus, parse_instance

TOL = 1e-6
BLANK = "__________"
OUTDIR = "handcheck_worksheets"
DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]

PURPOSE = {
    "slot_1": "minimization with a binding GE row - both sign conventions active at once",
    "slot_2": "binding upperbound-as-row - does the capped variable give equality or <=",
    "slot_3": "variable at zero or on a binding floor - the inequality branch",
    "slot_4": "multi-resource with a negative coefficient - full Check A sum",
    "slot_5": "single usable resource, FREE variable - the exact pinned price",
    "slot_6": "minimization with a binding lowerbound/upperbound row -- closes the "
              "untested min+at-zero cell and the no-binding-floor gap.",
}

FOOTER_BANNER = "---- DO NOT READ UNTIL WORKSHEET IS FILLED ----"

# n_usable traced to nl4opt_measure.py: classify_constraint() labels a row USABLE
# when `abs(pi) > TOL` with TOL = 1e-6; roll_up() then sets n_usable = len(usable).
GLOSSARY = '''---------------- GLOSSARY ----------------
Row / constraint fields:
  ctype           declaration type from the NL4Opt annotation (linear, sum,
                  lowerbound, upperbound, xby, ratio, xy). A search hint only --
                  NOT the definition of a shared row.
  operator        >= or <= as written in the annotation.
  rhs_limit       the number on the right-hand side of the constraint.
  coeff[X]        how much of this row's resource one unit of X consumes.
  n_vars_in_row   how many variables appear in this row structurally.
  n_vars_nonzero  how many of those have a nonzero coefficient. Differs from
                  n_vars_in_row only on the 2 known zero-coefficient rows.
  pi              the SHADOW PRICE of this row: how much the optimal objective
                  value changes if rhs_limit increases by exactly one unit.
                  Can be negative; that is expected, not a bug.
  slack           how much room is left in this row at the optimum. 0 means the
                  row is fully used up. Sign follows (rhs - Ax), so a GE row with
                  surplus shows a negative number.
  binding         True when the row is tight at the optimum (|slack| < 1e-6).
                  A non-binding row has pi = 0, so it contributes 0 to every sum.

Variable fields:
  objective coefficient   what one unit of this variable is worth (max) or
                          costs (min), from the objective.
  optimal value           how much of it the solver chose.
  status                  FREE      -- not sitting against either of its own
                                       single-variable bounds; set by the tradeoff.
                          AT_LOWER  -- sitting on its lower bound (often zero).
                          AT_UPPER  -- sitting on its upper bound (a cap).
                          PINNED    -- lower equals upper, no freedom.
  bound by                which row or bound holds it there. `own_lb` means the
                          variable's built-in >= 0, not a constraint row.

Instance fields:
  sense       MIN or MAX.
  n_usable    Traced to nl4opt_measure.py, classify_constraint(), which reads
              `if abs(pi) > TOL: return USABLE` with `TOL = 1e-6` set at module
              level; the count is taken in the same file by roll_up(), which does
              `usable = [c for c in cons if c.label == USABLE]` and then
              `n_usable=len(usable)`. In plain words: the number of rows in this
              instance whose shadow price has absolute value strictly greater
              than 1e-6.

The two checks:
  CHECK B   Sum (rhs_limit x pi) over ALL rows; compare to the optimal objective
            value. Should be equal. Uses only the limits.
  CHECK A   For ONE variable: sum (coeff x pi) over EVERY row that variable appears
            in; compare to that variable's objective coefficient.
-----------------------------------------'''

PRED_HEADER = '''Fill this sheet COMPLETELY before opening any slot worksheet.
Decide each relation from your own reasoning about optimality, not from the
method doc. Once filled, do not change it -- if a slot disagrees, that
disagreement is the result.'''

fixtures = json.load(open("data/handcheck_fixtures.json"))

# ==========================================================================
# STEP 0 -- GATE on slot 1
# ==========================================================================
print("=" * 74)
print("STEP 0 -- GATE: slot 1 must have a >= row that is binding with |Pi| > 1e-6")
print("=" * 74)
s1 = fixtures["slot_1"]
ge_rows = [c for c in s1["constraints"] if c["operator"] == ">="]
qualifying = [c for c in ge_rows if c["binding"] is True and abs(c["pi"]) > TOL]
print(f"  slot 1 instance: {s1['problem_id']}")
print(f"  >= rows: {len(ge_rows)}   qualifying: {len(qualifying)}")
if not qualifying:
    print("\n  GATE FAILED -- not generating worksheets. Every '>=' row:")
    for c in ge_rows:
        print(f"    {c['constraint_name']:<15} operator={c['operator']}  "
              f"pi={c['pi']!r}  slack={c['slack']!r}  binding={c['binding']}")
    sys.exit(1)
print("  GATE PASSED\n")

annot = {pid: (split, inst) for split, pid, inst in load_corpus(DATA)}


def mentions_used(inst):
    """Annotation term keys actually referenced by the declarations."""
    used = []
    od = inst["obj_declaration"]
    used += list(od.get("terms", {}))
    used += list(od.get("vars", []))
    for d in inst["const_declarations"]:
        used += list(d.get("terms", {}))
        for k in ("var", "x_var", "y_var"):
            if k in d:
                used.append(d[k])
    return used


def name_map(pid):
    """canonical name -> sorted annotation term keys. DISPLAY ONLY.

    handcheck_fixtures.json carries no name map, and it is read-only, so the map
    is rebuilt here from the source annotation's order_mapping -- the same source
    the VARIABLE NAME MAP section already used. No canonical name is changed.
    """
    split, inst = annot[pid]
    prob = parse_instance(pid, inst, split)
    order = inst["order_mapping"]
    used = mentions_used(inst)
    return {canon: sorted({m for m in used if order.get(m) == idx})
            for idx, canon in enumerate(prob.var_names)}


def disp(canon, keys):
    """canonical ("key", "key") -- or the canonical name alone if identical."""
    if not keys or set(keys) == {canon}:
        return canon
    return f'{canon} (' + ", ".join(f'"{k}"' for k in keys) + ')'


def build(slot, fx):
    pid = fx["problem_id"]
    nmap = name_map(pid)
    L = []
    w = L.append

    # 1. HEADER  (purpose deliberately absent -- it lives in the footer)
    w("=" * 78)
    w(f"HAND-CHECK WORKSHEET -- SLOT {slot[-1]}")
    w("=" * 78)
    w(f"problem_id   : {pid}")
    w(f"sense        : {fx['sense'].upper()}")
    w(f"optimal objective value : {fx['objective_value']!r}")
    w(f"n_usable     : {fx['n_usable']}")
    w(f"n_variables  : {len(fx['variables'])}")
    w("")

    # 2. GLOSSARY
    w(GLOSSARY)
    w("")

    # 3. PROBLEM TEXT
    w("-" * 78)
    w("PROBLEM TEXT")
    w("-" * 78)
    w(fx["document"])
    w("")

    # 4. RAW ANNOTATION
    w("-" * 78)
    w("RAW ANNOTATION")
    w("-" * 78)
    w("obj_declaration:")
    w("  " + json.dumps(fx["obj_declaration"]))
    w("const_declarations:")
    for d in fx["const_declarations"]:
        w("  " + json.dumps(d))
    w("")

    # 5. VARIABLE NAME MAP
    w("-" * 78)
    w("VARIABLE NAME MAP  (canonical name <- annotation term key(s))")
    w("-" * 78)
    any_mismatch = False
    for canon, keys in nmap.items():
        mismatch = any(k != canon for k in keys)
        any_mismatch = any_mismatch or mismatch
        w(f"  {canon:<26} <- {', '.join(repr(k) for k in keys)}"
          + (" ***" if mismatch else ""))
    if not any_mismatch:
        w("  no mismatches")
    w("")

    # 6. CONSTRAINT TABLE
    w("-" * 78)
    w("CONSTRAINT TABLE")
    w("-" * 78)
    for c in fx["constraints"]:
        w(f"  {c['constraint_name']}")
        w(f"    ctype          : {c['ctype']}")
        w(f"    n_vars_in_row  : {c['n_vars_in_row']}")
        w(f"    n_vars_nonzero : {c['n_vars_nonzero']}")
        w(f"    operator       : {c['operator']}")
        w(f"    rhs_limit      : {c['rhs_limit']!r}")
        for vn, co in c["coefficients"].items():
            w(f"    coeff[{disp(vn, nmap.get(vn, []))}] = {co!r}")
        w(f"    pi             : {c['pi']!r}")
        w(f"    slack          : {c['slack']!r}")
        w(f"    binding        : {c['binding']}")
        w("")

    # 7. VARIABLE TABLE
    w("-" * 78)
    w("VARIABLE TABLE")
    w("-" * 78)
    for v in fx["variables"]:
        w(f"  {disp(v['var_name'], nmap.get(v['var_name'], []))}")
        w(f"    objective coefficient : {v['objective_coefficient']!r}")
        w(f"    optimal value         : {v['value']!r}")
        w(f"    status                : {v['label']}")
        w(f"    bound by              : {v['restricted_by'] or '-'}")
        w("")

    # 8. CHECK B WORKSHEET
    w("-" * 78)
    w("CHECK B WORKSHEET")
    w("-" * 78)
    for c in fx["constraints"]:
        w(f"  {c['constraint_name']}:  limit = {c['rhs_limit']!r}   "
          f"Pi = {c['pi']!r}   product = {BLANK}")
    w("")
    w(f"  SUM OF PRODUCTS = {BLANK}")
    w(f"  OPTIMAL OBJECTIVE VALUE = {fx['objective_value']!r}")
    w(f"  DO THEY MATCH? {BLANK}")
    w("")

    # 9. CHECK A WORKSHEET -- relation predicted BEFORE any arithmetic
    w("-" * 78)
    w("CHECK A WORKSHEET")
    w("-" * 78)
    for v in fx["variables"]:
        name = v["var_name"]
        w(f"  variable: {disp(name, nmap.get(name, []))}   "
          f"status: {v['label']}   value: {v['value']!r}")
        w(f"    EXPECTED RELATION (=, >=, <=): {BLANK}")
        for c in fx["constraints"]:
            if name in c["coefficients"]:
                w(f"    {c['constraint_name']}:  coeff = {c['coefficients'][name]!r}   "
                  f"Pi = {c['pi']!r}   product = {BLANK}")
        w(f"    SUM OF PRODUCTS = {BLANK}")
        w(f"    OBJECTIVE COEFFICIENT = {v['objective_coefficient']!r}")
        w(f"    DOES IT HOLD? {BLANK}")
        w("")

    # 10. FOOTER -- purpose, quarantined
    w("")
    w(FOOTER_BANNER)
    w(f"purpose: {PURPOSE[slot]}")
    return "\n".join(L) + "\n"


def build_predictions():
    L = [PRED_HEADER, "", "=" * 78, ""]
    for slot in sorted(fixtures):
        fx = fixtures[slot]
        nmap = name_map(fx["problem_id"])
        for v in fx["variables"]:
            L.append(f"slot {slot[-1]}  {fx['problem_id']:>13}  "
                     f"{fx['sense'].upper():<4}  "
                     f"{disp(v['var_name'], nmap.get(v['var_name'], [])):<46} "
                     f"{v['label']:<9} relation: {BLANK}")
        L.append("")
    return "\n".join(L) + "\n"


os.makedirs(OUTDIR, exist_ok=True)
paths = []
for slot in sorted(fixtures):
    path = os.path.join(OUTDIR, f"slot_{slot[-1]}.txt")
    with open(path, "w") as fh:
        fh.write(build(slot, fixtures[slot]))
    paths.append(os.path.abspath(path))

pred_path = os.path.join(OUTDIR, "predictions.txt")
with open(pred_path, "w") as fh:
    fh.write(build_predictions())
paths.append(os.path.abspath(pred_path))

print("FILES WRITTEN")
for p in paths:
    print(f"  {p}")
