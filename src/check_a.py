"""STEP 2 -- Check A over the OPTIMAL corpus.

For each variable v:
    sum = SUM over every row where v appears of (coeff of v in row * pi of row)
compared against v's objective coefficient.

Two branches only, chosen by |value| vs ZERO_TOL. How a variable came to be
pinned is irrelevant: a variable at a binding cap and one at a nonzero floor
both take the equality branch.

    |value| >  ZERO_TOL   ->  sum == obj_coeff   (either sense)
    |value| <= ZERO_TOL   ->  max: sum >= obj_coeff
                              min: sum <= obj_coeff

Rows where v appears with pi == 0 are included; they contribute zero but count
toward n_terms.

TERM-LIST SOURCE (changed 2026-08-02)
-------------------------------------
`a_ij`, `c_j` and the sense are read from the CLEAN ANNOTATION via
`text_terms.load_text_terms()`, never from `constraints.csv`. Only `pi` (and
the variable values used for the branch rule) come from the solve. See
`text_terms.py` and amendment 2026-08-02 section 8 item 2: building the term
list from the model's own representation makes a type-6 mutation (coefficient
omission) invisible, because `constraints.csv` drops zero coefficients on write
and "absent from row" then looks identical to "present with coefficient 0".

On the clean corpus this is arithmetically identical to the old reading -- a
zero term contributes zero -- and it is the SAME code path a mutation harness
must use.
"""

import pandas as pd

from checks_common import REL_TOL, ZERO_TOL, load
from text_terms import load_text_terms

print("=" * 78)
print("STEP 2 -- CHECK A")
print("=" * 78)
print(f"  ZERO_TOL = {ZERO_TOL:g}   REL_TOL = {REL_TOL:g}")
cons, objs, vars_, optimal = load()
text, parse_failed = load_text_terms()

missing = optimal - set(text)
if missing:
    raise AssertionError(f"{len(missing)} OPTIMAL instance(s) have no text term "
                         f"list: {sorted(missing)[:10]}")

# pi is the only per-row quantity Check A takes from the solve.
pi_by_pid = {pid: dict(zip(g.constraint_name, g.pi.astype(float)))
             for pid, g in cons.groupby("problem_id")}

# The sense is text-derived; objectives.csv records what the model was built
# with. On the clean corpus they must agree.
sense_csv = objs.set_index("problem_id")["sense"].to_dict()
bad_sense = [p for p in optimal if text[p].sense != sense_csv[p]]
if bad_sense:
    raise AssertionError(f"sense disagreement text vs objectives.csv: {bad_sense[:10]}")

out = []
for v in vars_.itertuples():
    pid, name = v.problem_id, v.var_name
    t = text[pid]
    sense = t.sense
    obj_coeff = t.obj_coef(name)

    pis = pi_by_pid.get(pid, {})
    instance_n_rows = len(t.row_names)

    # the term list is the TEXT's, fixed per instance; pi comes from the solve
    total, n_terms = 0.0, 0
    for row_name, a_text in t.terms_for(name):
        total += a_text * pis[row_name]
        n_terms += 1

    value = float(v.value)
    scale = REL_TOL * max(1.0, abs(obj_coeff))

    if abs(value) > ZERO_TOL:
        branch, relation = "positive", "="
        fired = abs(total - obj_coeff) > scale
    else:
        branch = "zero"
        if sense == "max":
            relation = ">="
            fired = total < obj_coeff - scale
        else:
            relation = "<="
            fired = total > obj_coeff + scale

    out.append({
        "problem_id": pid, "variable": name, "status": v.status, "value": value,
        "branch": branch, "n_terms": n_terms, "sum": total,
        "obj_coeff": obj_coeff, "required_relation": relation,
        "diff": total - obj_coeff, "fired": fired,
        "instance_n_rows": instance_n_rows,
    })

df = pd.DataFrame(out).sort_values(["problem_id", "variable"]).reset_index(drop=True)
df.to_csv("data/checkA_results.csv", index=False)

# --------------------------------------------------------------------------
fired = df[df.fired]
vac_inst = df[df.instance_n_rows == 0]
vac_var = df[(df.instance_n_rows > 0) & (df.n_terms == 0)]
real = df[(df.instance_n_rows > 0) & (df.n_terms > 0)]

print("\n" + "-" * 78)
print("RESULT")
print("-" * 78)
print(f"  variable rows evaluated          {len(df)}")
print(f"  FIRED                            {len(fired)}")
print(f"  passed                           {len(df) - len(fired)}")
print(f"\n  by branch: {df.groupby('branch').size().to_dict()}")
print(f"  by required_relation: {df.groupby('required_relation').size().to_dict()}")
print(f"  by branch x sense: "
      f"{df.merge(objs[['problem_id','sense']], on='problem_id')
           .groupby(['branch','sense']).size().to_dict()}")

print("\n" + "-" * 78)
print("VACUOUS vs REAL PASSES  (n_rows == 0 flagged, not folded in)")
print("-" * 78)
print(f"  rows in zero-row instances (instance_n_rows == 0)   {len(vac_inst)}")
if len(vac_inst):
    print(f"    instances: {sorted(vac_inst.problem_id.unique())}")
    for r in vac_inst.to_dict("records"):
        print(f"      {r['problem_id']}  {r['variable']:<22} value={r['value']!r} "
              f"branch={r['branch']} n_terms={r['n_terms']} sum={r['sum']!r} "
              f"obj_coeff={r['obj_coeff']!r} rel={r['required_relation']} "
              f"fired={r['fired']}   <- VACUOUS (no rows exist)")
print(f"  rows in instances WITH rows but variable in none    {len(vac_var)}")
if len(vac_var):
    for r in vac_var.head(20).to_dict("records"):
        print(f"      {r['problem_id']}  {r['variable']:<22} "
              f"instance_n_rows={r['instance_n_rows']} n_terms=0 sum={r['sum']!r} "
              f"obj_coeff={r['obj_coeff']!r} rel={r['required_relation']} "
              f"fired={r['fired']}   <- VACUOUS (variable in no row)")
print(f"  rows with a genuine non-empty sum                   {len(real)}")
print(f"    of which fired                                   {int(real.fired.sum())}")

if len(fired):
    print("\n" + "-" * 78)
    print("FIRED ROWS")
    print("-" * 78)
    for r in fired.to_dict("records"):
        print(f"  {r['problem_id']}  {r['variable']:<24} branch={r['branch']:<8} "
              f"rel={r['required_relation']:<3} n_terms={r['n_terms']} "
              f"sum={r['sum']!r} obj_coeff={r['obj_coeff']!r} diff={r['diff']!r}")

# --------------------------------------------------------------------------
# ACCEPTANCE -- the twelve hand-worked variable rows
# --------------------------------------------------------------------------
HAND = [
    ("-451730029",  "Type A food",         "2.4",    "2.0",    "2.0",   "="),
    ("-451730029",  "Type B food",         "3.2",    "5.0",    "5.0",   "="),
    ("-503693411",  "easy puzzles",        "5.0",    "5.0",    "5.0",   "="),
    ("-503693411",  "hard puzzles",        "5.0",    "8.0",    "8.0",   "="),
    ("-2061445103", "PX7",                 "16.0",   "40.0",   "40.0",  "="),
    ("-2061445103", "Silent X",            "0.0",    "93.333", "75.0",  ">="),
    ("294897151",   "apples",              "16.667", "2.0",    "2.0",   "="),
    ("294897151",   "pears",               "33.333", "4.0",    "4.0",   "="),
    ("1727707716",  "small",               "0.0",    "3.333",  "3.0",   ">="),
    ("1727707716",  "large size",          "6.667",  "5.0",    "5.0",   "="),
    ("1866828133",  "newcomers",           "0.0",    "0.0",    "400.0", "<="),
    ("1866828133",  "full-time employees", "40.0",   "700.0",  "700.0", "="),
]


def agrees(code, hand_str):
    """Hand values are written to the precision shown; the code value must round
    to that same printed number. A strict 6-sig-fig equality would spuriously
    fail on e.g. 93.33333333 vs the table's 93.333."""
    d = len(hand_str.split(".")[1]) if "." in hand_str else 0
    return round(code, d) == float(hand_str)


idx = df.set_index(["problem_id", "variable"])
print("\n" + "=" * 78)
print("ACCEPTANCE -- twelve hand-worked variable rows")
print("=" * 78)
print(f"  {'instance':>12} {'variable':<21} {'status':<9} {'code sum':>20} "
      f"{'hand':>8} {'obj':>7} {'rel':>3} ok")
bad = []
for pid, var, h_val, h_sum, h_obj, h_rel in HAND:
    r = idx.loc[(pid, var)]
    ok_val = agrees(float(r["value"]), h_val)
    ok_sum = agrees(float(r["sum"]), h_sum)
    ok_obj = agrees(float(r["obj_coeff"]), h_obj)
    ok_rel = r["required_relation"] == h_rel
    ok = ok_val and ok_sum and ok_obj and ok_rel
    if not ok:
        bad.append((pid, var, dict(value=(float(r["value"]), h_val, ok_val),
                                   sum=(float(r["sum"]), h_sum, ok_sum),
                                   obj=(float(r["obj_coeff"]), h_obj, ok_obj),
                                   rel=(r["required_relation"], h_rel, ok_rel))))
    print(f"  {pid:>12} {var:<21} {r['status']:<9} {float(r['sum']):>20.12g} "
          f"{h_sum:>8} {h_obj:>7} {r['required_relation']:>3} {'OK' if ok else 'MISMATCH'}")

print(f"\n  all twelve reproduce: {not bad}")
if bad:
    print("\n  *** DISCREPANCY -- code NOT adjusted, tolerance NOT adjusted ***")
    for pid, var, fields in bad:
        print(f"    {pid} {var}")
        for k, (code, hand, okk) in fields.items():
            if not okk:
                print(f"      {k}: code={code!r}  hand={hand!r}")
    raise SystemExit(1)

print("\nwrote checkA_results.csv")
