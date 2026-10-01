"""STEP 1 -- Check B over the OPTIMAL corpus.

    sum over ALL rows of (rhs_limit * pi)   vs   obj_value

Every row is included, including pi == 0 and rhs_limit == 0 rows.
No sign adjustment for either sense.
Fires when |sum - obj_value| > 1e-6 * max(1, |obj_value|).
"""

import pandas as pd

from checks_common import REL_TOL, load

print("=" * 72)
print("STEP 1 -- CHECK B")
print("=" * 72)
cons, objs, vars_, optimal = load()

rows = []
by_pid = dict(list(cons.groupby("problem_id")))
for r in objs.itertuples():
    pid = r.problem_id
    g = by_pid.get(pid)
    # every row participates; no filtering on pi or rhs_limit
    total = float((g["rhs_limit"] * g["pi"]).sum()) if g is not None else 0.0
    n_rows = 0 if g is None else len(g)
    obj_value = float(r.obj_value)
    abs_diff = abs(total - obj_value)
    scale = max(1.0, abs(obj_value))
    rows.append({
        "problem_id": pid,
        "sum": total,
        "obj_value": obj_value,
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / scale,
        "fired": abs_diff > REL_TOL * scale,
        "_n_rows": n_rows,
        "_sense": r.sense,
    })

df = pd.DataFrame(rows).sort_values("problem_id").reset_index(drop=True)
df.drop(columns=["_n_rows", "_sense"]).to_csv("data/checkB_results.csv", index=False)

fired = df[df.fired]
print("\n" + "-" * 72)
print("RESULT")
print("-" * 72)
print(f"  instances evaluated              {len(df)}")
print(f"  rows consumed                    {int(df['_n_rows'].sum())}")
print(f"  FIRED                            {len(fired)}")
print(f"  passed                           {len(df) - len(fired)}")
print(f"  tolerance                        |sum-obj| > {REL_TOL:g} * max(1,|obj|)")

print(f"\n  by sense: " + str(df.groupby('_sense').size().to_dict()))
print(f"  instances with zero rows: {int((df['_n_rows'] == 0).sum())}"
      + (f"  -> {df.loc[df['_n_rows'] == 0, 'problem_id'].tolist()}"
         if (df["_n_rows"] == 0).any() else ""))

print("\n  worst residuals (largest rel_diff):")
print(f"    {'problem_id':>13} {'sense':>5} {'sum':>22} {'obj_value':>22} {'rel_diff':>12}")
for r in df.nlargest(8, "rel_diff").to_dict("records"):
    print(f"    {r['problem_id']:>13} {r['_sense']:>5} {r['sum']:>22.15g} "
          f"{r['obj_value']:>22.15g} {r['rel_diff']:>12.3e}")

if len(fired):
    print("\n  *** FIRED INSTANCES ***")
    for r in fired.to_dict("records"):
        print(f"    {r['problem_id']}  sum={r['sum']!r}  obj_value={r['obj_value']!r}  "
              f"abs_diff={r['abs_diff']!r}  rel_diff={r['rel_diff']!r}")

# acceptance: Check B must close on all six hand-worked instances
SIX = ["-451730029", "-503693411", "-2061445103", "294897151",
       "1727707716", "1866828133"]
print("\n" + "-" * 72)
print("ACCEPTANCE -- Check B on the six hand-worked instances")
print("-" * 72)
print(f"    {'problem_id':>13} {'sense':>5} {'sum':>20} {'obj_value':>20} "
      f"{'abs_diff':>12} fired")
sub = df[df.problem_id.isin(SIX)]
for pid in SIX:
    r = sub[sub.problem_id == pid].iloc[0]
    print(f"    {pid:>13} {r['_sense']:>5} {r['sum']:>20.15g} "
          f"{r['obj_value']:>20.15g} {r['abs_diff']:>12.3e} {r['fired']}")
print(f"\n  all six close: {not sub['fired'].any()}")
print("\nwrote checkB_results.csv")
