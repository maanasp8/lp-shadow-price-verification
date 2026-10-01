"""Zero-false-positive control: Checks A and B on CLEAN models, no mutation.

    venv/bin/python clean_control.py

The hard requirement in amendment §9 is **zero false positives given correct
extraction**. Every prior run has reported 0 fires. This re-runs that control
against the CURRENT codebase -- `text_terms.py`, the §12.3/§12.6 mutation
engine, and the code path that produced the full population -- rather than
against the CSV artifacts alone.

Two independent paths are exercised over all 1040 OPTIMAL instances:

  ENGINE PATH   `mutations.solve` + `mutations.check_a` / `check_b`, the exact
                functions that scored all 39 340 mutants, with the mutation
                step omitted. This is the one that matters: if the engine had
                acquired a false-positive bug, §13 would be invalid.
  ARTIFACT PATH `check_a.py` / `check_b.py` against the persisted clean CSVs,
                the historical control, run separately.

The engine path also asserts that each in-memory clean solve reproduces the
persisted `objectives.csv` objective value, so a divergence in the solve itself
cannot hide inside a passing check.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from checks_common import REL_TOL, ZERO_TOL
from mutations import check_a, check_b, solve
from text_terms import TextTermsStore, load_clean_problems

print("=" * 78)
print("CLEAN-MODEL ZERO-FPR CONTROL -- current codebase, no mutation")
print("=" * 78)
print(f"  REL_TOL {REL_TOL:g}   ZERO_TOL {ZERO_TOL:g}")

results = pd.read_csv("data/results.csv", dtype={"problem_id": str})
optimal = sorted(results.loc[results.solve_status == "OPTIMAL", "problem_id"],
                 key=int)
objs = pd.read_csv("data/objectives.csv", dtype={"problem_id": str})
obj_by_pid = dict(zip(objs.problem_id, objs.obj_value.astype(float)))

problems, failed = load_clean_problems()
store = TextTermsStore(problems)
print(f"  OPTIMAL instances {len(optimal)}   parse failures {len(failed)} "
      f"(none OPTIMAL: {not (set(failed) & set(optimal))})")

a_fired_rows, b_fired_inst, solve_mismatch, status_bad = [], [], [], []
n_var_rows = n_real_rows = 0
n_vac_instance = n_vac_var = 0

for pid in optimal:
    prob = problems[pid]
    text = store.get(pid)
    sol = solve(prob)                       # NO mutation applied
    if not sol.optimal:
        status_bad.append((pid, sol.status))
        continue
    if abs(sol.obj_value - obj_by_pid[pid]) > 1e-9 * max(1.0, abs(obj_by_pid[pid])):
        solve_mismatch.append((pid, sol.obj_value, obj_by_pid[pid]))

    _, a_detail = check_a(text, sol)
    for d in a_detail:
        n_var_rows += 1
        if len(text.row_names) == 0:
            n_vac_instance += 1
        elif d["n_terms"] == 0:
            n_vac_var += 1
        else:
            n_real_rows += 1
        if d["fired"]:
            a_fired_rows.append((pid, d["variable"], d["sum"], d["obj_coeff"],
                                 d["relation"], d["residual"]))

    b_fired, b_detail = check_b(text, sol)
    if b_fired:
        b_fired_inst.append((pid, b_detail["sum"], b_detail["obj_value"],
                             b_detail["abs_diff"]))

print("\n" + "-" * 78)
print("ENGINE PATH  (mutations.check_a / check_b -- the code that scored §13)")
print("-" * 78)
print(f"  instances solved                 {len(optimal) - len(status_bad)}")
print(f"  non-OPTIMAL re-solves            {len(status_bad)}")
print(f"  ObjVal disagreements vs CSV      {len(solve_mismatch)}")
print(f"\n  Check A variable rows evaluated  {n_var_rows}")
print(f"    vacuous (zero-row instance)    {n_vac_instance}")
print(f"    vacuous (variable in no row)   {n_vac_var}")
print(f"    real                           {n_real_rows}")
print(f"    FIRED                          {len(a_fired_rows)}")
print(f"\n  Check B instances evaluated      {len(optimal) - len(status_bad)}")
print(f"    FIRED                          {len(b_fired_inst)}")

bad = bool(a_fired_rows or b_fired_inst or solve_mismatch or status_bad)
if bad:
    print("\n" + "!" * 78)
    print("FALSE POSITIVE OR SOLVE DIVERGENCE -- §13 IS SUPERSEDED, STOP")
    print("!" * 78)
    for r in a_fired_rows[:20]:
        print(f"  A FIRED  {r}")
    for r in b_fired_inst[:20]:
        print(f"  B FIRED  {r}")
    for r in solve_mismatch[:20]:
        print(f"  OBJVAL   {r}")
    for r in status_bad[:20]:
        print(f"  STATUS   {r}")
    sys.exit(1)

print("\n  -> 0 fires on Check A, 0 fires on Check B, 0 solve divergences")
print(f"  -> reproduces the reported convention: {n_real_rows} / {n_var_rows} real "
      f"Check A rows, 0 fired; {len(optimal)} Check B instances, 0 fired")

EXPECTED = {"var_rows": 2146, "real_rows": 2141, "vac_instance": 2, "vac_var": 3,
            "instances": 1040}
got = {"var_rows": n_var_rows, "real_rows": n_real_rows,
       "vac_instance": n_vac_instance, "vac_var": n_vac_var,
       "instances": len(optimal)}
print("\n" + "-" * 78)
print("AGAINST THE HISTORICAL REPORTING CONVENTION")
print("-" * 78)
for k, v in EXPECTED.items():
    flag = "OK" if got[k] == v else "MISMATCH"
    print(f"  {k:<14} expected {v:<6} got {got[k]:<6} {flag}")
if got != EXPECTED:
    raise SystemExit("structure differs from the reported convention -- stopping")
print("\n  -> identical to every prior run")
