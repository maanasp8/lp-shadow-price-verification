"""Phase 3: solve every parsed model, classify constraints, write results.

Tolerances are exactly 1e-6 as specified. Every one of the 1101 instances gets
a row in results.csv -- parse failures included, marked PARSE_FAILED -- so the
qualifying fraction can never be read off a silently reduced corpus.
"""

import collections
import csv

import gurobipy as gp
from gurobipy import GRB

from nl4opt_measure import (SHARED, classify_constraints, classify_variables,
                            roll_up, row_scope, suspicious_values)
from nl4opt_parser import ParseError, build_model, load_corpus, parse_instance

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]

STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED", GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.NUMERIC: "NUMERIC",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT", GRB.TIME_LIMIT: "TIME_LIMIT",
}

RESULT_COLS = ["problem_id", "split", "solve_status", "n_constraints", "n_shared",
               "n_usable", "n_usable_shared", "n_usable_bound", "n_degenerate",
               "n_inactive", "n_unclassified", "qualifies", "suspicious"]

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

corpus = load_corpus(DATA)
total = len(corpus)
print(f"loaded {total} instances from {len(DATA)} files\n")

results, failures, variables, suspicious_rows = [], [], [], []
status_counts = collections.Counter()
usable_hist = collections.Counter()
unclassified_rows = []

for n, (split, pid, inst) in enumerate(corpus, 1):
    if n % 25 == 0 or n == total:
        print(f"  [{n}/{total}] processed")

    # ---- parse ----
    try:
        prob = parse_instance(pid, inst, split)
    except ParseError as exc:
        status_counts["PARSE_FAILED"] += 1
        failures.append({"problem_id": pid, "split": split, "stage": "parse",
                         "reason": str(exc)})
        results.append({"problem_id": pid, "split": split,
                        "solve_status": "PARSE_FAILED", "n_constraints": "",
                        "n_shared": "", "n_usable": "", "n_usable_shared": "",
                        "n_usable_bound": "", "n_degenerate": "", "n_inactive": "",
                        "n_unclassified": "", "qualifies": "", "suspicious": ""})
        continue
    except Exception as exc:  # noqa: BLE001 - never let one instance vanish
        status_counts["PARSE_FAILED"] += 1
        failures.append({"problem_id": pid, "split": split, "stage": "parse",
                         "reason": f"UNEXPECTED {type(exc).__name__}: {exc}"})
        results.append({"problem_id": pid, "split": split,
                        "solve_status": "PARSE_FAILED", "n_constraints": "",
                        "n_shared": "", "n_usable": "", "n_usable_shared": "",
                        "n_usable_bound": "", "n_degenerate": "", "n_inactive": "",
                        "n_unclassified": "", "qualifies": "", "suspicious": ""})
        continue

    n_shared = sum(1 for c in prob.constraints if row_scope(c) == SHARED)
    offenders = suspicious_values(prob)

    # ---- solve ----
    m = build_model(prob, env=env)
    m.setParam("DualReductions", 0)   # never collapse infeasible/unbounded
    m.optimize()
    status = STATUS_NAMES.get(m.Status, f"STATUS_{m.Status}")
    status_counts[status] += 1

    if offenders:
        suspicious_rows.append({
            "problem_id": pid, "split": split, "solve_status": status,
            "n_offending": len(offenders),
            "max_abs_value": f"{max(abs(v) for v in offenders):.6g}",
            "values": "; ".join(f"{v:g}" for v in sorted(offenders, key=abs, reverse=True)),
        })

    if status != "OPTIMAL":
        # no duals exist -- unsolvable for this measurement
        failures.append({"problem_id": pid, "split": split, "stage": "solve",
                         "reason": status})
        results.append({"problem_id": pid, "split": split, "solve_status": status,
                        "n_constraints": len(prob.constraints), "n_shared": n_shared,
                        "n_usable": "", "n_usable_shared": "", "n_usable_bound": "",
                        "n_degenerate": "", "n_inactive": "", "n_unclassified": "",
                        "qualifies": "", "suspicious": bool(offenders)})
        m.dispose()
        continue

    # ---- classify ----
    cons = classify_constraints(prob, m)
    r = roll_up(prob, status, cons)
    usable_hist[r.n_usable] += 1

    for c in cons:
        if c.label == "UNCLASSIFIED":
            unclassified_rows.append((pid, c.name, c.pi, c.slack))

    for v in classify_variables(prob, m):
        why = []
        if v.at_natural_lb:
            why.append("own_lb")
        if v.at_natural_ub:
            why.append("own_ub")
        why += [f"{rn}(lower)" for rn in v.lower_rows]
        why += [f"{rn}(upper)" for rn in v.upper_rows]
        variables.append({"problem_id": pid, "split": split, "var_name": v.name,
                          "value": f"{v.value:.10g}", "label": v.label,
                          "restricted_by": ",".join(why)})

    results.append({"problem_id": pid, "split": split, "solve_status": status,
                    "n_constraints": r.n_constraints, "n_shared": n_shared,
                    "n_usable": r.n_usable, "n_usable_shared": r.n_usable_shared,
                    "n_usable_bound": r.n_usable_bound,
                    "n_degenerate": r.n_degenerate, "n_inactive": r.n_inactive,
                    "n_unclassified": r.n_unclassified,
                    "qualifies": r.qualifies, "suspicious": bool(offenders)})
    m.dispose()


def write(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


write("data/results.csv", results, RESULT_COLS)
write("data/failures.csv", failures, ["problem_id", "split", "stage", "reason"])
write("data/variables.csv", variables,
      ["problem_id", "split", "var_name", "value", "label", "restricted_by"])
write("data/suspicious.csv", suspicious_rows,
      ["problem_id", "split", "solve_status", "n_offending", "max_abs_value", "values"])

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
solved = [r for r in results if r["solve_status"] == "OPTIMAL"]
n_parsed = total - status_counts["PARSE_FAILED"]
n_solved = len(solved)
qualifying = [r for r in solved if r["qualifies"]]
susp_total = sum(1 for r in results if r["suspicious"] is True)


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "n/a"


print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)
print(f"total instances found            {total}")
print(f"  parsed successfully            {n_parsed}")
print(f"  failed to parse                {status_counts['PARSE_FAILED']}")
print("\nsolve status (of parsed):")
for name in ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"):
    print(f"  {name:<18} {status_counts[name]}")
other = {k: v for k, v in status_counts.items()
         if k not in ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD", "PARSE_FAILED")}
print(f"  {'other':<18} {sum(other.values())}" + (f"  {other}" if other else ""))

print(f"\nQUALIFYING (>=1 USABLE constraint)")
print(f"  count                          {len(qualifying)}")
print(f"  fraction of solved             {pct(len(qualifying), n_solved)}  "
      f"({len(qualifying)}/{n_solved})")
print(f"  fraction of all instances      {pct(len(qualifying), total)}  "
      f"({len(qualifying)}/{total})")

tot_usable = sum(r["n_usable"] for r in solved)
tot_degen = sum(r["n_degenerate"] for r in solved)
tot_inact = sum(r["n_inactive"] for r in solved)
print("\nconstraint rows by label (must sum to total rows):")
print(f"  USABLE                         {tot_usable}")
print(f"  DEGENERATE                     {tot_degen}   "
      f"(in {sum(1 for r in solved if r['n_degenerate'] > 0)} instances)")
print(f"  INACTIVE                       {tot_inact}")
print(f"  sum / total rows               {tot_usable + tot_degen + tot_inact}"
      f" / {sum(r['n_constraints'] for r in solved)}")

print("\ndistribution of n_usable (solved instances):")
for k in (0, 1, 2):
    print(f"  exactly {k}                      {usable_hist[k]}")
print(f"  3 or more                      {sum(v for k, v in usable_hist.items() if k >= 3)}")

usable_shared = sum(r["n_usable_shared"] for r in solved)
usable_bound = sum(r["n_usable_bound"] for r in solved)
n_shared_only = sum(1 for r in solved if r["n_usable_shared"] >= 1)
print("\nusable constraints by scope (the split you asked for):")
print(f"  USABLE rows, shared-resource   {usable_shared}")
print(f"  USABLE rows, single-var bound  {usable_bound}")
print(f"  problems with >=1 USABLE SHARED {n_shared_only}  "
      f"({pct(n_shared_only, n_solved)} of solved)")
tot_shared = sum(r["n_shared"] for r in solved)
tot_rows = sum(r["n_constraints"] for r in solved)
print(f"  shared rows / all rows         {tot_shared}/{tot_rows}")

print(f"\nsuspicious (|value| > 1e4)        {susp_total}")
susp_solved = [r for r in solved if r["suspicious"] is True]
susp_qual = sum(1 for r in susp_solved if r["qualifies"])
print(f"  of which solved to optimality  {len(susp_solved)}")
print(f"  qualifying EXCLUDING suspicious {len(qualifying) - susp_qual}/"
      f"{n_solved - len(susp_solved)}  "
      f"({pct(len(qualifying) - susp_qual, n_solved - len(susp_solved))})")

print(f"\nUNCLASSIFIED rows (tolerance boundary)  {len(unclassified_rows)}")
for row in unclassified_rows[:10]:
    print(f"    {row}")

print(f"\nfailures.csv rows                {len(failures)}  "
      f"(parse {sum(1 for f in failures if f['stage'] == 'parse')}, "
      f"solve {sum(1 for f in failures if f['stage'] == 'solve')})")
print("wrote results.csv, failures.csv, variables.csv, suspicious.csv")
