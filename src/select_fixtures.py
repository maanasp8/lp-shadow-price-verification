"""Step 2 selection + Step 3 raw dump, driven by constraints.csv / objectives.csv.

No Check A, no Check B, no verdicts. Every printed number is either read from a
CSV or read from the annotation.
"""

import csv
import json
from collections import defaultdict

from nl4opt_parser import load_corpus

TOL = 1e-6
DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]

results = {r["problem_id"]: r for r in csv.DictReader(open("data/results.csv"))}
objectives = {r["problem_id"]: r for r in csv.DictReader(open("data/objectives.csv"))}
suspicious = {r["problem_id"] for r in csv.DictReader(open("data/suspicious.csv"))}

cons_by_pid = defaultdict(list)
for r in csv.DictReader(open("data/constraints.csv")):
    r["pi"] = float(r["pi"]); r["slack"] = float(r["slack"])
    r["rhs_limit"] = float(r["rhs_limit"])
    r["binding"] = r["binding"] == "True"
    r["n_vars_in_row"] = int(r["n_vars_in_row"])
    r["n_vars_nonzero"] = int(r["n_vars_nonzero"])
    r["coefs"] = json.loads(r["coefficients"])
    cons_by_pid[r["problem_id"]].append(r)

vars_by_pid = defaultdict(list)
for r in csv.DictReader(open("data/variables.csv")):
    r["value"] = float(r["value"])
    vars_by_pid[r["problem_id"]].append(r)

annot = {pid: inst for _, pid, inst in load_corpus(DATA)}

# ---- exclusions applied to every slot ------------------------------------
EXCL = {}
eligible = []
for pid, r in results.items():
    if r["solve_status"] != "OPTIMAL":
        EXCL[pid] = f"solve_status={r['solve_status']}"
    elif int(r["n_degenerate"]) > 0:
        EXCL[pid] = f"has {r['n_degenerate']} degenerate row(s)"
    elif pid in suspicious:
        EXCL[pid] = "listed in suspicious.csv"
    else:
        eligible.append(pid)
print(f"eligible after exclusions: {len(eligible)} of {len(results)}")


def all_values(pid):
    vals = [abs(v) for c in cons_by_pid[pid] for v in c["coefs"].values()]
    vals += [abs(c["rhs_limit"]) for c in cons_by_pid[pid]]
    vals += [abs(v) for v in json.loads(objectives[pid]["coefficients"]).values()]
    return vals


def prefer(pid):
    """Small integer coefficients and limits first; deterministic tie-break."""
    vals = all_values(pid)
    all_int = all(v == int(v) for v in vals)
    return (not all_int, max(vals) if vals else 0, pid)


def ctypes(pid):
    return sorted({c["ctype"] for c in cons_by_pid[pid]})


# ---- predicates ----------------------------------------------------------
def p1(pid):   # MIN with >=1 binding GE row
    if objectives[pid]["sense"] != "min":
        return None
    hits = [c for c in cons_by_pid[pid] if c["operator"] == ">=" and c["binding"]]
    return hits or None


def p2(pid):   # MAX with a binding upperbound row
    if objectives[pid]["sense"] != "max":
        return None
    hits = [c for c in cons_by_pid[pid] if c["ctype"] == "upperbound" and c["binding"]]
    return hits or None


def p3(pid):   # product at zero OR on a binding lowerbound row
    hits = [v for v in vars_by_pid[pid]
            if v["value"] == 0.0 or "lowerbound" in v["restricted_by"]]
    return hits or None


def p4(pid):   # structural: n_vars_in_row>1 AND |Pi|>TOL AND a negative coefficient
    if int(results[pid]["n_usable"]) != 2:
        return None
    hits = [c for c in cons_by_pid[pid]
            if c["n_vars_in_row"] > 1 and abs(c["pi"]) > TOL
            and any(v < 0 for v in c["coefs"].values())]
    return hits or None


def p5(pid):   # n_usable==1, MAX, a FREE product, all coefficients positive
    if int(results[pid]["n_usable"]) != 1 or objectives[pid]["sense"] != "max":
        return None
    if not any(v["label"] == "FREE" and v["value"] != 0.0 for v in vars_by_pid[pid]):
        return None
    vals = [v for c in cons_by_pid[pid] for v in c["coefs"].values()]
    vals += list(json.loads(objectives[pid]["coefficients"]).values())
    if any(v <= 0 for v in vals):
        return None
    return [v for v in vars_by_pid[pid] if v["label"] == "FREE"]


SLOTS = [
    ("1", "MIN + >=1 binding GE row", p1),
    ("2", "MAX + binding upperbound row", p2),
    ("3", "product at zero OR on binding lowerbound (prefer n_usable==1)", p3),
    ("4", "n_usable==2 + n_vars_in_row>1 + |Pi|>1e-6 + negative coefficient", p4),
    ("5", "n_usable==1, MAX, FREE product, all coefficients positive", p5),
]

picked, used = {}, set()
print("\n" + "=" * 74)
print("STEP 2: SELECTION")
print("=" * 74)
for slot, desc, pred in SLOTS:
    cands = [pid for pid in eligible if pid not in used and pred(pid)]
    if slot == "3":   # stated preference
        pref = [p for p in cands if int(results[p]["n_usable"]) == 1]
        cands = pref or cands
    if slot in ("3", "5"):   # spread types across slots 2/3/5
        used_types = {t for s in ("2", "3", "5") if s in picked
                      for t in ctypes(picked[s])}
        div = [p for p in cands if set(ctypes(p)) - used_types - {"linear"}]
        cands = div or cands
    if slot == "4":
        hint = [p for p in cands if {"xy", "xby"} & set(ctypes(p))]
        print(f"  slot 4 search hint: {len(hint)} of {len(cands)} structural "
              f"candidates contain an xy/xby row (hint only, filter is structural)")
    print(f"\n  slot {slot}: {desc}")
    print(f"    candidates after exclusions: {len(cands)}")
    if not cands:
        print("    NO MATCH")
        continue
    cands.sort(key=prefer)
    pid = cands[0]
    picked[slot] = pid
    used.add(pid)
    print(f"    selected {pid}   ctypes={ctypes(pid)}  "
          f"n_usable={results[pid]['n_usable']}  sense={objectives[pid]['sense']}  "
          f"max|value|={max(all_values(pid)):g}")
    print(f"    matched on:")
    for h in pred(pid)[:4]:
        if "constraint_name" in h:
            print(f"      row {h['constraint_name']:<14} {h['ctype']:<10} "
                  f"{h['operator']} {h['rhs_limit']:g}  Pi={h['pi']:.6g} "
                  f"binding={h['binding']}  coefs={h['coefs']}")
        else:
            print(f"      var {h['var_name']:<20} value={h['value']:g} "
                  f"label={h['label']} restricted_by={h['restricted_by']!r}")

types_235 = {s: ctypes(picked[s]) for s in ("2", "3", "5") if s in picked}
print(f"\n  slot 2/3/5 ctype spread: {types_235}")
print(f"  all three linear-only: "
      f"{all(v == ['linear'] for v in types_235.values())}")

# ==========================================================================
# STEP 3: raw dump
# ==========================================================================
print("\n" + "=" * 74)
print("STEP 3: RAW DUMP")
print("=" * 74)
fixtures = {}
for slot, _, _ in SLOTS:
    if slot not in picked:
        continue
    pid = picked[slot]
    inst, obj = annot[pid], objectives[pid]
    obj_coefs = json.loads(obj["coefficients"])
    vlist = vars_by_pid[pid]

    print("\n" + "-" * 74)
    print(f"SLOT {slot}   instance {pid}")
    print("-" * 74)
    print(f"  sense              {obj['sense']}")
    print(f"  n_variables        {len(vlist)}")
    print(f"  n_usable           {results[pid]['n_usable']}")
    print(f"  objective value    {obj['obj_value']}")

    print(f"\n  problem text:\n    {inst['document']}")
    print(f"\n  annotation:")
    print(f"    OBJ  {json.dumps(inst['obj_declaration'])}")
    for d in inst["const_declarations"]:
        print(f"    CON  {json.dumps(d)}")

    print(f"\n  constraints:")
    for c in cons_by_pid[pid]:
        print(f"    {c['constraint_name']:<15} {c['ctype']:<11} "
              f"coefs={c['coefs']}  {c['operator']} {c['rhs_limit']:g}")
        print(f"    {'':<15} {'':<11} Pi={c['pi']!r}  Slack={c['slack']!r}  "
              f"binding={c['binding']}")

    print(f"\n  variables:")
    for v in vlist:
        flags = []
        if v["value"] == 0.0:
            flags.append("at zero")
        if "(lower)" in v["restricted_by"]:
            flags.append("at binding floor")
        if "(upper)" in v["restricted_by"]:
            flags.append("at binding cap")
        if not flags and v["label"] == "FREE":
            flags.append("freely chosen")
        print(f"    {v['var_name']:<22} obj_coef={obj_coefs[v['var_name']]:<10g} "
              f"value={v['value']:<14g} {', '.join(flags)}"
              f"   [label={v['label']} restricted_by={v['restricted_by']!r}]")

    fixtures[f"slot_{slot}"] = {
        "problem_id": pid, "split": results[pid]["split"], "sense": obj["sense"],
        "objective_value": float(obj["obj_value"]),
        "n_usable": int(results[pid]["n_usable"]),
        "document": inst["document"],
        "obj_declaration": inst["obj_declaration"],
        "const_declarations": inst["const_declarations"],
        "objective_coefficients": obj_coefs,
        "constraints": [{k: c[k] for k in
                         ("constraint_name", "ctype", "n_vars_in_row",
                          "n_vars_nonzero", "operator", "rhs_limit", "pi",
                          "slack", "binding")} | {"coefficients": c["coefs"]}
                        for c in cons_by_pid[pid]],
        "variables": [{"var_name": v["var_name"], "value": v["value"],
                       "label": v["label"], "restricted_by": v["restricted_by"],
                       "objective_coefficient": obj_coefs[v["var_name"]]}
                      for v in vlist],
    }

with open("data/handcheck_fixtures.json", "w") as fh:
    json.dump(fixtures, fh, indent=2)
print(f"\n\nwrote handcheck_fixtures.json "
      f"({len(fixtures)} slots)")
