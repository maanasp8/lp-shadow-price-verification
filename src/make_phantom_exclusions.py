"""Derive phantom_variable_exclusions.csv from the corpus + constraints.csv.

A phantom variable is an entry in an instance's order_mapping that is not a
decision variable of the problem as written. Two detected signatures:

  numeric_order_mapping_key -- the key is a bare number lifted from the text
                               (the NER layer labels it LIMIT, not VAR)
  orphan_variable           -- the index is referenced by no declaration

"Live" means the phantom carries a nonzero coefficient in at least one row, so a
mutation could move it off zero. Only live phantoms are excluded from mutation.

Additive: writes one new file, reads everything else read-only.
"""

import csv
import json
import re
from collections import defaultdict

NUM = re.compile(r"^\s*[\$]?[\d][\d,]*\.?\d*\s*%?\s*$")
ZERO = 1e-12
DATA = [f"nl4opt-competition/generation_data/{s}.jsonl" for s in ("train", "dev", "test")]

optimal = {r["problem_id"] for r in csv.DictReader(open("data/results.csv"))
           if r["solve_status"] == "OPTIMAL"}
objective_coefs = {r["problem_id"]: json.loads(r["coefficients"])
                   for r in csv.DictReader(open("data/objectives.csv"))
                   if r["problem_id"] in optimal}
cons = defaultdict(list)
for r in csv.DictReader(open("data/constraints.csv")):
    if r["problem_id"] in optimal:
        r["coefs"] = json.loads(r["coefficients"])
        cons[r["problem_id"]].append(r)


def used_mentions(inst):
    u = []
    od = inst["obj_declaration"]
    u += list(od.get("terms", {}))
    u += list(od.get("vars", []))
    for d in inst["const_declarations"]:
        u += list(d.get("terms", {}))
        for k in ("var", "x_var", "y_var"):
            if k in d:
                u.append(d[k])
    return u


rows = []
for path in DATA:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        for pid, inst in json.loads(line).items():
            if pid not in optimal:
                continue
            om, doc = inst["order_mapping"], inst["document"]
            labels = defaultdict(set)
            for s in inst.get("spans", []):
                t = s.get("text")
                if t is None:                      # recover from offsets
                    t = doc[s["start"]:s["end"]]
                labels[t].add(s["label"])
            referenced = {om[m] for m in used_mentions(inst) if m in om}

            # one row per phantom INDEX, not per mention: several mentions can
            # share an index, and only the canonical one keys the coefficients
            by_index = defaultdict(list)
            for key, idx in om.items():
                by_index[idx].append(key)
            canonical_names = set(objective_coefs[pid])

            for idx, mentions in sorted(by_index.items()):
                sigs = []
                if any(NUM.match(m) for m in mentions):
                    sigs.append("numeric_order_mapping_key")
                if idx not in referenced:
                    sigs.append("orphan_variable")
                if not sigs:
                    continue
                canon = next((m for m in mentions if m in canonical_names), mentions[0])
                aliases = sorted(m for m in mentions if m != canon)
                live = [(c["constraint_name"], c["ctype"], c["coefs"][canon], float(c["pi"]))
                        for c in cons[pid]
                        if canon in c["coefs"] and abs(c["coefs"][canon]) > ZERO]
                rows.append({
                    "problem_id": pid,
                    "phantom_name": canon,
                    "aliases": "|".join(aliases),
                    "signature": "+".join(sigs),
                    "span_labels": "|".join(sorted(
                        set().union(*(labels.get(m, set()) for m in mentions)))) or "<none>",
                    "coefficients_live": bool(live),
                    "n_live_rows": len(live),
                    "live_detail": json.dumps(
                        [{"row": a, "ctype": b, "coef": c, "pi": d} for a, b, c, d in live]),
                    "excluded_from_mutation": bool(live),
                })

rows.sort(key=lambda r: (r["problem_id"], r["phantom_name"]))
COLS = ["problem_id", "phantom_name", "aliases", "signature", "span_labels",
        "coefficients_live", "n_live_rows", "live_detail", "excluded_from_mutation"]
with open("data/phantom_variable_exclusions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

print("=" * 78)
print("data/phantom_variable_exclusions.csv")
print("=" * 78)
print(f"  phantom variables found      {len(rows)}")
print(f"  distinct instances           {len({r['problem_id'] for r in rows})}")
excl = sorted({r["problem_id"] for r in rows if r["excluded_from_mutation"]})
print(f"  EXCLUDED from mutation       {len(excl)}  {excl}")
print(f"  retained (inert phantoms)    "
      f"{len({r['problem_id'] for r in rows}) - len(excl)}")
print()
for r in rows:
    print(f"  {r['problem_id']:>13}  {r['phantom_name']:<16} "
          f"aliases={r['aliases'] or '-':<12} {r['signature']:<42} "
          f"spans={r['span_labels']:<10} live={r['coefficients_live']!s:<5} "
          f"excluded={r['excluded_from_mutation']}")
