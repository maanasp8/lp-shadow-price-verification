#!/usr/bin/env python
"""Reconstruct the 9 objective + 13 row-term decomposition of the 22
variable-attached numeric cells in the extraction pilot.

Read-only. Reimplements no matching: variable and row alignment are taken as
recorded in `extraction_pilot/scores.json` (`variable_alignment.matched_keys`,
`row_matching.pairs`); cell values and manual-review routing are taken from
`extraction_pilot/ground_truth.json`. The extractor is NOT re-run.

    venv/bin/python reconstruct_22_cells.py

Writes `extraction_22_cell_split_2026-08-18.{csv,json}`.
"""
import csv, json

sc = json.load(open("data/extraction_pilot/scores.json"))
gt = json.load(open("data/extraction_pilot/ground_truth.json"))

cells = []
for rec in sc["per_instance"]:
    pid = rec["problem_id"]
    g = gt[pid]
    matched_vars = set(rec["variable_alignment"]["matched_keys"])
    matched_rows = {int(k) for k in rec["row_matching"]["pairs"]}

    for t in g["objective_terms"]:
        if t.get("non_numeric_raw") or t.get("value") is None:
            continue
        if t["var_index"] in matched_vars:
            cells.append(dict(problem_id=pid, cell_class="objective_coefficient",
                              row="(objective)", variable=t["variable"],
                              var_index=t["var_index"], raw=t["raw"], value=t["value"]))

    for i, c in enumerate(g["constraints"]):
        if i not in matched_rows:
            continue
        for t in (c.get("declared_terms") or []):
            if t.get("non_numeric_raw") or t.get("value") is None:
                continue
            if t["var_index"] in matched_vars:
                cells.append(dict(problem_id=pid, cell_class="row_term",
                                  row=c["name"], variable=t["variable"],
                                  var_index=t["var_index"], raw=t["raw"], value=t["value"]))

n_obj = sum(1 for c in cells if c["cell_class"] == "objective_coefficient")
n_row = sum(1 for c in cells if c["cell_class"] == "row_term")

with open("data/extraction_22_cell_split_2026-08-18.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
    w.writeheader(); w.writerows(cells)

json.dump(dict(
    generated="2026-08-18",
    method="derived from scores.json alignment + ground_truth.json values; "
           "no matching reimplemented, extractor not re-run",
    objective_coefficients=n_obj, row_terms=n_row, total=len(cells),
    instances_contributing=sorted({c["problem_id"] for c in cells}),
    variables_resolved=len({(c["problem_id"], c["var_index"]) for c in cells}),
    note="These 22 cells belong to the 9 order_mapping-resolved variables. The 54 "
         "unresolved variables carry the separate 124 content-gap cells. The "
         "remaining numeric-scorable cells are 54 row limits, which have no "
         "variable referent: 22 + 54 = 76 = scores.json numeric_correct + numeric_wrong.",
    cells=cells,
), open("data/extraction_22_cell_split_2026-08-18.json", "w"), indent=1)

print(f"objective coefficients {n_obj}   row terms {n_row}   total {len(cells)}")
print(f"instances contributing {sorted({c['problem_id'] for c in cells})}")
print(f"distinct variables     {len({(c['problem_id'], c['var_index']) for c in cells})}")
print("wrote extraction_22_cell_split_2026-08-18.{csv,json}")
