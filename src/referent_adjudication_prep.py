#!/usr/bin/env python
"""
Build a HUMAN review sheet for the 54 ground-truth variables that the frozen
scorer left unresolved in the extraction pilot.

READ-ONLY with respect to every frozen artefact. This script:
  * imports `extraction_score` and calls its matching functions unchanged;
  * never writes to extraction_score.py, the pre-registration, ground_truth.json,
    scores.json, or extraction_pilot/raw/;
  * emits two NEW files (a CSV and a markdown sheet) and nothing else.

It does NOT adjudicate. The `my_call` column is emitted blank by construction and
this script contains no rule that would fill it. Candidate pairing between a
ground-truth variable and an extracted name is MECHANICAL (substring containment,
or elimination where containment finds nothing) and is offered only so the
reviewer can see the two strings side by side. It is not a verdict on referent
correctness.

Outputs:
  referent_adjudication_sheet_2026-08-07.csv
  referent_adjudication_sheet_2026-08-07.md
"""

import csv
import json
import os
import re

import extraction_score as S  # imported, never modified

CORPUS = [f"nl4opt-competition/generation_data/{s}.jsonl"
          for s in ("train", "dev", "test")]
CSV_OUT = "data/referent_adjudication_sheet_2026-08-07.csv"
MD_OUT = "docs/referent_adjudication_sheet_2026-08-07.md"

# The five patterns named in the 2026-08-07 diagnostic, plus "other".
INVESTMENT = re.compile(r"^(amount invested in|money put on)$")


def load_corpus():
    out = {}
    for path in CORPUS:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            for pid, inst in json.loads(line).items():
                if isinstance(inst, dict) and "order_mapping" in inst:
                    out[str(pid)] = inst
    return out


def wrapper_category(key, extracted):
    """Classify the extracted name against the annotation key. Mirrors the
    2026-08-07 diagnostic exactly. Returns (category, prefix, suffix)."""
    if key not in extracted:
        return "head-noun substitution", "", ""
    i = extracted.index(key)
    pre = extracted[:i].strip()
    post = extracted[i + len(key):].strip()
    if INVESTMENT.match(pre):
        return "investment", pre, post
    if pre.startswith("number of"):
        return "count", pre, post
    if post:
        return "usage/temporal", pre, post
    if pre:
        return "unit-of-measure", pre, post
    return "other", pre, post


def rows_touching(gt, idx, varname):
    """Every constraint that references variable `idx`, rendered readably."""
    out = []
    for r in gt["constraints"]:
        terms = {t["variable"]: t for t in r["declared_terms"]}
        hit = (any(t.get("var_index") == idx for t in r["declared_terms"])
               or varname in (r.get("canonical_net_coefficients") or {})
               or varname in (r.get("subject_var"), r.get("x_var"), r.get("y_var")))
        if not hit:
            continue
        lim = r.get("declared_limit")
        limtxt = f"{lim['raw']}" if lim else (
            r["declared_param"]["raw"] if r.get("declared_param") else "-")
        if r["declared_terms"]:
            decl = ", ".join(f"{t['variable']}={t['raw']}" for t in r["declared_terms"])
        else:
            decl = "(no per-variable coefficients declared)"
        canon = r.get("canonical_net_coefficients") or {}
        mine = terms.get(varname)
        out.append(
            f"[{r['name']} · {r['ctype']}] declared: {decl} {r['operator']} {limtxt}"
            + (f"  | this variable's declared cell: {mine['raw']}" if mine else "")
            + (f"  | implicit-structural row (canonical {canon})"
               if r.get("implicit_structural_coefficient") else "")
        )
    return out


def main():
    gt_all = json.load(open(S.GT_PATH))
    corpus = load_corpus()
    records = []

    for pid, gt in gt_all.items():
        rec = json.load(open(os.path.join(S.RAW_DIR, f"{pid}.json")))
        ex = rec.get("extraction") if rec.get("ok") else None
        alias = S.alias_map(gt)
        ex_names = [v for v in (ex.get("variables") or [])]
        va = S.align_variables(gt["variables"], ex_names, alias)
        if not va["gt_unmatched"]:
            continue

        idx_keys = {}
        for k, i in alias.items():
            idx_keys.setdefault(i, []).append(k)

        # extracted names already consumed by a containment pairing
        taken = set()
        pairing = {}
        for nm in va["gt_unmatched"]:
            i = gt["variables"].index(nm)
            cand = [e for e in ex_names
                    if any(k in S.norm(e) for k in idx_keys.get(i, []))
                    and S.norm(e) not in taken]
            if cand:
                pairing[nm] = (cand[0], "containment")
                taken.add(S.norm(cand[0]))
        # anything left over pairs by elimination
        for nm in va["gt_unmatched"]:
            if nm in pairing:
                continue
            left = [e for e in ex_names if S.norm(e) not in taken
                    and S.norm(e) not in {S.norm(x) for x in va["matched"].values()}]
            pairing[nm] = ((left[0], "elimination") if left else ("(none)", "none"))
            if left:
                taken.add(S.norm(left[0]))

        for nm in va["gt_unmatched"]:
            i = gt["variables"].index(nm)
            extracted, how = pairing[nm]
            keys = sorted(idx_keys.get(i, []))
            best = next((k for k in sorted(keys, key=len, reverse=True)
                         if k in S.norm(extracted)), keys[0] if keys else "")
            cat, pre, post = wrapper_category(best, S.norm(extracted))

            n_obj = sum(1 for t in gt["objective_terms"]
                        if t.get("var_index") == i and not t["non_numeric_raw"]
                        and t["value"] is not None)
            n_row = sum(1 for r in gt["constraints"] for t in r["declared_terms"]
                        if t.get("var_index") == i and not t["non_numeric_raw"])

            implicit_obj = gt.get("objective_coefficients_implicit")

            def show(t):
                if t["raw"] is None and implicit_obj:
                    return f"{t['variable']}=(implicit 1)"
                return f"{t['variable']}={t['raw']}"

            obj = [t for t in gt["objective_terms"] if t.get("var_index") == i]
            objline = f"{gt['sense']}  " + ", ".join(
                show(t) for t in gt["objective_terms"])
            if implicit_obj:
                objline += (f"   [objective_type={gt['objective_type']}; no "
                            f"per-variable coefficients are declared in the text — "
                            f"every variable enters with coefficient 1]")

            records.append({
                "instance_id": pid,
                "gt_variable": nm,
                "order_mapping_keys": " | ".join(keys),
                "matched_key_used": best,
                "extracted_name": extracted,
                "pairing_basis": how,
                "wrapper_category": cat,
                "wrapper_prefix": pre,
                "wrapper_suffix": post,
                "cells_objective_coefficient": n_obj,
                "cells_row_term": n_row,
                "objective_line": objline,
                "this_variable_objective_cell": (show(obj[0]) .split("=", 1)[1]
                                                 if obj else "-"),
                "constraint_lines": "\n".join(rows_touching(gt, i, nm)) or "-",
                "problem_text": (corpus.get(pid, {}).get("document")
                                 or gt.get("document") or ""),
                "my_call": "",
                "notes": "",
            })

    order = {"investment": 0, "unit-of-measure": 1, "usage/temporal": 2,
             "count": 3, "other": 4, "head-noun substitution": 5}
    records.sort(key=lambda r: (order.get(r["wrapper_category"], 9),
                                r["instance_id"], r["gt_variable"]))

    cols = list(records[0].keys())
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    with open(MD_OUT, "w") as fh:
        fh.write(write_md(records))

    print(f"  cases written              {len(records)}")
    print(f"  objective-coefficient cells under them  "
          f"{sum(r['cells_objective_coefficient'] for r in records)}")
    print(f"  row-term cells under them               "
          f"{sum(r['cells_row_term'] for r in records)}")
    by = {}
    for r in records:
        by[r["wrapper_category"]] = by.get(r["wrapper_category"], 0) + 1
    for k in sorted(by, key=lambda x: order.get(x, 9)):
        print(f"    {k:26} {by[k]}")
    print(f"  wrote {CSV_OUT}, {MD_OUT}")


def write_md(records):
    L = []
    A = L.append
    A("# Referent adjudication sheet — 54 unresolved variables\n")
    A("Extraction pilot, 25-instance sample. Generated 2026-08-07 by "
      "`referent_adjudication_prep.py`.\n")
    A("**This sheet contains no adjudications.** Every `My call` field is blank "
      "and the generating script has no rule that would fill one. The "
      "ground-truth key / extracted name pairing is mechanical (substring "
      "containment, or elimination where containment finds nothing) and is not "
      "a judgment about referent correctness.\n")
    A("**Scope of what a call here can move.** Same asymmetry already on record "
      "for the `-114149479` numeric adjudication: a call can only ever move the "
      "referent-accuracy figure in the extractor's favour, never against it. "
      "The as-measured figure (9/63 resolved) stands unaltered in `scores.json` "
      "either way.\n")
    A("\n---\n")

    A("\n## Summary\n")
    A("| # | instance | ground-truth variable | extracted name | category | cells (obj/row) | my call |")
    A("|---|---|---|---|---|---|---|")
    for n, r in enumerate(records, 1):
        A(f"| {n} | `{r['instance_id']}` | `{r['gt_variable']}` | "
          f"`{r['extracted_name']}` | {r['wrapper_category']} | "
          f"{r['cells_objective_coefficient']}/{r['cells_row_term']} |  |")

    cur = None
    for n, r in enumerate(records, 1):
        if r["wrapper_category"] != cur:
            cur = r["wrapper_category"]
            A(f"\n---\n\n## Category: {cur}\n")
            if cur == "head-noun substitution":
                A("> **Flagged separately.** Containment finds no annotation key "
                  "inside the extracted name; the extractor replaced the head "
                  "noun. The pairing below is by elimination, not by any string "
                  "relation. This is the case where the referent question is "
                  "least mechanical.\n")
        A(f"\n### Case {n} — `{r['instance_id']}` · ground-truth variable "
          f"`{r['gt_variable']}`\n")
        A(f"- **`order_mapping` keys for this variable:** `{r['order_mapping_keys']}`")
        A(f"- **Extracted name:** `{r['extracted_name']}`  "
          f"*(paired by {r['pairing_basis']})*")
        A(f"- **Wrapper category:** {r['wrapper_category']}"
          + (f"  — prefix `{r['wrapper_prefix']}`" if r["wrapper_prefix"] else "")
          + (f", suffix `{r['wrapper_suffix']}`" if r["wrapper_suffix"] else ""))
        A(f"- **Cells under this variable:** "
          f"{r['cells_objective_coefficient']} objective coefficient, "
          f"{r['cells_row_term']} row term")
        A(f"\n**Objective line:** `{r['objective_line']}`  \n"
          f"**This variable's objective cell:** `{r['this_variable_objective_cell']}`\n")
        A("**Constraint rows referencing this variable:**\n")
        A("```")
        A(r["constraint_lines"])
        A("```\n")
        A("**Problem text:**\n")
        A("> " + r["problem_text"].replace("\n", "\n> ") + "\n")
        A("**My call:**\n")
        A("**Notes:**\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
