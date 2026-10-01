"""EXTRACTION PILOT — step 2: structured ground truth per sampled instance.

    venv/bin/python extraction_ground_truth.py [--write]

Writes `extraction_pilot/ground_truth.json`.

WHAT THE GROUND TRUTH IS
------------------------
The **declared annotation**, not the compiled model. The extractor's job is slot
extraction from the problem text — reading numbers off the page and attaching
them to the right variable and row — not formulation. So ground truth records
what the annotation *declares*, with the raw string preserved beside every
parsed value.

Canonical net coefficients (what `build_model` actually constructs) are recorded
alongside for reference, but are NOT the scoring target: reproducing them would
require the extractor to apply declaration semantics — `ratio` becomes
`(1-limit)` for the subject variable and `-limit` for every other — which is
exactly the autoformulation the prompt forbids.

TWO CONSEQUENCES, both logged rather than silently absorbed
-----------------------------------------------------------
1. **Non-numeric raw parameters are KEPT, never cleaned.** Percentages ("30%"),
   comma-formatted ("1,000"), word-numbers ("third", "twice", "half"),
   multipliers ("3 times"), currency ("$2") and unit-carrying strings
   ("8000 minutes") are real stress cases for the numeric-accuracy axis. Each is
   flagged `non_numeric_raw: true` with both `raw` and the parser's `value`, and
   routes to the scorer's MANUAL-REVIEW bucket — never an automatic pass or fail.

2. **Rows whose coefficients are structural, not declared** — `sum`, `upperbound`,
   `lowerbound`, `ratio`, `xby`, `xy` — carry no per-variable coefficient in the
   annotation. Their coefficient cells are marked
   `implicit_structural_coefficient: true` and route to the same MANUAL-REVIEW
   bucket. This is an interpretive call: the alternative (scoring the extractor
   against derived canonical coefficients) would penalise it for not
   autoformulating, which the prompt explicitly forbids.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

from gurobipy import GRB

from nl4opt_parser import parse_number, load_corpus, parse_instance
from text_terms import row_coefficients

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl"
        for s in ("train", "dev", "test")]
OUTDIR = "data/extraction_pilot"
SAMPLE_PATH = os.path.join(OUTDIR, "sample.json")
GT_PATH = os.path.join(OUTDIR, "ground_truth.json")

# Rows whose per-variable coefficients come from declaration semantics rather
# than from a number written in the text.
IMPLICIT_COEF_TYPES = {"sum", "upperbound", "lowerbound", "ratio", "xby", "xy"}


def classify_raw(raw) -> dict:
    """-> {raw, value, non_numeric_raw, reason}. Never cleans the raw string."""
    s = "" if raw is None else str(raw)
    try:
        value = parse_number(s)
    except Exception:                                            # noqa: BLE001
        value = None
    plain = False
    try:
        float(s.strip())
        plain = True
    except (ValueError, AttributeError):
        plain = False
    reason = None
    if not plain:
        t = s.strip().lower()
        if "%" in t or "percent" in t:
            reason = "percentage"
        elif "," in t:
            reason = "comma_formatted"
        elif re.search(r"\btimes\b", t):
            reason = "multiplier_phrase"
        elif re.search(r"[$¢£€]", t):
            reason = "currency_mark"
        elif re.search(r"\d", t):
            reason = "digits_with_units_or_punctuation"
        elif t:
            reason = "word_number"
        else:
            reason = "empty"
    return {"raw": s, "value": value, "non_numeric_raw": not plain,
            "reason": reason}


def var_index(mention: str, order: Dict[str, int]) -> Optional[int]:
    """Resolve a surface mention to its variable INDEX via the corpus's own
    `order_mapping` — the same table `nl4opt_parser._resolve` uses to build the
    model. An instance may refer to one variable by several names ('Standard'
    and 'Standard grade alloy' both -> 0); the index is the referent, the name
    is only a label. Returns None if the mention is not in the table at all.
    """
    return order.get(mention)


def declared_terms(decl: dict, ctype: str, order: Dict[str, int]) -> List[dict]:
    """Per-variable coefficients AS DECLARED. Empty for implicit-coef types."""
    if ctype != "linear":
        return []
    return [{"variable": var, "var_index": var_index(var, order), **classify_raw(raw)}
            for var, raw in (decl.get("terms") or {}).items()]


def declared_limit(decl: dict, ctype: str) -> Optional[dict]:
    if ctype in ("linear", "sum", "upperbound", "lowerbound", "ratio"):
        return classify_raw(decl.get("limit"))
    return None            # xby/xy carry a `param` or nothing, handled below


def declared_param(decl: dict, ctype: str) -> Optional[dict]:
    if ctype == "xby":
        return classify_raw(decl.get("param"))
    return None


def build_one(pid: str, split: str, inst: dict) -> dict:
    prob = parse_instance(pid, inst, split)
    order: Dict[str, int] = inst["order_mapping"]
    obj = inst["obj_declaration"]
    otype = obj.get("type")

    if otype == "objective":
        obj_terms = [{"variable": v, "var_index": var_index(v, order),
                      **classify_raw(raw)}
                     for v, raw in (obj.get("terms") or {}).items()]
        obj_implicit = False
    else:                                   # objvar -> every listed var, coef 1
        obj_terms = [{"variable": v, "var_index": var_index(v, order),
                      "raw": None, "value": 1.0,
                      "non_numeric_raw": False, "reason": None}
                     for v in (obj.get("vars") or [])]
        obj_implicit = True

    rows = []
    for i, decl in enumerate(inst["const_declarations"]):
        ctype = decl.get("type")
        name = f"c{i}_{ctype}"
        con = prob.constraints[i]
        rows.append({
            "name": name,
            "ctype": ctype,
            "operator": con.op,
            "operator_raw": decl.get("operator"),
            "direction_text": decl.get("direction"),
            "subject_var": decl.get("var"),
            "x_var": decl.get("x_var"),
            "y_var": decl.get("y_var"),
            "declared_limit": declared_limit(decl, ctype),
            "declared_param": declared_param(decl, ctype),
            "declared_terms": declared_terms(decl, ctype, order),
            "implicit_structural_coefficient": ctype in IMPLICIT_COEF_TYPES,
            # reference only — NOT the scoring target
            "canonical_net_coefficients": row_coefficients(con, prob.var_names),
            "canonical_rhs": con.rhs_const,
        })

    flagged = []
    for t in obj_terms:
        if t.get("non_numeric_raw"):
            flagged.append({"where": "objective", "variable": t["variable"],
                            "raw": t["raw"], "reason": t["reason"]})
    for r in rows:
        for key in ("declared_limit", "declared_param"):
            c = r[key]
            if c and c["non_numeric_raw"]:
                flagged.append({"where": f"{r['name']}.{key}", "raw": c["raw"],
                                "reason": c["reason"]})
        for t in r["declared_terms"]:
            if t["non_numeric_raw"]:
                flagged.append({"where": f"{r['name']}.term",
                                "variable": t["variable"], "raw": t["raw"],
                                "reason": t["reason"]})

    unresolved = sorted({t["variable"] for t in obj_terms
                         if t["var_index"] is None}
                        | {t["variable"] for r in rows
                           for t in r["declared_terms"] if t["var_index"] is None})

    return {
        "problem_id": pid,
        "split": split,
        "document": inst["document"],
        "variables": prob.var_names,
        # `variables[i]` is variable i; `variable_aliases` maps EVERY surface
        # mention the annotation recognises onto that same index. Both are
        # needed: the canonical name is for display, the index is the referent.
        "variable_aliases": dict(order),
        "unresolved_mentions": unresolved,
        "n_vars": len(prob.var_names),
        "sense": "max" if prob.sense == GRB.MAXIMIZE else "min",
        "objective_type": otype,
        "objective_coefficients_implicit": obj_implicit,
        "objective_terms": obj_terms,
        "constraints": rows,
        "non_numeric_cells": flagged,
        "parser_notes": prob.notes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    sample = json.load(open(SAMPLE_PATH))
    wanted = set(sample["sample_all"])
    corpus = {pid: (split, inst) for split, pid, inst in load_corpus(DATA)}

    print("=" * 78)
    print("EXTRACTION PILOT — GROUND TRUTH")
    print("=" * 78)
    print(f"  sample {len(wanted)} instances (seed {sample['seed']}, "
          f"frame {sample['frame']})")

    gt = {}
    for pid in sample["sample_all"]:
        split, inst = corpus[pid]
        gt[pid] = build_one(pid, split, inst)

    n_rows = sum(len(g["constraints"]) for g in gt.values())
    n_obj = sum(len(g["objective_terms"]) for g in gt.values())
    n_decl_terms = sum(len(r["declared_terms"])
                       for g in gt.values() for r in g["constraints"])
    n_implicit = sum(1 for g in gt.values() for r in g["constraints"]
                     if r["implicit_structural_coefficient"])
    flagged = [(pid, f) for pid, g in gt.items() for f in g["non_numeric_cells"]]

    print(f"\n  constraint rows                     {n_rows}")
    print(f"    of which implicit-coefficient     {n_implicit}  "
          f"-> MANUAL REVIEW for coefficient cells")
    print(f"  declared per-variable coef cells    {n_decl_terms}")
    print(f"  objective coefficient cells         {n_obj}")
    print(f"    implicit (objvar, coef 1)         "
          f"{sum(1 for g in gt.values() if g['objective_coefficients_implicit'])}"
          f" instance(s)")

    print(f"\n  NON-NUMERIC raw parameters (kept, never cleaned): {len(flagged)}")
    by_reason = {}
    for pid, f in flagged:
        by_reason.setdefault(f["reason"], []).append((pid, f))
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"    {reason:<34} {len(items)}")
        for pid, f in items[:6]:
            print(f"        {pid:>13} {f['where']:<22} {f['raw']!r}")

    n_alias = sum(1 for g in gt.values()
                  if len(g["variable_aliases"]) > len(g["variables"]))
    unres = {pid: g["unresolved_mentions"] for pid, g in gt.items()
             if g["unresolved_mentions"]}
    print(f"\n  instances carrying extra surface names for a variable "
          f"(order_mapping wider than var_names)   {n_alias}")
    print(f"  mentions NOT resolvable via order_mapping            "
          f"{sum(len(v) for v in unres.values())}"
          + (f"  {unres}" if unres else "   <- all resolve"))

    print(f"\n  by constraint ctype: "
          + str({t: sum(1 for g in gt.values() for r in g['constraints']
                        if r['ctype'] == t)
                 for t in sorted({r['ctype'] for g in gt.values()
                                  for r in g['constraints']})}))

    if args.write:
        os.makedirs(OUTDIR, exist_ok=True)
        with open(GT_PATH, "w") as fh:
            json.dump(gt, fh, indent=2)
        print(f"\n  wrote {GT_PATH}")
    else:
        print("\n  --write not given; nothing written")


if __name__ == "__main__":
    main()
