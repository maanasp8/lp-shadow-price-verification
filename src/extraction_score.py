"""EXTRACTION PILOT — step 4: score. MECHANICAL ONLY, runs after extraction.

    venv/bin/python extraction_score.py [--write]

No LLM judgment anywhere in this file. Every verdict is a comparison between the
extraction and the ground truth under fixed rules. Four axes, reported
SEPARATELY — there is no blended score, by design.

  1. NUMERIC ACCURACY      of the cells where extraction and ground truth agree
                           on the referent, how often is the VALUE right?
  2. ALIGNMENT ACCURACY    is the value attached to the right variable and the
                           right row? A right number on the wrong referent is an
                           ALIGNMENT ERROR even though the number itself is right.
  3. CONTENT-GAP RATE      ground-truth cells with no extracted counterpart.
  4. SPURIOUS-EXTRACTION   extracted cells with no ground-truth counterpart.

MANUAL-REVIEW BUCKET — never an automatic pass or fail
------------------------------------------------------
Two classes of ground-truth cell route to manual review instead of scoring:
  * `non_numeric_raw` cells (percentages, comma-formatted, word-numbers, …) —
    the pilot keeps them as stress cases, and what counts as "right" for them is
    a human call, not a float comparison;
  * `implicit_structural_coefficient` rows (`sum`, bound, `ratio`, `xby`, `xy`) —
    the annotation declares no per-variable coefficient there, so scoring the
    extractor against a derived one would penalise it for not autoformulating,
    which the prompt forbids.
They are counted, listed, and excluded from every rate's denominator.

MATCHING RULES, fixed here so they are not chosen after seeing results
----------------------------------------------------------------------
Variables       exact string match after casefold + whitespace collapse. A
                variable that matches nothing is an alignment failure, not a
                free pass.
Rows            ground-truth rows and extracted rows are matched greedily by
                (operator, limit) then by declared-term overlap. Extraction row
                ids are the model's own and carry no information, so they are
                never used for matching.
Numbers         equal when |a-b| <= 1e-9 * max(1,|b|) — the project's REL_TOL
                shape. "ABSENT" is not a number; it matches only absence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

OUTDIR = "data/extraction_pilot"
RAW_DIR = os.path.join(OUTDIR, "raw")
GT_PATH = os.path.join(OUTDIR, "ground_truth.json")
SCORE_PATH = os.path.join(OUTDIR, "scores.json")
REVIEW_PATH = os.path.join(OUTDIR, "manual_review.json")

TOL = 1e-9
ABSENT = "ABSENT"


def norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def num_eq(a, b) -> bool:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(a - b) <= TOL * max(1.0, abs(b))


def is_absent(v) -> bool:
    return isinstance(v, str) and v.strip().upper() == ABSENT


# ==========================================================================
# variable identity — the REFERENT is the index, not the string
# ==========================================================================
# An instance may name one variable several ways: `-1501301743` writes both
# 'Standard' and 'Standard grade alloy' for variable 0, and `-1921036549`
# absorbs a typo by mapping 'chorine' and 'chlorine' both to variable 0. The
# corpus ships that table as `order_mapping`; `nl4opt_parser._resolve` already
# uses it to build the model. Matching on the surface string instead would
# score a flawless extraction as wrong whenever it picked the other spelling —
# see the pre-registration's §5.3 amendment.
#
# A key is the integer index when the mention resolves, and ("name", <norm>)
# when it does not — so unrecognised names still compare consistently and can
# only match another unrecognised name, never a real variable.

def alias_map(gt: dict) -> Dict[str, int]:
    return {norm(k): v for k, v in (gt.get("variable_aliases") or {}).items()}


def gt_key(index: Optional[int], name: str):
    return index if index is not None else ("name", norm(name))


def ex_key(name: str, alias: Dict[str, int]):
    i = alias.get(norm(name))
    return i if i is not None else ("name", norm(name))


def key_label(key, gt: dict) -> str:
    """Human-readable label for a key, for the `detail` log only."""
    if isinstance(key, int):
        names = gt.get("variables") or []
        return names[key] if 0 <= key < len(names) else f"var_{key}"
    return f"{key[1]!r} (unrecognised)"


def classify_cell(key, gt_value, ex_value, gt_scope: Dict[object, float]) -> str:
    """The fixed rule, applied per cell within one scope (the objective, or one
    row). `gt_scope` is {variable key -> ground-truth value} for that scope.

        absent / missing                      -> content_gap
        matches this variable's value         -> aligned + numeric_correct
        matches a DIFFERENT variable's value  -> MISATTACHED  (alignment error,
                                                 NOT a numeric error — the
                                                 number was read correctly and
                                                 attached to the wrong referent)
        matches nothing                       -> aligned + numeric_wrong
    """
    if ex_value is None or is_absent(ex_value):
        return "content_gap"
    if num_eq(ex_value, gt_value):
        return "correct"
    for other, other_val in gt_scope.items():
        if other != key and num_eq(ex_value, other_val):
            return "misattached"
    return "numeric_wrong"


# ==========================================================================
# variable alignment
# ==========================================================================

def align_variables(gt_vars: List[str], ex_vars: List[str],
                    alias: Dict[str, int]) -> dict:
    """Match on referent, not spelling. `gt_vars[i]` IS variable i, so any
    extracted name the alias table resolves to i matches it. A second extracted
    name resolving to an already-claimed variable is spurious, not a match —
    listing one variable twice is over-extraction."""
    gmap = {i: nm for i, nm in enumerate(gt_vars)}
    matched, ex_unmatched, claimed = {}, [], set()
    for v in ex_vars:
        k = ex_key(v, alias)
        if k in gmap and k not in claimed:
            matched[gmap[k]] = v
            claimed.add(k)
        else:
            ex_unmatched.append(v)
    return {
        "matched": matched,                                  # gt name -> ex name
        # Indices of ground-truth variables some extracted name resolved to.
        # Always ints: `gmap` is keyed by position, so a tuple key never lands
        # here. Used to attribute content gaps (axis 3) by cause.
        "matched_keys": sorted(claimed),
        "gt_unmatched": sorted(nm for k, nm in gmap.items() if k not in claimed),
        "ex_unmatched": sorted(ex_unmatched),
    }


# ==========================================================================
# row matching — never by the model's own ids
# ==========================================================================

def row_signature_gt(r: dict) -> Tuple[str, Optional[float], frozenset]:
    lim = r.get("declared_limit") or {}
    terms = frozenset(gt_key(t.get("var_index"), t["variable"])
                      for t in r.get("declared_terms") or [])
    return r["operator"], lim.get("value"), terms


def row_signature_ex(r: dict, alias: Dict[str, int]) -> Tuple[str, Optional[float], frozenset]:
    lim = r.get("limit")
    lim = lim if isinstance(lim, (int, float)) else None
    terms = frozenset(ex_key(t["variable"], alias) for t in (r.get("terms") or [])
                      if not is_absent(t.get("coefficient")))
    return r.get("operator", ""), lim, terms


def match_rows(gt_rows: List[dict], ex_rows: List[dict],
               alias: Dict[str, int]) -> dict:
    """Greedy: exact (operator, limit) first, then best declared-term overlap."""
    pairs, used = {}, set()
    gsig = [row_signature_gt(r) for r in gt_rows]
    esig = [row_signature_ex(r, alias) for r in ex_rows]

    for gi, (gop, glim, gterms) in enumerate(gsig):
        for ei, (eop, elim, eterms) in enumerate(esig):
            if ei in used:
                continue
            if gop == eop and glim is not None and num_eq(elim, glim):
                pairs[gi] = ei
                used.add(ei)
                break
    for gi, (gop, glim, gterms) in enumerate(gsig):
        if gi in pairs:
            continue
        best, best_score = None, 0
        for ei, (eop, elim, eterms) in enumerate(esig):
            if ei in used:
                continue
            score = len(gterms & eterms) + (1 if gop == eop else 0)
            if score > best_score:
                best, best_score = ei, score
        if best is not None and best_score > 0:
            pairs[gi] = best
            used.add(best)
    return {"pairs": pairs,
            "gt_unmatched": [i for i in range(len(gt_rows)) if i not in pairs],
            "ex_unmatched": [i for i in range(len(ex_rows)) if i not in used]}


# ==========================================================================
# cell-level scoring
# ==========================================================================

def score_instance(gt: dict, ex: Optional[dict]) -> dict:
    """-> per-instance cell tallies across the four axes."""
    out = {
        "problem_id": gt["problem_id"],
        "extraction_present": ex is not None,
        "numeric": {"correct": 0, "wrong": 0},
        "alignment": {"correct": 0, "error": 0},
        "content_gap": 0,
        "spurious": 0,
        # Breakdowns only. They partition the `alignment.error`, `spurious` and
        # `content_gap` totals by CAUSE and change neither total, any
        # denominator, nor any rate. Axes 2 and 4 added revision 2 §0.1 item
        # (5); axis 3 added §0.1 item (6). Both post-data — see the log.
        "alignment_breakdown": {},
        "spurious_breakdown": {},
        "content_gap_breakdown": {},
        "manual_review": [],
        "detail": [],
    }

    def align_err(cause: str) -> None:
        out["alignment"]["error"] += 1
        out["alignment_breakdown"][cause] = \
            out["alignment_breakdown"].get(cause, 0) + 1

    def spur(cause: str, n: int = 1) -> None:
        out["spurious"] += n
        out["spurious_breakdown"][cause] = \
            out["spurious_breakdown"].get(cause, 0) + n

    def gap(cause: str, n: int = 1) -> None:
        out["content_gap"] += n
        out["content_gap_breakdown"][cause] = \
            out["content_gap_breakdown"].get(cause, 0) + n

    def cell_cause(key) -> str:
        """A spurious cell sitting under a name the annotation never uses is a
        vocabulary mismatch; one under a real variable is over-extraction."""
        return ("cell_under_unrecognised_name" if not isinstance(key, int)
                else "extra_cell_on_known_variable")
    if ex is None:
        # A failed extraction is a total content gap over every scorable cell.
        gaps = len([t for t in gt["objective_terms"] if not t["non_numeric_raw"]])
        for r in gt["constraints"]:
            lim = r.get("declared_limit")
            if lim and not lim["non_numeric_raw"]:
                gaps += 1
            if not r["implicit_structural_coefficient"]:
                gaps += len([t for t in r["declared_terms"]
                             if not t["non_numeric_raw"]])
        gap("failed_extraction", gaps)
        out["detail"].append({"cell": "<whole instance>", "verdict": "content_gap",
                              "cause": "failed_extraction",
                              "note": "no usable extraction"})
        return out

    alias = alias_map(gt)
    va = align_variables(gt["variables"], ex.get("variables") or [], alias)
    out["variable_alignment"] = va
    matched_keys = set(va["matched_keys"])

    def gap_cause(key) -> str:
        """A gap under a referent no extracted name resolved to is a vocabulary
        mismatch; one under a matched referent is a genuinely missing value."""
        return ("cell_under_unmatched_variable" if key not in matched_keys
                else "missing_value_on_matched_variable")
    for v in va["gt_unmatched"]:
        align_err("unmatched_variable_name")
        out["detail"].append({"cell": f"variable[{v}]", "verdict": "alignment_error",
                              "cause": "unmatched_variable_name",
                              "note": "ground-truth variable not extracted"})
    for v in va["ex_unmatched"]:
        spur("unrecognised_variable_name")
        out["detail"].append({"cell": f"variable[{v}]", "verdict": "spurious",
                              "cause": "unrecognised_variable_name",
                              "note": "extracted variable with no counterpart"})

    # ---- sense (alignment-class: a structural referent, not a number) -----
    ex_sense = (ex.get("objective") or {}).get("sense")
    if ex_sense == gt["sense"]:
        out["alignment"]["correct"] += 1
    else:
        align_err("objective_sense")
        out["detail"].append({"cell": "objective.sense", "verdict": "alignment_error",
                              "cause": "objective_sense",
                              "gt": gt["sense"], "extracted": ex_sense})

    # ---- objective coefficients -------------------------------------------
    ex_obj = {}
    for t in (ex.get("objective") or {}).get("terms") or []:
        ex_obj.setdefault(ex_key(t["variable"], alias), t.get("coefficient"))
    obj_scope = {gt_key(t.get("var_index"), t["variable"]): t["value"]
                 for t in gt["objective_terms"]
                 if not t["non_numeric_raw"] and t["value"] is not None}
    seen_obj = set()
    for t in gt["objective_terms"]:
        key = gt_key(t.get("var_index"), t["variable"])
        cell = f"objective[{t['variable']}]"
        if t["non_numeric_raw"]:
            out["manual_review"].append(
                {"cell": cell, "reason": "non_numeric_raw",
                 "raw": t["raw"], "gt_value": t["value"],
                 "extracted": ex_obj.get(key)})
            seen_obj.add(key)
            continue
        verdict = classify_cell(key, t["value"], ex_obj.get(key), obj_scope)
        if verdict == "content_gap":
            gap(gap_cause(key))
            out["detail"].append({"cell": cell, "verdict": "content_gap",
                                  "cause": gap_cause(key), "gt": t["value"]})
        else:
            seen_obj.add(key)
            if verdict == "misattached":
                align_err("misattached_value")
                out["detail"].append({"cell": cell, "verdict": "alignment_error",
                                      "cause": "misattached_value",
                                      "note": "value belongs to another variable "
                                              "in this scope",
                                      "gt": t["value"], "extracted": ex_obj[key]})
            else:
                out["alignment"]["correct"] += 1
                if verdict == "correct":
                    out["numeric"]["correct"] += 1
                else:
                    out["numeric"]["wrong"] += 1
                    out["detail"].append({"cell": cell, "verdict": "numeric_wrong",
                                          "gt": t["value"],
                                          "extracted": ex_obj[key]})
    for key, val in ex_obj.items():
        if key in seen_obj or is_absent(val):
            continue
        spur(cell_cause(key))
        out["detail"].append({"cell": f"objective[{key_label(key, gt)}]",
                              "verdict": "spurious", "cause": cell_cause(key),
                              "extracted": val})

    # ---- constraints -------------------------------------------------------
    ex_rows = ex.get("constraints") or []
    rm = match_rows(gt["constraints"], ex_rows, alias)
    out["row_matching"] = {"pairs": {str(k): v for k, v in rm["pairs"].items()},
                           "gt_unmatched": rm["gt_unmatched"],
                           "ex_unmatched": rm["ex_unmatched"]}

    for gi, r in enumerate(gt["constraints"]):
        ei = rm["pairs"].get(gi)
        prefix = r["name"]
        if ei is None:
            lim = r.get("declared_limit")
            if lim and not lim["non_numeric_raw"]:
                gap("row_not_extracted")
            if not r["implicit_structural_coefficient"]:
                gap("row_not_extracted",
                    len([t for t in r["declared_terms"]
                         if not t["non_numeric_raw"]]))
            out["detail"].append({"cell": prefix, "verdict": "content_gap",
                                  "cause": "row_not_extracted",
                                  "note": "row not extracted"})
            continue
        er = ex_rows[ei]

        # operator — alignment-class
        if er.get("operator") == r["operator"]:
            out["alignment"]["correct"] += 1
        else:
            align_err("row_operator")
            out["detail"].append({"cell": f"{prefix}.operator",
                                  "verdict": "alignment_error",
                                  "cause": "row_operator",
                                  "gt": r["operator"],
                                  "extracted": er.get("operator")})

        # limit
        lim = r.get("declared_limit")
        if lim is not None:
            if lim["non_numeric_raw"]:
                out["manual_review"].append(
                    {"cell": f"{prefix}.limit", "reason": "non_numeric_raw",
                     "raw": lim["raw"], "gt_value": lim["value"],
                     "extracted": er.get("limit")})
            elif is_absent(er.get("limit")) or er.get("limit") is None:
                gap("missing_limit")
                out["detail"].append({"cell": f"{prefix}.limit",
                                      "verdict": "content_gap",
                                      "cause": "missing_limit",
                                      "gt": lim["value"]})
            else:
                out["alignment"]["correct"] += 1
                if num_eq(er.get("limit"), lim["value"]):
                    out["numeric"]["correct"] += 1
                else:
                    out["numeric"]["wrong"] += 1
                    out["detail"].append({"cell": f"{prefix}.limit",
                                          "verdict": "numeric_wrong",
                                          "gt": lim["value"],
                                          "extracted": er.get("limit")})

        # xby rows carry their number in `declared_param`, not `declared_limit`.
        # Pre-registration §5.5 promises every non-numeric raw cell is counted
        # and listed; without this branch the two `xby` params in the sample
        # ('half', '3 times') would be dropped silently. Manual-review cells sit
        # outside every denominator, so this adds entries and moves no rate.
        prm = r.get("declared_param")
        if prm is not None:
            out["manual_review"].append(
                {"cell": f"{prefix}.param",
                 "reason": ("non_numeric_raw" if prm["non_numeric_raw"]
                            else "declared_param_structural"),
                 "ctype": r["ctype"], "raw": prm["raw"],
                 "gt_value": prm["value"], "extracted_limit": er.get("limit")})

        ex_terms = {}
        for t in (er.get("terms") or []):
            ex_terms.setdefault(ex_key(t["variable"], alias), t.get("coefficient"))
        if r["implicit_structural_coefficient"]:
            out["manual_review"].append(
                {"cell": f"{prefix}.coefficients",
                 "reason": "implicit_structural_coefficient",
                 "ctype": r["ctype"],
                 "canonical_reference": r["canonical_net_coefficients"],
                 "extracted": {key_label(k, gt): v for k, v in ex_terms.items()
                               if not is_absent(v)}})
            continue

        row_scope = {gt_key(t.get("var_index"), t["variable"]): t["value"]
                     for t in r["declared_terms"]
                     if not t["non_numeric_raw"] and t["value"] is not None}
        seen = set()
        for t in r["declared_terms"]:
            key = gt_key(t.get("var_index"), t["variable"])
            cell = f"{prefix}[{t['variable']}]"
            if t["non_numeric_raw"]:
                out["manual_review"].append(
                    {"cell": cell, "reason": "non_numeric_raw", "raw": t["raw"],
                     "gt_value": t["value"], "extracted": ex_terms.get(key)})
                seen.add(key)
                continue
            verdict = classify_cell(key, t["value"], ex_terms.get(key), row_scope)
            if verdict == "content_gap":
                gap(gap_cause(key))
                out["detail"].append({"cell": cell, "verdict": "content_gap",
                                      "cause": gap_cause(key), "gt": t["value"]})
            else:
                seen.add(key)
                if verdict == "misattached":
                    align_err("misattached_value")
                    out["detail"].append({"cell": cell,
                                          "verdict": "alignment_error",
                                          "cause": "misattached_value",
                                          "note": "value belongs to another "
                                                  "variable in this row",
                                          "gt": t["value"],
                                          "extracted": ex_terms[key]})
                else:
                    out["alignment"]["correct"] += 1
                    if verdict == "correct":
                        out["numeric"]["correct"] += 1
                    else:
                        out["numeric"]["wrong"] += 1
                        out["detail"].append({"cell": cell,
                                              "verdict": "numeric_wrong",
                                              "gt": t["value"],
                                              "extracted": ex_terms[key]})
        for key, val in ex_terms.items():
            if key in seen or is_absent(val):
                continue
            spur(cell_cause(key))
            out["detail"].append({"cell": f"{prefix}[{key_label(key, gt)}]",
                                  "verdict": "spurious", "cause": cell_cause(key),
                                  "extracted": val})

    for ei in rm["ex_unmatched"]:
        er = ex_rows[ei]
        live = sum(1 for t in (er.get("terms") or [])
                   if not is_absent(t.get("coefficient")))
        spur("extra_row", live + (0 if is_absent(er.get("limit")) else 1))
        out["detail"].append({"cell": f"<extracted row {er.get('id')}>",
                              "verdict": "spurious", "cause": "extra_row",
                              "note": "no ground-truth counterpart",
                              "quoted_text": er.get("quoted_text")})

    # The breakdowns must partition their totals exactly — they are a view of
    # the same events, never a second count.
    assert sum(out["alignment_breakdown"].values()) == out["alignment"]["error"]
    assert sum(out["spurious_breakdown"].values()) == out["spurious"]
    assert sum(out["content_gap_breakdown"].values()) == out["content_gap"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    gt_all = json.load(open(GT_PATH))
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(f"{RAW_DIR}/ does not exist — run extraction_extract.py "
                         f"--run first (after freezing the pre-registration).")

    results, missing = [], []
    for pid, gt in gt_all.items():
        path = os.path.join(RAW_DIR, f"{pid}.json")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        rec = json.load(open(path))
        results.append(score_instance(gt, rec.get("extraction")
                                      if rec.get("ok") else None))

    print("=" * 78)
    print("EXTRACTION PILOT — SCORES  (mechanical only; no LLM judgment)")
    print("=" * 78)
    print(f"  instances scored {len(results)}   not yet extracted {len(missing)}")
    if missing:
        print(f"    missing: {missing}")
    if not results:
        return

    nc = sum(r["numeric"]["correct"] for r in results)
    nw = sum(r["numeric"]["wrong"] for r in results)
    ac = sum(r["alignment"]["correct"] for r in results)
    ae = sum(r["alignment"]["error"] for r in results)
    cg = sum(r["content_gap"] for r in results)
    sp = sum(r["spurious"] for r in results)
    mr = sum(len(r["manual_review"]) for r in results)

    print("\n  FOUR AXES — reported separately, never blended")
    print("  " + "-" * 74)
    print(f"  1. numeric accuracy      {nc}/{nc+nw} = "
          f"{(100*nc/(nc+nw)) if nc+nw else float('nan'):.2f} %"
          f"   (wrong {nw})")
    print(f"  2. alignment accuracy    {ac}/{ac+ae} = "
          f"{(100*ac/(ac+ae)) if ac+ae else float('nan'):.2f} %"
          f"   (errors {ae})")
    print(f"  3. content-gap rate      {cg} ground-truth cells with no "
          f"extracted counterpart")
    print(f"  4. spurious extraction   {sp} extracted cells with no "
          f"ground-truth counterpart")

    # ---- breakdown by cause (§0.1 item 5) — partitions axes 2 and 4, ------
    # ---- alters no total, no denominator, no rate ------------------------
    ab, sb, gb = {}, {}, {}
    for r in results:
        for k, v in r.get("alignment_breakdown", {}).items():
            ab[k] = ab.get(k, 0) + v
        for k, v in r.get("spurious_breakdown", {}).items():
            sb[k] = sb.get(k, 0) + v
        for k, v in r.get("content_gap_breakdown", {}).items():
            gb[k] = gb.get(k, 0) + v
    NAMING = {"unmatched_variable_name", "unrecognised_variable_name",
              "cell_under_unrecognised_name", "cell_under_unmatched_variable"}
    print(f"\n  BY CAUSE — a partition of axes 2, 3 and 4, not a new number")
    print("  " + "-" * 74)
    for title, total, brk in (("alignment errors", ae, ab),
                              ("content gaps", cg, gb),
                              ("spurious cells", sp, sb)):
        print(f"  {title:<18}{total}")
        for k, v in sorted(brk.items(), key=lambda x: -x[1]):
            print(f"      {'[naming]' if k in NAMING else '[content]':<10} "
                  f"{k:<36} {v}")
    n_name = sum(v for brk in (ab, gb, sb)
                 for k, v in brk.items() if k in NAMING)
    print(f"\n      vocabulary mismatch (extractor's wording vs the annotation's,"
          f" not a misread number):\n      {n_name} of {ae + cg + sp} scoring "
          f"events across axes 2-4")

    print(f"\n  MANUAL REVIEW (not pass, not fail)   {mr} cells")
    reasons = {}
    for r in results:
        for m in r["manual_review"]:
            reasons[m["reason"]] = reasons.get(m["reason"], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"      {k:<38} {v}")

    print("\n  per instance:")
    print(f"    {'problem_id':>13} {'num ok/n':>10} {'align ok/n':>12}"
          f" {'gap':>5} {'spur':>5} {'review':>7}")
    for r in sorted(results, key=lambda x: int(x["problem_id"])):
        n = r["numeric"]; a = r["alignment"]
        print(f"    {r['problem_id']:>13} {n['correct']:>4}/"
              f"{n['correct']+n['wrong']:<5} {a['correct']:>5}/"
              f"{a['correct']+a['error']:<6} {r['content_gap']:>5}"
              f" {r['spurious']:>5} {len(r['manual_review']):>7}")

    if args.write:
        with open(SCORE_PATH, "w") as fh:
            json.dump({"axes": {"numeric_correct": nc, "numeric_wrong": nw,
                                "alignment_correct": ac, "alignment_error": ae,
                                "content_gap": cg, "spurious": sp,
                                "manual_review": mr},
                       "by_cause": {"alignment_error": ab, "content_gap": gb,
                                    "spurious": sb},
                       "per_instance": results}, fh, indent=2)
        with open(REVIEW_PATH, "w") as fh:
            json.dump([{"problem_id": r["problem_id"], **m}
                       for r in results for m in r["manual_review"]], fh, indent=2)
        print(f"\n  wrote {SCORE_PATH}, {REVIEW_PATH}")
    else:
        print("\n  --write not given; nothing written")


if __name__ == "__main__":
    main()
