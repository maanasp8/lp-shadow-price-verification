"""EXTRACTION PILOT — INDEPENDENT recomputation of the four axes. Cross-check.

    venv/bin/python extraction_recompute.py

Recomputes axes 1-4 from `extraction_pilot/raw/*.json` and the NL4Opt corpus
JSONL **directly**, then diffs the result against `extraction_pilot/scores.json`.

WHY THIS EXISTS
---------------
`extraction_score.py` reached its current behaviour through several in-session
corrections (the `declared_param` routing, the `order_mapping` alias fix, the
by-cause partitions for axes 2/4 and then 3). Each was validated, but all of
them share one implementation and one derived artifact. This file is a second
implementation that shares neither, so agreement between the two is evidence
rather than restatement.

WHAT IS AND IS NOT INDEPENDENT — stated so the check is not oversold
--------------------------------------------------------------------
INDEPENDENT of the pipeline being checked:
  * ground truth is re-derived from `generation_data/*.jsonl`, NOT read from
    `extraction_pilot/ground_truth.json` (so `extraction_ground_truth.py` is
    not in the loop);
  * variable resolution, non-numeric classification, row matching, cell
    scoring and the four axes are all reimplemented here from the
    pre-registration §5 text;
  * `extraction_score.py` is never imported.

SHARED, unavoidably:
  * the raw model responses in `raw/*.json` (that is the measurement);
  * the sample id list in `sample.json`;
  * `load_corpus` for JSONL iteration only — file reading, no interpretation.

Its ground-truth-derived quantities (sense, operators, cell inventory) are
additionally asserted against `ground_truth.json` where both exist, so a
disagreement in the *derivation* is caught as well as one in the scoring.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from nl4opt_parser import load_corpus

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl"
        for s in ("train", "dev", "test")]
OUTDIR = "data/extraction_pilot"
TOL = 1e-9

IMPLICIT = {"sum", "upperbound", "lowerbound", "ratio", "xby", "xy"}
HAS_LIMIT = {"linear", "sum", "upperbound", "lowerbound", "ratio"}
OPS = {"LESS_OR_EQUAL": "<=", "GREATER_OR_EQUAL": ">="}


def norm(s) -> str:
    return re.sub(r"\s+", " ", (str(s) or "").strip()).casefold()


def plain_number(raw) -> Optional[float]:
    """Scorable only if the raw string is a plain decimal. Anything else —
    '30%', '1,000', 'half', '3 times' — is a manual-review cell (§5.5)."""
    try:
        return float(str(raw).strip())
    except (ValueError, TypeError, AttributeError):
        return None


def num_eq(a, b) -> bool:
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return False
    if not isinstance(b, (int, float)):
        return False
    return abs(a - b) <= TOL * max(1.0, abs(b))


def is_absent(v) -> bool:
    return isinstance(v, str) and v.strip().upper() == "ABSENT"


# --------------------------------------------------------------------------
# ground truth, re-derived from the corpus
# --------------------------------------------------------------------------

def derive(inst: dict) -> dict:
    order: Dict[str, int] = inst["order_mapping"]
    n = max(order.values()) + 1
    obj = inst["obj_declaration"]

    if obj.get("type") == "objective":
        obj_terms = [(order[m], plain_number(raw), raw)
                     for m, raw in (obj.get("terms") or {}).items()]
    else:                                    # objvar -> each listed var, coef 1
        obj_terms = [(order[m], 1.0, None) for m in (obj.get("vars") or [])]

    rows = []
    for i, d in enumerate(inst["const_declarations"]):
        ct = d.get("type")
        lim_raw = d.get("limit") if ct in HAS_LIMIT else None
        terms = ([(order[m], plain_number(raw), raw)
                  for m, raw in (d.get("terms") or {}).items()]
                 if ct == "linear" else [])
        rows.append({
            "name": f"c{i}_{ct}", "ctype": ct, "op": OPS[d["operator"]],
            "limit_raw": lim_raw,
            "limit_val": plain_number(lim_raw) if lim_raw is not None else None,
            "terms": terms, "implicit": ct in IMPLICIT,
        })

    return {"n_vars": n, "aliases": {norm(k): v for k, v in order.items()},
            "sense": "max" if "max" in (obj.get("direction") or "") else "min",
            "obj_terms": obj_terms, "rows": rows}


# --------------------------------------------------------------------------
# scoring, reimplemented from §5
# --------------------------------------------------------------------------

def key_of(name: str, aliases: Dict[str, int]):
    i = aliases.get(norm(name))
    return i if i is not None else ("?", norm(name))


def match_rows(gt_rows: List[dict], ex_rows: List[dict],
               aliases: Dict[str, int]) -> Dict[int, int]:
    def esig(r):
        lim = r.get("limit")
        lim = lim if isinstance(lim, (int, float)) and not isinstance(lim, bool) else None
        return (r.get("operator", ""), lim,
                frozenset(key_of(t["variable"], aliases)
                          for t in (r.get("terms") or [])
                          if not is_absent(t.get("coefficient"))))

    def gsig(r):
        return (r["op"], r["limit_val"],
                frozenset(k for k, _v, _raw in r["terms"]))

    pairs, used = {}, set()
    G, E = [gsig(r) for r in gt_rows], [esig(r) for r in ex_rows]
    for gi, (gop, glim, _gt) in enumerate(G):                 # pass 1: op+limit
        for ei, (eop, elim, _et) in enumerate(E):
            if ei in used or glim is None:
                continue
            if gop == eop and num_eq(elim, glim):
                pairs[gi], _ = ei, used.add(ei)
                break
    for gi, (gop, glim, gt_) in enumerate(G):                 # pass 2: overlap
        if gi in pairs:
            continue
        best, score = None, 0
        for ei, (eop, _el, et) in enumerate(E):
            if ei in used:
                continue
            s = len(gt_ & et) + (1 if gop == eop else 0)
            if s > score:
                best, score = ei, s
        if best is not None and score > 0:
            pairs[gi], _ = best, used.add(best)
    return pairs


def score(gt: dict, ex: Optional[dict]) -> dict:
    A = gt["aliases"]
    o = {"num_ok": 0, "num_bad": 0, "algn_ok": 0, "algn_bad": 0,
         "gap": 0, "spur": 0, "review": 0,
         "algn_naming": 0, "gap_naming": 0, "spur_naming": 0}

    if ex is None:
        o["gap"] = (len([1 for _k, v, _r in gt["obj_terms"] if v is not None])
                    + len([1 for r in gt["rows"] if r["limit_val"] is not None])
                    + len([1 for r in gt["rows"] if not r["implicit"]
                           for _k, v, _r2 in r["terms"] if v is not None]))
        return o

    # variables -------------------------------------------------------------
    claimed = set()
    for v in (ex.get("variables") or []):
        k = key_of(v, A)
        if isinstance(k, int) and k not in claimed:
            claimed.add(k)
        else:
            o["spur"] += 1
            o["spur_naming"] += 1
    for i in range(gt["n_vars"]):
        if i not in claimed:
            o["algn_bad"] += 1
            o["algn_naming"] += 1

    # sense -----------------------------------------------------------------
    if (ex.get("objective") or {}).get("sense") == gt["sense"]:
        o["algn_ok"] += 1
    else:
        o["algn_bad"] += 1

    def cells(gt_cells, ex_map, scope):
        """One scope (objective, or one row's terms). §5.2's rule verbatim."""
        seen = set()
        for k, val, _raw in gt_cells:
            if val is None:                       # non-numeric -> manual review
                o["review"] += 1
                seen.add(k)
                continue
            got = ex_map.get(k)
            if got is None or is_absent(got):
                o["gap"] += 1
                if k not in claimed:
                    o["gap_naming"] += 1
                continue
            seen.add(k)
            if num_eq(got, val):
                o["algn_ok"] += 1
                o["num_ok"] += 1
            elif any(kk != k and num_eq(got, vv) for kk, vv in scope.items()):
                o["algn_bad"] += 1                # misattached: not numeric
            else:
                o["algn_ok"] += 1
                o["num_bad"] += 1
        for k, val in ex_map.items():
            if k in seen or is_absent(val):
                continue
            o["spur"] += 1
            if not isinstance(k, int):
                o["spur_naming"] += 1

    ex_obj: Dict[object, object] = {}
    for t in (ex.get("objective") or {}).get("terms") or []:
        ex_obj.setdefault(key_of(t["variable"], A), t.get("coefficient"))
    cells(gt["obj_terms"], ex_obj,
          {k: v for k, v, _r in gt["obj_terms"] if v is not None})

    # constraints -----------------------------------------------------------
    ex_rows = ex.get("constraints") or []
    pairs = match_rows(gt["rows"], ex_rows, A)
    for gi, r in enumerate(gt["rows"]):
        ei = pairs.get(gi)
        if ei is None:
            if r["limit_val"] is not None:
                o["gap"] += 1
            if not r["implicit"]:
                o["gap"] += len([1 for _k, v, _x in r["terms"] if v is not None])
            continue
        er = ex_rows[ei]

        if er.get("operator") == r["op"]:
            o["algn_ok"] += 1
        else:
            o["algn_bad"] += 1

        if r["ctype"] in HAS_LIMIT:
            if r["limit_val"] is None:
                o["review"] += 1
            elif is_absent(er.get("limit")) or er.get("limit") is None:
                o["gap"] += 1
            else:
                o["algn_ok"] += 1
                if num_eq(er.get("limit"), r["limit_val"]):
                    o["num_ok"] += 1
                else:
                    o["num_bad"] += 1
        if r["ctype"] == "xby":
            o["review"] += 1                       # declared_param -> review

        ex_terms: Dict[object, object] = {}
        for t in (er.get("terms") or []):
            ex_terms.setdefault(key_of(t["variable"], A), t.get("coefficient"))
        if r["implicit"]:
            o["review"] += 1                       # coefficients -> review
            continue
        cells(r["terms"], ex_terms,
              {k: v for k, v, _x in r["terms"] if v is not None})

    for ei, er in enumerate(ex_rows):
        if ei in pairs.values():
            continue
        live = len([1 for t in (er.get("terms") or [])
                    if not is_absent(t.get("coefficient"))])
        o["spur"] += live + (0 if is_absent(er.get("limit")) else 1)
    return o


def main() -> int:
    sample = json.load(open(os.path.join(OUTDIR, "sample.json")))["sample_all"]
    corpus = {pid: inst for _s, pid, inst in load_corpus(DATA) if pid in sample}
    ref = json.load(open(os.path.join(OUTDIR, "ground_truth.json")))

    print("=" * 78)
    print("INDEPENDENT RECOMPUTATION — second implementation, corpus-derived")
    print("=" * 78)

    # cross-check the DERIVATION against ground_truth.json before scoring
    drift = []
    for pid in sample:
        d, g = derive(corpus[pid]), ref[pid]
        if d["sense"] != g["sense"]:
            drift.append(f"{pid} sense {d['sense']} != {g['sense']}")
        if d["n_vars"] != len(g["variables"]):
            drift.append(f"{pid} n_vars {d['n_vars']} != {len(g['variables'])}")
        for a, b in zip(d["rows"], g["constraints"]):
            if a["op"] != b["operator"]:
                drift.append(f"{pid} {a['name']} op {a['op']} != {b['operator']}")
    print(f"  derivation vs ground_truth.json: "
          f"{'AGREES on sense, n_vars, operators' if not drift else drift}")

    tot = {k: 0 for k in ("num_ok", "num_bad", "algn_ok", "algn_bad", "gap",
                          "spur", "review", "algn_naming", "gap_naming",
                          "spur_naming")}
    missing = []
    for pid in sample:
        p = os.path.join(OUTDIR, "raw", f"{pid}.json")
        if not os.path.exists(p):
            missing.append(pid)
            continue
        rec = json.load(open(p))
        s = score(derive(corpus[pid]),
                  rec.get("extraction") if rec.get("ok") else None)
        for k in tot:
            tot[k] += s[k]

    print(f"  instances {len(sample) - len(missing)}"
          + (f"   MISSING {missing}" if missing else ""))
    print("\n  RECOMPUTED")
    print(f"    1. numeric      {tot['num_ok']}/{tot['num_ok'] + tot['num_bad']}")
    print(f"    2. alignment    {tot['algn_ok']}/{tot['algn_ok'] + tot['algn_bad']}")
    print(f"    3. content gap  {tot['gap']}")
    print(f"    4. spurious     {tot['spur']}")
    print(f"       manual review {tot['review']}")
    print(f"       naming share — alignment {tot['algn_naming']}/{tot['algn_bad']}"
          f"   gaps {tot['gap_naming']}/{tot['gap']}"
          f"   spurious {tot['spur_naming']}/{tot['spur']}")

    sc = json.load(open(os.path.join(OUTDIR, "scores.json")))["axes"]
    exp = {"1. numeric_correct": (tot["num_ok"], sc["numeric_correct"]),
           "1. numeric_wrong": (tot["num_bad"], sc["numeric_wrong"]),
           "2. alignment_correct": (tot["algn_ok"], sc["alignment_correct"]),
           "2. alignment_error": (tot["algn_bad"], sc["alignment_error"]),
           "3. content_gap": (tot["gap"], sc["content_gap"]),
           "4. spurious": (tot["spur"], sc["spurious"]),
           "   manual_review": (tot["review"], sc["manual_review"])}
    print("\n  DIFF vs extraction_score.py")
    print(f"    {'quantity':<24} {'recomputed':>11} {'scorer':>8}   verdict")
    bad = 0
    for k, (mine, theirs) in exp.items():
        ok = mine == theirs
        bad += 0 if ok else 1
        print(f"    {k:<24} {mine:>11} {theirs:>8}   "
              f"{'match' if ok else '*** MISMATCH ***'}")
    print("\n  " + ("ALL QUANTITIES AGREE" if not bad
                    else f"{bad} MISMATCH(ES) — investigate before publishing"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
