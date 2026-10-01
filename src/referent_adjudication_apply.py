#!/usr/bin/env python
"""
Apply the human referent adjudications to the extraction pilot and report the
adjudicated figures ALONGSIDE the as-measured ones.

WHAT THIS DOES NOT DO
  * does not modify extraction_score.py — it imports it and calls
    `score_instance` unchanged, so every frozen rule (classify_cell,
    misattachment, manual-review routing, by-cause partition, the assertion that
    each breakdown sums to its axis) applies exactly as pre-registered;
  * does not overwrite scores.json, ground_truth.json, the pre-registration, or
    extraction_pilot/raw/;
  * does not re-extract or re-sample.

HOW THE ADJUDICATION IS INJECTED
  `score_instance` resolves an extracted name through `alias_map(gt)`. For the
  duration of this run only, that function is wrapped so it returns the
  annotation's own `order_mapping` PLUS the adjudicated pairings — i.e. an
  extracted name the reviewer judged to denote variable i now resolves to i.
  Nothing else about the scorer changes. The wrapper is asserted to be additive:
  it may never redirect a key the annotation already defines.

ASYMMETRY, ON THE RECORD
  Same bound as the `-114149479` numeric adjudication already disclosed: a call
  can only move referent accuracy in the extractor's favour. The as-measured
  9/63 is the floor and stands unaltered in scores.json.

Outputs (new files only):
  referent_adjudication_result_2026-08-11.json
  referent_adjudication_result_2026-08-11.md
"""

import csv
import json
import os

import extraction_score as S  # imported, never modified

CALLS = "data/referent_adjudication_sheet_2026-08-07.csv"
OUT_JSON = "data/referent_adjudication_result_2026-08-11.json"
OUT_MD = "docs/referent_adjudication_result_2026-08-11.md"

AFFIRM = {"y", "yes", "correct", "true", "1"}
DENY = {"n", "no", "incorrect", "false", "0"}


def read_calls():
    rows = list(csv.DictReader(open(CALLS)))
    calls, unreadable = {}, []
    for r in rows:
        raw = (r.get("my_call") or "").strip()
        v = raw.casefold()
        if v in AFFIRM:
            verdict = True
        elif v in DENY:
            verdict = False
        else:
            unreadable.append((r["instance_id"], r["gt_variable"], raw))
            continue
        calls[(r["instance_id"], r["gt_variable"])] = {
            "verdict": verdict,
            "raw_call": raw,
            "extracted_name": r["extracted_name"],
            "wrapper_category": r["wrapper_category"],
            "pairing_basis": r["pairing_basis"],
        }
    return rows, calls, unreadable


def run(alias_extra=None):
    """Score all 25 instances. `alias_extra` maps problem_id -> {norm name: idx}
    of adjudicated pairings; None reproduces the frozen as-measured run."""
    base = S.alias_map
    if alias_extra is not None:
        def patched(gt):
            a = dict(base(gt))
            for name, idx in alias_extra.get(gt["problem_id"], {}).items():
                assert name not in a, (
                    f"adjudication would redirect an annotation-defined key "
                    f"{name!r} in {gt['problem_id']} — refusing")
                a[name] = idx
            return a
        S.alias_map = patched
    try:
        gt_all = json.load(open(S.GT_PATH))
        out = []
        for pid, gt in gt_all.items():
            rec = json.load(open(os.path.join(S.RAW_DIR, f"{pid}.json")))
            out.append(S.score_instance(
                gt, rec.get("extraction") if rec.get("ok") else None))
        return out
    finally:
        S.alias_map = base


def totals(results):
    agg = {"numeric_correct": 0, "numeric_wrong": 0, "alignment_correct": 0,
           "alignment_error": 0, "content_gap": 0, "spurious": 0,
           "manual_review": 0}
    cause = {"alignment_error": {}, "content_gap": {}, "spurious": {}}
    for r in results:
        agg["numeric_correct"] += r["numeric"]["correct"]
        agg["numeric_wrong"] += r["numeric"]["wrong"]
        agg["alignment_correct"] += r["alignment"]["correct"]
        agg["alignment_error"] += r["alignment"]["error"]
        agg["content_gap"] += r["content_gap"]
        agg["spurious"] += r["spurious"]
        agg["manual_review"] += len(r["manual_review"])
        for k, src in (("alignment_error", "alignment_breakdown"),
                       ("content_gap", "content_gap_breakdown"),
                       ("spurious", "spurious_breakdown")):
            for c, n in (r.get(src) or {}).items():
                cause[k][c] = cause[k].get(c, 0) + n
    return agg, cause


def referent_counts(results, gt_all):
    resolved = total = 0
    for r in results:
        gt = gt_all[r["problem_id"]]
        total += len(gt["variables"])
        resolved += len(r["variable_alignment"]["matched_keys"])
    return resolved, total


def main():
    rows, calls, unreadable = read_calls()
    if unreadable:
        raise SystemExit("unreadable my_call values: " + repr(unreadable))
    if len(calls) != 54:
        raise SystemExit(f"expected 54 calls, read {len(calls)}")

    gt_all = json.load(open(S.GT_PATH))

    # ---- as-measured (frozen) --------------------------------------------
    before = run(None)
    b_agg, b_cause = totals(before)
    b_res, b_tot = referent_counts(before, gt_all)

    # ---- adjudicated ------------------------------------------------------
    extra, accepted, rejected = {}, [], []
    for (pid, var), c in calls.items():
        if not c["verdict"]:
            rejected.append((pid, var))
            continue
        idx = gt_all[pid]["variables"].index(var)
        extra.setdefault(pid, {})[S.norm(c["extracted_name"])] = idx
        accepted.append((pid, var))
    after = run(extra)
    a_agg, a_cause = totals(after)
    a_res, a_tot = referent_counts(after, gt_all)

    payload = {
        "generated": "2026-08-11",
        "source_calls": CALLS,
        "n_calls": len(calls),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "raw_call_values": sorted({c["raw_call"] for c in calls.values()}),
        "note": ("Adjudicated figures are a derived layer. scores.json is "
                 "unaltered and remains the as-measured record. A call can only "
                 "move referent accuracy in the extractor's favour; the "
                 "as-measured figure is the floor."),
        "referent_accuracy": {
            "as_measured": {"resolved": b_res, "total": b_tot,
                            "pct": round(100 * b_res / b_tot, 2)},
            "adjudicated": {"resolved": a_res, "total": a_tot,
                            "pct": round(100 * a_res / a_tot, 2)},
        },
        "axes": {"as_measured": b_agg, "adjudicated": a_agg},
        "by_cause": {"as_measured": b_cause, "adjudicated": a_cause},
        "per_instance": {
            "as_measured": {r["problem_id"]: {
                "numeric": r["numeric"], "alignment": r["alignment"],
                "content_gap": r["content_gap"], "spurious": r["spurious"]}
                for r in before},
            "adjudicated": {r["problem_id"]: {
                "numeric": r["numeric"], "alignment": r["alignment"],
                "content_gap": r["content_gap"], "spurious": r["spurious"]}
                for r in after},
        },
        "accepted": sorted(accepted),
        "rejected": sorted(rejected),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(OUT_MD, "w") as fh:
        fh.write(report(payload, b_agg, a_agg, b_cause, a_cause))
    print(report(payload, b_agg, a_agg, b_cause, a_cause))
    print(f"  wrote {OUT_JSON}, {OUT_MD}")

    # scores.json must be untouched
    live = json.load(open(S.SCORE_PATH))["axes"]
    assert live == b_agg, "scores.json no longer matches the as-measured run"
    print("  scores.json verified unchanged and still equal to the "
          "as-measured recomputation")


def report(p, b, a, bc, ac):
    L = []
    A = L.append
    A("# Referent adjudication — result\n")
    A("Generated 2026-08-11 by `referent_adjudication_apply.py`. "
      "Derived layer only; `scores.json` is unaltered.\n")
    A(f"- Calls read: **{p['n_calls']}** "
      f"(accepted {p['n_accepted']}, rejected {p['n_rejected']})")
    A(f"- Distinct raw call values: {p['raw_call_values']}\n")
    A("## Referent accuracy\n")
    r = p["referent_accuracy"]
    A("| | resolved | total | rate |")
    A("|---|---|---|---|")
    A(f"| as-measured (automated, `order_mapping` exact) | "
      f"{r['as_measured']['resolved']} | {r['as_measured']['total']} | "
      f"{r['as_measured']['pct']} % |")
    A(f"| adjudicated (human call per variable) | "
      f"{r['adjudicated']['resolved']} | {r['adjudicated']['total']} | "
      f"{r['adjudicated']['pct']} % |")
    A("\n## Four axes\n")
    A("| axis | as-measured | adjudicated |")
    A("|---|---|---|")
    A(f"| 1. numeric accuracy | {b['numeric_correct']}/"
      f"{b['numeric_correct']+b['numeric_wrong']} | {a['numeric_correct']}/"
      f"{a['numeric_correct']+a['numeric_wrong']} |")
    A(f"| 2. alignment accuracy | {b['alignment_correct']}/"
      f"{b['alignment_correct']+b['alignment_error']} | "
      f"{a['alignment_correct']}/"
      f"{a['alignment_correct']+a['alignment_error']} |")
    A(f"| 3. content gap | {b['content_gap']} | {a['content_gap']} |")
    A(f"| 4. spurious | {b['spurious']} | {a['spurious']} |")
    A(f"| manual review | {b['manual_review']} | {a['manual_review']} |")
    A("\n## By cause\n")
    for axis in ("alignment_error", "content_gap", "spurious"):
        A(f"\n**{axis}**\n")
        A("| cause | as-measured | adjudicated |")
        A("|---|---|---|")
        for c in sorted(set(bc[axis]) | set(ac[axis])):
            A(f"| `{c}` | {bc[axis].get(c, 0)} | {ac[axis].get(c, 0)} |")
    A("\n## Disclosure\n")
    A("- " + p["note"])
    A("- Adjudication is asymmetric: it can only ever move a number in the "
      "extractor's favour. Both readings are reported and neither replaces "
      "the other.")
    A("- Unlike the `-114149479` numeric adjudication, no annotation defect is "
      "involved here: `order_mapping` is correct, and what was adjudicated is a "
      "naming-convention difference between the extractor and the annotation.")
    A("- AI-code disclosure: `referent_adjudication_prep.py` and "
      "`referent_adjudication_apply.py` are AI-authored (Claude Opus 5); "
      "prompt log is the 2026-08-07/08-11 session. The 54 calls are the "
      "author's own.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
