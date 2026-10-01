#!/usr/bin/env python
"""
Two independent pre-freeze checks on the 2026-08-11 referent adjudication.

(1) MECHANICAL — injectivity of the augmented alias map.
    For each of the 25 pilot instances, resolve every extracted variable name
    through the alias map that `referent_adjudication_apply.py` builds (the
    annotation's own `order_mapping` plus the adjudicated pairings) and verify:
      a. no two distinct extracted names resolve to the same variable index;
      b. no variable index is claimed by more than one extracted name;
      c. the augmentation is additive — it never redirects a key the annotation
         already defines.
    No judgment involved.

(2) BLIND SAMPLE — a stratified draw for human re-review.
    Case 54 (head-noun substitution, the only elimination-based pairing) is
    included unconditionally. ~20 % of the remaining 53 containment-based cases
    are drawn, allocated across the other four wrapper categories by largest
    remainder so the draw is spread rather than concentrated.

    SEED 20260815 — chosen for this draw only. Distinct from
    `extraction_sample.py`'s 20260805; the mutation harness is seed-free
    (deterministic enumeration), so no collision is possible there.

    The emitted sheet deliberately OMITS the original call, the pairing basis,
    and any resolution category, so the reviewer judges each cell fresh against
    the source text. No answer key is written; the comparison step re-reads the
    original adjudication at that time.

READ-ONLY against every existing adjudication artefact. Writes two new files:
  blind_review_sample_2026-08-15.csv
  blind_review_sample_2026-08-15.md
"""

import csv
import json
import os
import random

import extraction_score as S  # imported, never modified

CALLS = "data/referent_adjudication_sheet_2026-08-07.csv"
CSV_OUT = "data/blind_review_sample_2026-08-15.csv"
MD_OUT = "docs/blind_review_sample_2026-08-15.md"

SEED = 20260815
SAMPLE_FRACTION = 0.20

CONTAINMENT_CATEGORIES = ["investment", "unit-of-measure",
                          "usage/temporal", "count"]
ELIMINATION_CATEGORY = "head-noun substitution"

# Forced inclusions, added 2026-08-15 after inspecting the draw.
#
# The investment stratum holds 15 cases but only 3 of the 124 cells: 12 of its
# cases are percentage-return problems whose objective coefficients route to
# manual review as `non_numeric_raw`, and whose rows are all
# implicit-structural, so nothing under them was ever a content gap. Sampling a
# 3-cell stratum is not sampling. The stratum is therefore taken EXHAUSTIVELY:
# the seeded draw supplied two of its three cell-bearing cases, and the third is
# forced in here so the category is complete rather than sampled.
#
# This is additive to the seeded draw and changes no other stratum — every row
# from the 20260815 draw is preserved unchanged.
FORCED_INCLUSIONS = [("-1062928915", "soccer game", "investment")]


# ==========================================================================
# (1) mechanical injectivity check
# ==========================================================================

def injectivity_check(gt_all, calls):
    violations = []
    per_instance = []
    for pid, gt in gt_all.items():
        rec = json.load(open(os.path.join(S.RAW_DIR, f"{pid}.json")))
        ex = rec.get("extraction") if rec.get("ok") else None
        ex_names = list((ex.get("variables") or [])) if ex else []

        base = S.alias_map(gt)
        aug = dict(base)
        for (cpid, var), c in calls.items():
            if cpid != pid or not c["verdict"]:
                continue
            key = S.norm(c["extracted_name"])
            idx = gt["variables"].index(var)
            if key in base:
                violations.append({
                    "kind": "non_additive_augmentation", "instance": pid,
                    "detail": f"adjudicated key {key!r} already defined by the "
                              f"annotation (-> var {base[key]})"})
            aug[key] = idx

        claimed = {}
        for name in ex_names:
            idx = aug.get(S.norm(name))
            if idx is None:
                continue
            claimed.setdefault(idx, []).append(name)

        for idx, names in claimed.items():
            if len(names) > 1:
                violations.append({
                    "kind": "index_claimed_twice", "instance": pid,
                    "detail": f"variable {idx} ({gt['variables'][idx]!r}) claimed "
                              f"by {len(names)} extracted names: {names}"})

        norm_seen = {}
        for name in ex_names:
            n = S.norm(name)
            if n in norm_seen and norm_seen[n] != name:
                violations.append({
                    "kind": "duplicate_extracted_name", "instance": pid,
                    "detail": f"{name!r} and {norm_seen[n]!r} normalise alike"})
            norm_seen[n] = name

        per_instance.append({
            "instance": pid, "n_extracted": len(ex_names),
            "n_resolved": sum(len(v) for v in claimed.values()),
            "n_distinct_indices": len(claimed),
            "injective": all(len(v) == 1 for v in claimed.values()),
        })
    return violations, per_instance


# ==========================================================================
# (2) stratified blind sample
# ==========================================================================

def allocate(counts, total_target):
    """Largest-remainder allocation across categories."""
    total = sum(counts.values())
    exact = {k: total_target * n / total for k, n in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    short = total_target - sum(base.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)[:short]:
        base[k] += 1
    return base


def cells_for(gt, idx):
    """Every previously-gapped cell under variable `idx`."""
    out = []
    for t in gt["objective_terms"]:
        if t.get("var_index") == idx and not t["non_numeric_raw"] \
                and t["value"] is not None:
            out.append({"cell_type": "objective coefficient", "row": "-",
                        "row_context": f"objective ({gt['sense']})",
                        "raw": t["raw"], "value": t["value"]})
    for r in gt["constraints"]:
        for t in r["declared_terms"]:
            if t.get("var_index") == idx and not t["non_numeric_raw"]:
                lim = r.get("declared_limit")
                limtxt = lim["raw"] if lim else "-"
                decl = ", ".join(f"{x['variable']}={x['raw']}"
                                 for x in r["declared_terms"])
                out.append({
                    "cell_type": "row term", "row": r["name"],
                    "row_context": f"[{r['name']} · {r['ctype']}] {decl} "
                                   f"{r['operator']} {limtxt}",
                    "raw": t["raw"], "value": t["value"]})
    return out


def main():
    gt_all = json.load(open(S.GT_PATH))

    rows = list(csv.DictReader(open(CALLS)))
    calls = {}
    for r in rows:
        v = (r.get("my_call") or "").strip().casefold()
        calls[(r["instance_id"], r["gt_variable"])] = {
            "verdict": v in {"y", "yes", "correct", "true", "1"},
            "extracted_name": r["extracted_name"],
            "wrapper_category": r["wrapper_category"],
            "pairing_basis": r["pairing_basis"],
        }

    # ---- (1) ----------------------------------------------------------
    violations, per_instance = injectivity_check(gt_all, calls)
    print("=" * 74)
    print("(1) MECHANICAL — injectivity of the augmented alias map")
    print("=" * 74)
    print(f"  instances checked        {len(per_instance)}")
    print(f"  all injective            "
          f"{all(p['injective'] for p in per_instance)}")
    print(f"  extracted names resolved "
          f"{sum(p['n_resolved'] for p in per_instance)}")
    print(f"  distinct indices claimed "
          f"{sum(p['n_distinct_indices'] for p in per_instance)}")
    if violations:
        print(f"\n  VIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"    [{v['kind']}] {v['instance']}: {v['detail']}")
    else:
        print("\n  VIOLATIONS: none — mapping is injective in every instance,")
        print("  and every adjudicated key was additive.")

    # ---- (2) ----------------------------------------------------------
    by_cat = {}
    for (pid, var), c in calls.items():
        by_cat.setdefault(c["wrapper_category"], []).append((pid, var))

    containment = {k: sorted(v) for k, v in by_cat.items()
                   if k in CONTAINMENT_CATEGORIES}
    counts = {k: len(v) for k, v in containment.items()}
    n_remaining = sum(counts.values())
    target = round(n_remaining * SAMPLE_FRACTION)
    alloc = allocate(counts, target)

    rng = random.Random(SEED)
    picked = []
    for cat in CONTAINMENT_CATEGORIES:
        pool = sorted(containment.get(cat, []))
        picked += [(pid, var, cat) for pid, var in rng.sample(pool, alloc[cat])]
    for pid, var in sorted(by_cat.get(ELIMINATION_CATEGORY, [])):
        picked.append((pid, var, ELIMINATION_CATEGORY))

    already = {(p, v) for p, v, _ in picked}
    forced = []
    for pid, var, cat in FORCED_INCLUSIONS:
        assert (pid, var) in calls, f"forced inclusion {pid}/{var} is not a case"
        if (pid, var) not in already:
            picked.append((pid, var, cat))
            forced.append((pid, var, cat))

    print()
    print("=" * 74)
    print("(2) BLIND SAMPLE — stratified draw")
    print("=" * 74)
    print(f"  seed {SEED}  (distinct from extraction_sample.py's 20260805;")
    print(f"                the mutation harness is seed-free)")
    print(f"  containment cases {n_remaining}   target {SAMPLE_FRACTION:.0%} "
          f"-> {target}")
    for cat in CONTAINMENT_CATEGORIES:
        print(f"    {cat:20} {counts.get(cat,0):3} cases -> {alloc[cat]} drawn")
    print(f"    {ELIMINATION_CATEGORY:20} "
          f"{len(by_cat.get(ELIMINATION_CATEGORY,[])):3} cases -> "
          f"{len(by_cat.get(ELIMINATION_CATEGORY,[]))} drawn (unconditional)")
    for pid, var, cat in forced:
        print(f"    FORCED (exhaustive stratum): {cat} {pid} {var!r}")
    print(f"  cases drawn total   {len(picked)} "
          f"({len(picked)-len(forced)} seeded + {len(forced)} forced)")

    sheet = []
    for pid, var, cat in picked:
        gt = gt_all[pid]
        idx = gt["variables"].index(var)
        aliases = sorted(k for k, i in S.alias_map(gt).items() if i == idx)
        for cell in cells_for(gt, idx):
            sheet.append({
                "instance_id": pid,
                "ground_truth_key": " | ".join(aliases),
                "extracted_name": calls[(pid, var)]["extracted_name"],
                "wrapper_category": cat,
                "cell_type": cell["cell_type"],
                "row": cell["row"],
                "row_context": cell["row_context"],
                "value_raw": cell["raw"],
                "value_numeric": cell["value"],
                "problem_text": gt.get("document", ""),
                "my_call": "",
                "notes": "",
            })
    print(f"  review rows (cells) {len(sheet)}")
    by = {}
    for s in sheet:
        by[s["cell_type"]] = by.get(s["cell_type"], 0) + 1
    print(f"    {by}")

    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows(sheet)
    with open(MD_OUT, "w") as fh:
        fh.write(render(sheet, alloc, counts, target))
    print(f"\n  wrote {CSV_OUT}, {MD_OUT}")


def render(sheet, alloc, counts, target):
    L = []
    A = L.append
    A("# Blind re-review sample — referent adjudication\n")
    A(f"Generated 2026-08-15 by `blind_review_prep.py`. Draw seed "
      f"**{SEED}**, used for this sample only and distinct from "
      f"`extraction_sample.py`'s `20260805`; the mutation harness uses "
      f"deterministic enumeration with no seed, so no collision there.\n")
    A("**This sheet is blind by construction.** It carries no original call, no "
      "pairing basis, and no resolution category. Judge each cell fresh against "
      "the problem text, then the comparison step will diff your calls against "
      "the 2026-08-11 adjudication.\n")
    A("**The question for each row:** does the extracted variable name denote "
      "the same thing as the ground-truth key, such that this number belongs to "
      "that variable?\n")
    A("## Stratification\n")
    A("| wrapper category | cases available | cases drawn |")
    A("|---|---|---|")
    for cat in CONTAINMENT_CATEGORIES:
        A(f"| {cat} | {counts.get(cat,0)} | {alloc[cat]} |")
    A(f"| {ELIMINATION_CATEGORY} | 1 | 1 (unconditional) |")
    A(f"\nTarget was {target} of the 53 containment cases, allocated by largest "
      f"remainder.\n")
    A("\n---\n")
    cur = None
    for n, r in enumerate(sheet, 1):
        key = (r["instance_id"], r["extracted_name"])
        if key != cur:
            cur = key
            A(f"\n## `{r['instance_id']}` — extracted `{r['extracted_name']}`\n")
            A(f"- **Ground-truth key(s):** `{r['ground_truth_key']}`")
            A(f"- **Wrapper category:** {r['wrapper_category']}\n")
            A("**Problem text:**\n")
            A("> " + r["problem_text"].replace("\n", "\n> ") + "\n")
        A(f"\n### Row {n} — {r['cell_type']}"
          + (f" in `{r['row']}`" if r["row"] != "-" else "") + "\n")
        A(f"- **Context:** `{r['row_context']}`")
        A(f"- **Value in question:** `{r['value_raw']}` "
          f"(numeric `{r['value_numeric']}`)\n")
        A("**My call:**\n")
        A("**Notes:**\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
