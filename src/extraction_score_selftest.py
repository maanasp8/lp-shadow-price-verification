"""EXTRACTION PILOT — scorer self-test. Synthetic damage, known right answers.

    venv/bin/python extraction_score_selftest.py

Runs the scorer against hand-built extractions whose correct scores are known in
advance, and FAILS LOUDLY on any mismatch. No API calls; no real extraction is
read. Every expectation below is written from the ground truth by hand, not
copied from a previous run of the scorer.

WHY THIS FILE EXISTS
--------------------
The variable-alias bug (pre-registration §5.3 amendment, revision 2) was
introduced precisely because a change that "was obviously correct" went in
without a test that would have caught it: `extraction_ground_truth.py` recorded
variables under canonical names and coefficients under raw declaration names,
and nothing checked that a *flawless* extraction scored clean. Cases 1-3 and 8
are that missing check.

Cases 5 and 6 re-run the synthetic damage the pre-registration §5.2 already
claims to have verified, so the alias fix is shown not to have broken the
numeric/alignment split it depends on.

FIXTURE: `-1501301743` — 3 variables, 2 linear rows, no non-numeric cells, and
an `order_mapping` that names every variable twice:

    {'Standard': 0, 'Plus': 1, 'Premium': 2,
     'Standard grade alloy': 0, 'Plus grade alloy': 1, 'Premium grade alloy': 2}

    max  1500*Standard + 2500*Plus + 5500*Premium
    c0:  3*Standard + 5*Plus + 6*Premium <= 150
    c1:  1*Standard + 2*Plus + 3*Premium <= 35

11 scorable cells (3 objective + 2 limits + 6 declared terms).
14 alignment observations (1 sense + 2 operators + 2 limits + 3 objective
+ 6 terms) when every variable is matched.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional

from extraction_score import score_instance

GT_PATH = "data/extraction_pilot/ground_truth.json"
PID = "-1501301743"

CANON = ("Standard", "Plus", "Premium")
ALIAS = ("Standard grade alloy", "Plus grade alloy", "Premium grade alloy")
UNKNOWN = ("Widget A", "Widget B", "Widget C")

OBJ = (1500, 2500, 5500)
C0 = (3, 5, 6)
C1 = (1, 2, 3)


def build(var_names, obj=OBJ, c0=C0, c1=C1, term_names=None, extra_vars=()):
    """A structurally complete extraction. `term_names` lets the terms use a
    different spelling than the `variables` list, which is the exact situation
    the alias fix has to handle."""
    tn = term_names or var_names
    return {
        "variables": list(var_names) + list(extra_vars),
        "objective": {"sense": "max", "terms": [
            {"variable": tn[i], "coefficient": obj[i]} for i in range(3)]},
        "constraints": [
            {"id": "iron", "quoted_text": "150 tons of iron ore",
             "operator": "<=", "limit": 150,
             "terms": [{"variable": tn[i], "coefficient": c0[i]} for i in range(3)]},
            {"id": "zinc", "quoted_text": "35 tons of zinc ore",
             "operator": "<=", "limit": 35,
             "terms": [{"variable": tn[i], "coefficient": c1[i]} for i in range(3)]},
        ],
    }


# (name, extraction, expected) — expected is written by hand from the fixture.
#   num  = (numeric_correct, numeric_total)
#   algn = (alignment_correct, alignment_total)
CASES = [
    ("1. perfect, canonical names ('Standard')",
     build(CANON),
     {"num": (11, 11), "algn": (14, 14), "gap": 0, "spur": 0},
     "every referent resolves; spelling is the one in `vars[]`"),

    ("2. perfect, alias names ('Standard grade alloy')",
     build(ALIAS),
     {"num": (11, 11), "algn": (14, 14), "gap": 0, "spur": 0},
     "SAME score as case 1 — this is the bug the amendment fixes"),

    ("3. perfect, mixed (canonical list, alias terms)",
     build(CANON, term_names=ALIAS),
     {"num": (11, 11), "algn": (14, 14), "gap": 0, "spur": 0},
     "resolution is per cell, not inherited from the variables list"),

    ("4. names absent from order_mapping ('Widget A')",
     build(UNKNOWN),
     {"num": (2, 2), "algn": (5, 8), "gap": 9, "spur": 12},
     "unrecognised names must NOT match: 3 gt vars unmatched -> alignment "
     "errors; 9 cells unfilled -> content gaps; 3 vars + 9 filled cells with "
     "no gt counterpart -> spurious. Limits/operators/sense still score."),

    ("5. objective coefficients swapped (Standard<->Plus)",
     build(CANON, obj=(2500, 1500, 5500)),
     {"num": (9, 9), "algn": (12, 14), "gap": 0, "spur": 0},
     "§5.2: right number on the wrong referent is an ALIGNMENT error and "
     "leaves the numeric denominator (11 -> 9)"),

    ("6. one coefficient garbled (c0 Standard 3 -> 99)",
     build(CANON, c0=(99, 5, 6)),
     {"num": (10, 11), "algn": (14, 14), "gap": 0, "spur": 0},
     "§5.2: a value matching nothing in scope is a NUMERIC error only"),

    ("7. ABSENT where a real coefficient exists",
     build(CANON, c0=("ABSENT", 5, 6)),
     {"num": (10, 10), "algn": (13, 13), "gap": 1, "spur": 0},
     "'ABSENT' is not a number — it is a content gap, not a wrong value"),

    ("8. one variable listed twice under both spellings",
     build(CANON, extra_vars=("Standard grade alloy",)),
     {"num": (11, 11), "algn": (14, 14), "gap": 0, "spur": 1},
     "aliases resolve to one referent; the second mention is over-extraction, "
     "NOT a second match"),
]


def main() -> int:
    gt = json.load(open(GT_PATH))[PID]
    if "variable_aliases" not in gt:
        raise SystemExit(
            f"{GT_PATH} predates the alias fix — re-run "
            f"`extraction_ground_truth.py --write` first.")

    print("=" * 78)
    print(f"EXTRACTION PILOT — SCORER SELF-TEST   fixture {PID}")
    print("=" * 78)
    print(f"  order_mapping: {gt['variable_aliases']}")
    print(f"  variables:     {gt['variables']}\n")

    failures: List[str] = []
    for name, ex, exp, why in CASES:
        r = score_instance(gt, ex)
        got = {
            "num": (r["numeric"]["correct"],
                    r["numeric"]["correct"] + r["numeric"]["wrong"]),
            "algn": (r["alignment"]["correct"],
                     r["alignment"]["correct"] + r["alignment"]["error"]),
            "gap": r["content_gap"],
            "spur": r["spurious"],
        }
        ok = got == exp
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {why}")
        print(f"         expected  numeric {exp['num'][0]}/{exp['num'][1]}   "
              f"alignment {exp['algn'][0]}/{exp['algn'][1]}   "
              f"gaps {exp['gap']}   spurious {exp['spur']}")
        print(f"         got       numeric {got['num'][0]}/{got['num'][1]}   "
              f"alignment {got['algn'][0]}/{got['algn'][1]}   "
              f"gaps {got['gap']}   spurious {got['spur']}")
        if not ok:
            failures.append(name)
            for k in ("num", "algn", "gap", "spur"):
                if got[k] != exp[k]:
                    print(f"         *** {k}: expected {exp[k]}, got {got[k]}")
        print()

    print("-" * 78)
    if failures:
        print(f"  {len(failures)}/{len(CASES)} FAILED: {failures}")
        return 1
    print(f"  all {len(CASES)} cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
