"""Witness ledger for the degeneracy certificate, added 2026-09-30 (MATH-AI camera-ready).

THE OBJECTION THIS ANSWERS
--------------------------
Reviewers at three venues raised the same point about the degeneracy certificate:
the existence program is a relaxation, so its infeasibility soundly certifies a
report against every optimal basis, but its feasibility does not by itself prove
that a single valid optimal pair removes the report. When certification fails, the
paper should establish that some optimal solution actually silences the report,
rather than only that certification did not succeed.

WHY FEASIBILITY IS EXACT HERE
-----------------------------
The relaxation decides each variable's complementary-slackness branch over the
whole primal optimal face. It is only a relaxation of "some primal optimum and some
dual-optimal pi satisfy the checks" when the optimal face is non-trivial, that is
when the primal optimum is not unique. When the primal optimum is unique, x* is
fixed, every branch is decided by that single x*, and the program is exact rather
than relaxed. A feasible point is then a genuine dual-optimal pi paired with the
actual optimal x*, which is a witness that the report is a basis artifact.

The certificate program (reframe_measure.py) records, per report, whether a
feasible dual exists over the optimal face and whether the primal optimum is
unique. This script audits those outcomes into three disjoint classes:
    certified  infeasible over the face, unavoidable against every optimal basis
    witnessed  feasible over the face AND primal optimum unique, so exact
    unresolved feasible over the face but primal optimum not unique
and reports the counts the manuscript quotes. On this corpus every withdrawn report
has a unique primal optimum, so the unresolved class is empty and the four
non-unique mutants all fall in the certified class, where infeasibility holds
regardless of uniqueness.

REPRODUCIBILITY
---------------
Reads the persisted certificate outcomes, matching how the tables assert against
per-instance artefacts, so the counts match the manuscript exactly.

Input : data/reframe_face_2026-08-27.csv
        columns pi_exists_over_face (feasible dual over the optimal face) and
        primal_unique (primal optimum unique) for each of the 86 check reports.
Output: data/witness_ledger_<date>.{csv,json}
"""

from __future__ import annotations

import csv
import datetime
import json
import os

TODAY = datetime.date.today().isoformat()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
FACE = os.path.join(DATA, "reframe_face_2026-08-27.csv")


def _truthy(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "t")


def classify(pi_exists: bool, primal_unique: bool) -> str:
    if not pi_exists:
        return "certified"          # infeasible over the face: unavoidable
    if primal_unique:
        return "witnessed"          # feasible and exact: a genuine silencing pair
    return "unresolved"             # feasible but relaxed: not a proven witness


def main() -> None:
    rows = []
    counts = {"certified": 0, "witnessed": 0, "unresolved": 0}
    for r in csv.DictReader(open(FACE, newline="")):
        pi_exists = _truthy(r["pi_exists_over_face"])
        primal_unique = _truthy(r["primal_unique"])
        cls = classify(pi_exists, primal_unique)
        counts[cls] += 1
        rows.append({
            "mutant_id": r["mutant_id"],
            "pi_exists_over_face": pi_exists,
            "primal_unique": primal_unique,
            "class": cls,
        })

    n = len(rows)
    summary = {
        "n_reports": n,
        "certified": counts["certified"],
        "witnessed": counts["witnessed"],
        "unresolved": counts["unresolved"],
        "stamp": TODAY,
    }

    # The manuscript states 55 certified, 31 witnessed, 0 unresolved over the 86
    # check reports. A camera-ready number is not shipped on a silent drift.
    assert n == 86, f"expected 86 reports, got {n}"
    assert counts["certified"] == 55, f"certified {counts['certified']}, expected 55"
    assert counts["witnessed"] == 31, f"witnessed {counts['witnessed']}, expected 31"
    assert counts["unresolved"] == 0, f"unresolved {counts['unresolved']}, expected 0"

    with open(os.path.join(DATA, f"witness_ledger_{TODAY}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(DATA, f"witness_ledger_{TODAY}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
