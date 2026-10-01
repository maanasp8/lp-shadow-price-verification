"""Feasibility-plus-objective baseline, added 2026-09-30 for the MATH-AI camera-ready.

THE OBJECTION THIS ANSWERS
--------------------------
A MATH-AI reviewer proposed a simpler fix for the reference-solve baseline. Rather
than compare the candidate's returned solution against the particular optimum the
reference solver returned, solve the reference model for its optimal value z_ref
and then test the candidate's returned solution x* on two conditions:
    (1) x* is feasible for the reference (text) model, and
    (2) c^T x* equals z_ref.
If both hold, x* is optimal for the reference model. The reviewer's point is that
this test does not depend on which reference optimum the solver happened to return,
so it is not exposed to the basis-selection confound on the reference side.

WHAT THIS BASELINE ACTUALLY IS
------------------------------
Condition (1) evaluates every text row at the candidate's x* and asks whether that
point is feasible for the text. That is exactly Check C (labeled Check D in the
artifacts), evaluated at the returned x*. Condition (2) is the objective test the
reference-solve baseline already applies. So the reviewer's baseline is Check C
combined with an objective comparison, and it fires when
    the candidate's x* is infeasible for the text   (Check C / check_d fires), or
    the candidate's optimal value moved from z_ref   (obj_moved).

WHAT IS MEASURED
----------------
Detection over the 18,525 evaluable primary mutants, and coverage of the 55
certified answer-neutral class. The baseline is silent exactly when the candidate's
returned x* is feasible for the text and reaches z_ref, which is the answer-neutral
case, so it cannot see any of the 55 by construction. This is verified, not assumed.

REPRODUCIBILITY
---------------
Like the table generators, this reads the per-instance artefacts persisted with the
paper rather than re-solving, so the number it reports matches the manuscript
exactly. The check_d and obj_moved flags are taken from the same files the union
figure is built from, and the A|B|C union is recomputed here as a self-check
against the published 91.87%.

Inputs : data/checkD_full_population_2026-08-19.csv  (d_fired / a_fired / b_fired)
         data/reframe_2026-08-27.csv                 (obj_moved, evaluable arms)
         data/characterize_55_2026-08-28.csv         (the certified class)
Outputs: data/feas_obj_baseline_<date>.{csv,json}
"""

from __future__ import annotations

import csv
import datetime
import json
import os

TODAY = datetime.date.today().isoformat()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

CHECKD = os.path.join(DATA, "checkD_full_population_2026-08-19.csv")
REFRAME = os.path.join(DATA, "reframe_2026-08-27.csv")
CLASS55 = os.path.join(DATA, "characterize_55_2026-08-28.csv")

# The evaluable primary population is the two reframe arms that partition it: the
# mutants whose optimum moved and those that are answer-neutral but not inert.
EVALUABLE_ARMS = ("moved", "still")


def _truthy(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "t")


def load_check_flags() -> dict:
    """mutant_id -> {d, a, b} for the evaluable primary population, deduped."""
    flags = {}
    with open(CHECKD, newline="") as f:
        for r in csv.DictReader(f):
            flags[r["mutant_id"]] = {
                "d": _truthy(r["d_fired"]),
                "a": _truthy(r["a_fired"]),
                "b": _truthy(r["b_fired"]),
            }
    return flags


def load_obj_moved() -> dict:
    """mutant_id -> obj_moved for the evaluable arms only."""
    obj = {}
    with open(REFRAME, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] in EVALUABLE_ARMS:
                obj[r["id"]] = _truthy(r["obj_moved"])
    return obj


def main() -> None:
    flags = load_check_flags()
    obj = load_obj_moved()
    ids = sorted(set(flags) & set(obj))
    n = len(ids)

    rows = []
    detected = both = only_feas = only_obj = union_abc = 0
    for i in ids:
        infeasible = flags[i]["d"]           # Check C at x*: x* not feasible for text
        moved = obj[i]                        # objective moved from z_ref
        fire = infeasible or moved            # reviewer's baseline reports an error
        abc = flags[i]["d"] or flags[i]["a"] or flags[i]["b"]
        if fire:
            detected += 1
        if infeasible and moved:
            both += 1
        elif infeasible:
            only_feas += 1
        elif moved:
            only_obj += 1
        if abc:
            union_abc += 1
        rows.append({
            "mutant_id": i,
            "feasible_for_text": not infeasible,
            "obj_moved": moved,
            "feas_obj_detects": fire,
        })

    class55 = [r["mutant_id"] for r in csv.DictReader(open(CLASS55, newline=""))]
    det55 = sum(
        1 for i in class55
        if i in flags and i in obj and (flags[i]["d"] or obj[i])
    )

    summary = {
        "n_evaluable": n,
        "feas_obj_detected": detected,
        "feas_obj_rate": round(detected / n, 4),
        "breakdown": {
            "infeasible_and_moved": both,
            "infeasible_only": only_feas,
            "moved_only": only_obj,
        },
        "abc_union_selfcheck": union_abc,
        "abc_union_rate_selfcheck": round(union_abc / n, 4),
        "class55_size": len(class55),
        "class55_detected_by_feas_obj": det55,
        "stamp": TODAY,
    }

    # Self-checks: the join must cover the whole evaluable population, the union
    # self-check must reproduce the published rate, and the baseline must miss the
    # whole certified class. A camera-ready number is not shipped on a silent drift.
    assert n == 18525, f"evaluable population is {n}, expected 18525"
    assert union_abc == 17018, f"A|B|C union is {union_abc}, expected 17018 (91.87%)"
    assert det55 == 0, f"feas+obj detected {det55} of the 55, expected 0"

    with open(os.path.join(DATA, f"feas_obj_baseline_{TODAY}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(DATA, f"feas_obj_baseline_{TODAY}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
