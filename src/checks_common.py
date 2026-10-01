"""Shared loading and integrity assertions for Check A / Check B.

Reads only the persisted CSVs. Gurobi is never imported.

Scope everywhere is the OPTIMAL instances. Nothing here chooses a tolerance
other than the frozen relative tolerance below.
"""

import json

import pandas as pd

# Frozen by the method spec: relative, single, not configurable.
# fires when |a - b| > REL_TOL * max(1, |b|)
REL_TOL = 1e-6

# ZERO_TOL = 1e-4, set 2026-08-01 by the author.
# Chosen inside the empirical gap (0.0, 0.1666667) measured by pick_zero_tol.py --
# five orders of magnitude from either boundary -- and kept deliberately distinct
# from the 1e-6 fire tolerance so classification and detection stay separately
# named constants.
ZERO_TOL = 1e-4


def fires_equality(value, target):
    return abs(value - target) > REL_TOL * max(1.0, abs(target))


PHANTOM_EXCLUSIONS = "data/phantom_variable_exclusions.csv"


def load_mutation_exclusions(path=PHANTOM_EXCLUSIONS):
    """Instance ids the MUTATION HARNESS must skip. -> set[str]

    Scope: mutation only. Check A and Check B report the full OPTIMAL corpus and
    must not call this -- the excluded instances passed both checks cleanly and
    those results stand.

    An instance is excluded when it carries a phantom variable with a live
    (nonzero) coefficient, so a mutation could move it off zero and produce a
    result that describes nothing in the source text. Phantoms with no nonzero
    coefficient are structurally inert and are retained.

    Raises if the file is missing: an exclusion list that silently degrades to
    "exclude nothing" is worse than no exclusion list.
    """
    import csv
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- regenerate it with make_phantom_exclusions.py "
            f"before running the mutation harness")
    with open(path) as fh:
        return {r["problem_id"] for r in csv.DictReader(fh)
                if r["excluded_from_mutation"] == "True"}


def load(verbose=True):
    """-> (cons, objs, vars_, optimal_ids). coefficients parsed to dicts."""
    results = pd.read_csv("data/results.csv", dtype={"problem_id": str})
    optimal = set(results.loc[results.solve_status == "OPTIMAL", "problem_id"])

    cons = pd.read_csv("data/constraints.csv", dtype={"problem_id": str})
    objs = pd.read_csv("data/objectives.csv", dtype={"problem_id": str})
    vars_ = pd.read_csv("data/variables.csv", dtype={"problem_id": str})

    cons = cons[cons.problem_id.isin(optimal)].copy()
    objs = objs[objs.problem_id.isin(optimal)].copy()
    vars_ = vars_[vars_.problem_id.isin(optimal)].copy()

    cons["coefs"] = cons["coefficients"].map(json.loads)
    objs["coefs"] = objs["coefficients"].map(json.loads)

    # The spec calls this field `status`; variables.csv stores it as `label`.
    # Alias in memory only -- the CSV is read-only and is not rewritten.
    vars_["status"] = vars_["label"]

    if verbose:
        print(f"  OPTIMAL instances     {len(optimal)}")
        print(f"  constraint rows       {len(cons)}")
        print(f"  objective rows        {len(objs)}")
        print(f"  variable rows         {len(vars_)}")

    _assert_integrity(cons, objs, vars_, optimal, results)
    return cons, objs, vars_, optimal


def _assert_integrity(cons, objs, vars_, optimal, results):
    """Loud failures for structural problems; reports for judgement calls."""
    known = vars_.groupby("problem_id")["var_name"].apply(set).to_dict()

    # --- 1. every coefficient key must be a known variable of that instance ---
    bad = []
    for r in cons.itertuples():
        unknown = set(r.coefs) - known.get(r.problem_id, set())
        if unknown:
            bad.append(("constraint", r.problem_id, r.constraint_name, sorted(unknown)))
    for r in objs.itertuples():
        unknown = set(r.coefs) - known.get(r.problem_id, set())
        if unknown:
            bad.append(("objective", r.problem_id, "-", sorted(unknown)))
    if bad:
        for b in bad[:20]:
            print(f"    UNKNOWN VARIABLE {b}")
        raise AssertionError(
            f"{len(bad)} coefficient dict(s) reference variables not in "
            f"variables.csv for that instance -- names are not normalised, so "
            f"this is a real mismatch, not a typo")

    # --- 2. every OPTIMAL instance has a pi for every one of its rows --------
    if cons["pi"].isna().any():
        n = int(cons["pi"].isna().sum())
        raise AssertionError(f"{n} constraint row(s) have a missing pi")
    missing = optimal - set(cons.problem_id)
    have_rows = cons.groupby("problem_id").size()
    expect = results.set_index("problem_id")["n_constraints"]
    mismatched = [
        (pid, int(have_rows[pid]), int(float(expect[pid])))
        for pid in sorted(have_rows.index)
        if int(have_rows[pid]) != int(float(expect[pid]))
    ]
    if missing:
        # an instance with zero declared rows legitimately has no rows at all
        zero_row = {p for p in missing if int(float(expect[p])) == 0}
        real = missing - zero_row
        if real:
            raise AssertionError(
                f"{len(real)} OPTIMAL instance(s) have no rows in constraints.csv: "
                f"{sorted(real)[:10]}")
        print(f"    note: {len(zero_row)} OPTIMAL instance(s) declare zero rows")
    if mismatched:
        for m in mismatched[:20]:
            print(f"    ROW COUNT MISMATCH {m}")
        raise AssertionError(
            f"{len(mismatched)} instance(s) have a different number of rows in "
            f"constraints.csv than results.csv n_constraints")

    # --- 3. status vs value disagreements: REPORT, never resolve -------------
    v = vars_
    at_zero = v["value"].abs() == 0.0
    suspicious = {
        "value == 0 but flagged AT_UPPER": v[at_zero & (v.status == "AT_UPPER")],
        "value == 0 but flagged FREE": v[at_zero & (v.status == "FREE")],
        "value != 0 but flagged AT_LOWER": v[~at_zero & (v.status == "AT_LOWER")],
        "status FREE yet a binding row is named":
            v[(v.status == "FREE") & v.restricted_by.fillna("").ne("")],
    }
    print("\n  STATUS-vs-VALUE REPORT (report only, nothing resolved):")
    for label, frame in suspicious.items():
        print(f"    {label:<44} {len(frame)}")
        if label == "value == 0 but flagged AT_UPPER" and len(frame):
            for r in frame.head(10).itertuples():
                print(f"        {r.problem_id}  {r.var_name}  value={r.value}  "
                      f"restricted_by={r.restricted_by!r}")
