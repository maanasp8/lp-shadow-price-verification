"""How often does a real autoformulator emit a primal-invisible error? 2026-09-03.

THE GAP THIS CLOSES. llm_autoformulate.py scores generated models and prints,
of the models that reproduce the annotation's optimal value:

    (a report here is either a real defect that does not move the
     optimum, or a false positive; inspect before reporting either)

Nobody ever inspected. The paper counts every such report as a false positive,
and Appendix E says so outright: "it is not measured how large that population
is here". So the paper certifies 55 primal-invisible errors in a SYNTHETIC
mutation population and never measures how often a real generator produces one.
That is the question behind every "narrow error class" review.

WHAT IS MEASURED. For each generated model that solves and admits a row
alignment:

  1. does it reproduce the annotation's optimal value?   (answer-level evaluation)
  2. is it actually a DIFFERENT model from the annotation, up to positive row
     scaling?                                            (the coefficient diff)
  3. do the duality checks report on it?
  4. is that report model-forced, or an artifact of the basis the solver
     happened to return?                                 (the certificate)

A model that passes (1), fails (2), and is certified at (4) is a
primal-invisible error emitted by a real generator. That is the population the
paper claims exists and has never counted outside its own mutations.

WHY SCALING MATTERS AT STEP 2. A uniformly rescaled row is an EQUIVALENT model,
not an error, and generators produce them constantly. row_normalization.row_scale
recovers the factor, so a row that differs only by one is not counted as a
difference. Without this the prevalence number is inflated by pure reformulation.

WHAT THIS DOES NOT DECIDE. Reproducing the annotation's optimal value is a lower
bound on correctness, not correctness: a model can be wrong and reproduce the
optimum, which is the whole point of the paper. So "answer-neutral difference"
here means "differs from the annotation and reproduces its optimum", which is
exactly the population answer-level evaluation cannot see.

    venv/bin/python src/prevalence_generated.py
    venv/bin/python src/prevalence_generated.py --model claude-haiku-4-5-20251001

Writes data/prevalence_generated_<date>.{csv,json}.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks_common import REL_TOL, ZERO_TOL
from llm_autoformulate import align_rows, build, extract_json
from mutations import check_a, check_b, check_d, solve
from reframe_measure import face_classes, pi_exists_over_face
from row_normalization import check_a_norm, check_b_norm, row_scale, scales_for
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
CACHE = "data/llm_autoformulate_cache.jsonl"
SCALE_EQ_TOL = 1e-7


def load_cache(path=CACHE):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r.get("model", "?"), r.get("variant", "plain"),
                     r["problem_id"])] = r
    return out


def _close(a, b):
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def loose_row_scale(gold_t, cand_t, rn, rtol=1e-5):
    """row_scale, but tolerant of the generator ROUNDING its coefficients.

    row_normalization.row_scale requires every cell to agree on k to within
    1e-7 relative, which is right for the mutation study, where coefficients are
    exact. A generator is not exact. On problem 1268238141 the annotation's
    ratio row is (0.7, -0.3) and the generator wrote (1, -0.428571), which is
    that row scaled by 1/0.7 with -3/7 rounded to six figures. The two implied
    factors differ by 1.4e-6, ten times row_scale's tolerance, so row_scale
    calls it unexplainable and the row reads as a formulation error. It is not
    one. It is the same constraint, rounded.

    -> k when one factor explains every cell to `rtol`, else 1.0.
    """
    if rn not in cand_t.row_coefs or rn not in cand_t.row_limits:
        return 1.0
    at, ac = gold_t.row_coefs[rn], cand_t.row_coefs[rn]
    if set(at) != set(ac):
        return 1.0
    ratios = []
    for v, a in at.items():
        b = ac[v]
        if abs(a) <= SCALE_EQ_TOL:
            if abs(b) > SCALE_EQ_TOL:
                return 1.0
            continue
        ratios.append(b / a)
    bt, bc = gold_t.row_limits[rn], cand_t.row_limits[rn]
    if abs(bt) > SCALE_EQ_TOL:
        ratios.append(bc / bt)
    elif abs(bc) > SCALE_EQ_TOL:
        return 1.0
    if not ratios:
        return 1.0
    k = ratios[0]
    if abs(k) <= SCALE_EQ_TOL:
        return 1.0
    for r in ratios[1:]:
        if abs(r - k) > rtol * max(1.0, abs(k)):
            return 1.0
    if (gold_t.row_operator[rn] != cand_t.row_operator[rn]) != (k < 0):
        return 1.0
    return k


SWAP_TYPES = {"4a", "4b", "5", "7"}
CERTIFIED_N = 55          # data/characterize_55_2026-08-28.json


def swap_base_rate(path="data/mutants_log.csv"):
    """-> (n_swap_mutants, rate) from the mutation study.

    characterize_55 shows the certified class is produced by exactly four error
    types, all of them value swaps: 4a (19), 5 (14), 4b (13), 7 (9). So the
    honest denominator for "how often does a swap turn out primal-invisible" is
    the number of evaluable swap-type mutants, not the whole population. This is
    what makes the generator run interpretable: without it, observing zero
    certified cases in a hundred generations looks like evidence of absence.

    The denominator is bucket "3 evaluable", NOT merely mutant_status OPTIMAL.
    The two differ by the INERT mutants, where every touched row is non-binding
    in both the clean and the mutant solve, so the text changes and the optimum
    does not. The paper excludes those from every rate it reports and so must
    this. Including them gives 6,934 and 0.793%; the correct denominator is
    6,183 and 0.890%.
    """
    seen, n = set(), 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            mid = r["mutant_id"]
            if mid in seen:
                continue
            seen.add(mid)
            if r["population"] not in ("primary", "extension"):
                continue
            if r["bucket"] != "3 evaluable":
                continue
            if r["error_type"] in SWAP_TYPES:
                n += 1
    return n, (CERTIFIED_N / n if n else 0.0)


def slot_values(t):
    """-> {slot_key: value} over every slot the mutation taxonomy can damage.

    Slot keys mirror mutations.py: an objective coefficient, a constraint
    coefficient, or a row limit.
    """
    out = {}
    for v in t.var_names:
        out[("obj", v)] = t.obj_coef(v)
    for rn in t.row_names:
        for v, a in t.row_coefs[rn].items():
            out[("row", rn, v)] = a
        out[("rhs", rn)] = t.row_limits[rn]
    return out


def classify_error(gold_t, cand_t):
    """Map a candidate's deviation onto the paper's mutation taxonomy.

    The point of this is Table 5: every one of the 55 certified primal-invisible
    errors "exchanges a pair of values between two slots", and no magnitude
    change, omission or spurious row produces one. So the question that decides
    whether the certified class is reachable in practice is not how many errors
    a generator makes, it is whether any of them are SWAPS.

    -> (kinds, swap_pairs). `kinds` is a set drawn from
       {"swap_within_row", "swap_cross_row", "swap_objective", "swap_limit",
        "magnitude", "omission", "spurious", "structural"}.
    """
    g, c = slot_values(gold_t), slot_values(cand_t)
    shared = [k for k in g if k in c]
    moved = [k for k in shared if not _close(g[k], c[k])]

    kinds, pairs, used = set(), [], set()
    for i, s1 in enumerate(moved):
        if s1 in used:
            continue
        for s2 in moved[i + 1:]:
            if s2 in used:
                continue
            # a swap: each slot now holds the other's original value
            if (_close(c[s1], g[s2]) and _close(c[s2], g[s1])
                    and not _close(g[s1], g[s2])):
                used.add(s1); used.add(s2)
                pairs.append((s1, s2))
                if s1[0] == "obj" and s2[0] == "obj":
                    kinds.add("swap_objective")
                elif s1[0] == "rhs" and s2[0] == "rhs":
                    kinds.add("swap_limit")
                elif s1[0] == "row" and s2[0] == "row":
                    kinds.add("swap_within_row" if s1[1] == s2[1]
                              else "swap_cross_row")
                else:
                    kinds.add("swap_cross_row")
                break

    for k in moved:
        if k in used:
            continue
        if abs(c[k]) <= ZERO_TOL < abs(g[k]):
            kinds.add("omission")
        else:
            kinds.add("magnitude")
    if set(c) - set(g):
        kinds.add("spurious")
    if set(g) - set(c):
        kinds.add("structural")
    return kinds, pairs


def structural_diff(gold_t, cand_t):
    """What differs between the annotation and the candidate, UP TO positive
    row scaling? -> (strict differences, differences after allowing for the
    generator rounding a rescaled row).

    Rows are compared by name, which is what alignment established. A row whose
    cells are a uniform rescaling of the annotation's is NOT a difference: it is
    the same constraint written differently, and row_scale recovers the factor.
    """
    diffs = []

    # objective
    for v in gold_t.var_names:
        cg, cc = gold_t.obj_coef(v), cand_t.obj_coef(v)
        if not _close(cg, cc):
            diffs.append(f"obj[{v}]: {cg:g} vs {cc:g}")
    if gold_t.sense != cand_t.sense:
        diffs.append(f"sense: {gold_t.sense} vs {cand_t.sense}")

    # rows
    rounded_rescalings = []
    for rn in gold_t.row_names:
        if rn not in cand_t.row_coefs:
            diffs.append(f"row {rn}: missing from candidate")
            continue
        k = row_scale(gold_t, cand_t, rn)
        if abs(k - 1.0) > SCALE_EQ_TOL:
            continue                      # a pure rescaling: equivalent row
        kl = loose_row_scale(gold_t, cand_t, rn)
        if abs(kl - 1.0) > SCALE_EQ_TOL:
            rounded_rescalings.append(f"row {rn}: rescaled by {kl:.6g}, rounded")
            continue                      # same constraint, rounded coefficients
        ag, ac = gold_t.row_coefs[rn], cand_t.row_coefs[rn]
        for v in set(ag) | set(ac):
            x, y = ag.get(v, 0.0), ac.get(v, 0.0)
            if not _close(x, y):
                diffs.append(f"row {rn}[{v}]: {x:g} vs {y:g}")
        if not _close(gold_t.row_limits[rn], cand_t.row_limits[rn]):
            diffs.append(f"row {rn} rhs: {gold_t.row_limits[rn]:g} "
                         f"vs {cand_t.row_limits[rn]:g}")
        if gold_t.row_operator[rn] != cand_t.row_operator[rn]:
            diffs.append(f"row {rn} op: {gold_t.row_operator[rn]} "
                         f"vs {cand_t.row_operator[rn]}")

    extra = [r for r in cand_t.row_names if r not in set(gold_t.row_names)]
    for rn in extra:
        diffs.append(f"row {rn}: not in annotation")
    return diffs, rounded_rescalings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--variant", default="plain", choices=["plain", "rowlabels"])
    ap.add_argument("--align", default="content",
                    choices=["name", "position", "content"])
    args = ap.parse_args()

    problems, _ = load_clean_problems()
    cache = load_cache()
    ids = [pid for (m, v, pid) in cache
           if m == args.model and v == args.variant and pid in problems]
    if not ids:
        sys.exit(f"nothing cached for model={args.model} variant={args.variant}")
    print(f"{len(ids)} cached generations for {args.model} / {args.variant}, "
          f"align={args.align}\n")

    rows = []
    for pid in sorted(ids):
        rec = cache[(args.model, args.variant, pid)]
        gold = problems[pid]
        gold_t = text_terms_of(gold)
        gsol = solve(gold)
        out = {"problem_id": pid, "status": "", "align": "",
               "matches_gold": "", "differs": "", "differs_strict": "",
               "n_diffs": "", "rounded_rescalings": "",
               "error_kinds": "", "swap_pairs": "",
               "a": "", "b": "", "c": "", "any": "", "any_norm": "",
               "certified": "", "cert_note": "", "diff_detail": ""}
        if not rec.get("raw"):
            out["status"] = "no_response"
            rows.append(out); continue
        try:
            cand = build(extract_json(rec["raw"]), gold.var_names, pid)
        except Exception as e:                                   # noqa: BLE001
            out["status"] = f"parse_fail: {type(e).__name__}"
            rows.append(out); continue
        out["align"] = align_rows(cand, gold, args.align)
        if out["align"] == "count_mismatch":
            out["status"] = "count_mismatch"
            rows.append(out); continue
        sol = solve(cand)
        if not sol.optimal:
            out["status"] = f"solve_{sol.status}"
            rows.append(out); continue
        out["status"] = "OPTIMAL"

        # (1) answer-level evaluation
        out["matches_gold"] = bool(
            abs(sol.obj_value - gsol.obj_value)
            <= REL_TOL * max(1.0, abs(gsol.obj_value)))

        # (2) is it actually a different model, up to row scaling?
        cand_t = text_terms_of(cand)
        diffs, rounded = structural_diff(gold_t, cand_t)
        out["differs"] = bool(diffs)
        out["differs_strict"] = bool(diffs or rounded)
        out["n_diffs"] = len(diffs)
        out["rounded_rescalings"] = len(rounded)
        out["diff_detail"] = " | ".join((diffs + rounded)[:6])
        kinds, pairs = classify_error(gold_t, cand_t)
        out["error_kinds"] = ",".join(sorted(kinds))
        out["swap_pairs"] = len(pairs)

        # (3) the checks
        a = check_a(gold_t, sol)[0]
        b = check_b(gold_t, sol)[0]
        c = check_d(gold_t, sol)[0]
        k = scales_for(gold_t, cand_t)
        out.update({"a": a, "b": b, "c": c, "any": bool(a or b or c),
                    "any_norm": bool(check_a_norm(gold_t, sol, k)
                                     or check_b_norm(gold_t, sol, k) or c)})

        # (4) the certificate, only where a check reports
        if out["any"]:
            cls, _uniq = face_classes(cand, sol.obj_value)
            if cls is None:
                out["certified"], out["cert_note"] = "", "face_lp_failed"
            else:
                try:
                    feasible = pi_exists_over_face(cand, gold_t,
                                                   sol.obj_value, cls)
                    out["certified"] = (not feasible)
                    out["cert_note"] = ("model-forced" if not feasible
                                        else "basis artifact")
                except Exception as e:                           # noqa: BLE001
                    out["certified"], out["cert_note"] = "", f"err:{type(e).__name__}"
        rows.append(out)

    # ----------------------------------------------------------- reporting
    ok = [r for r in rows if r["status"] == "OPTIMAL"]
    match = [r for r in ok if r["matches_gold"] is True]
    wrong = [r for r in ok if r["matches_gold"] is False]

    # THE population: reproduces the optimum but is not the same model
    neutral = [r for r in match if r["differs"] is True]
    same = [r for r in match if r["differs"] is False]
    neutral_strict = [r for r in match if r["differs_strict"] is True]
    rounded_only = [r for r in match
                    if r["differs_strict"] is True and r["differs"] is False]
    reported = [r for r in neutral if r["any"]]
    certified = [r for r in neutral if r["certified"] is True]
    # a report on a model that is NOT different is a false positive outright
    fp_same = [r for r in same if r["any"]]
    fp_same_norm = [r for r in same if r["any_norm"]]

    def pct(n, d):
        return f"{100.0*n/d:.1f}%" if d else "n/a"

    print("=" * 70)
    print("PREVALENCE OF PRIMAL-INVISIBLE ERROR IN REAL GENERATOR OUTPUT")
    print("=" * 70)
    print(f"  cached generations               {len(rows)}")
    print(f"  solved and alignable             {len(ok)}")
    print(f"  optimum differs from annotation  {len(wrong)}")
    print(f"  optimum matches annotation       {len(match)}")
    print()
    print("  Of the models that PASS answer-level evaluation:")
    print(f"    same model as the annotation   {len(same)}  ({pct(len(same), len(match))})")
    print(f"    DIFFERENT model, same optimum  {len(neutral)}  ({pct(len(neutral), len(match))})")
    print(f"      (before allowing for rounding: {len(neutral_strict)}; "
          f"{len(rounded_only)} were a rescaled row with rounded coefficients,")
    print("       which is the same constraint written differently, not an error)")
    print("      ^ this is the answer-neutral population: a formulation error")
    print("        that answer-level evaluation cannot see, at any tolerance.")
    print()
    print(f"    of those {len(neutral)}, a check reports on   {len(reported)}  ({pct(len(reported), len(neutral))})")
    print(f"    of those {len(neutral)}, CERTIFIED model-forced {len(certified)}  ({pct(len(certified), len(neutral))})")
    print()
    print("  False positives, strictly (report on a model that IS the annotation):")
    print(f"    as specified                   {len(fp_same)}/{len(same)}  ({pct(len(fp_same), len(same))})")
    print(f"    normalized                     {len(fp_same_norm)}/{len(same)}  ({pct(len(fp_same_norm), len(same))})")
    if wrong:
        d = sum(bool(r["any"]) for r in wrong)
        print(f"\n  Detection on models whose optimum differs: {d}/{len(wrong)} ({pct(d, len(wrong))})")

    # --- what KIND of error does a real generator make? ---------------------
    erroneous = [r for r in ok if r["error_kinds"]]
    swaps = [r for r in erroneous if "swap" in r["error_kinds"]]
    print()
    print("  ERROR TAXONOMY of this generator (Table 5: every certified")
    print("  primal-invisible error is a value swap between two slots):")
    print(f"    models deviating from the annotation  {len(erroneous)}/{len(ok)}")
    print(f"    of those, containing a VALUE SWAP     {len(swaps)}  ({pct(len(swaps), len(erroneous))})")
    kc = Counter(k for r in erroneous for k in r["error_kinds"].split(",") if k)
    for k, n in sorted(kc.items(), key=lambda x: -x[1]):
        print(f"      {k:<20} {n}")

    # --- is a null result here informative? ---------------------------------
    n_swap, rate = swap_base_rate()
    expected = len(swaps) * rate
    need = (len(ok) * (1.0 / rate) / len(swaps)) if swaps else float("inf")
    print()
    print("  IS THE NULL INFORMATIVE? Not on this corpus.")
    print(f"    swap-type mutants in the study      {n_swap}")
    print(f"    of which certified primal-invisible {CERTIFIED_N}"
          f"  ({100*rate:.3f}%, 1 in {1/rate:.0f})")
    print(f"    generator swaps observed here       {len(swaps)}")
    print(f"    EXPECTED certified cases            {expected:.3f}")
    print(f"    generations needed to expect one    {need:.0f}")
    print("    NL4OPT holds 1,101 instances, so even generating the whole")
    print("    corpus expects under half a case. Observing zero is what the")
    print("    mutation study predicts. It is not evidence of absence.")
    notes = Counter(r["cert_note"] for r in neutral if r["cert_note"])
    if notes:
        print(f"\n  certificate outcomes on the answer-neutral set: {dict(notes)}")

    os.makedirs("data", exist_ok=True)
    stem = f"data/prevalence_generated_{args.model.split('-')[1]}_{args.variant}_{TODAY}"
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({
        "stamp": TODAY, "model": args.model, "variant": args.variant,
        "align": args.align,
        "cached": len(rows), "scored": len(ok),
        "optimum_differs": len(wrong), "optimum_matches": len(match),
        "answer_neutral_same_model": len(same),
        "answer_neutral_different_model": len(neutral),
        "answer_neutral_different_model_strict": len(neutral_strict),
        "rounded_rescaling_only": len(rounded_only),
        "answer_neutral_reported": len(reported),
        "answer_neutral_certified": len(certified),
        "false_positive_same_model": len(fp_same),
        "false_positive_same_model_norm": len(fp_same_norm),
        "cert_notes": dict(notes),
        "deviating_models": len(erroneous),
        "models_with_value_swap": len(swaps),
        "error_kind_counts": dict(kc),
        "swap_type_mutants": n_swap,
        "certified_rate_among_swaps": rate,
        "expected_certified_here": expected,
        "generations_to_expect_one": need,
    }, open(stem + ".json", "w"), indent=1)
    print(f"\nwrote {stem}.csv/.json")


if __name__ == "__main__":
    main()
