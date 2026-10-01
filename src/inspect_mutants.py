"""Full before/after inspection of individual mutants.

    venv/bin/python inspect_mutants.py type3     three of the logged type-3 pilot mutants
    venv/bin/python inspect_mutants.py type8     a fresh 6-mutant type-8 mini-pilot [--write]

For each mutant this prints, from a rebuild rather than from the log:

  * every cell of the CLEAN canonical model and every cell of the MUTANT's,
    diffed -- so "no other cell changed" is shown, not asserted;
  * the bucket, with the clean/mutant slack that decided it;
  * Check B's actual sum and target at full precision, term by term;
  * Check A's per-variable sum, target and residual.

The canonical cell view is `lhs - rhs_terms` per row plus the RHS constant, the
operator, the objective coefficients and the sense -- i.e. everything that can
differ between two models built by `build_model`. A diff of that view is a
complete statement of what a mutation touched.
"""

from __future__ import annotations

import argparse
import copy
import datetime
from typing import Dict, Tuple

from mutations import (DEFAULT_SCOPE, ENUMERATORS, binding, bucket_of, check_a,
                       check_b, is_noop, solve, spurious_fraction, usage)
from mutation_harness import (APPLICABLE, append_log, in_scope, run_candidate)
from text_terms import TextTermsStore, load_clean_problems, row_coefficients

Cell = Tuple[str, ...]


def model_cells(prob) -> Dict[Cell, object]:
    """Complete canonical view of a model: every cell `build_model` can read."""
    cells: Dict[Cell, object] = {("sense",): prob.sense}
    obj: Dict[str, float] = {nm: 0.0 for nm in prob.var_names}
    for c, i in prob.obj_terms:
        obj[prob.var_names[i]] += c
    for nm, c in obj.items():
        cells[("obj", nm)] = c
    for con in prob.constraints:
        cells[("row", con.name, "op")] = con.op
        cells[("row", con.name, "rhs")] = con.rhs_const
        for nm, a in row_coefficients(con, prob.var_names).items():
            cells[("row", con.name, "coef", nm)] = a
    return cells


def diff_cells(clean, mutant):
    a, b = model_cells(clean), model_cells(mutant)
    changed, added, removed = [], [], []
    for k in a.keys() | b.keys():
        if k not in b:
            removed.append((k, a[k]))
        elif k not in a:
            added.append((k, b[k]))
        elif a[k] != b[k]:
            changed.append((k, a[k], b[k]))
    key = lambda t: [str(x) for x in t[0]]
    return sorted(changed, key=key), sorted(added, key=key), sorted(removed, key=key)


def fmt(k: Cell) -> str:
    if k == ("sense",):
        return "sense"
    if k[0] == "obj":
        return f"obj c[{k[1]}]"
    if k[2] == "coef":
        return f"{k[1]}  a[{k[3]}]"
    return f"{k[1]}  {k[2]}"


def report(pid, meta, prob, clean, cand, text, n, total):
    mut_prob = copy.deepcopy(prob)
    cand.apply(mut_prob)
    sol = solve(mut_prob)

    print("\n" + "=" * 78)
    print(f"[{n}/{total}]  {pid}   type {cand.error_type}  direction {cand.direction}"
          f"   subclass {meta['subclass']}")
    print("=" * 78)
    print(f"  slot          {cand.slot}")
    print(f"  touched rows  {';'.join(cand.touched_rows) or '-'}")

    # ---- what changed, from a rebuild -------------------------------------
    changed, added, removed = diff_cells(prob, mut_prob)
    print("\n  MODEL DIFF  (canonical cells: sense, obj coefs, per-row op / rhs / coefs)")
    print(f"    clean model cells {len(model_cells(prob))}   "
          f"mutant model cells {len(model_cells(mut_prob))}")
    for k, old, new in changed:
        print(f"    CHANGED  {fmt(k):<34} {old!r:>22}  ->  {new!r}")
    for k, new in added:
        print(f"    ADDED    {fmt(k):<34} {'':>22}      {new!r}")
    for k, old in removed:
        print(f"    REMOVED  {fmt(k):<34} {old!r:>22}")
    print(f"    -> {len(changed)} changed, {len(added)} added, {len(removed)} removed")
    print(f"    -> every other cell identical: "
          f"{len(model_cells(prob)) - len(changed) - len(removed)} unchanged")
    print(f"    bucket-0 no-op test (is_noop): {is_noop(prob, mut_prob)}")

    # ---- bucket -----------------------------------------------------------
    bucket = bucket_of(cand.touched_rows, clean, sol)
    print(f"\n  BUCKET  {bucket}")
    print(f"    mutant status {sol.status}")
    for r in cand.touched_rows:
        cs = clean.slack.get(r)
        ms = sol.slack.get(r)
        print(f"    row {r:<16} clean slack "
              f"{('-' if cs is None else repr(cs)):>24}  binding={binding(clean,[r])}")
        print(f"    {'':<20} mutant slack "
              f"{('-' if ms is None else repr(ms)):>24}  binding={binding(sol,[r])}")
    if not sol.optimal:
        print("    bucket 1 -- never credited to Check A or B")
        return

    print(f"    clean objval  {clean.obj_value!r}")
    print(f"    mutant objval {sol.obj_value!r}")
    print(f"    obj change    "
          f"{abs(sol.obj_value-clean.obj_value)/max(1.0,abs(clean.obj_value))*100:.6f} %")

    # ---- Check B term by term ---------------------------------------------
    b_fired, b_detail = check_b(text, sol)
    print(f"\n  CHECK B   applicable={APPLICABLE[cand.error_type][1]}")
    print(f"    {'row':<18} {'b_i (TEXT)':>14} {'pi_i (MUTANT)':>22} {'product':>22}")
    for rn in text.row_names:
        if rn not in sol.pi:
            print(f"    {rn:<18} {text.row_limits[rn]:>14g} {'(row not in mutant)':>22}")
            continue
        print(f"    {rn:<18} {text.row_limits[rn]:>14g} {sol.pi[rn]:>22.15g} "
              f"{text.row_limits[rn]*sol.pi[rn]:>22.15g}")
    for rn in sol.pi:
        if rn not in text.row_limits:
            print(f"    {rn:<18} {'(not in text)':>14} {sol.pi[rn]:>22.15g} "
                  f"{'excluded from sum':>22}")
    print(f"    {'SUM':<18} {'':>14} {'':>22} {b_detail['sum']:>22.15g}")
    print(f"    {'TARGET (mutant ObjVal)':<40} {b_detail['obj_value']:>22.15g}")
    print(f"    {'abs_diff':<40} {b_detail['abs_diff']:>22.15g}")
    print(f"    fires when abs_diff > 1e-6 * max(1,|target|) = "
          f"{1e-6*max(1.0,abs(b_detail['obj_value'])):.15g}")
    print(f"    -> Check B {'FIRES' if b_fired else 'silent'}")

    # ---- Check A ----------------------------------------------------------
    a_fired, a_detail = check_a(text, sol)
    print(f"\n  CHECK A   applicable={APPLICABLE[cand.error_type][0]}")
    print(f"    {'variable':<20} {'x*_mut':>12} {'branch':>4} {'sum':>20} "
          f"{'c_j (TEXT)':>12} {'residual':>14}  fired")
    for d in a_detail:
        print(f"    {d['variable']:<20} {d['value']:>12.6g} {d['relation']:>4} "
              f"{d['sum']:>20.15g} {d['obj_coeff']:>12g} {d['residual']:>14.6g}  "
              f"{d['fired']}")
    print(f"    -> Check A {'FIRES' if a_fired else 'silent'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["type3", "type8"])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    scope = in_scope()
    problems, _ = load_clean_problems()
    store = TextTermsStore(problems)
    today = datetime.date.today().isoformat()

    if args.which == "type3":
        # Three of the twelve logged pilot mutants: one silent-with-a-real-miss,
        # one that fires, one bucket-2 inert. Identified by (instance, slot).
        picks = [("1727707716", "b[c0_linear]=20"),
                 ("-2105580086", "b[c0_linear]=25"),
                 ("-2094163363", "b[c0_upperbound]=5")]
        print("=" * 78)
        print("THREE OF THE TWELVE LOGGED TYPE-3 PILOT MUTANTS, REBUILT")
        print("=" * 78)
        print("  rebuilt from the annotation, not read back from mutants_log.csv")
        for n, (pid, slot) in enumerate(picks, 1):
            prob = problems[pid]
            clean = solve(prob)
            cand = next(c for c in ENUMERATORS["3"](prob, clean, DEFAULT_SCOPE)
                        if c.slot == slot)
            report(pid, scope[pid], prob, clean, cand, store.get(pid), n, len(picks))
        return

    # ---- fresh type-8 mini-pilot -----------------------------------------
    print("=" * 78)
    print("TYPE-8 MINI-PILOT -- 6 mutants, single-resource")
    print("=" * 78)
    print("  generation rule: amendment §1 as revised §12.3, CORRECTED §12.6,")
    print("  unit of enumeration ratified §12.7(d) --")
    print("    duplicate a row's coefficient pattern; zero-usage rows excluded;")
    print("    ONE mutant per instance, source row = median realised usage;")
    print("    limit = f*u with f sign-aware: 0.5 if (op=='<=') == (u>0) else 2.0.")

    picked, records = [], []
    for pid, meta in scope.items():
        if meta["subclass"] != "single":
            continue
        prob = problems[pid]
        clean = solve(prob)
        for cand in ENUMERATORS["8"](prob, clean, DEFAULT_SCOPE):
            picked.append((pid, meta, prob, clean, cand))
            if len(picked) >= 6:
                break
        if len(picked) >= 6:
            break

    for n, (pid, meta, prob, clean, cand) in enumerate(picked, 1):
        # show the usage the limit was derived from
        src = cand.slot.split("pattern of ")[1].split(" ")[0].rstrip(",")
        con = next(c for c in prob.constraints if c.name == src)
        print("\n" + "-" * 78)
        print(f"  source row {src}  op {con.op}  realised usage in clean optimum "
              f"{usage(prob, con, clean)!r}")
        report(pid, meta, prob, clean, cand, store.get(pid), n, len(picked))
        records.append(run_candidate(pid, meta, prob, clean, cand, store.get(pid),
                                     provenance="minipilot-type8-2026-08-02-post-12.6",
                                     today=today))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    import pandas as pd
    df = pd.DataFrame(records)
    print(df[["problem_id", "bucket", "mutant_status", "check_a_fired",
              "check_b_fired", "obj_change_pct"]].to_string(index=False))
    print(f"\n  TextTerms builds {store.builds} for "
          f"{len({r['problem_id'] for r in records})} distinct instances")
    if args.write:
        append_log(records)
        print(f"  appended {len(records)} rows to mutants_log.csv "
              f"(provenance minipilot-type8-2026-08-02-post-12.6)")
    else:
        print("  --write not given; nothing written")


if __name__ == "__main__":
    main()
