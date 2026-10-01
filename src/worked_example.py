"""Print a certified mutant in full, small enough to check by hand. 2026-08-28."""
from __future__ import annotations
import copy, sys
from mutation_harness import enumerate_candidates, in_scope
from mutations import check_a, check_b, check_d, solve
import mutations
from text_terms import load_clean_problems, text_terms_of

TYPE_ORDER = getattr(mutations, "TYPE_ORDER", ["1","2","3","4a","4b","5","6","7","8"])
WANT = sys.argv[1:] or ["-1176603373|7|swap|b[c0_linear]=30 <-> b[c1_linear]=50",
                        "-1742967011|5|swap|c[helicopters]=5 <-> c[trucks]=10",
                        "-1609306796|4a|swap|a[c0_linear,vans]=50 <-> a[c0_linear,trucks]=100"]

def show(tag, prob, sol):
    t = text_terms_of(prob)
    print(f"  {tag}: {t.sense} " + " + ".join(
        f"{t.obj_coefs[v]:g}*{v}" for v in t.var_names))
    for rn in t.row_names:
        body = " + ".join(f"{t.row_coefs[rn].get(v,0):g}*{v}" for v in t.var_names)
        print(f"      {rn:16s} {body} {t.row_operator[rn]} {t.row_limits[rn]:g}"
              f"     pi = {sol.pi.get(rn, float('nan')):+.4g}")
    print(f"      x* = " + ", ".join(f"{v}={sol.values[v]:g}" for v in t.var_names)
          + f"      z* = {sol.obj_value:g}")

def main():
    problems, _ = load_clean_problems()
    for mid in WANT:
        pid = mid.split("|")[0]
        if pid not in problems: continue
        prob = problems[pid]; text = text_terms_of(prob); ref = solve(prob)
        seen = set()
        for et in TYPE_ORDER:
            for scope in ("narrow","wide"):
                for cand in enumerate_candidates(pid, prob, ref, et,
                                                 both_directions=True, scope=scope):
                    m = f"{pid}|{cand.error_type}|{cand.direction}|{cand.slot}"
                    if m != mid or m in seen: continue
                    seen.add(m)
                    mut = copy.deepcopy(prob); cand.apply(mut); sol = solve(mut)
                    print("="*74); print(mid); print("="*74)
                    show("TEXT (annotation)", prob, ref)
                    show("CANDIDATE (mutant)", mut, sol)
                    a,b,c = check_a(text,sol)[0], check_b(text,sol)[0], check_d(text,sol)[0]
                    dz = abs(sol.obj_value-ref.obj_value)
                    dx = max(abs(sol.values[v]-ref.values[v]) for v in prob.var_names)
                    print(f"  |dz*| = {dz:.3g}   max|dx*| = {dx:.3g}"
                          f"   ->  reference solve is SILENT" if dz<1e-9 and dx<1e-9 else "")
                    print(f"  Check A {a}   Check B {b}   Check C {c}")
                    # the identity, written out
                    tv = text
                    print("  Check B:  sum_i b_i^text * pi_i^cand = " + " + ".join(
                        f"({tv.row_limits[r]:g})({sol.pi[r]:+.4g})" for r in tv.row_names
                        if r in sol.pi) + f" = "
                        f"{sum(tv.row_limits[r]*sol.pi[r] for r in tv.row_names if r in sol.pi):.6g}"
                        f"   vs z* = {sol.obj_value:g}")
                    for v in tv.var_names:
                        s = sum(a2*sol.pi[r] for r,a2 in tv.terms_for(v) if r in sol.pi)
                        print(f"  Check A [{v}]: s_j = {s:.6g}  c_j = {tv.obj_coef(v):g}"
                              f"   x*_j = {sol.values[v]:g}")
                    print()
main()
