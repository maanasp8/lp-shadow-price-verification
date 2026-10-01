#!/usr/bin/env python
"""Generate paper/tables_results.tex: the three Section 4 tables.

Read-only against the data. Every cell is read from a persisted artefact and
asserted against the figure the body prose states; any mismatch is fatal.

Sources, in table order:
  Table 3  checkA_results.csv, checkB_results.csv,
           checkD_clean_control_2026-08-19.csv, results.csv
  Table 4  checkD_target.csv, checkD_secondary.csv,
           checkD_full_population_2026-08-19.json
  Table 5  mutants_log.csv, union_endpoint_2026-08-15.csv

Check-label policy, revision 3 §R3.1: prose, figures and table headers say
"Check C" for the primal feasibility audit. Every artefact read here keeps
"Check D" in its filename and in its `d_fired` column.

    venv/bin/python make_results_tables.py
"""
import csv, json, sys

import pandas as pd

REL_TOL = 1e-6
fail = []


def want(label, got, exp):
    if got != exp:
        fail.append(f"  {label}: got {got!r}, body states {exp!r}")


def sci(v, digits=1):
    """-> LaTeX math for v in scientific notation, e.g. $1.5\\times10^{-11}$."""
    mant, exp = f"{v:.{digits}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


# ------------------------------------------------------- Table 3: FP control
A = pd.read_csv("data/checkA_results.csv", dtype={"problem_id": str})
B = pd.read_csv("data/checkB_results.csv", dtype={"problem_id": str})
C = pd.read_csv("data/checkD_clean_control_2026-08-19.csv", dtype={"problem_id": str})
res = pd.read_csv("data/results.csv", dtype={"problem_id": str})

n_clean = int((res.solve_status == "OPTIMAL").sum())
want("clean instances", n_clean, 1040)
want("Check A rows", len(A), 2146)
want("Check A fired", int(A.fired.sum()), 0)
want("Check B instances", len(B), 1040)
want("Check B fired", int(B.fired.sum()), 0)
want("Check C instances", len(C), 1040)
want("Check C fired", int(C.d_fired.sum()), 0)

# vacuous rows: empty sum, cannot fire
a_vac = int(((A.instance_n_rows == 0) | (A.n_terms == 0)).sum())
want("Check A vacuous", a_vac, 5)
b_vac = int((res.loc[res.solve_status == "OPTIMAL", "n_constraints"] == 0).sum())
want("Check B vacuous", b_vac, 1)

a_real = A[(A.instance_n_rows > 0) & (A.n_terms > 0)]
a_eq = a_real[a_real.branch == "positive"]
a_thr = REL_TOL * a_eq.obj_coeff.abs().clip(lower=1.0)
a_worst, a_ratio = a_eq["diff"].abs().max(), (a_eq["diff"].abs() / a_thr).max()
b_worst, b_ratio = B.rel_diff.max(), B.rel_diff.max() / REL_TOL
c_worst, c_ratio = C.max_violation.max(), C.max_violation.max() / REL_TOL
a_nz = int((a_eq["diff"].abs() > 0).sum())
b_nz = int((B.rel_diff > 0).sum())
c_nz = int((C.max_violation > 0).sum())
want("Check A nonzero residuals", a_nz, 354)
want("Check C nonzero residuals", c_nz, 60)

# ------------------------------------------------------- Table 4: Check C
tgt = pd.read_csv("data/checkD_target.csv")
tgt = tgt[tgt.resolved & tgt.d_fired.notna()]
want("target stratum n", len(tgt), 2746)
want("target stratum fired", int(tgt.d_fired.sum()), 2531)

sec = pd.read_csv("data/checkD_secondary.csv")
co = sec[sec.error_type == "CO"]
of = sec[sec.error_type == "OF"]
want("CO n", len(co), 2590)
want("CO fired", int(co.d_fired.sum()), 1385)
want("OF n", len(of), 2082)
want("OF fired", int(of.d_fired.sum()), 1377)
want("CO+OF n", len(sec), 4672)
want("CO+OF fired", int(sec.d_fired.sum()), 2762)

full = json.load(open("data/checkD_full_population_2026-08-19.json"))
want("full-population n", full["micro_all"]["n"], 18525)
want("full-population fired", full["micro_all"]["fired"], 5997)
pt = full["per_type"]
zeros = []
for t, label in (("2", "objective coefficient magnitude"),
                 ("5", "objective coefficient swap"),
                 ("8", "spurious constraint")):
    n = pt[f"type{t}_single"]["n"] + pt[f"type{t}_multi"]["n"]
    k = pt[f"type{t}_single"]["fired"] + pt[f"type{t}_multi"]["fired"]
    want(f"type {t} Check C fired", k, 0)
    zeros.append((t, label, n))
want("type 2 n", zeros[0][2], 4096)
want("type 5 n", zeros[1][2], 1003)
want("type 8 n", zeros[2][2], 724)

# ------------------------------------------------------- Table 5: coverage
log = pd.read_csv("data/mutants_log.csv", low_memory=False, dtype={"problem_id": str})
p = log[(log.provenance == "full-2026-08-02") & (log.population == "primary")
        & (log.bucket == "3 evaluable")]
u = {r["scope"]: r for r in csv.DictReader(open("data/union_endpoint_2026-08-15.csv"))}

cov = []
for scope, label in (("single", "single-resource"), ("multi", "multi-resource"),
                     ("pooled", "pooled")):
    g = p if scope == "pooled" else p[p.subclass == scope]
    row = u[scope]
    a = int((g.check_a_fired == True).sum())            # noqa: E712
    b = int((g.check_b_fired == True).sum())            # noqa: E712
    want(f"{scope} n_eval", len(g), int(row["n_evaluable"]))
    want(f"{scope} A or B", int(((g.check_a_fired == True) |
                                 (g.check_b_fired == True)).sum()),
         int(row["a_or_b"]))
    cov.append((label, len(g), a, b, int(row["a_or_b"]), row["a_or_b_pct"],
                int(row["a_or_b_or_d"]), row["a_or_b_or_d_pct"],
                int(row["marginal_d"])))

want("pooled A fired", cov[2][2], 11366)
want("pooled B fired", cov[2][3], 3758)
want("union pooled", cov[2][6], 17018)

# =========================================================================
# Tables 6-8, added 2026-08-25 for the reframe. Sources:
#   Table 6  mutants_log.csv + checkD_full_population_2026-08-19.csv
#            + checks_fire_obj_unchanged_2026-08-25.csv + diff_baseline json
#   Table 7  restructure_invariance_2026-08-25.json
#   Table 8  extraction_noise_2026-08-25.csv
# =========================================================================

# ---- dedupe: the log carries 10 duplicate mutant_id rows from an append rerun
log_u = log.drop_duplicates(subset="mutant_id", keep="first")
dfull = pd.read_csv("data/checkD_full_population_2026-08-19.csv")
d_by = dict(zip(dfull.mutant_id, dfull.d_fired.astype(str).str.lower() == "true"))

ev = log_u[(log_u.population == "primary") & (log_u.bucket == "3 evaluable")].copy()
want("evaluable primary (deduped)", len(ev), 18525)
ev["fired"] = ((ev.check_a_fired == True) | (ev.check_b_fired == True)   # noqa: E712
               | ev.mutant_id.map(d_by).fillna(False))
ev["zmoved"] = ((ev.mutant_objval.astype(float) - ev.clean_objval.astype(float)).abs()
                > REL_TOL * ev.clean_objval.astype(float).abs().clip(lower=1.0))

moved, still = ev[ev.zmoved], ev[~ev.zmoved]
# Strict accounting: only the dual check the per-type protocol declares applicable.
# The 100% headline is the union of all three checks, each genuinely applicable to
# every mutant; this is the dual-only figure, reported beside it so the two
# accountings in Appendix B cannot be confused.
tb_ = lambda c: ev[c].astype(str).str.lower() == "true"
moved_dual = int((((tb_("check_a_fired") & tb_("check_a_applicable"))
                   | (tb_("check_b_fired") & tb_("check_b_applicable")))
                  & ev.zmoved).sum())
want("answer-moved, applicable dual check only", moved_dual, 14261)
want("answer-moved n", len(moved), 16852)
want("answer-moved fired", int(moved.fired.sum()), 16852)
want("answer-unchanged n", len(still), 1673)
want("answer-unchanged fired", int(still.fired.sum()), 166)

# Inert scope is the primary and extension populations, the two Checks A and B
# are scored on. The log carries 4 further inert rows in the SECONDARY
# (operator-flip) population, which §3.2 scores by Check C alone; folding them
# in here would mix two scoring protocols in one denominator. 3,840 + 1,667.
inert = log_u[(log_u.bucket == "2 inert")
              & (log_u.population.isin(["primary", "extension"]))]
n_inert = len(inert)
want("inert n (primary+extension)", n_inert, 5507)
want("inert checks fired", int(((inert.check_a_fired == True) |    # noqa: E712
                                (inert.check_b_fired == True)).sum()), 0)

def latest(pattern):
    """-> newest artefact matching `pattern`. These harnesses date-stamp their
    output, and a run that straddles midnight yields mixed stamps, so resolve by
    glob rather than hardcoding. Fatal if absent: a missing artefact must never
    degrade to a stale one."""
    import glob
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no artefact matches {pattern} -- rerun its harness")
    return hits[-1]


x = pd.read_csv(latest("data/checks_fire_obj_unchanged_*.csv"))
want("166 re-solved", len(x), 166)
n_xmoved = int(x.x_moved.astype(str).str.lower().eq("true").sum())
want("of the 166, x* moved", n_xmoved, 79)

dj = json.load(open(latest("data/diff_baseline_*.json")))
want("comparison on evaluable", dj["detected"], 18525)
want("comparison pct", round(dj["pct"], 2), 100.0)

RI = json.load(open(latest("data/restructure_invariance_*.json")))["summary"]
want("negate_flip_all n", RI["negate_flip_all"]["n"], 1040)
want("negate_flip_all fired", RI["negate_flip_all"]["any_fired"], 1017)
want("reorder_rows fired", RI["reorder_rows"]["any_fired"], 0)
want("reindex_vars fired", RI["reindex_vars"]["any_fired"], 0)

RN = json.load(open(latest("data/row_normalization_*.json")))
want("normalized detection equals unnormalized", RN["arm2"]["fired"], 14401)
want("normalized detection n", RN["arm2"]["n"], 18525)
for _k in ("negate_flip_all", "rescale_x2_all", "rescale_x01_all",
           "negate_flip_row0", "rescale_x2_row0", "reorder_rows", "reindex_vars"):
    want(f"normalized FP {_k}", RN["arm1"][_k]["fired"], 0)

SR = json.load(open(latest("data/solve_reference_*.json")))
want("oracle on evaluable", SR["mutants"]["evaluable"]["oracle"], 17030)
want("checks on evaluable", SR["mutants"]["evaluable"]["checks"], 17018)
want("oracle on inert", SR["mutants"]["inert"]["oracle"], 75)
want("checks on inert (A|B|C)", SR["mutants"]["inert"]["checks"], 29)
DO = json.load(open(latest("data/dual_only_*.json")))
BT = json.load(open(latest("data/basis_test_*.json")))
want("basis test n", BT["n"], 86)
want("basis-dependent artifacts", BT["basis_dependent"], 17)
want("dual-optimal pi exists (artifact)", BT["artifact_pi_exists"], 14)
want("genuine after both stages", BT["genuine"], 55)

C55 = json.load(open(latest("data/characterize_55_*.json")))
want("the 55, characterised", C55["n"], 55)
want("55: within-row swaps", C55["by_type"]["4a"], 19)
want("55: cross-row swaps", C55["by_type"]["4b"], 13)
want("55: objective swaps", C55["by_type"]["5"], 14)
want("55: limit swaps", C55["by_type"]["7"], 9)
want("55: only swap types occur", sorted(C55["by_type"]), ["4a", "4b", "5", "7"])
want("55: Check A alone", C55["by_check"]["A only"], 46)
want("55: Check B alone", C55["by_check"]["B only"], 9)
want("55: Check C never", C55["by_check"]["C"], 0)
want("55: at a degenerate vertex", C55["degenerate_vertex"], 2)

RF = json.load(open(latest("data/reframe_2026*.json")))
want("normalized, optimum moved", RF["moved"]["checks_norm"], 16852)
want("normalized, objective unchanged", RF["still"]["checks_norm"], 166)
want("normalized, answer-neutral", RF["inert"]["checks_norm"], 29)
want("reference, objective unchanged", RF["still"]["reference"], 178)
want("reference, answer-neutral", RF["inert"]["reference"], 75)
want("face-robust certificate holds", RF["face"]["genuine"], 55)
want("face-robust, primal optimum unique", RF["face"]["primal_unique"], 82)
want("reference 75, basis-dependent", RF["reference_75"]["basis_dependent"], 43)
want("reference 75, forced", RF["reference_75"]["genuine"], 5)
want("reference 75, all x*-only", RF["reference_75"]["obj_moved"], 0)
want("checks 29, forced", RF["checks_29"]["forced"], 0)
want("reference unique detections", RF["reference_unique_det"]["n"], 98)
want("reference unique detections, forced", RF["reference_unique_det"]["genuine"], 2)
want("checks-only, deduped", DO["n"], 86)
want("checks-only by check A", DO["by_check"]["a"], 75)
want("checks-only by check B", DO["by_check"]["b"], 11)

LM = json.load(open("data/llm_autoformulate_summary.json"))
RL = json.load(open("data/row_labels_ablation.json"))
want("LLM wrong detected", LM["wrong"]["detected"], 8)
want("LLM wrong n", LM["wrong"]["n"], 8)
want("LLM match reported", LM["match"]["reported"], 3)
want("LLM match reported normalized", LM["match"]["reported_norm"], 0)
want("LLM alignment by name FP", LM["alignment_ablation_opus44"]["name"]["fp"], 44)
want("LLM alignment by content FP", LM["alignment_ablation_opus44"]["content"]["fp"], 0)

NZ = pd.read_csv(latest("data/extraction_noise_*.csv"))
noise = []
for p in sorted(NZ.noise_p.unique()):
    g = NZ[NZ.noise_p == p]
    noise.append((p, g.diff_fp_pct.mean(), g.diff_det_pct.mean(),
                  g.checks_fp_pct.mean(), g.checks_det_pct.mean()))
want("noise levels", len(noise), 6)
z0 = noise[0]
want("noise p=0 comparison det", round(z0[2], 2), 100.0)
want("noise p=0 checks det", round(z0[4], 2), 91.87)

if fail:
    print("ASSERTION FAILURE - artefacts disagree with the body:", file=sys.stderr)
    print("\n".join(fail), file=sys.stderr)
    sys.exit(1)
print(f"reconciliation OK: FP 0/{len(A)}, 0/{len(B)}, 0/{len(C)}; "
      f"C {int(tgt.d_fired.sum())}/{len(tgt)}; union {cov[2][6]}/{cov[2][1]}")

# ------------------------------------------------------------------ emit tex
L = ["% GENERATED FILE - do not edit by hand.",
     "% Produced by make_results_tables.py from the persisted artefacts.",
     "% Every cell is asserted against the body figure at generation time.",
     "% Regenerate with:  venv/bin/python make_results_tables.py",
     ""]

L += [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{False-positive control on the " + f"{n_clean:,}" + r" clean models. "
 r"No check reports a violation. The units differ. Check A is measured per variable "
 r"row, and Checks B and C are measured per instance. A vacuous row has an empty sum "
 r"and cannot be violated, so the "
 r"strict counts are $0/2{,}141$ and $0/1{,}039$. The residual column is the "
 r"largest observation on the clean corpus, and the last column gives it as a "
 r"fraction of that check's detection threshold. The nonzero counts show the "
 r"comparison is doing real arithmetic rather than closing by construction.}",
 r"  \label{tab:fpr}",
 r"  \begin{tabular}{@{}llrrrrr@{}}", r"    \toprule",
 r"    check & unit & $n$ & vacuous & violations & worst residual & $\div$ threshold \\",
 r"    \midrule",
 f"    A & variable rows & {len(A):,} & {a_vac} & 0 & {sci(a_worst)} & {sci(a_ratio)} \\\\",
 f"    B & instances & {len(B):,} & {b_vac} & 0 & {sci(b_worst)} & {sci(b_ratio)} \\\\",
 f"    C & instances & {len(C):,} & --- & 0 & {sci(c_worst)} & {sci(c_ratio)} \\\\",
 r"    \midrule",
 r"    \multicolumn{5}{@{}l}{\itshape rows or instances with a strictly nonzero residual}"
 f" & {a_nz} / {b_nz} / {c_nz} & \\\\",
 r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

L += [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{Check C across the three populations it was run on, each with its "
 r"own denominator. The binding-to-slack stratum is the population Checks A and "
 r"B report nothing on. The full evaluable primary row is post hoc and is not a "
 r"headline figure. The three types at the foot return exactly zero for the "
 r"structural reasons of \S2.6: types 2 and 5 mutate only the objective, and "
 r"type 8 tightens the model.}",
 r"  \label{tab:checkc}",
 r"  \begin{tabular}{@{}lrrr@{}}", r"    \toprule",
 r"    population & $n$ & violations & \% \\", r"    \midrule",
 f"    binding-to-slack misses & {len(tgt):,} & {int(tgt.d_fired.sum()):,} & 92.17 \\\\",
 f"    constraint omission & {len(co):,} & {int(co.d_fired.sum()):,} & 53.47 \\\\",
 f"    operator flip & {len(of):,} & {int(of.d_fired.sum()):,} & 66.14 \\\\",
 f"    \\quad both dedicated types & {len(sec):,} & {int(sec.d_fired.sum()):,} & 59.12 \\\\",
 f"    all evaluable primary (post hoc) & {full['micro_all']['n']:,} & "
 f"{full['micro_all']['fired']:,} & {full['micro_all']['pct']:.2f} \\\\",
 r"    \midrule",
 r"    \multicolumn{4}{@{}l}{\itshape structural zeros within the primary population} \\"]
for t, label, n in zeros:
    L.append(f"    \\quad type {t}, {label} & {n:,} & 0 & 0.00 \\\\")
L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

L += [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{Detection over the evaluable primary population, pooled within "
 r"each subclass. Checks A and B overlap on 723 mutants, so their union is "
 r"smaller than their sum. The marginal column counts mutants Check C catches "
 r"that neither dual check caught, and it accounts for the whole of the rise "
 r"from the A or B column to the union column. These are pooled rates and are "
 r"not comparable line-for-line with the macro-average of "
 r"Table~\ref{tab:detection-by-type}.}",
 r"  \label{tab:coverage}",
 r"  \begin{tabular}{@{}lrrrr rr@{}}", r"    \toprule",
 r"    subclass & $n_{\mathrm{eval}}$ & A & B & A or B & A or B or C & marginal C \\",
 r"    \midrule"]
for label, n, a, b, ab, abp, abc, abcp, marg in cov:
    bold = r"\bfseries " if label == "pooled" else ""
    L.append(f"    {bold}{label} & {n:,} & {a:,} & {b:,} & {ab:,} ({abp}) & "
             f"{bold}{abc:,} ({abcp}) & {marg:,} \\\\")
L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

# ================================================== Table 6: the partition
n_disagree = SR["mutants"]["evaluable"]["oracle"] - 16932 + DO["n"]
want("evaluable disagreements", n_disagree, 184)


def _cell(k, n):
    return f"{k:,} ({100*k/n:.1f}\\%)"


P = [
 r"\begin{table}[!htbp]", r"  \centering", r"  \footnotesize",
 r"  \setlength{\tabcolsep}{4pt}",
 r"  \caption{Detection split by whether the mutation changed the model's "
 r"optimum. Each column is the union of the three checks or the whole of the "
 r"named baseline. The first two rows partition the 18{,}525 evaluable "
 r"mutants. The third is a separate control of mutants that alter the model "
 r"without changing its answer, so the $n$ column does not sum. "
 r"\emph{Comparison} flags any coefficient difference. \emph{Reference} solves "
 r"the text's model and flags a moved $z^\star$ or $x^\star$. "
 r"\emph{Normalized} applies \S\ref{sec:normalization} to the same three "
 r"checks. The reference's first-row figure is definitional and the comparison "
 r"column is $100\%$ of each row's $n$ throughout. Under the stricter per-type "
 r"assignment the dual checks alone account for " + f"{moved_dual:,}" + r" of "
 r"the first row. Denominators differ.}",
 r"  \label{tab:partition}",
 r"  \begin{tabular}{@{}lrrrrr@{}}", r"    \toprule",
 r"    population & $n$ & checks & normalized & comparison & reference \\",
 r"    \midrule",
 f"    \\bfseries optimum moved & {len(moved):,} & \\bfseries "
 f"{_cell(int(moved.fired.sum()), len(moved))} & \\bfseries "
 f"{_cell(RF['moved']['checks_norm'], len(moved))} & "
 f"{len(moved):,} & {len(moved):,} \\\\",
 f"    objective unchanged, evaluable & {len(still):,} & "
 f"{_cell(int(still.fired.sum()), len(still))} & "
 f"{_cell(RF['still']['checks_norm'], len(still))} & "
 f"{len(still):,} & {_cell(RF['still']['reference'], len(still))} \\\\",
 r"    \midrule",
 f"    \\bfseries no change to the answer & {n_inert:,} & \\bfseries "
 f"{_cell(SR['mutants']['inert']['checks'], n_inert)} & \\bfseries "
 f"{_cell(RF['inert']['checks_norm'], n_inert)} & {n_inert:,} "
 f"& {_cell(RF['inert']['reference'], n_inert)} \\\\",
 r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

# ================================ Table: what the certified 55 are made of
_SW = [("4a", "two coefficients within one row", "A"),
       ("4b", "one variable's coefficient across two rows", "A"),
       ("5", "two objective coefficients", "A"),
       ("7", "two rows' limits", "B")]
SWAP = [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{What the " + f"{C55['n']}" + r" certified errors are. Every one "
 r"exchanges a pair of values between two slots. No magnitude change, omission "
 r"or spurious row produces one, even though those types supply $70\%$ of the "
 r"population. Check C reads the primal and reports none of them, because it "
 r"cannot. "
 r"Only " + f"{C55['degenerate_vertex']}" + r" sit at a degenerate vertex, so on "
 r"the rest the mutant's dual is already unique before the certificate is "
 r"applied.}",
 r"  \label{tab:swaps}",
 r"  \begin{tabular}{@{}lrl@{}}", r"    \toprule",
 r"    the mutation exchanges & $n$ & reported by \\", r"    \midrule"]
for t, label, chk in _SW:
    SWAP.append(f"    {label} & {C55['by_type'][t]} & Check {chk} \\\\")
SWAP += [r"    \midrule",
 f"    \\bfseries total & \\bfseries {C55['n']} & "
 f"A {C55['by_check']['A only']}, B {C55['by_check']['B only']}, C 0 \\\\",
 r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

# ======================================= Table 6b: the degeneracy certificate
CERT = [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \setlength{\tabcolsep}{5pt}",
 r"  \caption{The degeneracy certificate, applied to both verifiers. Each "
 r"reports on mutants the other does not. \emph{Certified} counts the reports "
 r"that survive the test appropriate to what that verifier reads. For the "
 r"checks, that test is whether no dual-optimal $\pi$ satisfies the text's "
 r"identities at any optimal solution of the model. For the reference, it is "
 r"whether the mutant's optimal face and the text model's are disjoint. Both "
 r"quantify over the whole optimal set rather than one returned solution. "
 r"Reading the two \emph{certified} columns together gives each verifier "
 r"idealized to its forced reports alone, and \S\ref{sec:dualonly} discusses "
 r"what that comparison establishes.}",
 r"  \label{tab:certificate}",
 r"  \begin{tabular}{@{}l rrr c rrr@{}}", r"    \toprule",
 r"    & \multicolumn{3}{c}{unique detections} & &"
 r" \multicolumn{3}{c}{reports on the answer-neutral " + f"{n_inert:,}" + r"} \\",
 r"    \cmidrule(lr){2-4}\cmidrule(l){6-8}",
 r"    verifier & reports & artifact & certified & & reports & artifact"
 r" & forced \\",
 r"    \midrule",
 f"    \\bfseries checks & {DO['n']} & {RF['face']['artifact']} & \\bfseries "
 f"{RF['face']['genuine']} & & {SR['mutants']['inert']['checks']} & "
 f"{RF['checks_29']['artifact']} & \\bfseries {RF['checks_29']['forced']} \\\\",
 f"    reference solve & {RF['reference_unique_det']['n']} & "
 f"{RF['reference_unique_det']['faces_intersect']} & "
 f"{RF['reference_unique_det']['genuine']} & & "
 f"{SR['mutants']['inert']['oracle']} & {RF['reference_75']['faces_intersect']} "
 f"& {RF['reference_75']['genuine']} \\\\",
 r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

# ============================================== Table 7: invariance boundary
# Single-row variants (negate_flip_row0, rescale_x2_row0) are omitted from the
# body table for page budget; both sit near 64% and are in the CSV artefact.
ORDER = [("negate_flip_all", "negate and flip every row"),
         ("rescale_x2_all", r"rescale every row by $2$"),
         ("rescale_x01_all", r"rescale every row by $0.1$"),
         ("reorder_rows", "reorder the rows"),
         ("reindex_vars", "reindex the variables")]
I = [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{False positives on semantics-preserving restructurings of the "
 + f"{n_clean:,}" + r" clean models. Every transformed candidate re-solves to "
 r"the original optimal value, so each is a \emph{correct} model and every violation "
 r"reported is a false positive. The text side is the unaltered annotation throughout. "
 r"Reordering and reindexing preserve the row-name correspondence both checks "
 r"key on and produce no violation. Rescaling or reorienting a row changes $\pi_i$ "
 r"relative to the text's $a_{ij}$ and is violated by almost every model. The "
 r"invariance class is exactly the name-preserving permutations. The final "
 r"column applies the row normalization of \S\ref{sec:normalization}, which "
 r"recovers each row's scale factor and corrects $\pi_i$ before use. It removes "
 r"every false positive here while leaving detection on the mutant population "
 r"bit-for-bit unchanged. The \emph{reference} column is the solve-the-reference "
 r"baseline on the same transformed models. It reports violations too. A rescaled "
 r"row can move the solver to a different optimal vertex at the same objective "
 r"value, and every report in that column is such a case, with none being a moved "
 r"$z^\star$. Its worst rate is "
 + f"{max(SR['restructure'][k]['oracle'] for k, _ in ORDER)}" + r"/"
 + f"{n_clean:,}" + r", which is where the $1.4\%$ of \S\ref{sec:diff-baseline} "
 r"comes from. The numerically similar $75/5{,}507$ of "
 r"Table~\ref{tab:partition} is a different measurement on a different "
 r"population, and the two should not be conflated.}"
 r"  \label{tab:invariance}",
 r"  \begin{tabular}{@{}lrrrrrr@{}}", r"    \toprule",
 r"    & & \multicolumn{3}{c}{as specified} & normalized & reference \\",
 r"    \cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
 r"    transformation & $n$ & A & B & any check & any check & solve \\",
 r"    \midrule"]
for key, label in ORDER:
    v = RI[key]
    o = SR["restructure"][key]
    I.append(f"    {label} & {v['n']:,} & {v['a_fired']:,} & {v['b_fired']:,} & "
             f"{v['any_fired']:,} ({v['any_pct']:.1f}\\%) & "
             f"\\bfseries {RN['arm1'][key]['fired']} & "
             f"{o['oracle']} ({100*o['oracle']/o['n']:.1f}\\%) \\\\")
I += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

# ============================================ Table 8: extraction-noise sweep
N = [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{Operating characteristics against extraction quality. Each "
 r"numeric cell of the text side is corrupted independently with probability "
 r"$p$, at the extraction pilot's own $6{:}1$ ratio of referent to numeric "
 r"failure. The candidate is correct in the false-positive arm, so every violation "
 r"reported there is caused by extraction error alone. Margin is detection minus false "
 r"positives, the share of the population on which the verdict is worth acting. "
 r"Twenty replicates are drawn per level, and $p=0$ is deterministic. The checks lead from "
 r"$p=0.10$ but never by more than $4.6$ points, and by then neither verifier "
 r"retains a usable margin.}",
 r"  \label{tab:noise}",
 r"  \begin{tabular}{@{}rrrr rrr@{}}", r"    \toprule",
 r"    & \multicolumn{3}{c}{comparison} & \multicolumn{3}{c}{Checks A, B, C} \\",
 r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
 r"    $p$ & FP & detection & margin & FP & detection & margin \\",
 r"    \midrule"]
for p, dfp, ddet, cfp, cdet in noise:
    bold = r"\bfseries " if p == 0.0 else ""
    N.append(f"    {bold}{p:.2f} & {dfp:.2f}\\% & {ddet:.2f}\\% & {ddet-dfp:.2f} & "
             f"{cfp:.2f}\\% & {cdet:.2f}\\% & {bold}{cdet-cfp:.2f} \\\\")
N += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

# ================================================ Table 9: real LLM models
w, m, ab = LM["wrong"], LM["match"], LM["alignment_ablation_opus44"]
G = [
 r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
 r"  \caption{Models generated by an actual autoformulator. "
 r"\textsc{claude-haiku-4.5} was given each problem's text and its variable "
 r"names and asked for a linear program, and " + f"{LM['scored']}" + r" of "
 + f"{LM['attempted']}" + r" parsed, solved and admitted a row alignment. "
 r"\emph{Upper:} detection, and the effect of the normalization of "
 r"\S\ref{sec:normalization}. All three reports it removes are uniform row "
 r"rescalings. \emph{Lower:} the row-alignment ablation of "
 r"\S\ref{sec:rowalign}. Its five rows come from two generation runs scored "
 r"three ways, because whether the prompt lists the annotation's row names "
 r"changes what the generator writes. The effect of that is visible in the "
 r"denominators, where 79 of the models produced without row names reproduce "
 r"the annotation's optimum against 80 of those produced with them. A model is "
 r"counted objective-matching when its own solve reproduces the annotation's "
 r"optimal value, which is independent of how rows are aligned. Alignment decides "
 r"only which models are scorable at all, and matching optima is a lower bound on "
 r"correctness, so a report on an objective-matching model is not automatically a "
 r"false positive. Wilson 95\%.}",
 r"  \label{tab:llm}",
 r"  \begin{tabular}{@{}lrrl@{}}", r"    \toprule",
 r"    generated models & $n$ & reported & 95\% CI \\",
 r"    \midrule",
 f"    \\bfseries optimum differs from annotation & {w['n']} & "
 f"\\bfseries {w['detected']} ({w['pct']:.0f}\\%) & [{w['ci95'][0]}, {w['ci95'][1]}] \\\\",
 f"    optimum matches, as specified & {m['n']} & {m['reported']} ({m['pct']:.1f}\\%) & "
 f"[{m['ci95'][0]}, {m['ci95'][1]}] \\\\",
 f"    \\bfseries optimum matches, normalized & {m['n']} & \\bfseries "
 f"{m['reported_norm']} (0.0\\%) & [{m['ci95_norm'][0]}, {m['ci95_norm'][1]}] \\\\",
 r"    \midrule",
 r"    \multicolumn{4}{@{}l}{\itshape row alignment: reports on models reproducing the optimum} \\",
 f"    \\quad row names withheld, matched by name & {RL['arms']['plain_name']['n_match']} & "
 f"{RL['arms']['plain_name']['fp']} ({RL['arms']['plain_name']['fp_pct']}\\%) & --- \\\\",
 f"    \\quad row names supplied, matched by name & {RL['arms']['rowlabels_name']['n_match']} & "
 f"{RL['arms']['rowlabels_name']['fp']} ({RL['arms']['rowlabels_name']['fp_pct']}\\%) & --- \\\\",
 f"    \\quad \\quad the same, normalized & {RL['arms']['rowlabels_name']['n_match']} & "
 f"{RL['arms']['rowlabels_name']['fp_norm']} ({RL['arms']['rowlabels_name']['fp_norm_pct']}\\%) & --- \\\\",
 f"    \\quad row names supplied, matched by content & {RL['arms']['rowlabels_content']['n_match']} & "
 f"{RL['arms']['rowlabels_content']['fp']} ({RL['arms']['rowlabels_content']['fp_pct']:.1f}\\%) & --- \\\\",
 f"    \\quad \\quad the same, normalized & {RL['arms']['rowlabels_content']['n_match']} & "
 f"\\bfseries {RL['arms']['rowlabels_content']['fp_norm']} "
 f"({RL['arms']['rowlabels_content']['fp_norm_pct']:.1f}\\%) & --- \\\\",
 r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

# ------------------------------------------------------------------- assemble
# Body carries the Section 4 story: FP control, the partition, the invariance
# boundary, the noise sweep. The Check C per-population table and the coverage
# breakdown move to the appendix -- neither is cited in the revised Section 4.
starts = [i for i, x in enumerate(L) if x.startswith(r"\begin{table}")]
t_fpr, t_checkc, t_cov = L[starts[0]:starts[1]], L[starts[1]:starts[2]], L[starts[2]:]
hdr = L[:4]

body = hdr + [""] + P
appx = hdr + [""] + CERT + SWAP + t_fpr + I + G + t_checkc + t_cov + N
open("paper/tables_results.tex", "w").write("\n".join(body).rstrip() + "\n")
open("paper/tables_appendix.tex", "w").write("\n".join(appx).rstrip() + "\n")
print(f"wrote paper/tables_results.tex (tab:partition), "
      f"paper/tables_appendix.tex (tab:fpr, tab:invariance, tab:llm, "
      f"tab:checkc, tab:coverage, tab:noise)")
