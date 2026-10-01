#!/usr/bin/env python
"""Generate paper/tables.tex (Table 1) from table1_detection_by_type.csv.

Read-only against the data. Every per-type generated count is asserted against
the §12.10 re-reported Step 0 table transcribed below; any mismatch is fatal.

NOTE: the transcription below is from `preregistration_amendment_2026-08-02.md`
§12.10, the table that SUPERSEDES §11. It is deliberately NOT taken from the
`STEP0` dict at mutation_harness.py:202, which transcribes the retired §11 and is
stale by construction (see revision 3 §R3.1 discussion and the §12.9/§12.8 change
log entries).

Check-label policy, revision 3 §R3.1: prose and figure labels say "Check C" for
the primal feasibility audit; artefacts, columns and code keep "Check D". Table 1
covers Checks A and B only, so no remapping arises here.

    venv/bin/python make_tables_tex.py
"""
import csv, sys

# --- §12.10 PRIMARY POPULATION, narrow scope, both directions on types 1/2/3 ---
S12_10 = {"1": (2974, 2942), "2": (1590, 2506), "3": (1690, 3114),
          "4a": (774, 733), "4b": (812, 697), "5": (427, 576),
          "6": (1481, 1459), "7": (539, 1425), "8": (383, 622)}
S12_10_TOTAL = 24744
POOLS = {"4": ["4a", "4b"]}
ENDPOINT = ["1", "2", "3", "4", "5", "6", "7", "8"]
VARIANTS = ["4a", "4b"]

DESC = {"1": "constraint coefficient magnitude", "2": "objective coefficient magnitude",
        "3": "limit magnitude", "4": "coefficient swap", "5": "objective coefficient swap",
        "6": "coefficient omission", "7": "limit swap between rows", "8": "spurious constraint",
        "4a": "\\quad within-row", "4b": "\\quad cross-row, same variable"}

rows = list(csv.DictReader(open("data/table1_detection_by_type.csv")))
idx = {(r["error_type"], r["subclass"]): r for r in rows}

# ---------------------------------------------------------------- assertions
fail = []
for t in ENDPOINT + VARIANTS:
    members = POOLS.get(t, [t])
    for sc, pos in (("single", 0), ("multi", 1)):
        got = int(idx[(t, sc)]["n_generated"])
        want = sum(S12_10[m][pos] for m in members)
        if got != want:
            fail.append(f"  type {t} {sc}: CSV n_generated={got}, §12.10={want}")

gen_s = sum(int(idx[(t, "single")]["n_generated"]) for t in ENDPOINT)
gen_m = sum(int(idx[(t, "multi")]["n_generated"]) for t in ENDPOINT)
if (gen_s, gen_m) != (10670, 14074):
    fail.append(f"  column totals: got {gen_s}/{gen_m}, expected 10670/14074")
if gen_s + gen_m != S12_10_TOTAL:
    fail.append(f"  grand total: got {gen_s+gen_m}, §12.10 says {S12_10_TOTAL}")

ev_s = sum(int(idx[(t, "single")]["n_evaluable"]) for t in ENDPOINT)
ev_m = sum(int(idx[(t, "multi")]["n_evaluable"]) for t in ENDPOINT)
if (ev_s, ev_m) != (7753, 10772):
    fail.append(f"  evaluable totals: got {ev_s}/{ev_m}, expected 7753/10772")

macro_s = float(idx[("MACRO-AVERAGE", "single")]["detection_rate_pct"])
macro_m = float(idx[("MACRO-AVERAGE", "multi")]["detection_rate_pct"])
if (macro_s, macro_m) != (77.30, 85.56):
    fail.append(f"  macro-averages: got {macro_s}/{macro_m}, expected 77.30/85.56")

# Pooled (micro) rate, derived here rather than read: the CSV carries no pooled
# row. It is emitted as its own table row so the macro/micro distinction is
# visible in the table itself and not only in the caption.
fired_s = sum(int(idx[(t, "single")]["n_fired"]) for t in ENDPOINT)
fired_m = sum(int(idx[(t, "multi")]["n_fired"]) for t in ENDPOINT)
micro_s = round(100 * fired_s / ev_s, 2)
micro_m = round(100 * fired_m / ev_m, 2)
if (fired_s, fired_m) != (5481, 8920):
    fail.append(f"  fired totals: got {fired_s}/{fired_m}, expected 5481/8920")
if (micro_s, micro_m) != (70.70, 82.81):
    fail.append(f"  pooled rates: got {micro_s}/{micro_m}, expected 70.70/82.81")

if fail:
    print("ASSERTION FAILURE — §12.10 reconciliation broke:", file=sys.stderr)
    print("\n".join(fail), file=sys.stderr)
    sys.exit(1)

print(f"§12.10 reconciliation OK: generated {gen_s}+{gen_m}={gen_s+gen_m}; "
      f"evaluable {ev_s}+{ev_m}={ev_s+ev_m}; macro {macro_s}/{macro_m}")

# ------------------------------------------------------------------- emit tex
def line(t, indent=False):
    s, m = idx[(t, "single")], idx[(t, "multi")]
    label = t if not indent else t
    return (f"{label} & {DESC[t]} & {s['check']} & "
            f"{int(s['n_generated']):,} & {int(s['n_evaluable']):,} & {s['detection_rate_pct']} & "
            f"{int(m['n_generated']):,} & {int(m['n_evaluable']):,} & {m['detection_rate_pct']} \\\\")

L = []
L.append("% GENERATED FILE — do not edit by hand.")
L.append("% Produced by make_tables_tex.py from table1_detection_by_type.csv.")
L.append("% Every n_generated cell is asserted against amendment §12.10 at generation time.")
L.append("% Regenerate with:  venv/bin/python make_tables_tex.py")
L.append("")
L.append("\\begin{table}[t]")
L.append("  \\centering")
L.append("  \\small")
L.append("  \\setlength{\\tabcolsep}{4pt}")
L.append("  \\caption{Detection by error type for Checks A and B. $n_{\\mathrm{gen}}$ counts "
         "every mutant generated and $n_{\\mathrm{eval}}$ the evaluable subset remaining after "
         "the exclusions of \\S2.1, and every rate is taken over $n_{\\mathrm{eval}}$. Each type is "
         "scored by the check its error slot makes applicable, type 8 being the one type "
         "visible to both. The macro-average weights the eight types equally, so it differs "
         "from the pooled rate in the final row. Types 4a and 4b are pooled into type 4 for "
         "the macro-average and listed only for transparency.}")
L.append("  \\label{tab:detection-by-type}")
L.append("  \\begin{tabular}{@{}llc rrr rrr@{}}")
L.append("    \\toprule")
L.append("    & & & \\multicolumn{3}{c}{single-resource} & \\multicolumn{3}{c}{multi-resource} \\\\")
L.append("    \\cmidrule(lr){4-6}\\cmidrule(l){7-9}")
L.append("    \\# & error type & check & $n_{\\mathrm{gen}}$ & $n_{\\mathrm{eval}}$ & \\% "
         "& $n_{\\mathrm{gen}}$ & $n_{\\mathrm{eval}}$ & \\% \\\\")
L.append("    \\midrule")
for t in ENDPOINT:
    L.append("    " + line(t))
L.append("    \\midrule")
L.append("    \\multicolumn{9}{@{}l}{\\itshape variants of type 4, reported only, pooled into "
         "type 4 for the macro-average} \\\\")
for t in VARIANTS:
    L.append("    " + line(t))
L.append("    \\midrule")
s, m = idx[("MACRO-AVERAGE", "single")], idx[("MACRO-AVERAGE", "multi")]
L.append(f"    \\multicolumn{{3}}{{@{{}}l}}{{\\bfseries macro-average}} & "
         f"{int(s['n_generated']):,} & {int(s['n_evaluable']):,} & \\bfseries {s['detection_rate_pct']} & "
         f"{int(m['n_generated']):,} & {int(m['n_evaluable']):,} & \\bfseries {m['detection_rate_pct']} \\\\")
L.append(f"    \\multicolumn{{3}}{{@{{}}l}}{{pooled over all evaluable}} & "
         f"& {ev_s:,} & {micro_s:.2f} & & {ev_m:,} & {micro_m:.2f} \\\\")
L.append("    \\bottomrule")
L.append("  \\end{tabular}")
L.append("\\end{table}")

open("paper/tables.tex", "w").write("\n".join(L) + "\n")
print(f"wrote paper/tables.tex ({len(L)} lines)")
