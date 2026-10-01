"""Emit the scale tables from the basis-artifact sweeps. 2026-09-05.

Follows the convention of make_results_tables.py: the caption lives here, not in
the .tex, and every cell is asserted against the persisted artefact at
generation time so a stale number cannot reach the PDF.

    venv/bin/python src/make_scale_table.py
"""

from __future__ import annotations

import json
import os
import sys
from math import ceil

OUT = os.environ.get("SCALE_TABLE_OUT", "tables_scale.tex")

# The certified class, from the same synthesized programs. Separate artefacts
# because scale_sweep.py also runs the checks and the face LP, which the
# basis-artifact sweep does not need.
CLASS_RUNS = [
    (2, "data/scale_sweep_v2_2026-09-03.json"),
    (4, "data/scale_sweep_v4_2026-09-03.json"),
    (6, "data/scale_sweep_v6_2026-09-03.json"),
]

# (label, file, n_vars). The blend run is the structural control: it differs
# from resource in objective sense, operator direction, coefficient sign and
# whether rows are typed.
RUNS = [
    ("resource, 4 vars", "data/basis_artifact_scale_v4_2026-09-05.json", 4),
    ("blend, 4 vars", "data/basis_artifact_scale_v4_blend_2026-09-04.json", 4),
    ("resource, 6 vars", "data/basis_artifact_scale_v6_2026-09-05.json", 6),
]
RATIOS = [1, 2, 3, 4]


def load():
    cells, alldet = {}, {}
    for lab, path, nv in RUNS:
        if not os.path.exists(path):
            sys.exit(f"missing artefact {path}; run src/basis_artifact_scale.py")
        d = json.load(open(path))
        assert d["n_vars"] == nv, (path, d["n_vars"], nv)
        for r in d["summary"]:
            cv = round(r["n_rows"] / nv)
            cells[(lab, cv)] = (r["basis_artifact_answer_neutral"],
                                r["reference_detects_answer_neutral"])
            alldet[(lab, cv)] = (r["basis_artifact"], r["reference_detects"])
    return cells, alldet


def class_rows():
    """-> [(n_vars, constraints/var, swaps, certified, rate)] for the class."""
    out = []
    for nv, path in CLASS_RUNS:
        if not os.path.exists(path):
            sys.exit(f"missing artefact {path}; run src/scale_sweep.py")
        d = json.load(open(path))
        assert d["n_vars"] == nv
        for r in d["summary"]:
            cv = r["n_rows"] / nv
            if cv < 1.0 or cv > 4.0 or r["swaps"] < 1000:
                continue          # keep the range the three runs have in common
            out.append((nv, cv, r["swaps"], r["certified"],
                        r["certified"] / r["swaps"]))
    return out


def class_table():
    rows = class_rows()
    by = {(nv, round(cv)): rate for nv, cv, _s, _c, rate in rows
          if abs(cv - round(cv)) < 1e-9}
    # the claim the body makes: the class thins as variables are added
    for cv in (1, 2):
        seq = [by[(nv, cv)] for nv in (2, 4, 6) if (nv, cv) in by]
        assert len(seq) == 3, f"need all three n_vars at {cv} constraints/var"
        assert seq[0] > seq[1] > seq[2], f"class rate not falling at {cv}: {seq}"

    body = "\n".join(
        "    %d & %s & %s & %d & %.2f \\\\" % (
            nv, ("%g" % cv), f"{sw:,}", ce, 100.0 * rate)
        for nv, cv, sw, ce, rate in rows)
    caption = (
        "The certified primal-invisible class on the same synthesized programs. "
        "The rate is over evaluable swap mutants, the denominator NL4OPT gives "
        "as 6,183 with 55 certified, or 0.89\\%. The class thins as variables "
        "are added, by roughly a factor of ten per two variables at matched "
        "constraints per variable, and adding rows without adding variables "
        "does not revive it. This is the opposite direction from "
        "Table~\\ref{tab:scale}, and \\S\\ref{app:scale} gives the reason: the class "
        "needs a swap to leave the optimum untouched, and larger programs make "
        "that rarer, while basis ambiguity among the reports that remain grows."
    )
    return f"""
\\begin{{table}}[t]
  \\caption{{{caption}}}
  \\label{{tab:classscale}}
  \\centering
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{6pt}}
  \\begin{{tabular}}{{@{{}}rrrrr@{{}}}}
    \\toprule
    variables & constraints per variable & swap mutants & certified & \\% \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def main():
    cells, alldet = load()

    # every cell the table prints must exist and be adequately powered
    for lab, _p, _v in RUNS:
        for cv in RATIOS:
            k = (lab, cv)
            assert k in cells, f"no cell for {k}"
            art, det = cells[k]
            assert det >= 200, f"{k} has only {det} reports; too thin to print"
            assert 0 <= art <= det

    # the claims the body makes, asserted here so the two cannot drift
    top = {lab: cells[(lab, 4)] for lab, _p, _v in RUNS}
    for lab, (art, det) in top.items():
        assert 100.0 * art / det > 92.0, f"{lab} at 4 constraints/var is not >92%"
    for lab, _p, _v in RUNS:
        seq = [cells[(lab, cv)][0] / cells[(lab, cv)][1] for cv in RATIOS]
        assert all(x < y for x, y in zip(seq, seq[1:])), \
            f"{lab} is not monotone in constraints per variable: {seq}"

    # The spread claims. These used to be one hardcoded "13 points" attributed
    # to all three configurations, which was wrong: 13 is the gap between the
    # two FAMILIES at four variables, and the three configurations spread more
    # than twice that below four constraints per variable, because the third
    # differs in variable count rather than in construction. Both numbers are
    # computed and asserted here so neither can drift again.
    def pct(lab, cv):
        art, det = cells[(lab, cv)]
        return 100.0 * art / det

    def spread(labs, cv):
        vals = [pct(lab, cv) for lab in labs]
        return max(vals) - min(vals)

    all_labs = [lab for lab, _p, _v in RUNS]
    fam_labs = [lab for lab, _p, v in RUNS if v == 4]
    assert len(fam_labs) == 2, "the family comparison needs exactly two 4-var runs"

    agree_at_4 = spread(all_labs, 4)
    below = [cv for cv in RATIOS if cv < 4]
    spread_all = max(spread(all_labs, cv) for cv in below)
    spread_fam = max(spread(fam_labs, cv) for cv in below)

    # the caption says agreement is at the top of the range and not throughout
    assert agree_at_4 < spread_all, \
        f"the three do not agree more closely at 4 ({agree_at_4}) than below ({spread_all})"
    assert spread_fam < spread_all, \
        f"family spread {spread_fam} is not below the all-configuration spread {spread_all}"
    # ceiling, not nearest: each of these is printed as a bound ("to within
    # N", "as much as N"), so rounding down would make the sentence false.
    n_agree, n_all, n_fam = (ceil(agree_at_4), ceil(spread_all), ceil(spread_fam))

    rows = []
    for cv in RATIOS:
        cs = []
        for lab, _p, _v in RUNS:
            art, det = cells[(lab, cv)]
            cs.append(f"{100.0*art/det:.1f} & {det}")
        rows.append(f"    {cv} & " + " & ".join(cs) + r" \\")

    caption = (
        "Share of a reference solve's objective-unchanged reports that are "
        "basis artifacts, on synthesized programs. A report is a basis artifact "
        "when the candidate's optimal face and the reference's intersect, so "
        "some solver could have returned the same point from both and the "
        "report records vertex selection rather than anything the models force. "
        "The share rises with constraints per variable in all three "
        "configurations and exceeds 92\\% at four, which is where the three "
        f"agree to within {n_agree} points. Below that they spread by as much "
        f"as {n_all}, and the two families at four variables, which differ "
        f"only in construction, by as much as {n_fam}. The agreement is at the "
        "top of the range and not throughout. "
        "\\emph{Blend} differs from \\emph{resource} in objective sense, "
        "operator direction, coefficient sign and whether rows are typed. "
        "$n$ is the number of objective-unchanged reports in the cell. "
        "Over all reports rather than this stratum the share falls instead, "
        "from about 10\\% to under 2\\%, for the reason given in "
        "\\S\\ref{app:scale}."
    )

    tex = f"""% GENERATED by src/make_scale_table.py. Do not edit by hand.
% Every cell is asserted against data/basis_artifact_scale_*.json at generation
% time, including monotonicity and the >92%% claim the body makes.
\\begin{{table}}[t]
  \\caption{{{caption}}}
  \\label{{tab:scale}}
  \\centering
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{5pt}}
  \\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
    \\toprule
    & \\multicolumn{{2}}{{c}}{{resource, 4 vars}}
    & \\multicolumn{{2}}{{c}}{{blend, 4 vars}}
    & \\multicolumn{{2}}{{c}}{{resource, 6 vars}} \\\\
    \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
    constraints per variable & \\% & $n$ & \\% & $n$ & \\% & $n$ \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    d = os.path.dirname(OUT)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(tex)
        fh.write(class_table())
    print(f"wrote {OUT}")
    for cv in RATIOS:
        line = f"  {cv} constraints/var: "
        for lab, _p, _v in RUNS:
            art, det = cells[(lab, cv)]
            line += f"{lab} {100.0*art/det:.1f}% (n={det})   "
        print(line)


if __name__ == "__main__":
    main()
