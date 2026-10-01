#!/usr/bin/env python
"""Generate paper/fig_wrapper_taxonomy.tex (Figure 2) from the referent
adjudication sheet.

Counts are read from `referent_adjudication_sheet_2026-08-07.csv`
(`wrapper_category`), never hardcoded. The expected distribution is asserted and
the total is asserted to equal 54; any mismatch is fatal.

    venv/bin/python make_fig2_tex.py
"""
import csv, collections, sys

SHEET = "data/referent_adjudication_sheet_2026-08-07.csv"
EXPECT = {"investment": 15, "unit-of-measure": 15, "usage/temporal": 12,
          "count": 11, "head-noun substitution": 1}
EXAMPLE = {  # illustrative gloss per category, for the axis label only
    "investment": r"``amount invested in $X$''",
    "unit-of-measure": r"``acres of $X$''",
    "usage/temporal": r"``minutes $X$ is used''",
    "count": r"``number of $X$''",
    "head-noun substitution": r"head-noun swap",
}

rows = list(csv.DictReader(open(SHEET)))
counts = collections.Counter(r["wrapper_category"] for r in rows)

fail = []
if sum(counts.values()) != 54:
    fail.append(f"  total is {sum(counts.values())}, expected 54")
if dict(counts) != EXPECT:
    fail.append(f"  distribution mismatch\n    artefact: {dict(counts)}\n    expected: {EXPECT}")
if fail:
    print("ASSERTION FAILURE — wrapper taxonomy did not reproduce:", file=sys.stderr)
    print("\n".join(fail), file=sys.stderr)
    sys.exit(1)
print(f"wrapper taxonomy OK: {dict(counts)} total {sum(counts.values())}")

order = sorted(counts, key=lambda k: (-counts[k], k))
bars = "\n".join(f"        ({counts[k]},{i}) [{counts[k]}]" for i, k in enumerate(order))
ticks = ",".join(str(i) for i in range(len(order)))
labels = ",".join("{%s}" % k.replace("&", r"\&") for k in order)

TEX = r"""%% Figure 2 -- wrapper taxonomy of the 54 hand-adjudicated referent cases.
%% GENERATED FILE -- do not edit by hand. Produced by make_fig2_tex.py from
%% referent_adjudication_sheet_2026-08-07.csv; counts are asserted at generation
%% time against a total of 54. Regenerate with:
%%   venv/bin/python make_fig2_tex.py
%%
%% Requires in the preamble:
%%   \usepackage{pgfplots}
%%   \pgfplotsset{compat=1.18}
%%
%% Width: the axis is sized in \linewidth units, so the same source fits the 8pp
%% build and the 4pp cut without editing coordinates.

\begin{figure}[t]
  \centering
  \begin{tikzpicture}
    \begin{axis}[
        xbar,
        width=0.80\linewidth,
        height=42mm,
        font=\small,
        bar width=3.4mm,
        xmin=0, xmax=%(xmax)d,
        y=6.2mm,
        ytick={%(ticks)s},
        yticklabels={%(labels)s},
        yticklabel style={font=\small},
        y dir=reverse,
        xlabel={variables (of $n=54$)},
        xlabel style={font=\small},
        xtick align=inside,
        nodes near coords,
        nodes near coords style={font=\scriptsize},
        every node near coord/.append style={xshift=1pt},
        axis x line*=bottom,
        axis y line*=left,
        tick style={draw=none},
        enlarge y limits={abs=0.6},
      ]
      \addplot[fill=black!20, draw=black!55] coordinates {
%(bars)s
      };
    \end{axis}
  \end{tikzpicture}
  %% TODO: caption must state --
  %%  (1) that these 54 are the ground-truth variables that exact match did NOT
  %%      resolve, out of 63 total -- 9 resolved by exact match, 63/63 after
  %%      hand adjudication;
  %%  (2) that in 53 of 54 cases the annotation key is a literal SUBSTRING of the
  %%      extractor's name, i.e. a naming-convention collision, not synonymy;
  %%  (3) that the single head-noun-substitution case (Case 54) was paired BY
  %%      ELIMINATION from the problem text, not by containment evidence, and is
  %%      therefore weaker evidence than the other 53 -- report as 53 + 1;
  %%  (4) that the clustering is instance-level, not noise: 4 of 25 instances
  %%      matched all their variables and 21 of 25 matched none, with zero
  %%      partial instances (Wilson 95%% CI on the 4/25 rate [6.4%%, 34.7%%]);
  %%  (5) that adjudication is asymmetric -- it can only move a figure in the
  %%      extractor's favour -- so the as-measured 9/63 is the floor and
  %%      scores.json is unaltered.
  \caption{}
  \label{fig:wrapper-taxonomy}
\end{figure}
""" % dict(xmax=max(counts.values()) + 3, ticks=ticks, labels=labels, bars=bars)

open("paper/fig_wrapper_taxonomy.tex", "w").write(TEX)
print("wrote paper/fig_wrapper_taxonomy.tex")
