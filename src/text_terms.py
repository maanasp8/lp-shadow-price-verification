"""Text-side term lists for Check A -- read from the CLEAN ANNOTATION ONLY.

WHY THIS MODULE EXISTS
----------------------
Check A's identity is

    sum over every row i of ( a_ij_TEXT * pi_i )   vs   c_j_TEXT

Both `a_ij` and `c_j` are *text-derived* quantities. Under mutation only `pi`
(and the objective value, and the variable values) may come from the mutant
solve; every coefficient must come from the clean annotation.

`check_a.py` previously built its per-variable term list by reading
`constraints.csv`, which is written with zero coefficients filtered out at
|a| > 1e-12 (see resolve_pass.py). That is correct arithmetic on the clean
corpus -- a zero term contributes zero -- but it is the wrong SOURCE. If a
mutation harness re-derived the term list from the mutant's own representation,
a type-6 mutation (coefficient omission, a_ij -> 0) would delete the term from
the list instead of setting it to zero: the sum would equal c_j by construction
and the mutation would be invisible, with no error raised. Amendment
2026-08-02 section 8 item 2.

The loader here fixes the term list ONCE PER INSTANCE from the annotation, so
it cannot depend on what a mutant did to any cell.

WHAT COUNTS AS A TERM
---------------------
A variable "appears in" a row when it appears in the row's declared structure,
i.e. `var_indices(con) = {i for _, i in con.lhs} | {i for _, i in con.rhs_terms}`
-- the same function that produced `n_vars_in_row` in `constraints.csv`. That
includes variables whose NET coefficient is zero (a declared "0" coefficient,
or exact cancellation between the two sides). Such terms contribute 0.0 to the
sum but are present in the list and counted in `n_terms`.

The coefficient is the CANONICAL NET coefficient, `lhs - rhs_terms`, matching
gurobipy's row after `addConstr(lhs op rhs)` moves the rhs across -- so these
values are directly comparable to `constraints.csv.coefficients`, and
`row_limit` is comparable to `constraints.csv.rhs_limit`.

Declaration semantics are NOT reimplemented here: everything comes from
`nl4opt_parser.parse_instance`, the single source of truth. That is why this
module imports gurobipy transitively (nl4opt_parser does, for GRB constants);
`check_a.py` therefore no longer satisfies the old "reads CSVs only" property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nl4opt_parser import (ParseError, ParsedConstraint, ParsedProblem,
                           load_corpus, parse_instance)

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl"
        for s in ("train", "dev", "test")]


# --------------------------------------------------------------------------
# per-row canonical form
# --------------------------------------------------------------------------

def row_coefficients(con: ParsedConstraint, var_names: List[str]) -> Dict[str, float]:
    """-> {var_name: net coefficient}, one entry per variable STRUCTURALLY in
    the row, including entries whose net coefficient is 0.0.

    Net = sum(lhs coefficients) - sum(rhs_terms coefficients), matching the
    gurobipy row built by `build_model`.
    """
    net: Dict[int, float] = {}
    for c, i in con.lhs:
        net[i] = net.get(i, 0.0) + c
    for c, i in con.rhs_terms:
        net[i] = net.get(i, 0.0) - c
    return {var_names[i]: a for i, a in net.items()}


def row_limit(con: ParsedConstraint) -> float:
    """-> the row's canonical RHS constant (rhs_terms have been moved left)."""
    return float(con.rhs_const)


# --------------------------------------------------------------------------
# per-instance text facts
# --------------------------------------------------------------------------

@dataclass
class TextTerms:
    """Every text-derived quantity Check A and Check B are allowed to use.

    Fixed at construction from the clean annotation. A mutation harness holds
    this object constant and varies only the solver output.
    """
    problem_id: str
    var_names: List[str]
    sense: str                                   # "max" / "min"
    obj_coefs: Dict[str, float]                  # every variable, 0.0 if absent
    row_names: List[str]                         # declaration order
    row_ctype: Dict[str, str]
    row_operator: Dict[str, str]                 # "<=" / ">="
    row_coefs: Dict[str, Dict[str, float]]       # row -> {var: net a_ij}
    row_limits: Dict[str, float]                 # row -> b_i

    # --- the two accessors Check A is meant to use --------------------------

    def terms_for(self, var_name: str) -> List[Tuple[str, float]]:
        """-> [(row_name, a_ij_text), ...] for every row the variable appears
        in, in declaration order. Zero-valued terms ARE included."""
        return [(rn, self.row_coefs[rn][var_name])
                for rn in self.row_names
                if var_name in self.row_coefs[rn]]

    def obj_coef(self, var_name: str) -> float:
        return self.obj_coefs[var_name]


# Instrumentation. `TextTerms` must be built ONCE PER INSTANCE and reused
# across every mutant of that instance -- it is a function of the clean
# annotation alone, so rebuilding it per mutant is pure waste and, worse, an
# invitation to accidentally build it from a mutant. `TextTermsStore` below is
# the accessor the harness uses; this counter makes "once per instance"
# checkable rather than asserted.
BUILD_COUNT = {"n": 0}


def text_terms_of(prob: ParsedProblem) -> TextTerms:
    """ParsedProblem (clean) -> TextTerms."""
    BUILD_COUNT["n"] += 1
    obj_coefs: Dict[str, float] = {nm: 0.0 for nm in prob.var_names}
    for c, i in prob.obj_terms:
        obj_coefs[prob.var_names[i]] += c

    row_names, ctype, operator, coefs, limits = [], {}, {}, {}, {}
    for con in prob.constraints:
        row_names.append(con.name)
        ctype[con.name] = con.ctype
        operator[con.name] = con.op
        coefs[con.name] = row_coefficients(con, prob.var_names)
        limits[con.name] = row_limit(con)

    from gurobipy import GRB
    return TextTerms(
        problem_id=prob.pid,
        var_names=list(prob.var_names),
        sense="max" if prob.sense == GRB.MAXIMIZE else "min",
        obj_coefs=obj_coefs,
        row_names=row_names, row_ctype=ctype, row_operator=operator,
        row_coefs=coefs, row_limits=limits,
    )


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

def load_text_terms(paths=DATA, verbose: bool = True
                    ) -> Tuple[Dict[str, TextTerms], Dict[str, str]]:
    """-> ({problem_id: TextTerms}, {problem_id: parse-error reason}).

    Every instance in the corpus is attempted. Instances that do not parse are
    returned in the second dict rather than dropped silently -- they are the
    same 9 that `results.csv` records as PARSE_FAILED and none of them is
    OPTIMAL, so no check consumes them.
    """
    terms: Dict[str, TextTerms] = {}
    failed: Dict[str, str] = {}
    for split, pid, inst in load_corpus(paths):
        try:
            prob = parse_instance(pid, inst, split)
        except ParseError as e:
            failed[pid] = str(e)
            continue
        except Exception as e:                                   # noqa: BLE001
            failed[pid] = f"{type(e).__name__}: {e}"
            continue
        terms[pid] = text_terms_of(prob)
    if verbose:
        print(f"  text term lists built  {len(terms)} instances "
              f"({len(failed)} did not parse)")
    return terms, failed


class TextTermsStore:
    """Per-instance cache of clean text quantities.

    Built once from the corpus; `get(pid)` is memoised, so an instance's
    `TextTerms` is constructed on first use and reused by every mutant of that
    instance. `builds` reports how many were actually constructed -- the harness
    asserts it equals the number of DISTINCT instances touched, never the number
    of mutants.
    """

    def __init__(self, problems: Dict[str, ParsedProblem]):
        self._problems = problems
        self._cache: Dict[str, TextTerms] = {}
        self._built = 0

    def get(self, pid: str) -> TextTerms:
        t = self._cache.get(pid)
        if t is None:
            t = text_terms_of(self._problems[pid])
            self._cache[pid] = t
            self._built += 1
        return t

    @property
    def builds(self) -> int:
        return self._built

    @property
    def cached(self) -> int:
        return len(self._cache)


def load_clean_problems(paths=DATA) -> Tuple[Dict[str, ParsedProblem], Dict[str, str]]:
    """Same sweep, but keeps the whole ParsedProblem. Used by the mutation
    probes, which need the declaration structure to build a mutant."""
    probs: Dict[str, ParsedProblem] = {}
    failed: Dict[str, str] = {}
    for split, pid, inst in load_corpus(paths):
        try:
            probs[pid] = parse_instance(pid, inst, split)
        except Exception as e:                                   # noqa: BLE001
            failed[pid] = f"{type(e).__name__}: {e}"
    return probs, failed
