"""Mutation engine -- candidate enumeration, mutant construction, buckets.

Single source of truth for what a mutant IS. Both the section-5 confirmation
probe (`confirm_applicability_real.py`) and the Step 1 harness
(`mutation_harness.py`) import from here, so a mutant scored in the probe and a
mutant scored in the harness are built by the same code.

Nothing here writes to disk and nothing here decides how many mutants to
generate. Enumeration is pure counting; construction is per candidate.

CANONICAL FORM
--------------
Coefficient mutations (types 1, 4a, 4b, 6) address a row's CANONICAL NET
coefficients -- `lhs - rhs_terms`, the vector gurobipy actually builds -- and
not the declared `lhs`/`rhs_terms` split. Applying one rewrites the row into an
all-lhs canonical form first. That rewrite is model-identical (same row, same
duals) and makes every cell individually addressable, including the structural
coefficients of `sum`, bound, `ratio`, `xby` and `xy` rows.

This is the definition §11's Step 0 counts are built on -- see
`mutation_harness.py --counts`, which reproduces all nine cells of the §11
table exactly. It is WIDER than a "written number in the text" reading; see the
note in `mutation_harness.py`.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from checks_common import REL_TOL, ZERO_TOL
from nl4opt_parser import ParsedConstraint, ParsedProblem, build_model
from text_terms import TextTerms, row_coefficients

# --------------------------------------------------------------------------
# type-8 generation rule -- amendment §1, revised §12.3, corrected §12.6
# --------------------------------------------------------------------------
# The invented row duplicates an existing row's coefficient pattern and takes
# `f * u` as its limit, where `u` is that row's realised usage in the clean
# optimum. For the row to BIND, the new limit must sit strictly on the
# tightening side of `u`: below `u` for a `<=` row, above `u` for a `>=` row.
# With a MULTIPLICATIVE fraction that depends on the sign of `u` as well as the
# operator -- f<1 scales toward zero, which tightens a `<=` row only when u>0.
#
#   <=  : f < 1 when u > 0,  f > 1 when u < 0
#   >=  : f > 1 when u > 0,  f < 1 when u < 0
#
# §12.3 conditioned on the operator alone and was wrong for the 57 in-scope
# rows that are `<=` with negative usage; see §12.6.
FRACTION_TIGHTEN = 0.5          # magnitudes only; the sign logic is below
FRACTION_LOOSEN = 2.0


def spurious_fraction(op: str, u: float) -> float:
    """Multiplier that makes `f * u` bind for a row with operator `op` at
    realised usage `u`. Undefined at u == 0, which is excluded upstream."""
    return FRACTION_TIGHTEN if (op == "<=") == (u > 0) else FRACTION_LOOSEN

# A candidate row whose realised usage in the clean optimum is zero is dropped:
# any fraction of zero is zero, so the invented row cannot be made to bind in a
# way that changes the answer. Threshold sits inside the measured empirical gap
# (5.820766e-11, 0.3) -- every cutoff from 1e-10 to 1e-2 yields the identical
# 276 / 2546 partition of in-scope rows -- and coincides with gurobi's default
# FeasibilityTol. Same idiom as ZERO_TOL (see pick_zero_tol.py).
USAGE_ZERO_TOL = 1e-6

_STATUS = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
           GRB.UNBOUNDED: "UNBOUNDED", GRB.INF_OR_UNBD: "INF_OR_UNBD",
           GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.NUMERIC: "NUMERIC"}

_ENV = None


def env() -> gp.Env:
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


# ==========================================================================
# solving
# ==========================================================================

@dataclass
class Solved:
    status: str
    obj_value: Optional[float]
    pi: Dict[str, float]
    slack: Dict[str, float]
    values: Dict[str, float]

    @property
    def optimal(self) -> bool:
        return self.status == "OPTIMAL"


def solve(prob: ParsedProblem) -> Solved:
    m = build_model(prob, env=env())
    m.optimize()
    status = _STATUS.get(m.Status, f"STATUS_{m.Status}")
    if status != "OPTIMAL":
        m.dispose()
        return Solved(status, None, {}, {}, {})
    out = Solved(status, m.ObjVal,
                 {r.ConstrName: r.Pi for r in m.getConstrs()},
                 {r.ConstrName: r.Slack for r in m.getConstrs()},
                 {nm: gv.X for gv, nm in zip(m.getVars(), prob.var_names)})
    m.dispose()
    return out


def usage(prob: ParsedProblem, con: ParsedConstraint, sol: Solved) -> float:
    """Realised usage of a row at a solution: canonical LHS evaluated at x*."""
    return (sum(a * sol.values[prob.var_names[i]] for a, i in con.lhs)
            - sum(a * sol.values[prob.var_names[i]] for a, i in con.rhs_terms))


# ==========================================================================
# the two checks, against a mutant solve with clean text quantities
# ==========================================================================

def check_a(text: TextTerms, sol: Solved) -> Tuple[bool, List[dict]]:
    """Text term list and text sense, mutant pi and mutant variable values."""
    detail = []
    for name in text.var_names:
        c_j = text.obj_coef(name)
        total, n_terms = 0.0, 0
        for row_name, a_text in text.terms_for(name):
            if row_name not in sol.pi:          # text row absent from the mutant
                continue
            total += a_text * sol.pi[row_name]
            n_terms += 1
        value = sol.values[name]
        scale = REL_TOL * max(1.0, abs(c_j))
        if abs(value) > ZERO_TOL:
            rel, fired = "=", abs(total - c_j) > scale
        elif text.sense == "max":               # TEXT sense, not the mutant's
            rel, fired = ">=", total < c_j - scale
        else:
            rel, fired = "<=", total > c_j + scale
        detail.append({"variable": name, "value": value, "n_terms": n_terms,
                       "sum": total, "obj_coeff": c_j, "relation": rel,
                       "residual": total - c_j, "fired": fired})
    return any(d["fired"] for d in detail), detail


def check_b(text: TextTerms, sol: Solved) -> Tuple[bool, dict]:
    """Text limits, mutant pi, mutant ObjVal. Rows absent from the text (an
    invented row) are not in the sum -- that is how B sees type 8."""
    total = sum(text.row_limits[rn] * sol.pi[rn]
                for rn in text.row_names if rn in sol.pi)
    obj = sol.obj_value
    scale = max(1.0, abs(obj))
    return (abs(total - obj) > REL_TOL * scale,
            {"sum": total, "obj_value": obj, "abs_diff": abs(total - obj)})


def check_d(text: TextTerms, sol: Solved) -> Tuple[bool, List[dict]]:
    """CHECK D -- primal feasibility of the mutant's optimum against the TEXT.

    Evaluate every row of the CLEAN annotation at the mutant's optimal x* and
    ask whether it is satisfied. Text-derived: a_ij, b_i and the operator.
    Mutant-derived: x* only. The objective is never touched, which is why §5
    marks D N/A for every objective-slot error.

    EVERY text row is evaluated, including rows the mutant no longer has -- that
    is exactly how D sees a deleted row (constraint omission).

    Fire tolerance is the frozen one (§9): a violation counts when it exceeds
    `REL_TOL * max(1, |b_i|)`. Same constant, same relative form as A and B.
    """
    detail = []
    for rn in text.row_names:
        b = text.row_limits[rn]
        lhs = sum(a * sol.values[v] for v, a in text.row_coefs[rn].items())
        op = text.row_operator[rn]
        violation = (lhs - b) if op == "<=" else (b - lhs)
        tol = REL_TOL * max(1.0, abs(b))
        detail.append({"row": rn, "op": op, "lhs": lhs, "limit": b,
                       "violation": violation, "tol": tol,
                       "fired": violation > tol})
    return any(d["fired"] for d in detail), detail


# ==========================================================================
# candidates
# ==========================================================================

@dataclass
class Candidate:
    error_type: str                 # "1" "2" "3" "4a" "4b" "5" "6" "7" "8"
    direction: str                  # "x10" "/10" "swap" "omit" "invent" "flip"
    slot: str                       # textual identity of the cell(s) touched
    touched_rows: List[str]
    apply: Callable[[ParsedProblem], ParsedProblem] = field(repr=False)


def _canonicalise(p: ParsedProblem, k: int) -> None:
    """Rewrite row k into all-lhs canonical form in place. Model-identical."""
    con = p.constraints[k]
    net: Dict[int, float] = {}
    for a, i in con.lhs:
        net[i] = net.get(i, 0.0) + a
    for a, i in con.rhs_terms:
        net[i] = net.get(i, 0.0) - a
    con.lhs = [(a, i) for i, a in sorted(net.items())]
    con.rhs_terms = []


def _set_coef(p: ParsedProblem, k: int, idx: int, value: float) -> None:
    _canonicalise(p, k)
    con = p.constraints[k]
    con.lhs = [(value if i == idx else a, i) for a, i in con.lhs]


# --------------------------------------------------------------------------
# candidate scope for types 1, 4a, 4b, 6 -- amendment §12.9, ratified 2026-08-02
# --------------------------------------------------------------------------
# NARROW is PRIMARY and feeds the §6 macro-average: only coefficients declared
# as text in a `linear` row's `terms`. That is the only population §0's realism
# argument -- "an LLM copying a number that is present in the source text as a
# token" -- actually covers.
#
# WIDE additionally addresses the structural coefficients of `sum`, bound,
# `ratio`, `xby` and `xy` rows, which come from declaration semantics rather
# than from a written number. Reported as a labelled secondary extension, the
# same treatment as the sense-flip footnote; NOT in the primary endpoint.
#
# NARROW is a strict subset of WIDE: in a `linear` row `rhs_terms` is empty, so
# each declared term IS the structural cell.
NARROW = "narrow"
WIDE = "wide"
DEFAULT_SCOPE = NARROW


def _cells(prob: ParsedProblem, scope: str = DEFAULT_SCOPE
           ) -> List[Tuple[int, int, float, str, str]]:
    """Coefficient cells with a NONZERO net value.
    -> [(row index, var index, coefficient, row name, var name)]
    Zero cells are exclusion list 1 (x10 and /10 both leave zero unchanged)."""
    out = []
    for k, con in enumerate(prob.constraints):
        if scope == NARROW and con.ctype != "linear":
            continue
        coefs = row_coefficients(con, prob.var_names)
        for j, nm in enumerate(prob.var_names):
            if nm in coefs and coefs[nm] != 0.0:
                out.append((k, j, coefs[nm], con.name, nm))
    return out


def _nonzero_limit_rows(prob: ParsedProblem) -> List[int]:
    """Limit mutations skip zero-RHS rows -- exclusion list 2."""
    return [k for k, c in enumerate(prob.constraints) if c.rhs_const != 0.0]


def _mut(prob, fn):
    p = copy.deepcopy(prob)
    fn(p)
    return p


# ---- type 1 / 6 : one candidate per structural nonzero coefficient cell ----

def cand_coef_magnitude(prob, factor: float, scope: str = DEFAULT_SCOPE
                        ) -> List[Candidate]:
    d = "x10" if factor == 10.0 else "/10"
    return [Candidate("1", d, f"a[{rn},{vn}]={a:g}", [rn],
                      (lambda k=k, j=j, a=a: lambda p: _set_coef(p, k, j, a * factor))())
            for k, j, a, rn, vn in _cells(prob, scope)]


def cand_coef_omission(prob, scope: str = DEFAULT_SCOPE) -> List[Candidate]:
    """a_ij -> 0 with the row surviving: the row needs another variable.

    The "row survives" requirement of §1 is applied as written -- a cell in a
    single-variable row is NOT a candidate, because zeroing the only
    coefficient leaves `0 op b`, a constant, not a constraint. §11's original
    type-6 count did not apply it; corrected in §12.8.
    """
    out = []
    for k, j, a, rn, vn in _cells(prob, scope):
        if len(row_coefficients(prob.constraints[k], prob.var_names)) < 2:
            continue
        out.append(Candidate("6", "omit", f"a[{rn},{vn}]={a:g}->0", [rn],
                             (lambda k=k, j=j: lambda p: _set_coef(p, k, j, 0.0))()))
    return out


# ---- type 2 / 5 : objective coefficients ----------------------------------

def cand_obj_magnitude(prob, factor: float) -> List[Candidate]:
    d = "x10" if factor == 10.0 else "/10"
    out = []
    for t, (c, i) in enumerate(prob.obj_terms):
        if c == 0.0:
            continue
        out.append(Candidate("2", d, f"c[{prob.var_names[i]}]={c:g}", [],
                             (lambda t=t, c=c, i=i:
                              lambda p: p.obj_terms.__setitem__(t, (c * factor, i)))()))
    return out


def cand_obj_swap(prob) -> List[Candidate]:
    out = []
    for s, t in itertools.combinations(range(len(prob.obj_terms)), 2):
        (c1, i1), (c2, i2) = prob.obj_terms[s], prob.obj_terms[t]
        if c1 == c2:                    # no-op, bucket 0 at generation
            continue

        def ap(p, s=s, t=t, c1=c1, i1=i1, c2=c2, i2=i2):
            p.obj_terms[s] = (c2, i1)
            p.obj_terms[t] = (c1, i2)
        out.append(Candidate("5", "swap",
                             f"c[{prob.var_names[i1]}]={c1:g} <-> "
                             f"c[{prob.var_names[i2]}]={c2:g}", [], ap))
    return out


# ---- type 3 / 7 : limits --------------------------------------------------

def cand_limit_magnitude(prob, factor: float) -> List[Candidate]:
    d = "x10" if factor == 10.0 else "/10"
    out = []
    for k in _nonzero_limit_rows(prob):
        b = prob.constraints[k].rhs_const
        rn = prob.constraints[k].name
        out.append(Candidate("3", d, f"b[{rn}]={b:g}", [rn],
                             (lambda k=k, b=b:
                              lambda p: setattr(p.constraints[k], "rhs_const",
                                                b * factor))()))
    return out


def cand_limit_swap(prob) -> List[Candidate]:
    out = []
    rows = _nonzero_limit_rows(prob)
    for k1, k2 in itertools.combinations(rows, 2):
        b1, b2 = prob.constraints[k1].rhs_const, prob.constraints[k2].rhs_const
        if b1 == b2:
            continue
        n1, n2 = prob.constraints[k1].name, prob.constraints[k2].name

        def ap(p, k1=k1, k2=k2, b1=b1, b2=b2):
            p.constraints[k1].rhs_const = b2
            p.constraints[k2].rhs_const = b1
        out.append(Candidate("7", "swap", f"b[{n1}]={b1:g} <-> b[{n2}]={b2:g}",
                             [n1, n2], ap))
    return out


# ---- type 4 : coefficient swaps -------------------------------------------

def cand_coef_swap_within(prob, scope: str = DEFAULT_SCOPE) -> List[Candidate]:
    """Two unequal coefficients inside one row exchange places."""
    out = []
    by_row: Dict[int, List] = {}
    for k, j, a, rn, vn in _cells(prob, scope):
        by_row.setdefault(k, []).append((j, a, rn, vn))
    for k, cs in by_row.items():
        for (j1, a1, rn, v1), (j2, a2, _, v2) in itertools.combinations(cs, 2):
            if a1 == a2:
                continue

            def ap(p, k=k, j1=j1, j2=j2, a1=a1, a2=a2):
                _set_coef(p, k, j1, a2)
                _set_coef(p, k, j2, a1)
            out.append(Candidate("4a", "swap",
                                 f"a[{rn},{v1}]={a1:g} <-> a[{rn},{v2}]={a2:g}",
                                 [rn], ap))
    return out


def cand_coef_swap_cross(prob, scope: str = DEFAULT_SCOPE) -> List[Candidate]:
    """The SAME variable's coefficients in two different rows exchange places
    -- the reading error where a value is attached to the wrong row."""
    out = []
    by_var: Dict[int, List] = {}
    for k, j, a, rn, vn in _cells(prob, scope):
        by_var.setdefault(j, []).append((k, a, rn, vn))
    for j, cs in by_var.items():
        for (k1, a1, n1, vn), (k2, a2, n2, _) in itertools.combinations(cs, 2):
            if a1 == a2:
                continue

            def ap(p, k1=k1, k2=k2, j=j, a1=a1, a2=a2):
                _set_coef(p, k1, j, a2)
                _set_coef(p, k2, j, a1)
            out.append(Candidate("4b", "swap",
                                 f"a[{n1},{vn}]={a1:g} <-> a[{n2},{vn}]={a2:g}",
                                 [n1, n2], ap))
    return out


# ---- type 8 : spurious constraint -----------------------------------------

def eligible_spurious_rows(prob, clean: Solved) -> List[Tuple[int, float]]:
    """-> [(row index, realised usage)] for rows that can seed an invented row.

    Rows whose realised usage is zero are DROPPED (§12.3): any fraction of zero
    is zero, so the invented row either duplicates the original exactly or sits
    at the origin, and in neither case can it change the answer.
    """
    out = []
    for k, con in enumerate(prob.constraints):
        if not con.lhs and not con.rhs_terms:
            continue
        u = usage(prob, con, clean)
        if abs(u) <= USAGE_ZERO_TOL:
            continue
        out.append((k, u))
    return out


def cand_spurious(prob, clean: Solved) -> List[Candidate]:
    """ONE candidate per instance -- amendment §12.7(d), ratified 2026-08-02.

    Source row = the eligible row with MEDIAN realised usage in the clean
    optimum. Deterministic and reproducible, and it does not select toward
    whichever row is easiest or hardest to detect. Ties and even counts:
    eligible rows are sorted by `(usage, row name)` and the LOWER median,
    index `(n-1)//2`, is taken -- so the choice is total-ordered even when two
    rows carry identical usage.

    Limit := `spurious_fraction(op, u) * u`, which binds for either operator and
    either sign of `u` (§12.6).
    """
    elig = eligible_spurious_rows(prob, clean)
    if not elig:
        return []
    elig.sort(key=lambda t: (t[1], prob.constraints[t[0]].name))
    k, u = elig[(len(elig) - 1) // 2]
    con = prob.constraints[k]
    f = spurious_fraction(con.op, u)
    limit = f * u
    new_name = f"c{len(prob.constraints)}_spurious"

    def ap(p, k=k, limit=limit, new_name=new_name):
        src = p.constraints[k]
        p.constraints.append(ParsedConstraint(
            name=new_name, ctype="linear", lhs=copy.deepcopy(src.lhs),
            op=src.op, rhs_terms=copy.deepcopy(src.rhs_terms),
            rhs_const=limit))
    return [Candidate("8", "invent",
                      f"{new_name} = pattern of {con.name} "
                      f"(median of {len(elig)} eligible), "
                      f"limit {f:g}*usage={limit:g}", [new_name], ap)]


# ---- footnote : sense flip -------------------------------------------------

def cand_sense_flip(prob) -> List[Candidate]:
    was = "max" if prob.sense == GRB.MAXIMIZE else "min"
    now = "min" if was == "max" else "max"

    def ap(p):
        p.sense = GRB.MINIMIZE if p.sense == GRB.MAXIMIZE else GRB.MAXIMIZE
    return [Candidate("SF", "flip", f"sense {was}->{now}", [], ap)]


# `scope` is honoured only by the four types §12.9 splits; the rest ignore it.
# ---- §2 secondary types: DETECTION AGAINST CHECK D ONLY -------------------
# Both are provably invisible to A and B (§5). They are enumerated so their
# candidate and bucket counts can be reported; NO detection rate is computed for
# them while Check D does not exist.

def cand_constraint_omission(prob) -> List[Candidate]:
    """Delete a whole row."""
    out = []
    for k, con in enumerate(prob.constraints):
        def ap(p, k=k):
            del p.constraints[k]
        out.append(Candidate("CO", "delete", f"delete {con.name}", [con.name], ap))
    return out


def cand_operator_flip(prob) -> List[Candidate]:
    """Flip a row's operator, <= <-> >=. No numeric error required."""
    out = []
    for k, con in enumerate(prob.constraints):
        new = ">=" if con.op == "<=" else "<="
        def ap(p, k=k, new=new):
            p.constraints[k].op = new
        out.append(Candidate("OF", "flip", f"{con.name} {con.op} -> {new}",
                             [con.name], ap))
    return out


SECONDARY_ENUMERATORS = {
    "CO": lambda p, c, s=None: cand_constraint_omission(p),
    "OF": lambda p, c, s=None: cand_operator_flip(p),
}


ENUMERATORS = {
    "1":  lambda p, c, s=DEFAULT_SCOPE: cand_coef_magnitude(p, 10.0, s),
    "2":  lambda p, c, s=DEFAULT_SCOPE: cand_obj_magnitude(p, 10.0),
    "3":  lambda p, c, s=DEFAULT_SCOPE: cand_limit_magnitude(p, 10.0),
    "4a": lambda p, c, s=DEFAULT_SCOPE: cand_coef_swap_within(p, s),
    "4b": lambda p, c, s=DEFAULT_SCOPE: cand_coef_swap_cross(p, s),
    "5":  lambda p, c, s=DEFAULT_SCOPE: cand_obj_swap(p),
    "6":  lambda p, c, s=DEFAULT_SCOPE: cand_coef_omission(p, s),
    "7":  lambda p, c, s=DEFAULT_SCOPE: cand_limit_swap(p),
    "8":  lambda p, c, s=DEFAULT_SCOPE: cand_spurious(p, c),
    "SF": lambda p, c, s=DEFAULT_SCOPE: cand_sense_flip(p),
}

# the ÷10 arm of the three magnitude types. §12.7(b): BOTH directions are
# generated, as separate mutants.
ENUMERATORS_DIV = {
    "1": lambda p, c, s=DEFAULT_SCOPE: cand_coef_magnitude(p, 0.1, s),
    "2": lambda p, c, s=DEFAULT_SCOPE: cand_obj_magnitude(p, 0.1),
    "3": lambda p, c, s=DEFAULT_SCOPE: cand_limit_magnitude(p, 0.1),
}

# types whose candidate set depends on the narrow/wide scope (§12.9)
SCOPED_TYPES = {"1", "4a", "4b", "6"}


# ==========================================================================
# buckets -- §7, assigned before any check runs
# ==========================================================================

def signature(p: ParsedProblem):
    return (p.sense, sorted(p.obj_terms),
            sorted((c.op, c.rhs_const,
                    tuple(sorted(row_coefficients(c, p.var_names).items())))
                   for c in p.constraints))


def is_noop(clean_prob: ParsedProblem, mut_prob: ParsedProblem) -> bool:
    """Bucket 0: the mutant model is not structurally different from clean.

    NOTE the limit of this test, established empirically on 2026-08-02: it
    compares the SET of rows, so a mutant carrying an extra row that adds
    nothing to the feasible region -- a duplicated row -- is NOT caught here.
    That is why type 8's zero-usage exclusion above is a real filter and not a
    belt-and-braces restatement of bucket 0.
    """
    return signature(clean_prob) == signature(mut_prob)


def bucket_of(touched: List[str], clean: Solved, mut: Solved) -> str:
    if not mut.optimal:
        return "1 solver refused"
    live = [r for r in touched if r in clean.slack and r in mut.slack]
    if live and all(abs(clean.slack[r]) > 1e-6 and abs(mut.slack[r]) > 1e-6
                    for r in live):
        return "2 inert"
    return "3 evaluable"


def binding(sol: Solved, rows: List[str]) -> str:
    return ";".join("-" if r not in sol.slack else str(abs(sol.slack[r]) < 1e-6)
                    for r in rows) or "-"
