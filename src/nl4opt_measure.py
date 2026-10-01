"""Classification logic for the shadow-price measurement.

Tolerances are fixed at 1e-6 exactly as specified and are not to be tuned.

Two structural notions are used throughout:

  SHARED row  -- involves >= 2 distinct variables. Its dual prices a resource
                 that variables compete for.
  BOUND row   -- involves exactly 1 distinct variable. Its dual prices that one
                 variable's own limit, not a shared resource.

The split is STRUCTURAL, not by declaration type: 38 `linear` declarations in
the corpus reduce to a single variable (e.g. `8*x <= 65`) and are counted as
BOUND rows, while all 393 `lowerbound` / 291 `upperbound` declarations are
single-variable as expected. Set SPLIT_BY_DECLARATION_TYPE to switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from gurobipy import GRB

from nl4opt_parser import ParsedConstraint, ParsedProblem

TOL = 1e-6

SPLIT_BY_DECLARATION_TYPE = False
BOUND_DECLARATION_TYPES = {"upperbound", "lowerbound"}

USABLE, DEGENERATE, INACTIVE, UNCLASSIFIED = (
    "USABLE", "DEGENERATE", "INACTIVE", "UNCLASSIFIED")
SHARED, BOUND = "SHARED", "BOUND"
FREE, AT_LOWER, AT_UPPER, PINNED = "FREE", "AT_LOWER", "AT_UPPER", "PINNED"


# --------------------------------------------------------------------------
# row structure
# --------------------------------------------------------------------------

def var_indices(con: ParsedConstraint) -> Set[int]:
    return {i for _, i in con.lhs} | {i for _, i in con.rhs_terms}


def row_scope(con: ParsedConstraint) -> str:
    if SPLIT_BY_DECLARATION_TYPE:
        return BOUND if con.ctype in BOUND_DECLARATION_TYPES else SHARED
    return BOUND if len(var_indices(con)) == 1 else SHARED


@dataclass
class SingleVarForm:
    """A one-variable row reduced to `coef * x  op  rhs`."""
    index: int
    coef: float
    op: str
    rhs: float

    @property
    def restricts(self) -> Optional[str]:
        """'upper' / 'lower' / None -- which side of x this row pins."""
        if abs(self.coef) < TOL:
            return None  # row says nothing about x
        upper = (self.op == "<=") == (self.coef > 0)
        return "upper" if upper else "lower"


def single_var_form(con: ParsedConstraint) -> Optional[SingleVarForm]:
    """Reduce a one-variable row to canonical form, netting both sides.

    Returns None for rows touching more than one variable.
    """
    idx = var_indices(con)
    if len(idx) != 1:
        return None
    i = next(iter(idx))
    coef = sum(c for c, j in con.lhs if j == i) - \
        sum(c for c, j in con.rhs_terms if j == i)
    return SingleVarForm(index=i, coef=coef, op=con.op, rhs=con.rhs_const)


# --------------------------------------------------------------------------
# constraint classification
# --------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    name: str
    ctype: str
    scope: str          # SHARED | BOUND
    pi: float
    slack: float
    label: str          # USABLE | DEGENERATE | INACTIVE | UNCLASSIFIED


def classify_constraint(pi: float, slack: float) -> str:
    """Exactly the specified rules -- no reinterpretation of the boundaries.

    A row landing outside all three predicates (possible only on the exact
    tolerance boundary) is reported as UNCLASSIFIED rather than reassigned.
    """
    if abs(pi) > TOL:
        return USABLE
    if abs(slack) < TOL and abs(pi) < TOL:
        return DEGENERATE
    if abs(slack) > TOL and abs(pi) < TOL:
        return INACTIVE
    return UNCLASSIFIED


def classify_constraints(prob: ParsedProblem, model) -> List[ConstraintResult]:
    out = []
    rows = {c.ConstrName: c for c in model.getConstrs()}
    for con in prob.constraints:
        row = rows[con.name]
        pi, slack = row.Pi, row.Slack
        out.append(ConstraintResult(name=con.name, ctype=con.ctype,
                                    scope=row_scope(con), pi=pi, slack=slack,
                                    label=classify_constraint(pi, slack)))
    return out


# --------------------------------------------------------------------------
# variable classification
# --------------------------------------------------------------------------

@dataclass
class VariableResult:
    name: str
    value: float
    label: str                  # FREE | AT_LOWER | AT_UPPER | PINNED
    lower_rows: List[str]       # binding single-variable rows pinning from below
    upper_rows: List[str]       # ... from above
    at_natural_lb: bool         # value sits on the var's own lb (0)
    at_natural_ub: bool


def classify_variables(prob: ParsedProblem, model) -> List[VariableResult]:
    """Is each variable freely chosen, or pinned at a limit of its own?

    Because upper/lower bounds are modelled as ROWS, `var.LB`/`var.UB` stay at
    0/inf and cannot be trusted to reveal a restriction. So a variable counts as
    restricted when EITHER its own gurobi bound is active OR a binding
    single-variable row pins it on that side. Binding uses the same |Slack| <
    1e-6 test as the constraint classification; distance-to-own-bound is the
    slack of the implicit `x >= 0` row, so the identical tolerance applies.

    FREE means "not at a limit of its own" -- the variable may still be fully
    determined by shared constraints.
    """
    gvars = model.getVars()
    rows = {c.ConstrName: c for c in model.getConstrs()}

    lower_by_var: Dict[int, List[str]] = {}
    upper_by_var: Dict[int, List[str]] = {}
    for con in prob.constraints:
        form = single_var_form(con)
        if form is None:
            continue
        side = form.restricts
        if side is None:
            continue
        if abs(rows[con.name].Slack) >= TOL:      # not binding -> not pinning
            continue
        target = upper_by_var if side == "upper" else lower_by_var
        target.setdefault(form.index, []).append(con.name)

    out = []
    for i, name in enumerate(prob.var_names):
        gv = gvars[i]
        value = gv.X
        at_lb = abs(value - gv.LB) < TOL
        at_ub = gv.UB < GRB.INFINITY and abs(value - gv.UB) < TOL
        lower_rows = lower_by_var.get(i, [])
        upper_rows = upper_by_var.get(i, [])

        below = at_lb or bool(lower_rows)
        above = at_ub or bool(upper_rows)
        if below and above:
            label = PINNED
        elif above:
            label = AT_UPPER
        elif below:
            label = AT_LOWER
        else:
            label = FREE

        out.append(VariableResult(name=name, value=value, label=label,
                                  lower_rows=lower_rows, upper_rows=upper_rows,
                                  at_natural_lb=at_lb, at_natural_ub=at_ub))
    return out


# --------------------------------------------------------------------------
# per-problem roll-up
# --------------------------------------------------------------------------

@dataclass
class ProblemResult:
    problem_id: str
    solve_status: str
    n_constraints: int
    n_usable: int
    n_usable_shared: int
    n_usable_bound: int
    n_degenerate: int
    n_inactive: int
    n_unclassified: int
    qualifies: bool
    suspicious: bool


def suspicious_values(prob: ParsedProblem) -> List[float]:
    """Objective coefficients, constraint coefficients and RHS with |v| > 1e4."""
    vals = [c for c, _ in prob.obj_terms]
    for con in prob.constraints:
        vals += con.coefficients()
        vals.append(con.rhs_const)
    return [v for v in vals if abs(v) > 1e4]


def roll_up(prob: ParsedProblem, status: str,
            cons: List[ConstraintResult]) -> ProblemResult:
    usable = [c for c in cons if c.label == USABLE]
    return ProblemResult(
        problem_id=prob.pid,
        solve_status=status,
        n_constraints=len(cons),
        n_usable=len(usable),
        n_usable_shared=sum(1 for c in usable if c.scope == SHARED),
        n_usable_bound=sum(1 for c in usable if c.scope == BOUND),
        n_degenerate=sum(1 for c in cons if c.label == DEGENERATE),
        n_inactive=sum(1 for c in cons if c.label == INACTIVE),
        n_unclassified=sum(1 for c in cons if c.label == UNCLASSIFIED),
        # unchanged definition: >= 1 USABLE constraint of either scope
        qualifies=len(usable) >= 1,
        suspicious=bool(suspicious_values(prob)),
    )
