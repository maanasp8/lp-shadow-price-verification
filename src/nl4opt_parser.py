"""
Parse NL4Opt sub-task 2 (generation) annotations into gurobipy models.

Semantics follow the organizers' canonical converter
(nl4opt-subtask2-baseline/parsers.py::convert_to_canonical), with two
deliberate departures documented in DEPARTURES below.

Every instance either parses or raises ParseError with a reason. Nothing is
skipped silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

DEPARTURES = """
1. `linear`: the baseline initialises constraint rows to np.ones(), so variables
   absent from `terms` silently receive coefficient 1 (affects 64 constraints).
   We use coefficient 0 -- i.e. only the listed terms appear.
2. `xy`: the baseline also initialises from np.ones() and never sets the RHS,
   yielding `x - y <= 1` plus stray coefficient-1 terms. We use `x op y`.
Both baseline quirks cancel out in prediction-vs-truth scoring but would
corrupt dual values.
"""


class ParseError(Exception):
    """Raised when an instance cannot be faithfully translated."""


# --------------------------------------------------------------------------
# number parsing
# --------------------------------------------------------------------------

# Fraction / multiplier words. The first block is the organizers' NUMS_DICT
# (parsing_utils/constants.py) verbatim; the rest are equivalents observed in
# the corpus scan.
WORD_FRACTIONS = {
    "a half": 0.5, "half": 0.5, "one half": 0.5,
    "one third": 1 / 3, "third": 1 / 3, "a third": 1 / 3,
    "one fourth": 0.25, "fourth": 0.25, "a fourth": 0.25,
    "one fifth": 0.2, "fifth": 0.2, "a fifth": 0.2,
    "twice": 2.0, "thrice": 3.0,
    "quarter": 0.25, "a quarter": 0.25, "one quarter": 0.25,
}

WORD_INTS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _words_to_number(s: str) -> Optional[float]:
    """Compound English numeral -> float, or None if not a numeral."""
    total, current, found = 0.0, 0.0, False
    for tok in s.replace("-", " ").split():
        if tok in WORD_INTS:
            current += WORD_INTS[tok]
            found = True
        elif tok == "hundred":
            current = (current or 1) * 100
            found = True
        elif tok in SCALES:
            total += (current or 1) * SCALES[tok]
            current = 0.0
            found = True
        elif tok == "and":
            continue
        else:
            return None
    return total + current if found else None


def parse_number(raw: str) -> float:
    """Number-like string -> float. Raises ParseError rather than guessing.

    Mirrors the organizers' Parser.parse_number cleanup order (strip commas and
    currency marks, percent/cents -> /100, then digit extraction, then words)
    but raises instead of falling back to 0.
    """
    if raw is None:
        raise ParseError("missing number")
    x = str(raw).strip().replace(",", "").strip(";,$")
    x = re.sub(r"\.+$", "", x)

    multiplier = 1.0
    stripped = re.sub(r"(percent)|%|¢", "", x)
    if stripped != x:
        multiplier = 0.01
        x = stripped.strip()

    try:
        return float(x) * multiplier
    except ValueError:
        pass

    low = x.lower().strip()

    # "3 times", "four times", "1.5 times"
    if low.endswith(" times"):
        return parse_number(low[: -len(" times")]) * multiplier

    if low in WORD_FRACTIONS:
        return WORD_FRACTIONS[low] * multiplier

    # embedded digits: "2$.", "8000 minutes"
    if re.search(r"\d", x):
        digits = re.sub(r"\.+$", "", re.sub(r"[^0-9.]", "", x))
        try:
            return float(digits) * multiplier
        except ValueError:
            pass

    val = _words_to_number(low)
    if val is not None:
        return val * multiplier

    raise ParseError(f"unparseable number {raw!r}")


# --------------------------------------------------------------------------
# objective direction
# --------------------------------------------------------------------------

MAX_WORDS = {"maximize", "maximum", "maximized", "maximizing", "maximise",
             "most", "highest", "largest"}
MIN_WORDS = {"minimize", "minimum", "minimizes", "minimizing", "minimal",
             "minimise", "lowest", "reduce", "decrease", "least", "smallest"}


def parse_direction(raw: str) -> int:
    low = str(raw).strip().lower()
    if low in MAX_WORDS:
        return GRB.MAXIMIZE
    if low in MIN_WORDS:
        return GRB.MINIMIZE
    raise ParseError(f"unknown objective direction {raw!r}")


# --------------------------------------------------------------------------
# parsed representation
# --------------------------------------------------------------------------

KNOWN_TYPES = {"linear", "sum", "upperbound", "lowerbound", "ratio", "xby", "xy"}

Term = Tuple[float, int]  # (coefficient, variable index)


@dataclass
class ParsedConstraint:
    """`sum(lhs) op sum(rhs_terms) + rhs_const`, op in {<=, >=}."""
    name: str
    ctype: str
    lhs: List[Term]
    op: str
    rhs_terms: List[Term]
    rhs_const: float

    def coefficients(self) -> List[float]:
        return [c for c, _ in self.lhs] + [c for c, _ in self.rhs_terms]


@dataclass
class ParsedProblem:
    pid: str
    split: str
    document: str
    var_names: List[str]
    sense: int
    obj_terms: List[Term]
    constraints: List[ParsedConstraint]
    notes: List[str] = field(default_factory=list)

    @property
    def n_vars(self) -> int:
        return len(self.var_names)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _variable_names(inst: dict) -> List[str]:
    order = inst["order_mapping"]
    if not order:
        raise ParseError("empty order_mapping")
    n = max(order.values()) + 1
    names: List[Optional[str]] = [None] * n
    # canonical `vars` entries win; other mentions are fallbacks
    for mention, idx in order.items():
        if not 0 <= idx < n:
            raise ParseError(f"order_mapping index {idx} out of range for {mention!r}")
        if names[idx] is None:
            names[idx] = mention
    for name in inst.get("vars", []):
        if name in order:
            names[order[name]] = name
    return [nm if nm is not None else f"var_{i}" for i, nm in enumerate(names)]


def _resolve(mention: str, order: Dict[str, int]) -> int:
    if mention not in order:
        raise ParseError(f"variable mention {mention!r} not in order_mapping")
    return order[mention]


def _check_collision(mentions: List[str], order: Dict[str, int], where: str) -> None:
    """Distinct mentions in one declaration must map to distinct variables."""
    distinct = set(mentions)
    if len(distinct) != len({order[m] for m in distinct}):
        raise ParseError(
            f"order_mapping_collision: distinct mentions in {where} collapse onto "
            f"the same variable index ({sorted(distinct)})"
        )


def _op_of(decl: dict) -> str:
    operator = decl.get("operator")
    if operator == "LESS_OR_EQUAL":
        return "<="
    if operator == "GREATER_OR_EQUAL":
        return ">="
    raise ParseError(f"unknown operator {operator!r}")


def _parse_ratio_limit(raw: str, notes: List[str]) -> float:
    """Ratio limits are shares of the total. Bare numbers are percentages the
    annotator wrote without a '%'; rescale, then reject anything still absurd."""
    value = parse_number(raw)
    if value > 1.0:
        rescaled = value / 100.0
        notes.append(f"ratio limit {raw!r} rescaled {value:g} -> {rescaled:g}")
        value = rescaled
    if value > 1.0 or value < 0.0:
        raise ParseError(f"ratio_limit_out_of_range: {raw!r} -> {value:g} after rescale")
    return value


def parse_instance(pid: str, inst: dict, split: str = "") -> ParsedProblem:
    """Annotation dict -> ParsedProblem. Raises ParseError on any doubt."""
    order: Dict[str, int] = inst["order_mapping"]
    var_names = _variable_names(inst)
    n = len(var_names)
    all_idx = list(range(n))
    notes: List[str] = []

    # ---- objective ----
    obj = inst["obj_declaration"]
    sense = parse_direction(obj["direction"])
    otype = obj.get("type")
    if otype == "objective":
        terms_raw = obj.get("terms")
        if terms_raw is None:
            raise ParseError("objective of type 'objective' has no 'terms'")
        _check_collision(list(terms_raw), order, "obj_declaration.terms")
        obj_terms = [(parse_number(v), _resolve(k, order)) for k, v in terms_raw.items()]
        if not obj_terms:
            notes.append("empty objective (obj_declaration.terms == {})")
    elif otype == "objvar":
        mentions = obj.get("vars")
        if mentions is None:
            raise ParseError("objective of type 'objvar' has no 'vars'")
        _check_collision(list(mentions), order, "obj_declaration.vars")
        # objvar => every listed variable enters with coefficient 1
        obj_terms = [(1.0, _resolve(m, order)) for m in mentions]
    else:
        raise ParseError(f"unknown objective type {otype!r}")

    # ---- constraints ----
    constraints: List[ParsedConstraint] = []
    for i, decl in enumerate(inst["const_declarations"]):
        ctype = decl.get("type")
        if ctype not in KNOWN_TYPES:
            raise ParseError(f"unknown constraint type {ctype!r}")
        op = _op_of(decl)
        name = f"c{i}_{ctype}"

        if ctype == "linear":
            terms_raw = decl["terms"]
            _check_collision(list(terms_raw), order, f"{name}.terms")
            lhs = [(parse_number(v), _resolve(k, order)) for k, v in terms_raw.items()]
            rhs_terms, rhs_const = [], parse_number(decl["limit"])

        elif ctype == "sum":
            # constrains the sum of every variable; carries no var/terms field
            lhs = [(1.0, j) for j in all_idx]
            rhs_terms, rhs_const = [], parse_number(decl["limit"])

        elif ctype in ("upperbound", "lowerbound"):
            lhs = [(1.0, _resolve(decl["var"], order))]
            rhs_terms, rhs_const = [], parse_number(decl["limit"])

        elif ctype == "ratio":
            limit = _parse_ratio_limit(decl["limit"], notes)
            lhs = [(1.0, _resolve(decl["var"], order))]
            rhs_terms, rhs_const = [(limit, j) for j in all_idx], 0.0

        elif ctype == "xby":
            _check_collision([decl["x_var"], decl["y_var"]], order, name)
            lhs = [(1.0, _resolve(decl["x_var"], order))]
            rhs_terms = [(parse_number(decl["param"]), _resolve(decl["y_var"], order))]
            rhs_const = 0.0

        else:  # xy
            _check_collision([decl["x_var"], decl["y_var"]], order, name)
            lhs = [(1.0, _resolve(decl["x_var"], order))]
            rhs_terms = [(1.0, _resolve(decl["y_var"], order))]
            rhs_const = 0.0

        constraints.append(ParsedConstraint(name, ctype, lhs, op, rhs_terms, rhs_const))

    if not constraints:
        notes.append("no constraint declarations")

    return ParsedProblem(pid=pid, split=split, document=inst["document"],
                         var_names=var_names, sense=sense, obj_terms=obj_terms,
                         constraints=constraints, notes=notes)


# --------------------------------------------------------------------------
# gurobipy model construction
# --------------------------------------------------------------------------

def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def build_model(prob: ParsedProblem, env: Optional[gp.Env] = None) -> gp.Model:
    """ParsedProblem -> gurobipy Model. Bounds are added as named constraints,
    not variable bounds, so every declaration yields a dual."""
    m = gp.Model(f"nl4opt_{_safe(prob.pid)}", env=env) if env is not None \
        else gp.Model(f"nl4opt_{_safe(prob.pid)}")
    m.setParam("OutputFlag", 0)
    # Presolve off: presolve may aggregate or drop rows, so the basis reported
    # back is not necessarily the one the untouched model would yield. Duals are
    # still uncrushed correctly either way, but degeneracy detection depends on
    # WHICH optimal basis is returned, so solve the model as written.
    m.setParam("Presolve", 0)
    # Never let INF_OR_UNBD collapse infeasible and unbounded into one bucket.
    m.setParam("DualReductions", 0)

    x = [m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                  name=_safe(nm)) for nm in prob.var_names]
    m.update()

    m.setObjective(gp.quicksum(c * x[i] for c, i in prob.obj_terms), prob.sense)

    for con in prob.constraints:
        lhs = gp.quicksum(c * x[i] for c, i in con.lhs)
        rhs = gp.quicksum(c * x[i] for c, i in con.rhs_terms) + con.rhs_const
        if con.op == "<=":
            m.addConstr(lhs <= rhs, name=con.name)
        else:
            m.addConstr(lhs >= rhs, name=con.name)
    m.update()
    return m


# --------------------------------------------------------------------------
# readable algebra
# --------------------------------------------------------------------------

def fmt_num(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def _render_side(terms: List[Term], names: List[str], const: float = 0.0) -> str:
    if not terms:
        return fmt_num(const)
    coefs = {c for c, _ in terms}
    if len(terms) > 1 and len(coefs) == 1:
        # factored form reads better: 0.7*(a + b + c)
        coef = terms[0][0]
        inner = " + ".join(names[i] for _, i in terms)
        body = inner if coef == 1.0 else f"{fmt_num(coef)}*({inner})"
    else:
        body = " + ".join(
            names[i] if c == 1.0 else f"{fmt_num(c)}*{names[i]}" for c, i in terms
        )
    if const:
        body += f" + {fmt_num(const)}"
    return body


def render_algebra(prob: ParsedProblem) -> str:
    names = prob.var_names
    sense = "maximize" if prob.sense == GRB.MAXIMIZE else "minimize"
    obj = _render_side(prob.obj_terms, names) if prob.obj_terms else "0"
    lines = [f"{sense} {obj}"]
    width = max((len(c.name) for c in prob.constraints), default=0)
    for k, con in enumerate(prob.constraints):
        prefix = "s.t. " if k == 0 else "     "
        lhs = _render_side(con.lhs, names)
        rhs = _render_side(con.rhs_terms, names, con.rhs_const)
        lines.append(f"{prefix}{con.name:<{width}}  {lhs} {con.op} {rhs}")
    prefix = "s.t. " if not prob.constraints else "     "
    lines.append(f"{prefix}{'':<{width}}  {', '.join(names)} >= 0")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

def load_corpus(paths) -> List[Tuple[str, str, dict]]:
    """-> [(split, instance_id, annotation), ...] in file order."""
    out = []
    for path in paths:
        split = str(path).split("/")[-1].replace(".jsonl", "")
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if len(obj) != 1:
                    raise ParseError(f"{path}:{lineno} has {len(obj)} top-level keys")
                for pid, inst in obj.items():
                    out.append((split, pid, inst))
    return out
