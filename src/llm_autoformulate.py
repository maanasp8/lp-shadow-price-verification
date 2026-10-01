"""Real LLM autoformulation -> run the checks on errors a generator actually makes.

WHY THIS EXISTS
---------------
Every rate in the paper is measured on mechanically mutated annotations. No
LLM-generated model appears anywhere in the evaluation, which is the single
gap most likely to sink a review of a paper about verifying LLM-generated
models. This script closes it.

DESIGN
------
Variable names are GIVEN to the model in the prompt and it is required to use
them verbatim. That is deliberate. The paper measures the CHECKS, and the
checks are conditional on referent alignment, which Appendix C measures
separately and finds unreliable. Supplying the names isolates formulation error
from alignment error, so a detection rate here describes the checks rather than
the extractor. Any model that returns an unknown name is recorded as a parse
failure, never silently repaired.

A generated model is scored WRONG when its optimal objective value differs from
the annotation's, or it fails to solve. That is a lower bound on wrongness: a
model can be wrong and coincidentally right here, exactly the population
Section 4.2 calls answer-neutral. We report both, and never call a matching optimum
"correct" -- the column is named `matches_gold`.

COST CONTROL
------------
One request per instance, no retries beyond `--retries`, and `--limit` caps the
run. Results append to a JSONL cache keyed by problem_id, so a re-run resumes
and never pays twice. Start with `--limit 10` to see the cost before committing.

USAGE
-----
    export ANTHROPIC_API_KEY=sk-...
    venv/bin/python llm_autoformulate.py --limit 10          # smoke test
    venv/bin/python llm_autoformulate.py --limit 100         # the real run
    venv/bin/python llm_autoformulate.py --score-only        # re-score cache
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import re
import sys

from checks_common import REL_TOL
from mutations import check_a, check_b, check_d, solve
from nl4opt_parser import ParsedConstraint, ParsedProblem
from row_normalization import check_a_norm, check_b_norm, scales_for
from text_terms import load_clean_problems, text_terms_of

TODAY = datetime.date.today().isoformat()
CACHE = "data/llm_autoformulate_cache.jsonl"
DEFAULT_MODEL = "claude-opus-5"

PROMPT = """You are formulating a linear program from a word problem.

Use EXACTLY these decision variable names, in this order:
{names}

Return ONLY a JSON object, no prose and no code fence, with this shape:
{{
  "sense": "max" or "min",
  "objective": {{"<var>": <number>, ...}},
  "constraints": [
    {{"name": "c0", "coefs": {{"<var>": <number>, ...}}, "op": "<=" or ">=",
      "rhs": <number>}}
  ]
}}

Rules:
- Every key in "objective" and in each "coefs" must be one of the given names.
- Express every stated restriction, including upper and lower bounds on a
  variable, as an entry in "constraints". Do not rely on implicit bounds.
- Variables are continuous and nonnegative; do not add x >= 0 rows.
- Use only "<=" and ">=". Convert any equality into the pair of inequalities.

Problem:
{doc}"""

# Variant used by --with-row-labels. Supplying row names mirrors what we already
# do for variable names, and tests whether the alignment failure of the row-name
# experiment is a property of the method or of the harness.
#
# READ THE RESULT AS AN UPPER BOUND, NOT A DEPLOYABLE CONFIGURATION. The
# annotation names rows by DECLARATION TYPE (c0_linear, c1_upperbound, c2_sum),
# so listing them tells the generator how many constraints to emit and what kind
# each is. That is a substantial part of the formulation task, given away. Any
# alignment or detection rate measured here bounds what would be achievable if a
# perfect row correspondence were somehow available; it does not describe a
# system anyone could deploy, because in deployment nobody holds the annotation's
# row names.
PROMPT_ROWS = """You are formulating a linear program from a word problem.

Use EXACTLY these decision variable names, in this order:
{names}

Use EXACTLY these constraint names, one per constraint, in this order:
{rows}

Return ONLY a JSON object, no prose and no code fence, with this shape:
{{
  "sense": "max" or "min",
  "objective": {{"<var>": <number>, ...}},
  "constraints": [
    {{"name": "<one of the constraint names above>",
      "coefs": {{"<var>": <number>, ...}}, "op": "<=" or ">=",
      "rhs": <number>}}
  ]
}}

Rules:
- Every key in "objective" and in each "coefs" must be one of the given variable
  names, and every "name" must be one of the given constraint names, used once.
- Express every stated restriction, including upper and lower bounds on a
  variable, as an entry in "constraints". Do not rely on implicit bounds.
- Variables are continuous and nonnegative; do not add x >= 0 rows.
- Use only "<=" and ">=". Convert any equality into the pair of inequalities.

Problem:
{doc}"""


# --------------------------------------------------------------------------

def build(spec: dict, names: list, pid: str) -> ParsedProblem:
    """LLM JSON -> ParsedProblem. Raises ValueError on anything unrecognised."""
    from gurobipy import GRB
    idx = {n: i for i, n in enumerate(names)}
    sense_s = str(spec.get("sense", "")).lower()
    if sense_s not in ("max", "min"):
        raise ValueError(f"bad sense {spec.get('sense')!r}")
    obj = []
    for v, c in (spec.get("objective") or {}).items():
        if v not in idx:
            raise ValueError(f"unknown objective variable {v!r}")
        obj.append((float(c), idx[v]))
    cons = []
    for k, c in enumerate(spec.get("constraints") or []):
        op = c.get("op")
        if op not in ("<=", ">="):
            raise ValueError(f"bad operator {op!r}")
        lhs = []
        for v, a in (c.get("coefs") or {}).items():
            if v not in idx:
                raise ValueError(f"unknown constraint variable {v!r}")
            lhs.append((float(a), idx[v]))
        if not lhs:
            continue
        cons.append(ParsedConstraint(name=str(c.get("name") or f"llm{k}"),
                                     ctype="linear", lhs=lhs, op=op,
                                     rhs_terms=[], rhs_const=float(c.get("rhs", 0.0))))
    return ParsedProblem(pid=pid, split="llm", document="", var_names=list(names),
                         sense=GRB.MAXIMIZE if sense_s == "max" else GRB.MINIMIZE,
                         obj_terms=obj, constraints=cons)


def align_rows(cand: ParsedProblem, gold: ParsedProblem, how: str = "content") -> str:
    """Rename the candidate's rows to the annotation's, IN ORDER.

    Checks A and B key a shadow price to a row by NAME, so they require a
    correspondence between the text's rows and the candidate's. The annotation
    names rows by declaration type (c0_linear, c1_upperbound); a generator names
    them whatever it likes. On this run the name overlap was zero on every
    instance, which is the row-level half of the referent-alignment problem of
    Appendix C -- the half the body of the paper does not discuss.

    We align positionally, which is the weakest assumption that yields any
    alignment at all: both the annotation and the generator tend to emit
    constraints in the order the text states them. It is a HEURISTIC. When it
    is wrong it produces a violation that looks like a detection, so every rate
    computed downstream of it must be reported as conditional on it.

    -> "positional" when the counts match and renaming happened,
       "count_mismatch" otherwise (no alignment is possible, nothing renamed).
    """
    if how == "name":
        return "name"                      # leave names alone; checks will miss
    if len(cand.constraints) != len(gold.constraints):
        return "count_mismatch"
    if how == "position":
        for cc, gc in zip(cand.constraints, gold.constraints):
            cc.name = gc.name
        return "positional"

    # content: pair each annotation row with the unused candidate row whose
    # coefficients, operator and limit agree best. This is the only strategy
    # that works -- and it IS a coefficient comparison, which is the baseline
    # these checks are meant to replace. Reported, not hidden.
    gt, ct = text_terms_of(gold), text_terms_of(cand)
    names, pool = list(ct.row_names), list(range(len(cand.constraints)))

    def cost(gn, ci):
        cn = names[ci]
        gc, cc = gt.row_coefs[gn], ct.row_coefs[cn]
        d = sum(abs(gc.get(k, 0.0) - cc.get(k, 0.0))
                for k in set(gc) | set(cc))
        d += abs(gt.row_limits[gn] - ct.row_limits[cn])
        return d + (0.0 if gt.row_operator[gn] == ct.row_operator[cn] else 1e6)

    for gn in gt.row_names:
        if not pool:
            return "count_mismatch"
        best = min(pool, key=lambda ci: cost(gn, ci))
        cand.constraints[best].name = gn
        pool.remove(best)
    return "content"


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in response")
    return json.loads(t[i:j + 1])


def load_cache() -> dict:
    if not os.path.exists(CACHE):
        return {}
    out = {}
    with open(CACHE) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r.get("model", DEFAULT_MODEL), r.get("variant", "plain"),
                     r["problem_id"])] = r
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="generator to audit; use a weaker one to obtain errors")
    ap.add_argument("--with-row-labels", action="store_true",
                    help="supply the annotation's row names in the prompt; see "
                         "PROMPT_ROWS for why this is an upper bound only")
    ap.add_argument("--align", default="content",
                    choices=["name", "position", "content"],
                    help="row correspondence; see align_rows()")
    args = ap.parse_args()

    problems, _ = load_clean_problems()
    with open("data/results.csv") as fh:
        opt = [r["problem_id"] for r in csv.DictReader(fh)
               if r["solve_status"] == "OPTIMAL"]
    ids = [p for p in opt if p in problems and problems[p].document]
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.limit]

    cache = load_cache()
    variant = "rowlabels" if args.with_row_labels else "plain"
    todo = [p for p in ids if (args.model, variant, p) not in cache]
    print(f"{len(ids)} instances selected; {len(cache)} cached; {len(todo)} to fetch")

    if todo and not args.score_only:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            sys.exit("REFUSING TO RUN: set ANTHROPIC_API_KEY first. "
                     "Re-run with --score-only to score what is already cached.")
        import anthropic
        client = anthropic.Anthropic()
        with open(CACHE, "a") as fh:
            for n, pid in enumerate(todo, 1):
                prob = problems[pid]
                if args.with_row_labels:
                    msg = PROMPT_ROWS.format(
                        names="\n".join(f"- {v}" for v in prob.var_names),
                        rows="\n".join(f"- {c.name}" for c in prob.constraints),
                        doc=prob.document)
                else:
                    msg = PROMPT.format(
                        names="\n".join(f"- {v}" for v in prob.var_names),
                        doc=prob.document)
                rec = {"problem_id": pid, "model": args.model,
                       "variant": variant, "raw": None, "error": None}
                for attempt in range(args.retries + 1):
                    try:
                        r = client.messages.create(
                            model=args.model, max_tokens=2000,
                            messages=[{"role": "user", "content": msg}])
                        if getattr(r, "stop_reason", None) == "refusal":
                            rec["error"] = "refusal"
                            break
                        txt = next((b.text for b in r.content
                                    if getattr(b, "type", "") == "text"), None)
                        if txt is None:
                            rec["error"] = "no text block in response"
                            break
                        rec["raw"] = txt
                        break
                    except Exception as e:                       # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if n % 10 == 0:
                    print(f"  fetched {n}/{len(todo)}")
        cache = load_cache()

    # ------------------------------------------------------------- scoring
    rows = []
    for pid in ids:
        rec = cache.get((args.model, variant, pid))
        if not rec:
            continue
        gold = problems[pid]
        text = text_terms_of(gold)
        gsol = solve(gold)
        out = {"problem_id": pid, "status": "", "align": "", "n_rows_gold": "",
               "n_rows_llm": "", "matches_gold": "",
               "a": "", "b": "", "c": "", "any": "", "any_norm": ""}
        if not rec.get("raw"):
            out["status"] = "no_response"
            rows.append(out)
            continue
        try:
            cand = build(extract_json(rec["raw"]), gold.var_names, pid)
        except Exception as e:                                   # noqa: BLE001
            out["status"] = f"parse_fail: {type(e).__name__}"
            rows.append(out)
            continue
        out["n_rows_gold"] = len(gold.constraints)
        out["n_rows_llm"] = len(cand.constraints)
        out["align"] = align_rows(cand, gold, args.align)
        sol = solve(cand)
        if not sol.optimal:
            out["status"] = f"solve_{sol.status}"
            rows.append(out)
            continue
        out["status"] = "OPTIMAL"
        out["matches_gold"] = (abs(sol.obj_value - gsol.obj_value)
                               <= REL_TOL * max(1.0, abs(gsol.obj_value)))
        a = check_a(text, sol)[0]
        b = check_b(text, sol)[0]
        c = check_d(text, sol)[0]
        k = scales_for(text, text_terms_of(cand))
        out.update({"a": a, "b": b, "c": c, "any": a or b or c,
                    "any_norm": check_a_norm(text, sol, k)
                                or check_b_norm(text, sol, k) or c})
        rows.append(out)

    if not rows:
        print("nothing scored: cache is empty. Set ANTHROPIC_API_KEY and run "
              "without --score-only.")
        return
    tag = f"{args.model.split('-')[1]}_{args.align}_{variant}"
    out_csv = f"llm_autoformulate_{tag}_{TODAY}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print("\n  alignment: " + str(dict(Counter(
        r["align"] for r in rows if r["align"]))))
    if args.with_row_labels:
        named = [r for r in rows if r["n_rows_llm"] != ""]
        print(f"  row labels supplied: {len(named)} models parsed; run this with "
              f"--align name to test\n  whether a supplied correspondence "
              f"dissolves the alignment failure.")
    ok = [r for r in rows if r["status"] == "OPTIMAL"
          and r["align"] in ("positional", "content", "name")]
    wrong = [r for r in ok if r["matches_gold"] is False]
    match = [r for r in ok if r["matches_gold"] is True]
    print("\n" + "=" * 66)
    print("REAL LLM AUTOFORMULATION")
    print("=" * 66)
    print(f"  attempted            {len(rows)}")
    print(f"  solved AND alignable {len(ok)}")
    print(f"  optimum != gold      {len(wrong)}")
    print(f"  optimum == gold      {len(match)}")
    if wrong:
        d = sum(bool(r["any"]) for r in wrong)
        dn = sum(bool(r["any_norm"]) for r in wrong)
        print(f"\n  DETECTION on models whose optimum differs from gold:")
        print(f"    checks            {d}/{len(wrong)} = {100*d/len(wrong):.1f}%")
        print(f"    normalized        {dn}/{len(wrong)} = {100*dn/len(wrong):.1f}%")
    if match:
        f = sum(bool(r["any"]) for r in match)
        fn = sum(bool(r["any_norm"]) for r in match)
        print(f"\n  On models reproducing the gold optimum (NOT proof of correctness):")
        print(f"    checks report     {f}/{len(match)} = {100*f/len(match):.1f}%")
        print(f"    normalized        {fn}/{len(match)} = {100*fn/len(match):.1f}%")
        print("    (a report here is either a real defect that does not move the")
        print("     optimum, or a false positive; inspect before reporting either)")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
