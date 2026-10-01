"""EXTRACTION PILOT — step 1: draw the stratified sample. Deterministic.

    venv/bin/python extraction_sample.py [--frame literal|native] [--write]

Writes `extraction_pilot/sample.json`. Re-running with the same seed and frame
reproduces the identical sample; the file records seed, frame, and the full
exclusion funnel so the draw is auditable without re-running.

EXCLUSIONS, applied BEFORE sampling, in this order
--------------------------------------------------
(a) the project's three exclusion lists. Under the frozen `native` frame only
    the third is an instance-level exclusion; the other two are scoped by
    amendment §7 to cells and rows respectively and are logged, not applied:
      1. zero-coefficient cells   -> 2 instances   (logged only under `native`)
      2. zero-RHS rows            -> 398 instances (logged only under `native`)
      3. live-phantom `73102943`  -> 1 instance    (applied)
(b) instances with zero `const_declarations`
(c) instances with DUPLICATE entries in their own `vars[]` list — self-
    contradictory ground truth. Every such case is scanned for and logged by
    id, never dropped silently.

FRAME CHOICE — flagged, not silent
----------------------------------
`--frame native` (default, and what the pre-registration freezes as of its
revision 2) applies (a) at the scope the amendment actually defines for it
(§7: zero-RHS is "limit mutations only", zero coefficients are *cells*), so only
the live-phantom instance is an instance-level exclusion. 1084 eligible, 50 in
the >=3-var bucket — a 20% draw.

`--frame literal` (rejected; frozen by revision 1 of the pre-registration and
superseded before any extraction ran) applies (a) at INSTANCE level: any
instance containing a zero-RHS row is excluded outright. It removes **398 of
1092 parsed instances (36%)** and leaves only **19** instances with >=3
variables, so the 10-instance multi-variable stratum becomes a 53% draw — closer
to a census than a sample. The rows it removes are all `xby`/`ratio`/`xy`, which
are among the better extraction stress cases. Retained as a reproducible
alternative, not as the design.

Neither is chosen silently: the frame is a recorded field of `sample.json` and
of the pre-registration.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from typing import Dict, List

import pandas as pd

from checks_common import load_mutation_exclusions
from nl4opt_parser import ParseError, load_corpus, parse_instance

DATA = [f"nl4opt-competition/generation_data/{s}.jsonl"
        for s in ("train", "dev", "test")]
OUTDIR = "data/extraction_pilot"
SAMPLE_PATH = os.path.join(OUTDIR, "sample.json")

# Frozen by the pre-registration. Changing either changes the sample.
SEED = 20260805
N_MULTI = 10      # instances with >= 3 variables
N_TWO = 15        # instances with exactly 2 variables
N_TOTAL = N_MULTI + N_TWO


def scan_duplicate_vars(corpus) -> Dict[str, Dict[str, int]]:
    """(c) — instances whose own `vars[]` repeats a name. Logged, not hidden."""
    out = {}
    for _split, pid, inst in corpus:
        vs = inst.get("vars") or []
        rep = {k: v for k, v in collections.Counter(vs).items() if v > 1}
        if rep:
            out[pid] = rep
    return out


def scan_zero_coefficient_instances(corpus) -> Dict[str, List[str]]:
    """Exclusion list 1, found BY VALUE (never by hardcoded id)."""
    out: Dict[str, List[str]] = {}
    for _split, pid, inst in corpus:
        for d in (inst.get("const_declarations") or []):
            if d.get("type") != "linear":
                continue
            for var, raw in (d.get("terms") or {}).items():
                try:
                    if float(str(raw).replace(",", "")) == 0.0:
                        out.setdefault(pid, []).append(var)
                except ValueError:
                    pass
    return out


def build_frame(frame: str) -> dict:
    corpus = load_corpus(DATA)
    n_vars, parse_failed = {}, {}
    for split, pid, inst in corpus:
        try:
            n_vars[pid] = len(parse_instance(pid, inst, split).var_names)
        except ParseError as e:
            parse_failed[pid] = str(e)
        except Exception as e:                                   # noqa: BLE001
            parse_failed[pid] = f"{type(e).__name__}: {e}"

    dup_vars = scan_duplicate_vars(corpus)
    zero_coef = scan_zero_coefficient_instances(corpus)
    zero_const = [pid for _s, pid, inst in corpus
                  if not inst.get("const_declarations")]
    phantom = load_mutation_exclusions()

    cons = pd.read_csv("data/constraints.csv", dtype={"problem_id": str})
    zero_rhs = sorted(set(cons.loc[cons.rhs_limit == 0.0, "problem_id"]), key=int)

    if frame == "literal":
        list_a = set(zero_coef) | set(zero_rhs) | set(phantom)
    else:
        list_a = set(phantom)

    excluded = list_a | set(zero_const) | set(dup_vars)
    eligible = sorted(set(n_vars) - excluded, key=int)

    return {
        "corpus_total": len(corpus),
        "parse_failed": parse_failed,
        "n_vars": n_vars,
        "scans": {
            "duplicate_vars_instances": dup_vars,
            "zero_coefficient_instances": zero_coef,
            "zero_const_declarations": zero_const,
            "zero_rhs_instances_n": len(zero_rhs),
            "phantom_excluded": sorted(phantom),
        },
        "frame": frame,
        "excluded_n": len(excluded),
        "eligible": eligible,
    }


def draw(frame_data: dict) -> dict:
    n_vars = frame_data["n_vars"]
    elig = frame_data["eligible"]
    multi = [p for p in elig if n_vars[p] >= 3]
    two = [p for p in elig if n_vars[p] == 2]

    if len(multi) < N_MULTI or len(two) < N_TWO:
        raise SystemExit(f"stratum too small: >=3 vars {len(multi)} (need "
                         f"{N_MULTI}), ==2 vars {len(two)} (need {N_TWO})")

    rng = random.Random(SEED)
    pick_multi = sorted(rng.sample(multi, N_MULTI), key=int)
    pick_two = sorted(rng.sample(two, N_TWO), key=int)
    return {"multi": pick_multi, "two": pick_two,
            "stratum_sizes": {"multi": len(multi), "two": len(two)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", choices=["literal", "native"], default="native")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("EXTRACTION PILOT — SAMPLE")
    print("=" * 78)
    print(f"  seed {SEED}   strata: {N_MULTI} x (>=3 vars) + {N_TWO} x (==2 vars)"
          f" = {N_TOTAL}")
    print(f"  frame: {args.frame}")

    fd = build_frame(args.frame)
    s = fd["scans"]

    print(f"\n  corpus                              {fd['corpus_total']}")
    print(f"  parse failures (excluded, no GT)    {len(fd['parse_failed'])}")
    print(f"\n  (c) DUPLICATE vars[] — logged, all cases:")
    for pid, rep in sorted(s["duplicate_vars_instances"].items(), key=lambda x: int(x[0])):
        print(f"        {pid:>13}  {rep}")
    print(f"  (b) zero const_declarations         {s['zero_const_declarations']}")
    print(f"  (a1) zero-coefficient instances     "
          f"{ {k: v for k, v in s['zero_coefficient_instances'].items()} }")
    print(f"  (a2) zero-RHS instances             {s['zero_rhs_instances_n']}"
          + ("  [EXCLUDED under this frame]" if args.frame == "literal"
             else "  [NOT instance-excluded under this frame]"))
    print(f"  (a3) live phantom                   {s['phantom_excluded']}")

    print(f"\n  excluded (union)                    {fd['excluded_n']}")
    print(f"  ELIGIBLE                            {len(fd['eligible'])}")

    picks = draw(fd)
    ss = picks["stratum_sizes"]
    print(f"\n  stratum sizes: >=3 vars {ss['multi']}   ==2 vars {ss['two']}")
    print(f"  draw rate:     >=3 vars {100*N_MULTI/ss['multi']:.0f}%   "
          f"==2 vars {100*N_TWO/ss['two']:.1f}%")
    if ss["multi"] < 3 * N_MULTI:
        print(f"\n  *** NOTE: the >=3-var stratum is small ({ss['multi']}); the draw "
              f"takes {100*N_MULTI/ss['multi']:.0f}% of it. ***")

    nv = fd["n_vars"]
    print(f"\n  SAMPLE ({N_TOTAL}):")
    print(f"    {'problem_id':>13}  {'n_vars':>6}")
    for pid in picks["multi"]:
        print(f"    {pid:>13}  {nv[pid]:>6}   [multi]")
    for pid in picks["two"]:
        print(f"    {pid:>13}  {nv[pid]:>6}   [two]")

    doc = {
        "generated_on": "2026-08-05",
        "seed": SEED,
        "frame": args.frame,
        "strata": {"multi_ge3": N_MULTI, "two": N_TWO},
        "stratum_sizes": ss,
        "corpus_total": fd["corpus_total"],
        "eligible_n": len(fd["eligible"]),
        "excluded_n": fd["excluded_n"],
        "scans": s,
        "parse_failed": sorted(fd["parse_failed"], key=int),
        "sample": {"multi": picks["multi"], "two": picks["two"]},
        "sample_all": picks["multi"] + picks["two"],
        "n_vars": {p: nv[p] for p in picks["multi"] + picks["two"]},
    }
    if args.write:
        os.makedirs(OUTDIR, exist_ok=True)
        with open(SAMPLE_PATH, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\n  wrote {SAMPLE_PATH}")
    else:
        print("\n  --write not given; nothing written")


if __name__ == "__main__":
    main()
