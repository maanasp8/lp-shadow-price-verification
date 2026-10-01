"""EXTRACTION PILOT — step 3: run the extractor. ONE API call per instance.

    venv/bin/python extraction_extract.py --dry-run     print prompts, call nothing
    venv/bin/python extraction_extract.py --run         call the API

Writes one JSON file per instance under `extraction_pilot/raw/`, each containing
the **raw prompt and the raw response** — required for the STS AI-code
disclosure checklist item — plus model, parameters, stop_reason and usage.

ISOLATION
---------
The extractor sees the problem `document` and nothing else. It has no access to
this project's annotations, parser, derivations, known bugs, prior results, or
the conversation that produced it. The prompt below is the complete input.

RUN ORDER IS ENFORCED
---------------------
`--run` refuses to start unless the pre-registration file exists and carries its
FROZEN marker. The pilot's design must be frozen before any extraction happens;
this makes that ordering a precondition rather than a promise.

TEMPERATURE — a decision, not an omission
-----------------------------------------
The design called for `temperature=0`. **`temperature` is not an accepted
parameter on `claude-opus-5`** — sampling parameters were removed on Opus 4.7
and later, and sending one returns a 400. It is therefore omitted, and this is
recorded rather than worked around by silently switching to an older model.
Two notes for the record: `temperature=0` never guaranteed identical outputs on
models that did accept it, and the determinism the design wanted is approached
here instead by pinning every other knob — model id, effort, thinking mode,
max_tokens, and a byte-frozen prompt.

REFUSALS ARE NOT RETRIED ON ANOTHER MODEL
-----------------------------------------
`claude-opus-5` can decline a request (HTTP 200, `stop_reason: "refusal"`). The
usual remedy is a server-side fallback to another model — deliberately NOT used
here: silently serving some instances from a different model would make the
measurement a mixture over two extractors. A refusal is recorded as a failed
extraction with its category and scored as such.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict

OUTDIR = "data/extraction_pilot"
RAW_DIR = os.path.join(OUTDIR, "raw")
SAMPLE_PATH = os.path.join(OUTDIR, "sample.json")
GT_PATH = os.path.join(OUTDIR, "ground_truth.json")
PREREG_PATH = "docs/extraction_pilot_prereg_2026-08-05.md"
FROZEN_MARKER = "STATUS: FROZEN"

# ---- pinned and logged; changing any of these changes the measurement -------
MODEL = "claude-opus-5"
MAX_TOKENS = 16000
EFFORT = "high"
THINKING = {"type": "adaptive"}
# temperature: NOT SENT — rejected on this model (see module docstring)

NUMBER_OR_ABSENT = {
    "anyOf": [{"type": "number"}, {"type": "string", "const": "ABSENT"}]
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "variables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every decision quantity the problem asks the reader to choose.",
        },
        "objective": {
            "type": "object",
            "properties": {
                "sense": {"type": "string", "enum": ["max", "min"]},
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "variable": {"type": "string"},
                            "coefficient": NUMBER_OR_ABSENT,
                        },
                        "required": ["variable", "coefficient"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["sense", "terms"],
            "additionalProperties": False,
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "quoted_text": {"type": "string"},
                    "operator": {"type": "string", "enum": ["<=", ">=", "=="]},
                    "limit": NUMBER_OR_ABSENT,
                    "terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "variable": {"type": "string"},
                                "coefficient": NUMBER_OR_ABSENT,
                            },
                            "required": ["variable", "coefficient"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "quoted_text", "operator", "limit", "terms"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["variables", "objective", "constraints"],
    "additionalProperties": False,
}

# ===========================================================================
# THE FIXED PROMPT — frozen by the pre-registration. Byte-identical per call
# except for the problem text substituted at {document}.
# ===========================================================================
SYSTEM_PROMPT = """\
You extract the numbers of a linear-programming word problem into a typed slot \
table. You are not building or solving a model.

Report only what the text states. Follow these rules exactly.

VARIABLES
- List every decision quantity the problem asks the reader to choose, using the \
wording the text uses.
- Do not invent variables for quantities that are given, not chosen.

OBJECTIVE
- `sense` is "max" or "min", whichever the text asks for.
- Give one term per variable that carries an objective coefficient in the text.
- If a variable appears in the problem but the text gives it no objective \
coefficient, emit a term for it with coefficient "ABSENT".

CONSTRAINTS
- One entry per constraint the text states. Give each a short id of your \
choosing and quote the span of the source text it comes from in `quoted_text`.
- `operator` is "<=", ">=" or "==" as the text implies.
- `limit` is the number on the other side of the comparison.
- Give one term per variable, for every variable you listed. If a variable does \
not appear in that constraint, emit a term for it with coefficient "ABSENT". \
Never omit a variable from a constraint's term list — say "ABSENT" instead.

NUMBERS
- Normalize every number to a plain decimal: no thousands separators, no \
currency marks, no percent signs, no units, no words.
- Write a percentage as the number the text writes, not as a fraction: "30%" \
becomes 30, not 0.3.
- Write a word-number as its digits: "twice" becomes 2, "a third" becomes 0.333.
- If a number is stated but you cannot normalize it, use "ABSENT".

DO NOT
- Do not autoformulate. Do not rearrange a constraint, move terms across the \
comparison, combine constraints, or introduce a coefficient the text does not \
state.
- Do not fold units, rates, or conversions into a coefficient.
- Do not add non-negativity, integrality, or any constraint the text does not \
state.
- Do not fill a missing number with a default such as 1. Use "ABSENT"."""

USER_TEMPLATE = """\
Extract the slot table for this problem.

<problem>
{document}
</problem>"""


def build_prompt(document: str) -> Dict[str, object]:
    return {"system": SYSTEM_PROMPT,
            "messages": [{"role": "user",
                          "content": USER_TEMPLATE.format(document=document)}]}


def check_prereg_frozen() -> None:
    if not os.path.exists(PREREG_PATH):
        raise SystemExit(
            f"REFUSING TO RUN: {PREREG_PATH} does not exist.\n"
            f"The pilot's design must be pre-registered and frozen before any "
            f"extraction happens.")
    text = open(PREREG_PATH).read()
    if FROZEN_MARKER not in text:
        raise SystemExit(
            f"REFUSING TO RUN: {PREREG_PATH} does not carry '{FROZEN_MARKER}'.\n"
            f"Freeze the pre-registration before extracting.")
    print(f"  pre-registration frozen: {PREREG_PATH}  OK")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="print the exact prompts; make no API calls")
    g.add_argument("--run", action="store_true", help="call the API")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt instances whose stored record has "
                         "ok=false (API errors, unparseable output). The prior "
                         "record is archived, never discarded.")
    ap.add_argument("--retry-refusals", action="store_true",
                    help="with --retry-failed, ALSO re-attempt recorded "
                         "refusals. Off by default: §1/§5.4 treat a refusal as "
                         "a measured result, not an error.")
    args = ap.parse_args()

    gt = json.load(open(GT_PATH))
    sample = json.load(open(SAMPLE_PATH))
    ids = sample["sample_all"][: args.limit] if args.limit else sample["sample_all"]

    print("=" * 78)
    print("EXTRACTION PILOT — EXTRACTOR")
    print("=" * 78)
    print(f"  model        {MODEL}")
    print(f"  max_tokens   {MAX_TOKENS}")
    print(f"  effort       {EFFORT}")
    print(f"  thinking     {THINKING}")
    print(f"  temperature  NOT SENT — rejected on {MODEL} (see module docstring)")
    print(f"  fallbacks    NOT USED — a refusal is recorded, never re-run on "
          f"another model")
    print(f"  instances    {len(ids)}")

    if args.dry_run:
        pid = ids[0]
        p = build_prompt(gt[pid]["document"])
        print("\n" + "-" * 78)
        print(f"EXACT PROMPT for {pid} (system, then user)")
        print("-" * 78)
        print(p["system"])
        print("\n--- user ---")
        print(p["messages"][0]["content"])
        print("\n" + "-" * 78)
        print(f"output_config.format schema keys: "
              f"{list(OUTPUT_SCHEMA['properties'])}")
        print(f"\n  dry run — no API call made, nothing written")
        return

    check_prereg_frozen()

    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "REFUSING TO RUN: the `anthropic` package is not installed.\n"
            "  venv/bin/python -m pip install anthropic")

    client = anthropic.Anthropic()
    os.makedirs(RAW_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()

    done = failed = skipped = retried = 0
    skipped_failures: Dict[str, str] = {}
    for n, pid in enumerate(ids, 1):
        path = os.path.join(RAW_DIR, f"{pid}.json")
        prior = None
        if os.path.exists(path):
            try:
                prior = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                prior = {"ok": False, "error": "stored record unreadable"}

        if prior is not None:
            # Resume must key on the RECORD, not on the file's existence. An
            # environment failure (unset credentials, transport error) writes
            # ok=false for every instance, and an exists()-only check would
            # then skip all of them forever on the retry — a failed run would
            # masquerade as a completed one.
            if prior.get("ok"):
                skipped += 1
                continue
            is_refusal = prior.get("error") == "refusal"
            retry = args.retry_failed and (args.retry_refusals or not is_refusal)
            if not retry:
                skipped += 1
                skipped_failures[pid] = (
                    "refusal" if is_refusal else str(prior.get("error"))[:60])
                continue
            # Re-attempting: archive the prior record rather than overwrite it.
            # `raw/failed/` is a subdirectory so `raw/*.json` stays exactly the
            # set of current records — the scorer reads by problem id, but the
            # disclosure artifact should not silently lose a refusal either.
            fdir = os.path.join(RAW_DIR, "failed")
            os.makedirs(fdir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            os.replace(path, os.path.join(fdir, f"{pid}.{stamp}.json"))
            retried += 1
            print(f"  [{n}/{len(ids)}] {pid:>13}  RETRY (prior: "
                  f"{str(prior.get('error'))[:40]}) -> archived")
        prompt = build_prompt(gt[pid]["document"])
        record = {
            "problem_id": pid, "run_on": today, "model": MODEL,
            "params": {"max_tokens": MAX_TOKENS, "effort": EFFORT,
                       "thinking": THINKING, "temperature": None,
                       "temperature_note": "not sent; rejected on this model"},
            "raw_prompt": prompt,
            "output_schema": OUTPUT_SCHEMA,
        }
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking=THINKING,
                output_config={"effort": EFFORT,
                               "format": {"type": "json_schema",
                                          "schema": OUTPUT_SCHEMA}},
                system=prompt["system"],
                messages=prompt["messages"],
            )
        except Exception as e:                                   # noqa: BLE001
            record.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                           "elapsed_s": time.perf_counter() - t0})
            failed += 1
            print(f"  [{n}/{len(ids)}] {pid:>13}  ERROR {type(e).__name__}")
            with open(path, "w") as fh:
                json.dump(record, fh, indent=2)
            continue

        elapsed = time.perf_counter() - t0
        record.update({
            "elapsed_s": elapsed,
            "stop_reason": resp.stop_reason,
            "stop_details": (resp.stop_details.model_dump()
                             if getattr(resp, "stop_details", None) else None),
            "usage": resp.usage.model_dump(),
            "response_model": resp.model,
            "request_id": getattr(resp, "_request_id", None),
            "raw_response": [b.model_dump() for b in resp.content],
        })
        if resp.stop_reason == "refusal":
            record.update({"ok": False, "error": "refusal", "extraction": None})
            failed += 1
            print(f"  [{n}/{len(ids)}] {pid:>13}  REFUSAL")
        else:
            text = next((b.text for b in resp.content if b.type == "text"), None)
            try:
                record.update({"ok": True, "extraction": json.loads(text)})
                done += 1
                print(f"  [{n}/{len(ids)}] {pid:>13}  ok  {elapsed:5.1f}s  "
                      f"{resp.usage.output_tokens} out-tok")
            except (TypeError, json.JSONDecodeError) as e:
                record.update({"ok": False, "error": f"unparseable: {e}",
                               "extraction": None})
                failed += 1
                print(f"  [{n}/{len(ids)}] {pid:>13}  UNPARSEABLE")
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)

    print(f"\n  ok {done}   failed {failed}   skipped (already present) {skipped}"
          + (f"   retried {retried}" if retried else ""))
    if skipped_failures:
        print(f"\n  *** {len(skipped_failures)} SKIPPED RECORD(S) HOLD A FAILURE, "
              f"NOT A RESULT ***")
        for pid, why in skipped_failures.items():
            print(f"        {pid:>13}  {why}")
        n_ref = sum(1 for w in skipped_failures.values() if w == "refusal")
        n_err = len(skipped_failures) - n_ref
        if n_err:
            print(f"      re-attempt the {n_err} non-refusal failure(s) with "
                  f"--retry-failed")
        if n_ref:
            print(f"      the {n_ref} refusal(s) are MEASURED RESULTS under "
                  f"§1/§5.4 and are not re-attempted by default;")
            print(f"      --retry-failed --retry-refusals overrides this — "
                  f"re-rolling a recorded refusal changes what was measured.")
    print(f"  raw prompts + responses in {RAW_DIR}/")


if __name__ == "__main__":
    main()
