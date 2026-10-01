# Basis Artifacts in Solution-Level Verification of LLM-Generated Linear Programs

Code and artefacts for the MATH-AI 2026 paper "Basis Artifacts in Solution-Level
Verification of LLM-Generated Linear Programs" by Maanas Punuru (Wake Tech
Community College).

The paper studies how verifiers for LLM-generated linear programs are scored. A
linear program has one optimal value but often many optimal solutions, so a solver
returns a choice among them. Comparing what a solver returns for a candidate and a
reference records that choice wherever the two disagree. This repository holds the
checks, the baselines, the certificate that decides whether a report follows from
the model or from the returned basis, and every artefact behind the numbers in the
paper.

## Layout

- `src/` the pipeline. Checks, mutation harness, baselines, certificate, table
  generators, and the experiment scripts.
- `data/` the persisted per-instance artefacts. Every table and every rate in the
  paper is recomputed from these, and the table generators assert their cells
  against them at build time.
- `fetch_corpus.py` retrieves the NL4OPT corpus from upstream at a pinned commit.
  No copy of the corpus is stored here.

## Map from paper to code

- Checks A, B and C: `src/check_a.py`, `src/check_b.py`, and the primal-feasibility
  audit `mutations.check_d` (labeled Check D in the artifacts, Check C in the
  paper). Shared helpers in `src/checks_common.py`.
- Mutation harness: `src/mutation_harness.py`, `src/mutations.py`.
- Baselines: `src/diff_baseline.py` (coefficient comparison), the reference solve
  `src/solve_reference_baseline.py`, and the feasibility-plus-objective baseline
  `src/feas_obj_baseline.py`.
- Certificate: `src/reframe_measure.py` runs the degeneracy certificate.
  `src/witness_report.py` audits its per-report outcomes into the 55 certified,
  31 witnessed, and 0 unresolved partition.
- Scale experiments: `src/basis_artifact_scale.py`, `src/scale_sweep.py`.
- Table generators: `src/make_results_tables.py`, `src/make_tables_tex.py`,
  `src/make_scale_table.py`, `src/make_swap_table.py`.

## Reproducing the numbers

The two scripts added for the camera-ready read the persisted artefacts and assert
their published counts, so they run without the corpus:

```
python3 src/feas_obj_baseline.py    # 16,897 / 18,525 (91.21%), misses all 55
python3 src/witness_report.py       # 55 certified, 31 witnessed, 0 unresolved
```

To rerun the pipeline from the corpus, fetch it first:

```
python3 fetch_corpus.py
```

Then run the check, baseline, and certificate scripts. Each writes its artefacts
into `data/`. The table generators recompute the paper's tables from `data/`.

## Dependencies

Python 3, Gurobi (with a license) for the linear-program solves, and NumPy. See
`VENDORED.md` for third-party components and their provenance.

## License

MIT. See `LICENSE`.
