# Upstream sources

Two third-party trees are used by this project. They are handled differently,
and the difference matters if you are trying to reproduce a number.

- **`nl4opt-competition/` is NOT in this repository.** It is 15.5 MB of corpus
  and was removed on 2026-08-29 to keep the clone small. Restore it with
  `python fetch_corpus.py`, which clones upstream at the pinned commit below and
  verifies it carries 1101 instances. The fetched tree was checked
  byte-identical to the one previously committed here.
- **`nl4opt-subtask2-baseline/` is NOT in this repository either**, as of
  2026-08-29. Nothing in this project reads it: only a provenance comment in
  `nl4opt_parser.py` names it, recording where the canonicalization convention
  came from. Clone it from upstream at the commit below if you want to check
  that convention.

| directory | upstream origin | commit vendored from |
|---|---|---|
| `nl4opt-competition/` | https://github.com/nl4opt/nl4opt-competition.git | `49f1e0d66b7fdcd33305a7f281c2a7c13f5620ea` |
| `nl4opt-subtask2-baseline/` | https://github.com/nl4opt/nl4opt-subtask2-baseline.git | `18a54bcb2c34be7d034d994c9e065e59cdf1ddfe` |

The SHA for `nl4opt-competition/` **is** the restore mechanism: `fetch_corpus.py`
checks out exactly that commit. The SHA for `nl4opt-subtask2-baseline/` is
provenance only, since those files are in this repository's own history.

## History

Both directories were originally clones carrying their own `.git`, which made
them invisible to this repository: `git add` on a path inside a nested
repository is a silent no-op, and adding the directory itself records an empty
gitlink rather than the files. The initial commit therefore excluded them in
`.gitignore` and pinned them by SHA here.

That approach was **superseded on 2026-08-15**. The inner `.git` directories were
deleted, detaching both trees from upstream history, and the raw files were
committed directly so the repository is self-contained. The corresponding
`.gitignore` entries were removed at the same time.

Both clones were verified pristine before detaching — no local modifications, no
submodules, no symlinks, no worktrees, and no borrowed object stores — so
detaching lost no work beyond the upstream commit history itself.

## What depends on these files

`nl4opt-competition/generation_data/{train,dev,test}.jsonl` is the corpus every
result in this project is computed from: 1101 instances, read by
`nl4opt_parser.py`, `extraction_ground_truth.py`, and the corpus-wide scans.
`nl4opt-competition/ner_data/` carries the NER annotations.

`nl4opt-subtask2-baseline/` is the reference implementation consulted when
building the parser and the CPT/OPT baseline.

A derived representation of the corpus is committed independently in
`results.csv`, `constraints.csv`, `variables.csv`, `objectives.csv`, and
`extraction_pilot/ground_truth.json`, so the corpus is now recorded twice: raw
and parsed.

## Licensing

Both upstream projects ship their own licence, vendored unchanged alongside the
files: `nl4opt-competition/LICENSE` and `nl4opt-subtask2-baseline/LICENSE.txt`.
Their terms continue to govern the vendored contents. Nothing in this repository
relicenses them, and any redistribution of this repository carries those terms
with it.


## 2026-08-29: the corpus left the repository

`nl4opt-competition/` was purged from the working tree *and from git history*,
along with `backups/` (a duplicate snapshot git already versioned) and two
third-party PDFs. This reversed the 2026-08-15 decision recorded above, in
exchange for a much smaller clone.

Consequences to know about:

- **Old commits no longer build the corpus-dependent scripts.** Checking out any
  historical commit gives you a tree without `nl4opt-competition/`. Run
  `fetch_corpus.py` there too.
- **Every commit hash before 2026-08-29 changed.** History was rewritten.
- Appendix F.2 of the paper states the arrangement and no longer claims we
  redistribute the corpus.
