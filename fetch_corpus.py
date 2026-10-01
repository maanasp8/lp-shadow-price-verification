#!/usr/bin/env python
"""Restore the NL4OPT corpus this project computes on.

The corpus is third-party and is no longer committed here. This script clones it
from upstream at the exact commit every result was computed against and leaves
it at the path the analysis scripts expect.

    python fetch_corpus.py

Nothing else in the project fetches anything. Run this once after cloning.
"""
import os, shutil, subprocess, sys

CORPUS = {
    "path": "nl4opt-competition",
    "url": "https://github.com/nl4opt/nl4opt-competition.git",
    "commit": "49f1e0d66b7fdcd33305a7f281c2a7c13f5620ea",
    # every result in this project reads these three files
    "verify": ["generation_data/train.jsonl",
               "generation_data/dev.jsonl",
               "generation_data/test.jsonl"],
    "instances": 1101,
}


def run(*args, **kw):
    return subprocess.run(args, check=True, **kw)


def main() -> int:
    dest, commit = CORPUS["path"], CORPUS["commit"]

    if os.path.isdir(dest):
        missing = [f for f in CORPUS["verify"]
                   if not os.path.isfile(os.path.join(dest, f))]
        if not missing:
            print(f"{dest}/ already present, nothing to do")
            return 0
        print(f"{dest}/ exists but is missing {missing}", file=sys.stderr)
        print("Remove it and rerun to get a clean checkout.", file=sys.stderr)
        return 1

    print(f"cloning {CORPUS['url']}")
    run("git", "clone", "--quiet", "--no-checkout", CORPUS["url"], dest)
    run("git", "-C", dest, "checkout", "--quiet", commit)

    # The clone's own history is not ours to carry. Detaching it keeps this
    # repository's status output clean and matches how the tree was used before.
    shutil.rmtree(os.path.join(dest, ".git"))

    missing = [f for f in CORPUS["verify"]
               if not os.path.isfile(os.path.join(dest, f))]
    if missing:
        print(f"checkout at {commit[:8]} is missing {missing}", file=sys.stderr)
        return 1

    n = sum(sum(1 for line in open(os.path.join(dest, f), encoding="utf-8") if line.strip())
            for f in CORPUS["verify"])
    print(f"{dest}/ restored at {commit[:8]}, {n} instances")
    if n != CORPUS["instances"]:
        print(f"WARNING: expected {CORPUS['instances']} instances, found {n}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
