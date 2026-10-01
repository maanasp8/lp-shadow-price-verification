#!/usr/bin/env python
"""Structural lint for a directory of .tex files (default: paper/).

Not a substitute for compiling. No LaTeX engine is installed on this machine, so
this checks what can be checked without one: comment-aware brace balance,
\\begin/\\end environment matching, and tabular column arity against the column
spec. Exits non-zero on any problem.

    venv/bin/python lint_tex.py [DIR]
"""
import glob, os, re, sys


def strip_comments(line: str) -> str:
    out, esc = [], False
    for ch in line:
        if ch == "\\" and not esc:
            esc = True; out.append(ch); continue
        if ch == "%" and not esc:
            break
        esc = False; out.append(ch)
    return "".join(out)


def main(root: str = "paper") -> int:
    bad = 0
    files = sorted(glob.glob(os.path.join(root, "*.tex")))
    if not files:
        print(f"no .tex files under {root}/", file=sys.stderr)
        return 1
    print(f"linting {len(files)} file(s) under {root}/\n")
    for f in files:
        text = "\n".join(strip_comments(l) for l in open(f).read().splitlines())

        depth, esc = 0, False
        for ch in text:
            if ch == "\\" and not esc:
                esc = True; continue
            if not esc:
                depth += (ch == "{") - (ch == "}")
            esc = False

        stack, envok = [], True
        for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
            if kind == "begin":
                stack.append(name)
            elif not stack or stack.pop() != name:
                envok = False
        envok = envok and not stack

        notes = []
        if depth: notes.append(f"BRACE IMBALANCE {depth:+d}"); bad += 1
        if not envok: notes.append("ENV MISMATCH"); bad += 1
        print(f"{f:36} braces {'OK' if not depth else depth:>4}   "
              f"envs {'OK' if envok else 'BAD'}   {' '.join(notes)}")

    tables = os.path.join(root, "tables.tex")
    if not os.path.isfile(tables):
        print(f"\n(no tables.tex under {root}/, skipping arity check)")
        print("LINT RESULT:", "PASS" if not bad else f"{bad} PROBLEM(S)")
        return 1 if bad else 0
    t = open(tables).read()
    spec = re.search(r"\\begin\{tabular\}\{@\{\}(.*?)@\{\}\}", t).group(1)
    ncol = len(re.sub(r"[^lcr]", "", spec))
    prob = 0
    rows = [l for l in t.splitlines()
            if l.strip().endswith(r"\\") and "\\begin" not in l]
    for line in rows:
        core = strip_comments(line).strip()
        if not core.endswith("\\\\"):
            continue
        core = core[:-2]
        eff = (len(re.split(r"(?<!\\)&", core))
               + sum(int(m) - 1 for m in re.findall(r"\\multicolumn\{(\d+)\}", core)))
        if eff != ncol:
            prob += 1
            print(f"   ARITY {eff} != {ncol}: {core[:80]}")
    print(f"\n{tables} spec '{spec}' -> {ncol} columns; "
          f"{len(rows)} data rows; {prob} arity problems")
    bad += bool(prob)
    print("LINT RESULT:", "PASS" if not bad else f"{bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper"))
