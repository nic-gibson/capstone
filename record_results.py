#!/usr/bin/env python3
"""Sync the weekly result files from outputs.txt into the per-function CSVs.

    python record_results.py                  # rewrite all 8 outputs.csv
    python record_results.py --bootstrap       # also build inputs.csv from inputs.txt
    python record_results.py --check          # validate alignment, write nothing

outputs.txt holds one Python list of 8 floats per line, oldest first -- line k
(1-indexed) is week k+1, one value per function. This script rewrites each
weekly_data/function_N/outputs.csv from ALL of those lines, so running it twice
is a no-op rather than a duplication. Same reasoning as `save_proposal` in
bayes_tools: rewrite, never append, and alignment cannot drift.

There is no week column anywhere. Alignment between inputs.csv and outputs.csv
is positional, and the invariant is len(inputs) - len(outputs) in (0, 1).
"""
import argparse
import ast
import csv
import os
import re
import sys

ROOT = "weekly_data"
N_FUNCTIONS = 8


def read_outputs_txt(path="outputs.txt"):
    """-> list of rows, each a list of N_FUNCTIONS floats, oldest first."""
    rows = []
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        vals = ast.literal_eval(line)
        if len(vals) != N_FUNCTIONS:
            raise ValueError(f"{path}:{lineno}: expected {N_FUNCTIONS} values, got {len(vals)}")
        rows.append([float(v) for v in vals])
    return rows


def read_inputs_txt(path="inputs.txt"):
    """-> list of weeks, each a list of N_FUNCTIONS coordinate lists, oldest first.

    inputs.txt holds one logical record per week -- a list of 8 numpy array
    reprs -- line-wrapped across raw lines, so records are matched by regex
    rather than by line.
    """
    raw = open(path).read()
    records = re.findall(r"\[array\(.*?\)\s*\]", raw, re.S)
    weeks = []
    for rec in records:
        arrays = re.findall(r"array\(\s*\[(.*?)\]\s*\)", rec, re.S)
        if len(arrays) != N_FUNCTIONS:
            raise ValueError(f"{path}: a record holds {len(arrays)} arrays, expected {N_FUNCTIONS}")
        weeks.append([[float(v) for v in re.split(r"[,\s]+", a.strip()) if v] for a in arrays])
    return weeks


def write_csv(path, header, rows, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt % v for v in r])


def count_rows(path):
    if not os.path.exists(path):
        return 0
    n = 0
    for rec in csv.reader(open(path, newline="")):
        if not rec:
            continue
        try:
            float(rec[0])
        except ValueError:
            continue                    # header
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true",
                    help="also rebuild inputs.csv from inputs.txt (one-time migration)")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    out_rows = read_outputs_txt()
    print(f"outputs.txt: {len(out_rows)} week(s) recorded (weeks 2-{len(out_rows) + 1})")

    if args.bootstrap and not args.check:
        in_weeks = read_inputs_txt()
        print(f"inputs.txt:  {len(in_weeks)} week(s) recorded")
        if len(in_weeks) != len(out_rows):
            print(f"  NOTE inputs.txt has {len(in_weeks)} weeks and outputs.txt "
                  f"{len(out_rows)}; a trailing input week with no result is expected "
                  "mid-cycle.")
        for n in range(1, N_FUNCTIONS + 1):
            coords = [wk[n - 1] for wk in in_weeks]
            D = len(coords[0])
            write_csv(os.path.join(ROOT, f"function_{n}", "inputs.csv"),
                      [f"x{j}" for j in range(D)], coords, "%.6f")
        print(f"  rebuilt {N_FUNCTIONS} inputs.csv from inputs.txt")

    if not args.check:
        for n in range(1, N_FUNCTIONS + 1):
            ys = [[wk[n - 1]] for wk in out_rows]
            write_csv(os.path.join(ROOT, f"function_{n}", "outputs.csv"), ["y"], ys, "%.17g")
        print(f"  rewrote {N_FUNCTIONS} outputs.csv from outputs.txt")

    print("\nalignment check (inputs - outputs must be 0 or 1):")
    bad = 0
    for n in range(1, N_FUNCTIONS + 1):
        ip = os.path.join(ROOT, f"function_{n}", "inputs.csv")
        op = os.path.join(ROOT, f"function_{n}", "outputs.csv")
        ni, no = count_rows(ip), count_rows(op)
        gap = ni - no
        ok = gap in (0, 1)
        bad += not ok
        state = "pending proposal" if gap == 1 else ("all resulted" if gap == 0 else "MISALIGNED")
        print(f"  function {n}: inputs {ni:>2}  outputs {no:>2}  gap {gap:+d}  {state}"
              + ("" if ok else "   <-- FIX THIS"))
    if bad:
        print(f"\n{bad} function(s) misaligned.")
        return 1
    print("\nall aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
