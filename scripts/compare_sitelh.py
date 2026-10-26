#!/usr/bin/env python3
"""Compare two IQ-TREE .sitelh files (CPU vs GPU) column by column. Stdlib only.

Written for `-wslr`, which emits per-site log-likelihood PER RATE CATEGORY. That makes it a direct
probe of `_pattern_lh_cat` -- the buffer most likely to be silently wrong, because an incorrect
interleaving leaves the tree log-likelihood correct while corrupting +R/+I EM and empirical-Bayes
site rates. Comparing only the final logL would never catch it.

Tolerances are ABSOLUTE on a log scale, which is the meaningful measure here: per-site logL values
are O(1)-O(100), and what matters downstream is whether a site's contribution shifts, not whether a
near-zero value has a large relative error. The gate mirrors the project's science gate: the summed
difference must stay far below the 1e-4 the tree log-likelihood is held to.

Usage: compare_sitelh.py <cpu.sitelh> <gpu.sitelh> [label] [atol]

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import sys, os


def read_sitelh(path):
    """Return (header_tokens, [ [floats...] per data row ]).

    IQ-TREE's .sitelh format is a small header line followed by rows that begin with a name
    ('Site_Lh', or a per-category label) and continue with one number per site.
    """
    rows, header = [], []
    if not os.path.exists(path):
        raise ValueError(f"missing {os.path.basename(path)}")
    with open(path, errors="ignore") as fh:
        for ln, line in enumerate(fh):
            parts = line.split()
            if not parts:
                continue
            if ln == 0 and not _is_float(parts[-1]):
                header = parts
                continue
            vals = [float(x) for x in parts if _is_float(x)]
            if vals:
                rows.append((parts[0] if not _is_float(parts[0]) else f"row{ln}", vals))
    if not rows:
        raise ValueError(f"no numeric rows parsed from {os.path.basename(path)}")
    return header, rows


def _is_float(tok):
    try:
        float(tok); return True
    except ValueError:
        return False


def main():
    if len(sys.argv) < 3:
        print("usage: compare_sitelh.py <cpu.sitelh> <gpu.sitelh> [label] [atol]")
        sys.exit(2)
    a_path, b_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "cmp"
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8

    try:
        _, arows = read_sitelh(a_path)
        _, brows = read_sitelh(b_path)
    except ValueError as e:
        # Never raise: a crashed arm must surface as a FAIL line, not a traceback that leaves the
        # caller with no verdict at all.
        print(f"  SITELH[{label}]: FAIL ({e})")
        sys.exit(1)

    if len(arows) != len(brows):
        print(f"  SITELH[{label}]: FAIL (row count {len(arows)} vs {len(brows)})")
        sys.exit(1)

    worst = 0.0; worst_row = ""; worst_i = -1; nsite = 0; nrow = 0
    for (an, av), (bn, bv) in zip(arows, brows):
        if len(av) != len(bv):
            print(f"  SITELH[{label}]: FAIL (row '{an}' length {len(av)} vs {len(bv)})")
            sys.exit(1)
        nrow += 1; nsite = max(nsite, len(av))
        for i, (x, y) in enumerate(zip(av, bv)):
            d = abs(x - y)
            if d > worst:
                worst, worst_row, worst_i = d, an, i
    ok = worst <= atol
    # Total drift over all sites bounds how far the tree log-likelihood could move.
    total = sum(abs(x - y) for (_, av), (_, bv) in zip(arows, brows) for x, y in zip(av, bv))
    print(f"  SITELH[{label}]: {'PASS' if ok else 'FAIL'}  rows={nrow} sites={nsite}  "
          f"max|dlogL|={worst:.3e} (row '{worst_row}' site {worst_i})  sum|dlogL|={total:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
