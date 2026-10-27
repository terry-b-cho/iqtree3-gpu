"""headline.py -- the ONE definition of which benchmark rows the manuscript reports.

WHY THIS EXISTS. `benchmark/runs/RUNS.tsv` records every paired run ever made, including 44 rows
produced by a SUPERSEDED build of the GPU binary (`07c70cae...`). Those rows are kept deliberately:
`binary_sha256` makes the build provenance of every measurement auditable directly from the
distributed data. They must not, however, contribute to any published figure or statistic, because
pooling two builds mixes a build difference into what is presented as measurement variation.

THE TRAP THIS FILE EXISTS TO PREVENT. The obvious way to exclude them is to swap the namespace
filter from `__l40s` to `__l40s_b2` (the namespace the re-runs were written to). That is WRONG: the
`__l40s` namespace is MIXED. It holds the 44 superseded cells AND four nucleotide 20-thread cells
(n06, n14, n20, n30) that were always on the current build and therefore have no `_b2` twin. A
namespace swap silently drops those four and leaves an 8-point nucleotide ladder instead of 12.

Select on the BUILD, never on the namespace. The headline sweep is then exactly 48 cells -- 12 in
each of the four (alphabet x thread-count) strata.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import csv, os, re

SUPERSEDED_SHA_PREFIX = "07c70cae"       # the build whose rows are excluded from all analysis
HEADLINE_NAMESPACES = ("l40s", "l40s_b2")  # the size-ladder namespaces, before the build filter


def namespace(cell):
    """Benchmark namespace = the trailing token of the cell name (NOT the physical card)."""
    return (cell or "").rsplit("__", 1)[-1]


def is_current_build(row):
    """True if this row was measured with the build the manuscript reports."""
    return not (row.get("binary_sha256") or "").startswith(SUPERSEDED_SHA_PREFIX)


def is_headline(row):
    """True if this row belongs to the current-build size ladder the headline figures draw."""
    return is_current_build(row) and namespace(row.get("cell")) in HEADLINE_NAMESPACES


# Repeat measurements of the SAME dataset on the same card class. `l40s_b2` must be here: it is the
# namespace the re-runs were written to, and the previous regex `__l40s(q|_r[0-9]+)?$` did not match
# it, so every re-run was silently excluded from replicate statistics.
REPLICATE_NAMESPACES = ("l40s", "l40s_b2", "l40sq")
_REPLICATE_RE = re.compile(r"^l40s_r[0-9]+$")


def is_replicate(row):
    """True if this row is a current-build repeat measurement usable in replicate statistics."""
    if not is_current_build(row):
        return False
    ns = namespace(row.get("cell"))
    return ns in REPLICATE_NAMESPACES or bool(_REPLICATE_RE.match(ns))


def load_runs(path):
    """Every row, unfiltered -- for equivalence and provenance reporting."""
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def current_rows(path):
    """Only rows measured with the reported build."""
    return [r for r in load_runs(path) if is_current_build(r)]


def headline_cells(path):
    """Set of cell NAMES in the current-build size ladder.

    Needed by generators that key on a directory name (e.g. PHASES.tsv under
    benchmark/runs/paired/<cell>/) and so cannot see `binary_sha256` themselves.
    """
    return {r["cell"] for r in load_runs(path) if is_headline(r)}


def runs_tsv(mod):
    return os.path.join(mod, "benchmark", "runs", "RUNS.tsv")
