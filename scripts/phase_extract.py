#!/usr/bin/env python3
"""phase_extract.py - per-STEP wall-clock and CPU time, parsed from each run's own IQ-TREE log.

WHY THIS EXISTS. `PHASES.tsv` records only three phases (modelfinder, tree_search, total), and the
`cpu_bootstrap_s` / `gpu_bootstrap_s` columns of RUNS.tsv were never populated -- they are `NA` in
every row. But IQ-TREE's own log reports considerably more than the harness captured, including
**CPU time as well as wall-clock**, and every one of the recorded cells has both arms' logs on disk.
So the finer breakdown readers want -- which step the speedup comes from, and the raw compute hours
each step costs -- is recoverable with no re-running whatsoever.

WHAT CPU TIME BUYS YOU. Wall-clock says how long you waited; CPU time says how much machine you
consumed, which is what a shared cluster actually bills. Their ratio is parallel efficiency, and the
contrast between the arms is stark: on a 17-taxon protein run the CPU arm spent 863,100 s of CPU over
43,501 s of wall (19.8x, near-perfect 20-thread scaling) while the GPU arm spent 12,199 s over
9,492 s (1.29x). Same tree, ~71x less CPU compute.

THREE HONESTY CONSTRAINTS, carried into the output rather than glossed:

  1. The steps are HIERARCHICAL and do not sum to the total. `modelfinder` and `tree_search` are
     the disjoint top-level phases (`level=top`); the rest are setup work nested inside them
     (`level=nested`) and must never be stacked alongside them. An explicit `unattributed` row
     carries what IQ-TREE left untimed; it is never folded into a named step.
  2. UFBoot is NOT separately timed. Bootstrap replicates run inside the tree-search phase, so
     `tree_search` means "tree search including bootstrap and support", not a pure search phase.
  3. DNA runs have no `modelfinder` or `fast_ml_search` step at all -- they use a fixed model. An
     absent step is absent, not zero.

Output: results/benchmarks/PHASE_STEPS.tsv  (cell, arm, data_type, threads, step, level, wall_s, cpu_s)

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, re, csv, glob

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = f"{REPO}/results/benchmarks"
OUT = f"{R}/PHASE_STEPS.tsv"

# step -> (regex, wall group, cpu group).  None = that step reports no CPU time.
PATTERNS = [
    ("parsimony_tree",  re.compile(r"Create initial parsimony tree.*?([\d.]+) seconds"), 1, None),
    ("fast_ml_search",  re.compile(r"Time for fast ML tree search:\s*([\d.]+) seconds"), 1, None),
    ("modelfinder",     re.compile(r"Wall-clock time for ModelFinder:\s*([\d.]+) seconds"), 1, None),
    ("modelfinder_cpu", re.compile(r"CPU time for ModelFinder:\s*([\d.]+) seconds"), None, 1),
    ("ml_distances",    re.compile(r"Computing ML distances took ([\d.]+) sec \(of wall-clock time\) ([\d.]+) sec \(of CPU time\)"), 1, 2),
    ("rapidnj_tree",    re.compile(r"Computing RapidNJ tree took ([\d.]+) sec \(of wall-clock time\) ([\d.]+) sec \(of CPU time\)"), 1, 2),
    ("initial_tree_ll", re.compile(r"Computing log-likelihood of \d+ initial trees \.\.\. ([\d.]+) seconds"), 1, None),
    ("tree_search",     re.compile(r"Wall-clock time used for tree search:\s*([\d.]+) sec"), 1, None),
    ("tree_search_cpu", re.compile(r"CPU time used for tree search:\s*([\d.]+) sec"), None, 1),
    ("total",           re.compile(r"Total wall-clock time used:\s*([\d.]+) sec"), 1, None),
    ("total_cpu",       re.compile(r"Total CPU time used:\s*([\d.]+) sec"), None, 1),
]
# THE STEPS ARE HIERARCHICAL, NOT A FLAT LIST -- established by measurement, not assumption.
# Across 218 arms carrying the full step set, modelfinder + tree_search NEVER exceeds the reported
# total (0/218), so those two are the disjoint top-level phases. Adding the small steps overflows
# the total in 62/218 arms, which means they are NESTED INSIDE those phases (the parsimony tree,
# fast ML search, ML distances, RapidNJ tree and initial-tree likelihoods are all setup work that
# IQ-TREE performs and times within a larger phase).
#
# Only TOP steps may be summed or stacked. Stacking the nested ones alongside them double-counts
# and, on 62 of our arms, would have drawn bars taller than the run itself.
TOP = ["modelfinder", "tree_search"]
NESTED = ["parsimony_tree", "fast_ml_search", "ml_distances", "rapidnj_tree", "initial_tree_ll"]
ACCOUNTED = TOP + NESTED
LEVEL = {**{s: "top" for s in TOP}, **{s: "nested" for s in NESTED}}


def parse_log(path):
    """-> {step: {'wall': x, 'cpu': y}}. Absent steps stay absent; they are not zero."""
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return {}
    out = {}
    for name, rx, gw, gc in PATTERNS:
        m = rx.search(txt)
        if not m:
            continue
        base = name.replace("_cpu", "")
        rec = out.setdefault(base, {})
        if gw:
            rec["wall"] = float(m.group(gw))
        if gc:
            rec["cpu"] = float(m.group(gc))
    return out


def main():
    meta = {}
    with open(f"{R}/RUNS.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            meta[r["cell"]] = (r.get("data_type", "NA"), r.get("threads", "NA"))

    rows, ncell, neg = [], 0, 0
    for d in sorted(glob.glob(f"{R}/paired/*/")):
        cell = os.path.basename(d.rstrip("/"))
        dt, thr = meta.get(cell, ("NA", "NA"))
        got = False
        for arm in ("cpu", "gpu"):
            steps = parse_log(os.path.join(d, f"{arm}.log"))
            if not steps:
                continue
            got = True
            for name in ACCOUNTED:
                if name in steps:
                    rows.append((cell, arm, dt, thr, name, LEVEL[name],
                                 steps[name].get("wall", ""), steps[name].get("cpu", "")))
            tot = steps.get("total", {})
            if "wall" in tot:
                # residual is total minus the TOP-level phases only -- nested steps are already
                # inside them and subtracting them again would invent time that does not exist.
                acc = sum(steps[n].get("wall", 0.0) for n in TOP if n in steps)
                resid = tot["wall"] - acc
                if resid < -1e-6:
                    neg += 1
                # Reported even when tiny: a silent residual is how a fake partition is born.
                rows.append((cell, arm, dt, thr, "unattributed", "top", f"{resid:.3f}", ""))
                rows.append((cell, arm, dt, thr, "total", "total",
                             tot.get("wall", ""), tot.get("cpu", "")))
        ncell += 1 if got else 0

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["cell", "arm", "data_type", "threads", "step", "level", "wall_s", "cpu_s"])
        w.writerows(rows)
    print(f"  phase steps: {len(rows)} rows from {ncell} cells x 2 arms -> "
          f"results/benchmarks/PHASE_STEPS.tsv")
    if neg:
        print(f"    WARNING: {neg} arm(s) have a NEGATIVE unattributed residual - steps overlap, "
              f"so they cannot be read as disjoint")


if __name__ == "__main__":
    main()
