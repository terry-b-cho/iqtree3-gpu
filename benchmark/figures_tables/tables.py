#!/usr/bin/env python3
"""tables.py - every table, number, workbook, map and cross-check of the benchmark, in one file.

Sections, each a former script: key_numbers (runs/KEY_NUMBERS.tsv and .md, every number quoted in
the text), source_data (figures/Source_Data.xlsx from the per-panel CSVs), supplementary_data
(runs/Supplementary_Data_benchmark.xlsx, the Supplementary Data sheets), style_workbooks (the
look of both workbooks, values untouched), benchmark_map (the figure and table map in
benchmark/README.md) and crosscheck (cross-file checks on the public tree). The shared data
helpers are in lib/ (figdata, headline).

Run one section or all, from the repository root:

    python3 benchmark/figures_tables/tables.py                       # every section, in order
    python3 benchmark/figures_tables/tables.py key_numbers           # one section
    python3 benchmark/figures_tables/tables.py source_data out.xlsx  # a section with its arguments
    python3 benchmark/figures_tables/tables.py benchmark_map --check

Nothing here runs IQ-TREE: the inputs are the measured record in benchmark/runs/ and the panels
written by figures.py. The benchmark harness lives in benchmark/scripts/.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
"""
import os, sys, re, csv, glob, zlib, math, random, subprocess, collections, statistics as st
import importlib, importlib.util as ilu
from datetime import datetime, timezone

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PANELS = os.path.join(REPO, "benchmark", "figures", "panels")
MANIFEST = os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv")
PAIRED = os.path.join(REPO, "benchmark", "runs", "paired")


def _load(n, path=None):
    """Load one of the shared library modules (lib/<n>.py) by path, so no PYTHONPATH is needed."""
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m


FS = _load("figstyle"); FD = _load("figdata"); H = _load("headline"); F = FD
try:
    import pandas as pd
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError:                       # the two workbooks need pandas + openpyxl; nothing else does
    pd = openpyxl = None


# ================================================================================================
# ==== section: key_numbers ====
# ================================================================================================
# key_numbers.py - every number the manuscript quotes, derived once, from the data.
#
# WHY THIS EXISTS. Headline numbers used to be hand-typed into the prose and went stale silently.
# This script makes them a build product: the manuscript quotes KEY_NUMBERS.tsv and a refresh
# regenerates it. `tables.py crosscheck` then FAILS if the abstract quotes a number this file
# does not contain at the same precision, so prose and data cannot drift apart again.
#
# TWO HISTORICAL BUGS THIS REWRITE FIXES.
# 1. The script imported `complexity_scaling.py` for its complexity definition. That script was
#    retired into `scripts/retired/`, so this one raised FileNotFoundError and stopped running: the
#    committed KEY_NUMBERS.tsv was stale output from before the retirement. The definition now comes
#    from `lib/figdata.py`, which is also what every panel uses, so they cannot diverge.
# 2. The speedup statistics excluded the `__mdl_*` correctness cells but NOT the gate-failure cell,
#    while the panels exclude both via `figdata.for_fits()`. KEY_NUMBERS therefore reported
#    protein_20T n=68 median 3.75 where the figures showed n=67 median 3.73. Both now use for_fits().
#
# Outputs: benchmark/runs/KEY_NUMBERS.md  (quotable block)
#          benchmark/runs/KEY_NUMBERS.tsv (key<TAB>value, for scripted assertions)
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

RUNS = f"{REPO}/benchmark/runs/RUNS.tsv"
STEPS = f"{REPO}/benchmark/runs/PHASE_STEPS.tsv"
OUTMD = f"{REPO}/benchmark/runs/KEY_NUMBERS.md"
OUTTSV = f"{REPO}/benchmark/runs/KEY_NUMBERS.tsv"

num = F.num
# The engine prints three lines per run (phylotreegpu.cpp); all three are read, because the
# per-entry figure depends on the ALLOCATED rate-category capacity, not the selected model's count.
ARENA_CAP_RE = re.compile(r"(\d+) rate categor(?:y|ies) \(arena sized for up to (\d+)\)")
ARENA_SLOT_RE = re.compile(r"partial slot = ([\d.]+) MB \+ ([\d.]+) MB counters; pooling (\d+) slots "
                           r"\(IQ-TREE arena (\d+) \+ 2 NNI\) -> ([\d.]+) GB VRAM")
GUARD_RE = re.compile(r"rate-category capacity reduced from (\d+) to (\d+): (\d+) slots at "
                      r"ncat=(\d+) would need ([\d.]+) GB of the ([\d.]+) GB")
ARENA_FRACTION = 0.70          # GPU_ARENA_VRAM_FRACTION, fixed in phylotreegpu.cpp:314


def q(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    lo = int(k)
    return v[lo] if lo == k else v[lo] + (k - lo) * (v[lo + 1] - v[lo])


def phase_totals():
    """(cell, arm) -> (wall_s, cpu_s) for the whole run, from each arm's own log."""
    out = {}
    if not os.path.exists(STEPS):
        return out
    for r in csv.DictReader(open(STEPS), delimiter="\t"):
        if r.get("step") == "total":
            out[(r["cell"], r["arm"])] = (num(r.get("wall_s")), num(r.get("cpu_s")))
    return out


def replicate_determinism(rows):
    """Same dataset, measured more than once: does an arm reproduce its own support values?

    This is the direct test of whether the 17 cross-arm support differences need a GPU explanation.
    Each arm is compared against ITSELF across replicate runs, so a difference can only come from
    that arm's own non-determinism.

    THE GROUPING KEY MATTERS AND AN EARLIER VERSION OF IT WAS WRONG. Grouping only on the dataset
    and thread count (the cell name minus its namespace) pools the `__mdl_*` correctness cells,
    which deliberately pin DIFFERENT models on the same alignment -- `dna_n06__T20` alone spans
    GTR+F+G4, GTR+F+I+G4, GTR+F+R4, HKY+F+G4 and JC. Comparing across those is a model difference,
    not a reduction-order difference, and it inflated the pair count from 341 to 391. The key below
    therefore requires the dataset, thread count, seed AND selected model to match.

    Groups whose SELECTED model varies are counted separately: for `n26_g__Ruminococcus__T20`
    ModelFinder chose VT+F+I+R7 on one run and VT+F+R8 on another. That is a real observation about
    model selection, but it is not a test of reduction determinism.
    """
    groups, model_varies = {}, set()
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["cell"].rsplit("__", 1)[0], []).append(r)
    for k, v in by_dataset.items():
        if len(v) > 1 and len({x.get("model") for x in v}) > 1:
            model_varies.add(k)
    for r in rows:
        key = (r.get("genus"), r.get("taxa"), r.get("patterns"), r.get("data_type"),
               r.get("threads"), r.get("seed"), r.get("model"))
        groups.setdefault(key, []).append(r["cell"])
    cache, out = {}, {}
    for arm in ("cpu", "gpu"):
        pairs = diff = 0
        worst = 0.0
        for cells in groups.values():
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    pa = f"{PAIRED}/{cells[i]}/{arm}.contree"
                    pb = f"{PAIRED}/{cells[j]}/{arm}.contree"
                    if not (os.path.exists(pa) and os.path.exists(pb)):
                        continue
                    for p in (pa, pb):
                        if p not in cache:
                            try:
                                cache[p] = F.parse_newick(open(p).read())
                            except OSError:
                                cache[p] = (None, None)
                    (ta, sa), (tb, sb) = cache[pa], cache[pb]
                    if not ta or not tb or ta != tb:
                        continue
                    shared = set(sa) & set(sb)
                    d = [abs(sa[k] - sb[k]) for k in shared
                         if sa[k] is not None and sb[k] is not None]
                    m = max(d) if d else 0.0
                    pairs += 1
                    if m > 0:
                        diff += 1
                    worst = max(worst, m)
        out[arm] = (pairs, diff, worst)
    out["_model_varies"] = sorted(model_varies)
    return out


def arena_report(r):
    """The engine's own arena report for one run: slot MB, counters MB, slots, footprint GB, the
    selected and the allocated (capacity) rate-category counts, and the guard event if any."""
    try:
        txt = open(f"{PAIRED}/{r['cell']}/gpu.stdout", errors="ignore").read()
    except OSError:
        return None
    m = ARENA_SLOT_RE.search(txt); c = ARENA_CAP_RE.search(txt)
    if not (m and c):
        return None
    g = GUARD_RE.search(txt)
    out = dict(slot_mb=float(m.group(1)), counters_mb=float(m.group(2)), slots=int(m.group(3)),
               iqtree_slots=int(m.group(4)), footprint_gb=float(m.group(5)),
               ncat_selected=int(c.group(1)), ncat_cap=int(c.group(2)))
    if g:
        out["guard"] = dict(frm=int(g.group(1)), to=int(g.group(2)), slots=int(g.group(3)),
                            need_gb=float(g.group(5)), have_gb=float(g.group(6)),
                            budget_gb=ARENA_FRACTION * float(g.group(6)))
    return out


def bytes_per_entry(rows):
    """Bytes of device arena per conditional-likelihood entry, from the engine's own report.

    Denominator = patterns x states x ALLOCATED rate-category capacity, per slot (the slot MB is
    already a per-slot figure). Round-8 correction (R13): the previous denominator used the
    selected model's category count, so a +G4 run whose arena was sized for ten categories under
    ModelFinder reported 8 x 10/4 = 20 bytes; the remainder was pre-allocated capacity, not padding.
    Returns (per_entry_bytes, per_selected_entry_bytes) lists, plus the parsed reports."""
    v, v_sel, reps = [], [], {}
    for r in rows:
        a = arena_report(r)
        if not a:
            continue
        reps[r["cell"]] = a
        m, k = num(r.get("patterns")) or 0, num(r.get("num_states")) or 0
        if m and k:
            v.append(a["slot_mb"] * 1048576 / (m * k * a["ncat_cap"]))
            v_sel.append(a["slot_mb"] * 1048576 / (m * k * F.rate_cats(r.get("model"))))
    return v, v_sel, reps


def kexp_table(rows):
    """R^2 of the cost law when the alphabet weight is k^p, for p = 0..3.

    k takes only two values (4 and 20), so this 'exponent' is a two-group contrast confounded with
    everything else that differs between the workflows. Reported so the k^2 weighting is visibly a
    DEFINITION rather than a fitted result -- the board asked for exactly this comparison.
    """
    out = {}
    for p in (0, 1, 2, 3):
        for arm in ("cpu", "gpu"):
            pts = []
            for r in rows:
                t, pat, k = num(r.get("taxa")), num(r.get("patterns")), num(r.get("num_states"))
                w = num(r.get(f"{arm}_wall_s"))
                if t and pat and k and w:
                    pts.append(((2 * t - 3) * pat * F.rate_cats(r.get("model")) * (k ** p), w))
            f = F.loglog_fit(pts, B=200)
            if f:
                out[(p, arm)] = f["r2"]
    return out


def main():
    all_rows = F.load(REPO, current_build_only=False)
    rows = F.load(REPO)                      # current build: the 182 analysed
    fits = F.for_fits(rows)                 # minus gate failure and __mdl_* cells: the 175
    ph = phase_totals()
    kv = []
    add = lambda k, v: kv.append((k, str(v)))

    # ---- provenance ---------------------------------------------------------------
    shas = {}
    for r in all_rows:
        shas.setdefault(r.get("binary_sha256", ""), []).append(r)
    prior = [s for s in shas if s.startswith(H.PRIOR_BUILD_SHA_PREFIX)]
    reported = [s for s in shas if s and s not in prior]
    add("total_rows_recorded", len(all_rows))
    add("total_rows", len(rows))
    add("rows_prior_build", len(all_rows) - len(rows))
    add("rows_analysed_for_fits", len(fits))
    add("build_sha_reported", reported[0] if reported else "NA")
    add("build_sha_prior", prior[0] if prior else "NA")
    add("gate_failure_cell", F.GATE_FAILURE_CELL)
    add("correctness_cells_excluded", sum(1 for r in rows if F.is_correctness_cell(r)))

    # ---- the alphabet/workflow aliasing -------------------------------------------
    ok, msg = F.aliasing_holds(rows)
    add("workflow_aliasing_total", "TRUE" if ok else "FALSE")
    for wf in (F.WORKFLOW_AA_MF, F.WORKFLOW_NT_FIXED):
        tag = "aa_mf" if wf == F.WORKFLOW_AA_MF else "nt_fixed"
        add(f"workflow_{tag}_n", sum(1 for r in rows if r["_workflow"] == wf))
    add("workflow_aa_with_fixed_model_n",
        sum(1 for r in rows if r["data_type"] == "protein" and r["model_search"] == "fixed_model"))
    add("workflow_nt_with_modelfinder_n",
        sum(1 for r in rows if r["data_type"] == "dna"
            and r["model_search"] == "full_ModelFinder_TESTNEW"))

    # ---- equivalence ---------------------------------------------------------------
    eq = F.for_equivalence(rows)
    rel = [num(r.get("logl_reldiff")) for r in eq]
    rel = [x for x in rel if x is not None]
    sup = [(num(r.get("support_max_absdiff")) or 0.0, r.get("cell")) for r in eq]
    sup_max, sup_cell = max(sup)
    sup_excl = max(v for v, c in sup if c != F.GATE_FAILURE_CELL)
    add("engaged_ok", sum(1 for r in eq if r.get("engaged") == "OK"))
    add("gate_pass_true", sum(1 for r in eq if r.get("gate_pass") == "TRUE"))
    add("gate_pass_not_true", sum(1 for r in eq if r.get("gate_pass") != "TRUE"))
    add("rf_nonzero", sum(1 for r in eq if (num(r.get("rf_dist")) or 0) != 0))
    add("rf_zero_of_total", f"{sum(1 for r in eq if (num(r.get('rf_dist')) or 0) == 0)}/{len(eq)}")
    add("logl_bitidentical_n", sum(1 for x in rel if x == 0))
    add("logl_nonzero_n", sum(1 for x in rel if x > 0))
    add("logl_reldiff_max", "%.3e" % max(rel))
    add("logl_reldiff_nonzero_min", "%.3e" % min(x for x in rel if x > 0))
    add("support_max_absdiff", "%.1f" % sup_max)
    add("support_max_absdiff_cell", sup_cell)
    add("support_max_absdiff_excl_gate_failure", "%.1f" % sup_excl)
    add("rows_with_support_diff", sum(1 for v, _ in sup if v > 0))
    gf = [r for r in eq if F.is_gate_failure(r)]
    if gf:
        add("gate_failure_support_diff", "%.1f" % (num(gf[0]["support_max_absdiff"]) or 0))
        add("gate_failure_logl_reldiff", "%.3e" % (num(gf[0]["logl_reldiff"]) or 0))
        add("gate_failure_rf", int(num(gf[0]["rf_dist"]) or 0))
        add("gate_failure_speedup", "%.2f" % (num(gf[0]["speedup"]) or 0))
    add("support_gate_tolerance", "%.1f" % F.SUPPORT_GATE_TOL)
    add("logl_contract", "%.0e" % F.LOGL_CONTRACT)
    # how far the largest observed difference sits below the tolerance (quoted in the text, R12)
    add("logl_margin_fold", "%.1f" % (F.LOGL_CONTRACT / max(rel)))

    # ---- replicate determinism (the board's item 2) --------------------------------
    if os.environ.get("KEYNUM_SKIP_REPLICATES") != "1":
        rep = replicate_determinism(rows)
        for arm in ("cpu", "gpu"):
            pairs, diff, worst = rep[arm]
            add(f"replicate_{arm}_vs_{arm}_pairs", pairs)
            add(f"replicate_{arm}_vs_{arm}_differing", diff)
            add(f"replicate_{arm}_vs_{arm}_worst", "%.1f" % worst)
        add("replicate_groups_with_varying_model", len(rep["_model_varies"]))

    # ---- speedup, by workflow x threads --------------------------------------------
    groups = {}
    for r in fits:
        sp, dt, th = r["_speedup"], r.get("data_type"), r.get("threads")
        if sp and sp > 0 and dt in ("protein", "dna"):
            groups.setdefault((dt, th), []).append(sp)
    for (dt, th), v in sorted(groups.items()):
        v.sort()
        tag = f"{dt}_{th}T"
        slower = sum(1 for x in v if x < 1.0)
        add(f"{tag}_n", len(v))
        add(f"{tag}_min", "%.2f" % v[0])
        add(f"{tag}_max", "%.2f" % v[-1])
        add(f"{tag}_median", "%.2f" % st.median(v))
        add(f"{tag}_iqr_lo", "%.2f" % q(v, .25))
        add(f"{tag}_iqr_hi", "%.2f" % q(v, .75))
        add(f"{tag}_gpu_slower_n", slower)
        add(f"{tag}_gpu_slower_pct", "%.0f" % (100.0 * slower / len(v)))

    # The 20-thread maximum must come from the largest PASSING cell, not from the fastest run
    # anywhere: the old "7.8-fold" was a Tesla V100 cell, and speedup is host-confounded across
    # cards, so quoting it as the headline mixed a card difference into a workload claim.
    for dt in ("protein", "dna"):
        c = [r for r in fits if r["data_type"] == dt and r["threads"] == "20" and r["_complexity"]]
        if not c:
            continue
        big = max(c, key=lambda r: r["_complexity"])
        add(f"{dt}_20T_largest_cell", big["cell"])
        add(f"{dt}_20T_largest_complexity", "%.2e" % big["_complexity"])
        add(f"{dt}_20T_largest_speedup", "%.2f" % big["_speedup"])
        add(f"{dt}_20T_largest_cpu_wall_h", "%.2f" % (num(big["cpu_wall_s"]) / 3600.0))
        add(f"{dt}_20T_largest_gpu_wall_h", "%.2f" % (num(big["gpu_wall_s"]) / 3600.0))
        fastest = max(c, key=lambda r: r["_speedup"])
        add(f"{dt}_20T_fastest_speedup", "%.2f" % fastest["_speedup"])
        add(f"{dt}_20T_fastest_cell", fastest["cell"])
        add(f"{dt}_20T_fastest_card", fastest["card"])

    # ---- crossovers -----------------------------------------------------------------
    _cx = {}
    for dt in ("protein", "dna"):
        pts = [(r["_complexity"], r["_speedup"]) for r in fits
               if r["data_type"] == dt and r["threads"] == "20"]
        c = F.crossover(pts)
        if not c:
            continue
        add(f"crossover_{dt}_t20", "%.2e" % c["crossover"]); _cx[dt] = c["crossover"]
        add(f"crossover_{dt}_t20_ci_lo", "%.2e" % c["ci"][0])
        add(f"crossover_{dt}_t20_ci_hi", "%.2e" % c["ci"][1])
        add(f"crossover_{dt}_t20_n", c["n"])
        add(f"crossover_{dt}_t20_n_below_parity", c["n_below"])
        below = sorted([(x, y) for x, y in pts if y < 1.0])
        atpar = sorted([(x, y) for x, y in pts if y >= 1.0])
        if below:
            add(f"crossover_{dt}_t20_highest_subparity_complexity", "%.2e" % below[-1][0])
            add(f"crossover_{dt}_t20_highest_subparity_speedup", "%.2f" % below[-1][1])
        if atpar:
            add(f"crossover_{dt}_t20_lowest_atparity_complexity", "%.2e" % atpar[0][0])
            add(f"crossover_{dt}_t20_lowest_atparity_speedup", "%.2f" % atpar[0][1])
        # interpolated == bracketed by measured runs on both sides
        add(f"crossover_{dt}_t20_interpolated",
            "TRUE" if (below and atpar and below[-1][0] <= c["crossover"] <= atpar[-1][0]) else "FALSE")

    # ratio of the two 20-thread break-even points, as the Abstract states it ("about 90-fold"):
    # exact, and rounded to one significant figure because both crossovers are quoted at one
    if "protein" in _cx and "dna" in _cx and _cx["protein"]:
        ratio = _cx["dna"] / _cx["protein"]
        add("crossover_ratio_dna_over_protein_t20", "%.1f" % ratio)
        add("crossover_ratio_dna_over_protein_t20_1sf", "%.0f" % float("%.1g" % ratio))
    # ---- cost law, T20 stratum, all cards pooled ---------------------------------------
    # The abscissa is WORK UNITS (likelihood calls x m k^2 r), which is what Figure 3a plots -- NOT
    # `complexity`, which carries the 2n-3 branch term and no call count and is the abscissa
    # of Figure 2a. Fitting against the two gives different exponents (0.81 vs 1.08 for the
    # amino-acid CPU arm), so quoting one as the other would put a number in the text that no
    # figure shows. Both definitions live in figdata so they cannot drift.
    L = [r for r in fits if r["threads"] == "20"]          # all six cards pooled, as Fig. 2B
    for dt in ("protein", "dna"):
        sub = [r for r in L if r["data_type"] == dt]
        for arm in ("cpu", "gpu"):
            pts = [(r["_work_units"], num(r[f"{arm}_wall_s"])) for r in sub]
            # straight log-log fit: ONE average exponent over the measured range
            f = F.loglog_fit(pts)
            if f:
                add(f"costlaw_{dt}_{arm}_exponent", "%.3f" % f["slope"])
                add(f"costlaw_{dt}_{arm}_ci_lo", "%.3f" % f["ci"][0])
                add(f"costlaw_{dt}_{arm}_ci_hi", "%.3f" % f["ci"][1])
                add(f"costlaw_{dt}_{arm}_r2", "%.3f" % f["r2"])
                add(f"costlaw_{dt}_{arm}_n", f["n"])
            # quadratic log-log fit: the LOCAL exponent at the smallest and the largest workload,
            # which is what Fig. 2B draws and tabulates (every series is concave-up on this scale)
            qf = F.quadratic_fit(pts)
            if qf:
                add(f"costlaw_{dt}_{arm}_exponent_smallest", "%.3f" % qf["slope_lo"])
                add(f"costlaw_{dt}_{arm}_exponent_smallest_ci_lo", "%.3f" % qf["ci_lo"][0])
                add(f"costlaw_{dt}_{arm}_exponent_smallest_ci_hi", "%.3f" % qf["ci_lo"][1])
                add(f"costlaw_{dt}_{arm}_exponent_largest", "%.3f" % qf["slope_hi"])
                add(f"costlaw_{dt}_{arm}_exponent_largest_ci_lo", "%.3f" % qf["ci_hi"][0])
                add(f"costlaw_{dt}_{arm}_exponent_largest_ci_hi", "%.3f" % qf["ci_hi"][1])
                add(f"costlaw_{dt}_{arm}_curvature", "%+.4f" % qf["curvature"])
                add(f"costlaw_{dt}_{arm}_r2_quadratic", "%.3f" % qf["r2"])
        pa = [(r["_work_units"], num(r["cpu_wall_s"])) for r in sub]
        pb = [(r["_work_units"], num(r["gpu_wall_s"])) for r in sub]
        pa, pb = zip(*[(a, b) for a, b in zip(pa, pb) if a[0] and a[1] and b[1]]) if pa else ((), ())
        pa, pb = list(pa), list(pb)
        if len(pa) >= 5:
            d = F.slope_difference(pa, pb)
            add(f"costlaw_{dt}_paired_diff", "%+.3f" % d["diff"])
            add(f"costlaw_{dt}_paired_diff_ci_lo", "%+.3f" % d["ci"][0])
            add(f"costlaw_{dt}_paired_diff_ci_hi", "%+.3f" % d["ci"][1])
            add(f"costlaw_{dt}_paired_diff_n", d["n"])
            # the same paired difference for the LOCAL exponents at the two ends of the range
            e = F.end_slope_difference(pa, pb)
            add(f"costlaw_{dt}_paired_diff_smallest", "%+.3f" % e["diff_lo"])
            add(f"costlaw_{dt}_paired_diff_smallest_ci_lo", "%+.3f" % e["ci_lo"][0])
            add(f"costlaw_{dt}_paired_diff_smallest_ci_hi", "%+.3f" % e["ci_lo"][1])
            add(f"costlaw_{dt}_paired_diff_largest", "%+.3f" % e["diff_hi"])
            add(f"costlaw_{dt}_paired_diff_largest_ci_lo", "%+.3f" % e["ci_hi"][0])
            add(f"costlaw_{dt}_paired_diff_largest_ci_hi", "%+.3f" % e["ci_hi"][1])
    add("costlaw_stratum", "20 threads, all cards pooled")
    add("costlaw_abscissa", "work units = likelihood calls x patterns x states^2 x rate categories")

    # k^p comparison -- shows the k^2 weighting is a definition, not a fitted result
    for (p, arm), r2 in sorted(kexp_table(fits).items()):
        add(f"kexp_p{p}_{arm}_r2", "%.3f" % r2)

    # ---- run strata, so every denominator the prose quotes has a key -------------------
    add("runs_current_build", len(rows))
    add("runs_for_fits", len(fits))
    add("runs_t20", len(F.at_threads(fits)))
    add("runs_t20_protein", len([r for r in F.at_threads(fits) if r["data_type"] == "protein"]))
    add("runs_t20_dna", len([r for r in F.at_threads(fits) if r["data_type"] == "dna"]))
    add("runs_t1", len(F.at_threads(fits, "1")))

    # ---- phases, processor time, cores ------------------------------------------------
    mf = [num(r["cpu_modelfinder_s"]) / num(r["gpu_modelfinder_s"]) for r in fits
          if r["data_type"] == "protein" and r["threads"] == "20"
          and num(r.get("cpu_modelfinder_s")) and num(r.get("gpu_modelfinder_s"))]
    if mf:
        add("modelfinder_speedup_n", len(mf))
        add("modelfinder_speedup_median", "%.2f" % st.median(mf))
        add("modelfinder_speedup_min", "%.2f" % min(mf))
        add("modelfinder_speedup_max", "%.2f" % max(mf))

    for dt in ("protein", "dna"):
        v = []
        for r in fits:
            if r["data_type"] != dt or r["threads"] != "20":
                continue
            _, cc = ph.get((r["cell"], "cpu"), (None, None))
            _, gc = ph.get((r["cell"], "gpu"), (None, None))
            if cc and gc and gc > 0:
                v.append(cc / gc)
        if v:
            add(f"cputime_ratio_{dt}_n", len(v))
            add(f"cputime_ratio_{dt}_median", "%.1f" % st.median(v))
            add(f"cputime_ratio_{dt}_iqr_lo", "%.1f" % q(v, .25))
            add(f"cputime_ratio_{dt}_iqr_hi", "%.1f" % q(v, .75))
        for arm in ("cpu", "gpu"):
            b = []
            for r in fits:
                if r["data_type"] != dt or r["threads"] != "20":
                    continue
                w, c = ph.get((r["cell"], arm), (None, None))
                if w and c and w > 0:
                    b.append(c / w)
            if b:
                add(f"cores_busy_{dt}_{arm}_median", "%.1f" % st.median(b))
                add(f"cores_busy_{dt}_{arm}_p90", "%.1f" % q(b, .90))
                add(f"cores_busy_{dt}_{arm}_n", len(b))

    # campaign core-hours: an AGGREGATE, dominated by the largest cells. The abstract quotes the
    # per-run median instead; this stays for the Results, labelled as an aggregate.
    ac = ag = 0.0
    per = []
    for r in fits:
        cw, gw, nc = num(r["cpu_wall_s"]), num(r["gpu_wall_s"]), num(r["threads"])
        if cw and gw and nc:
            ac += cw * nc / 3600.0
            ag += gw / 3600.0
            per.append((cw * nc) / gw)
    if per:
        add("corehours_cpu_total", "%.0f" % ac)
        add("corehours_gpu_total", "%.0f" % ag)
        add("corehours_aggregate_ratio", "%.1f" % (ac / ag))
        add("corehours_perrun_median", "%.1f" % st.median(per))
        add("corehours_perrun_n", len(per))

    # ---- cards: wall time on the datasets measured on every card ---------------------------
    # Speedup is NOT comparable across cards (the host CPU differs by up to 2.0x between sweep
    # nodes), so the cross-card statement is made on absolute GPU-arm wall time, and quoted
    # relative to the fastest card.
    shared = {}
    for r in fits:
        if r["data_type"] != "protein" or r["threads"] != "20":
            continue
        shared.setdefault((r["genus"], r["taxa"]), {})[r["card"]] = num(r["gpu_wall_s"])
    shared = {k: v for k, v in shared.items() if len(v) >= 6}
    if shared:
        add("card_shared_datasets", len(shared))
        base = "NVIDIA_A100_80GB_PCIe"
        rel = {}
        for v in shared.values():
            if not v.get(base):
                continue
            for c, w in v.items():
                rel.setdefault(c, []).append(w / v[base])
        for c, v in rel.items():
            add(f"card_wall_rel_a100_median_{c}", "%.2f" % st.median(v))
    # host-CPU spread across the sweep nodes on the same shared datasets
    hs = {}
    for r in fits:
        if r["data_type"] != "protein" or r["threads"] != "20":
            continue
        hs.setdefault((r["genus"], r["taxa"]), {})[r["card"]] = num(r["cpu_wall_s"])
    hs = [v for v in hs.values() if len(v) >= 6]
    if hs:
        sp = [max(v.values()) / min(v.values()) for v in hs]
        add("card_host_cpu_spread_min", "%.2f" % min(sp))
        add("card_host_cpu_spread_max", "%.2f" % max(sp))
        add("card_host_cpu_spread_median", "%.2f" % st.median(sp))

    # ---- memory ------------------------------------------------------------------------
    bpe, bpe_sel, reps = bytes_per_entry(fits)
    if bpe:
        add("arena_bytes_per_entry_median", "%.1f" % st.median(bpe))
        add("arena_bytes_per_entry_min", "%.2f" % min(bpe))
        add("arena_bytes_per_entry_max", "%.2f" % max(bpe))
        add("arena_bytes_per_entry_n", len(bpe))
        # The engine prints the slot size to one decimal in MB; for arenas under 1 MB (the 4-taxon
        # ladder cells) that rounding alone moves the per-entry figure by up to 80%. Reported
        # separately over the runs whose slot is at least 1 MB, where the print is exact to <1%.
        big = [a["slot_mb"] * 1048576 / ((num(r.get("patterns")) or 0) * (num(r.get("num_states")) or 0) * a["ncat_cap"])
               for r in fits for a in [reps.get(r["cell"])] if a and a["slot_mb"] >= 1.0
               and num(r.get("patterns")) and num(r.get("num_states"))]
        if big:
            add("arena_bytes_per_entry_ge1mb_min", "%.2f" % min(big))
            add("arena_bytes_per_entry_ge1mb_max", "%.2f" % max(big))
            add("arena_bytes_per_entry_ge1mb_n", len(big))
        # the quantity reported before round 8, kept so the correction can be stated
        add("arena_bytes_per_selected_entry_median", "%.1f" % st.median(bpe_sel))
        caps = [a["ncat_cap"] for a in reps.values()]
        add("arena_ncat_cap_mode", max(set(caps), key=caps.count))
        add("arena_vram_fraction", "%.2f" % ARENA_FRACTION)
        # Capacity table: arena = slots x patterns x states x capacity x 8 B; slots = taxa + 2 (the
        # engine pools IQ-TREE's own slot count, one per internal node when memory allows, plus two
        # for NNI). Patterns (millions) that fit the 70% budget; the 0.70 bounds only the category
        # HEADROOM in the engine, so this is a planning figure, not a guarantee (R13/R17).
        for gb in (16, 24, 48, 80):
            for tag, states, cap in (("aa_cap10", 20, 10), ("aa_cap4", 20, 4), ("nt_cap4", 4, 4)):
                for taxa in (50, 100, 200):
                    ent = gb * 1e9 * ARENA_FRACTION / ((taxa + 2) * states * cap * 8.0)
                    add(f"capacity_{gb}gb_{tag}_n{taxa}_patterns_m", "%.2f" % (ent / 1e6))
        # arena population (Supplementary Data 6): arena parsed, footprint above 0.00 GB, sampled peak VRAM present
        b6 = [r for r in fits if r["cell"] in reps and reps[r["cell"]]["footprint_gb"] > 0
              and (num(r.get("peak_vram_mb")) or 0) > 0]
        add("fig6b_n", len(b6))
        add("fig6b_n_protein", sum(1 for r in b6 if r.get("data_type") == "protein"))
        add("fig6b_n_dna", sum(1 for r in b6 if r.get("data_type") == "dna"))
        add("fig6b_dropped_tiny_arena_n", sum(1 for r in fits if r["cell"] in reps
                                             and reps[r["cell"]]["footprint_gb"] <= 0))
        # Guard events, in full: the engine lowers the category CAPACITY one step at a time while
        # slots x slot_bytes(capacity) exceeds 0.70 x VRAM, never below the model's own count, and
        # never reduces the slot count (phylotreegpu.cpp:1226-1235). The post-reduction footprint
        # can therefore exceed the budget; the extreme run did.
        gruns = [(c, a) for c, a in reps.items() if "guard" in a]
        add("guard_events_n", len(gruns))
        if gruns:
            add("guard_post_reduction_footprint_max_gb", "%.2f" % max(a["footprint_gb"] for _, a in gruns))
            add("guard_post_reduction_footprint_min_gb", "%.2f" % min(a["footprint_gb"] for _, a in gruns))
            add("guard_runs_over_budget_n", sum(1 for _, a in gruns if a["footprint_gb"] > a["guard"]["budget_gb"]))
            cx, ax_ = max(gruns, key=lambda t: t[1]["footprint_gb"])
            rx = next(r for r in fits if r["cell"] == cx)
            add("guard_extreme_cell", cx)
            add("guard_extreme_taxa", rx.get("taxa"))
            add("guard_extreme_patterns", rx.get("patterns"))
            add("guard_extreme_slots", ax_["slots"])
            add("guard_extreme_cap_from", ax_["guard"]["frm"])
            add("guard_extreme_cap_to", ax_["guard"]["to"])
            add("guard_extreme_need_gb", "%.2f" % ax_["guard"]["need_gb"])
            add("guard_extreme_have_gb", "%.2f" % ax_["guard"]["have_gb"])
            add("guard_extreme_budget_gb", "%.2f" % ax_["guard"]["budget_gb"])
            add("guard_extreme_footprint_gb", "%.2f" % ax_["footprint_gb"])
            add("guard_extreme_slot_mb", "%.1f" % ax_["slot_mb"])
            add("guard_extreme_peak_vram_mb", "%.0f" % (num(rx.get("peak_vram_mb")) or 0))
            add("guard_extreme_peak_ram_mb", "%.0f" % (num(rx.get("peak_ram_mb")) or 0))

    # ---- alignment properties -----------------------------------------------------------
    for dt in ("protein", "dna"):
        gp = [num(r["gap_ambiguity_pct"]) for r in fits
              if r["data_type"] == dt and num(r.get("gap_ambiguity_pct")) is not None]
        if gp:
            add(f"gap_ambiguity_{dt}_min", "%.1f" % min(gp))
            add(f"gap_ambiguity_{dt}_max", "%.1f" % max(gp))

    # ---- phases, profiling, energy, replicates, build regression -------------------------
    steps = {}
    if os.path.exists(STEPS):
        for r in csv.DictReader(open(STEPS), delimiter="\t"):
            steps.setdefault((r["cell"], r["arm"]), {})[r["step"]] = num(r.get("wall_s"))
    for step in ("modelfinder", "tree_search"):
        sp, share = [], []
        for r in fits:
            if r["data_type"] != "protein" or r["threads"] != "20":
                continue
            c, g = steps.get((r["cell"], "cpu"), {}), steps.get((r["cell"], "gpu"), {})
            if c.get(step) and g.get(step):
                sp.append(c[step] / g[step])
                if c.get("total"):
                    share.append(c[step] / c["total"])
        if sp:
            add(f"phase_{step}_speedup_median", "%.2f" % st.median(sp))
            add(f"phase_{step}_speedup_n", len(sp))
        if share:
            add(f"phase_{step}_cpu_wall_share_mean_pct", "%.1f" % (100 * st.mean(share)))
            add(f"phase_{step}_cpu_wall_share_min_pct", "%.1f" % (100 * min(share)))
            add(f"phase_{step}_cpu_wall_share_max_pct", "%.1f" % (100 * max(share)))

    gc = [num(r.get("gpu_calls")) for r in fits if num(r.get("gpu_calls"))]
    if gc:
        add("gpu_calls_min", "%.2e" % min(gc))
        add("gpu_calls_max", "%.2e" % max(gc))
    add("cpu_calls_max_host_fallbacks", "%.0f" % max(num(r.get("cpu_calls")) or 0 for r in fits))
    # profiling: n = 1 per alphabet, from the nsys reports
    prof = os.path.join(REPO, "benchmark", "profiling", "nsys_kernel_summary.tsv")
    if os.path.exists(prof):
        for r in csv.DictReader(open(prof), delimiter="\t"):
            dt = "protein" if r["case"] == "protein" else "dna"
            w, k, nk = float(r["total_wall_s"]), float(r["kernel_time_s"]), int(r["num_kernels"])
            add(f"nsys_{dt}_kernel_pct_of_wall", "%.1f" % (100 * k / w))
            add(f"nsys_{dt}_kernel_s", "%.2f" % k)
            add(f"nsys_{dt}_launches", nk)
            add(f"nsys_{dt}_mean_kernel_us", "%.2f" % (k / nk * 1e6))
        add("nsys_runs_per_alphabet", 1)

    # empty-kernel launch latency on the L40S (R14), from benchmark/scripts/harness/launch_latency.sh
    ll = os.path.join(REPO, "benchmark", "profiling", "launch_latency_l40s.tsv")
    if os.path.exists(ll):
        for r in csv.DictReader(open(ll), delimiter="\t"):
            tag = "" if r["variant"] == "sync" else "_async"
            add(f"launch_latency{tag}_us_l40s_median", "%.2f" % float(r["median_us"]))
            add(f"launch_latency{tag}_us_l40s_p25", "%.2f" % float(r["p25_us"]))
            add(f"launch_latency{tag}_us_l40s_p75", "%.2f" % float(r["p75_us"]))
            add(f"launch_latency{tag}_n", r["n"])
        if "launch_latency_us_l40s_median" in d and "nsys_dna_mean_kernel_us" in d:
            add("launch_latency_ratio_to_nt_kernel", "%.2f" % (float(d["launch_latency_us_l40s_median"]) / float(d["nsys_dna_mean_kernel_us"])))
        if "launch_latency_us_l40s_median" in d and "nsys_protein_mean_kernel_us" in d:
            add("launch_latency_ratio_to_aa_kernel", "%.3f" % (float(d["launch_latency_us_l40s_median"]) / float(d["nsys_protein_mean_kernel_us"])))


    # replicate dispersion, from the merged replicate table (one row per alphabet x size)
    f = os.path.join(REPO, "benchmark", "runs", "REPLICATES_20T.tsv")
    reps = list(csv.DictReader(open(f), delimiter="\t")) if os.path.exists(f) else []
    for tag in ("protein", "dna"):
        cv = [float(x["speedup_cv"]) for x in reps if x.get("data_type") == tag and x.get("speedup_cv")]
        if cv:
            add(f"replicate_cv_{tag}_min", "%.2f" % min(cv))
            add(f"replicate_cv_{tag}_max", "%.2f" % max(cv))
            add(f"replicate_cv_{tag}_median", "%.2f" % st.median(cv))
            add(f"replicate_cv_{tag}_n_sizes", len(cv))

    # ---- complexity coverage, hardware ---------------------------------------------------
    span = {}
    for r in rows:
        if r["_complexity"]:
            lo, hi = span.get(r["data_type"], (r["_complexity"], r["_complexity"]))
            span[r["data_type"]] = (min(lo, r["_complexity"]), max(hi, r["_complexity"]))
    for dt, (lo, hi) in sorted(span.items()):
        add(f"complexity_{dt}_min", "%.2e" % lo)
        add(f"complexity_{dt}_max", "%.2e" % hi)
    if "protein" in span and "dna" in span:
        lo = max(span["protein"][0], span["dna"][0])
        hi = min(span["protein"][1], span["dna"][1])
        add("complexity_shared_min", "%.2e" % lo)
        add("complexity_shared_max", "%.2e" % hi)
        add("complexity_shared_decades", "%.1f" % math.log10(hi / lo))
    add("distinct_cards", len({r["card"] for r in rows if r.get("card") not in (None, "", "NA")}))
    add("distinct_nodes", len({r["node"] for r in rows if r.get("node") not in (None, "", "NA")}))
    add("distinct_host_cpu_models",
        len({r["cpu_model"] for r in rows if r.get("cpu_model") not in (None, "", "NA")}))
    add("distinct_models", len({(r["data_type"], r["model"]) for r in rows
                                if r.get("model") not in (None, "", "NA")}))
    add("distinct_dna_models", len({r["model"] for r in rows
                                    if r["data_type"] == "dna" and r.get("model") not in (None, "", "NA")}))
    add("taxa_min", int(min(num(r["taxa"]) or 1e9 for r in rows)))
    add("taxa_max", int(max(num(r["taxa"]) or 0 for r in rows)))

    # ---- write ---------------------------------------------------------------------------
    d = dict(kv)
    md = ["<!-- GENERATED by benchmark/figures_tables/tables.py key_numbers - DO NOT EDIT BY HAND. -->",
          "<!-- Regenerate: python3 benchmark/figures_tables/tables.py key_numbers (runs inside refresh_all.sh) -->",
          "", "# Key numbers", "",
          "_Derived from `benchmark/runs/RUNS.tsv`._", "",
          f"**{d['total_rows']} paired runs** on the reported build, of which "
          f"**{d['rows_analysed_for_fits']}** enter fits and speedup summaries "
          f"({d['total_rows']} - 1 gate failure - {d['correctness_cells_excluded']} `__mdl_*` "
          f"correctness cells). {d['distinct_cards']} GPUs, {d['distinct_nodes']} nodes, "
          f"{d['distinct_host_cpu_models']} host CPU models, taxa {d['taxa_min']}-{d['taxa_max']}.", "",
          "> **Alphabet is perfectly aliased with workflow.** All "
          f"{d['workflow_aa_mf_n']} amino-acid runs use full ModelFinder and all "
          f"{d['workflow_nt_fixed_n']} nucleotide runs pin a model "
          f"({d['workflow_aa_with_fixed_model_n']} amino-acid runs with a fixed model, "
          f"{d['workflow_nt_with_modelfinder_n']} nucleotide runs with ModelFinder). No comparison "
          "in these data separates the two, so results are labelled by WORKFLOW, not alphabet.", "",
          "## Speedup (CPU wall / GPU wall, same node, same job, one binary)", "",
          "| workflow | threads | n | range | median | GPU slower |", "|---|---|---|---|---|---|"]
    label = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}
    for (dt, th) in sorted(groups):
        t = f"{dt}_{th}T"
        md.append(f"| {label[dt]} | {th} | {d[t+'_n']} | {d[t+'_min']}-{d[t+'_max']}x | "
                  f"{d[t+'_median']}x | {d[t+'_gpu_slower_n']}/{d[t+'_n']} "
                  f"({d[t+'_gpu_slower_pct']}%) |")
    md += ["",
           f"20-thread maximum from the largest **passing** cell: "
           f"**{d.get('protein_20T_largest_speedup','NA')}x** "
           f"(`{d.get('protein_20T_largest_cell','NA')}`, "
           f"{d.get('protein_20T_largest_cpu_wall_h','NA')} h -> "
           f"{d.get('protein_20T_largest_gpu_wall_h','NA')} h) for amino acid + ModelFinder, and "
           f"**{d.get('dna_20T_largest_speedup','NA')}x** for nucleotide, fixed model.", "",
           "## Break-even", "",
           "| workflow (20 threads) | crossover complexity | 95% CI | n | below parity |",
           "|---|---|---|---|---|"]
    for dt in ("protein", "dna"):
        if f"crossover_{dt}_t20" in d:
            md.append(f"| {label[dt]} | {d[f'crossover_{dt}_t20']} | "
                      f"{d[f'crossover_{dt}_t20_ci_lo']} - {d[f'crossover_{dt}_t20_ci_hi']} | "
                      f"{d[f'crossover_{dt}_t20_n']} | {d[f'crossover_{dt}_t20_n_below_parity']} |")
    md += ["", "## Equivalence", "",
           f"- topology: **{d['rf_nonzero']} of {d['total_rows']} runs differ** "
           f"(RF = 0 in {d['rf_zero_of_total']})",
           f"- likelihood: **{d['logl_bitidentical_n']} of {d['total_rows']} bit-identical**; the "
           f"remaining {d['logl_nonzero_n']} span {d['logl_reldiff_nonzero_min']} to "
           f"{d['logl_reldiff_max']} (contract <= {d['logl_contract']})",
           f"- support: {d['rows_with_support_diff']} of {d['total_rows']} differ at all; worst "
           f"**{d['support_max_absdiff']}** (`{d['support_max_absdiff_cell']}`), or "
           f"{d['support_max_absdiff_excl_gate_failure']} excluding it",
           f"- gate: **{d['gate_pass_true']} of {d['total_rows']} pass**; "
           f"{d['gate_pass_not_true']} does not"]
    if "replicate_cpu_vs_cpu_pairs" in d:
        md += ["",
               f"- **same-arm reproducibility**: CPU vs CPU "
               f"{d['replicate_cpu_vs_cpu_differing']}/{d['replicate_cpu_vs_cpu_pairs']} replicate "
               f"pairs differ (worst {d['replicate_cpu_vs_cpu_worst']}); GPU vs GPU "
               f"{d['replicate_gpu_vs_gpu_differing']}/{d['replicate_gpu_vs_gpu_pairs']} "
               f"(worst {d['replicate_gpu_vs_gpu_worst']}). Multi-threaded CPU reduction is "
               "observably non-deterministic; no GPU-side non-determinism was observed."]
    md += ["", "## Processor time", "",
           f"- per-run processor-time ratio, 20 threads: **{d.get('cputime_ratio_protein_median','NA')}x** "
           f"amino acid (n={d.get('cputime_ratio_protein_n','NA')}), "
           f"**{d.get('cputime_ratio_dna_median','NA')}x** nucleotide "
           f"(n={d.get('cputime_ratio_dna_n','NA')})",
           f"- campaign aggregate: {d.get('corehours_cpu_total','NA')} -> "
           f"{d.get('corehours_gpu_total','NA')} core-hours "
           f"({d.get('corehours_aggregate_ratio','NA')}x) - an AGGREGATE, dominated by the largest "
           f"cells; the median run is {d.get('corehours_perrun_median','NA')}x",
           f"- cores kept busy: amino acid CPU {d.get('cores_busy_protein_cpu_median','NA')} / GPU "
           f"{d.get('cores_busy_protein_gpu_median','NA')} (p90 "
           f"{d.get('cores_busy_protein_gpu_p90','NA')}); nucleotide CPU "
           f"{d.get('cores_busy_dna_cpu_median','NA')} / GPU "
           f"{d.get('cores_busy_dna_gpu_median','NA')} (p90 {d.get('cores_busy_dna_gpu_p90','NA')})",
           "", "## Tree complexity covered", "",
           "Complexity = `branches(2n-3) x patterns x rate categories x states^2`. The states^2 "
           "weighting is a DEFINITION, not a fitted result: refitting with k^p gives "
           f"R^2 {d.get('kexp_p0_gpu_r2','NA')} (p=0) and {d.get('kexp_p1_cpu_r2','NA')} (p=1) "
           f"against {d.get('kexp_p2_cpu_r2','NA')} (CPU) / {d.get('kexp_p2_gpu_r2','NA')} (GPU) at "
           "p=2, and k takes only two values so the exponent is a two-group contrast.", ""]
    for dt in ("protein", "dna"):
        if f"complexity_{dt}_min" in d:
            md.append(f"- **{label[dt]}**: {d[f'complexity_{dt}_min']} - {d[f'complexity_{dt}_max']}")
    if "complexity_shared_decades" in d:
        md.append(f"- **shared range**: {d['complexity_shared_min']} - {d['complexity_shared_max']} "
                  f"= **{d['complexity_shared_decades']} decades**")
    md.append("")

    open(OUTMD, "w").write("\n".join(md) + "\n")
    with open(OUTTSV, "w") as fh:
        fh.write("key\tvalue\n")
        for k, v in kv:
            fh.write(f"{k}\t{v}\n")
    print(f"key numbers: {len(kv)} keys from {d['total_rows']} rows "
          f"({d['rows_analysed_for_fits']} in fits) -> KEY_NUMBERS.{{md,tsv}}")
    if not ok:
        print("WARNING: " + msg)


key_numbers_main = main


# ================================================================================================
# ==== section: source_data ====
# ================================================================================================
# source_data.py - assemble the Source Data workbook from the per-panel CSVs.
#
#     python3 benchmark/figures_tables/tables.py source_data [<out.xlsx>]
#
# The panel writer (`benchmark/figures_tables/lib/figstyle.py`, `write_source_data`) records, for every published
# panel, the values behind every drawn mark, read back from the matplotlib artists so the file cannot
# disagree with the figure. This script collects those CSVs into one workbook with a README sheet and
# one sheet per panel, named by the panel's figure and letter (Fig1a, FigS8a, ...), in manifest order.
# Panels whose marks are drawings rather than data (the tree montages) carry a sheet that says which
# tree files they were drawn from.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.

SD = os.path.join(REPO, "benchmark", "figures", "source_data")
source_data_OUT = os.path.join(REPO, "benchmark", "figures", "Source_Data.xlsx")   # or `tables.py source_data <path>`
TREE_SCRIPT = "figures.py (trees)"


def main():
    man = pd.read_csv(MANIFEST, sep="\t", dtype=str).fillna("")
    readme, sheets = [], []
    # Two boxes that share one letter (Fig. 2A left and right, Fig. 2B left and right) would both
    # be "Fig2a"; such sheets carry the panel's slug so the name says which box it is.
    shared = man.groupby(["figure", "panel"])["file"].transform("count")
    for (_, r), nshare in zip(man.iterrows(), shared):
        stem = r["file"]; fig = r["figure"]; panel = r["panel"]
        # Alternative drafts (<panel>_alternative<k>_<slug>) are not published panels: they keep
        # their per-panel CSV but get no sheet, or the draft that sorts first would take the
        # published panel's sheet name (Fig1a was the ECDF draft once).
        if "_alternative" in stem:
            continue
        is_page = "_p" in stem[-4:] or stem.rsplit("_", 1)[-1].startswith("p") and stem.rsplit("_", 1)[-1][1:].isdigit()
        sheet = f"Fig{fig}{panel}" + (f"_{stem.rsplit('_p', 1)[1]}" if is_page else "")
        if nshare > 1 and not is_page:
            sheet += "_" + stem.split("_", 1)[1]
        sheet = sheet[:31]
        csv_path = os.path.join(SD, stem + ".csv")
        if r["source_script"] == TREE_SCRIPT:
            df = pd.DataFrame({"note": [f"Tree drawings; the plotted objects are the maximum-likelihood tree files of the "
                                        f"{r['rows_used']} datasets shown, distributed per cell in the archived record "
                                        f"(Supplementary Note 2, §11.3: <cell>/gpu.treefile and <cell>/cpu.treefile)."]})
            kind = "tree files (see note)"
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path); kind = f"{len(df)} plotted values"
        else:
            df = pd.DataFrame({"note": ["no plotted values were recorded for this panel"]}); kind = "none recorded"
        readme.append(dict(sheet=sheet, figure=fig, panel=panel, title=r["title"], generating_script=r["source_script"],
                           source_table=r["source_table"], rows_used=r["rows_used"], data_date=r["data_date"], contents=kind))
        sheets.append((sheet, df))
    with pd.ExcelWriter(source_data_OUT, engine="openpyxl") as xw:
        intro = pd.DataFrame({"Source Data": [
            "One sheet per published panel. Columns: axes (index of the axes within the panel), axes_title, kind "
            "(scatter, line, bar, segment, band, image), series (legend label where the artist has one), x_label, "
            "y_label, x, y, width (bars), height_or_value (bar height, or the cell value of an image).",
            "Values are in data coordinates as drawn; a log axis does not change them. Fitted lines and confidence "
            "bands appear as 'line' and 'band' rows with their series label.",
            "The values are read back from the drawn artists by the panel writer (benchmark/figures_tables/lib/figstyle.py), so a sheet "
            "cannot disagree with its panel. The underlying run table is in benchmark/runs/ of the archived record.",
        ]})
        intro.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(readme).to_excel(xw, sheet_name="README", index=False, startrow=5)
        seen = set()
        for sheet, df in sheets:
            name = sheet; k = 2
            while name in seen:
                name = f"{sheet[:28]}_{k}"; k += 1
            seen.add(name)
            df.to_excel(xw, sheet_name=name, index=False)
    print(f"  wrote {os.path.relpath(source_data_OUT, REPO)}: {len(sheets)} panel sheets")


source_data_main = main


# ================================================================================================
# ==== section: supplementary_data ====
# ================================================================================================
# supplementary_data.py - the Supplementary Data workbook of the tool paper, and its check.
#
#     python3 benchmark/figures_tables/tables.py supplementary_data [<out.xlsx>]
#
# Nature Communications distinguishes Supplementary Tables (in the SI document) from Supplementary
# Data (machine-readable files). This workbook holds the tables behind the main text and
# Supplementary Note 2, numbered Supplementary Data 1-12, one sheet each, from the archived
# record's files without alteration, plus a README sheet and a column dictionary for the run
# table taken from benchmark/BENCHMARK_SCHEMA.md. Data 7 (the engine's arena report per run) is assembled here
# from every run's gpu.stdout with the same parser the key numbers use. After writing, the workbook is
# read back and every cell compared with its source; any difference fails the build.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.

supplementary_data_OUT = os.path.join(REPO, "benchmark", "runs", "Supplementary_Data_benchmark.xlsx")   # or `tables.py supplementary_data <path>`




def read_tsv(path):
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    rows = list(csv.reader(lines, delimiter="\t"))
    hdr = rows[0]
    body = [r + [""] * (len(hdr) - len(r)) for r in rows[1:] if any(c.strip() for c in r)]
    return pd.DataFrame(body, columns=hdr, dtype=str).fillna("")


def arena_table():
    """Supplementary Data 7: the engine's arena report for every recorded run."""
    K = sys.modules[__name__]                   # the key_numbers section of this file
    FD = _load("figdata")                       # the single definition of the excluded build
    runs = read_tsv(os.path.join(REPO, "benchmark", "runs", "RUNS.tsv"))
    out = []
    for _, r in runs.iterrows():
        rep = K.arena_report(r)
        build = "prior" if (r.get("binary_sha256") or "").startswith(FD.PRIOR_BUILD_SHA_PREFIX) else "reported"
        row = dict(cell=r["cell"], run_id=r["run_id"], build=build, data_type=r["data_type"],
                   genus=r["genus"], taxa=r["taxa"], patterns=r["patterns"], model=r["model"], card=r["card"],
                   threads=r["threads"])
        if rep:
            g = rep.get("guard") or {}
            row.update(ncat_selected=rep["ncat_selected"], ncat_capacity_allocated=rep["ncat_cap"],
                       capacity_reduced="yes" if g else "no", capacity_requested=g.get("frm", rep["ncat_cap"]),
                       slots=rep["slots"], iqtree_lh_slots=rep["iqtree_slots"], slot_partials_mib=rep["slot_mb"],
                       slot_counters_mib=rep["counters_mb"], arena_footprint_gib=rep["footprint_gb"],
                       need_at_requested_gib=g.get("need_gb", ""), device_usable_gib=g.get("have_gb", ""),
                       budget_gib=("%.2f" % g["budget_gb"]) if g else "")
        else:
            row.update(ncat_selected="", ncat_capacity_allocated="", capacity_reduced="", capacity_requested="", slots="",
                       iqtree_lh_slots="", slot_partials_mib="", slot_counters_mib="", arena_footprint_gib="",
                       need_at_requested_gib="", device_usable_gib="", budget_gib="")
        row.update(peak_vram_sampled_mib=r.get("peak_vram_mb", ""), peak_ram_job_mib=r.get("peak_ram_mb", ""))
        out.append(row)
    d = pd.DataFrame(out).astype(str).replace({"nan": "", "None": ""})
    return d


def mag_sources_table():
    """Supplementary Data 0: every source genome with its accession, deposit and citation.

    Built from the curated metadata table (one row per MAG) joined to the hand-maintained study
    citation table on `dataset`; nothing is inferred to fill a blank, so a blank accession means
    the study's public deposit does not supply that identifier (see PROVENANCE.md)."""
    G = os.path.join(REPO, "benchmark", "data", "genomes")
    meta = read_tsv(os.path.join(G, "genome_metadata.tsv"))
    cites = read_tsv(os.path.join(G, "study_citations.tsv")).set_index("dataset")
    missing = sorted(set(meta["dataset"]) - set(cites.index))
    if missing:
        sys.exit(f"  study_citations.tsv has no row for dataset(s) {missing}")
    cols = ["stable_genome_id", "GTDB_tk_genus", "GTDB_tk_species", "dataset", "citation", "doi",
            "genome_deposit", "genome_id", "source_filename", "source_accession", "source_sample",
            "source_bioproject", "source_biosample", "genome_assembly_accession", "host_species",
            "host_common_name_group", "checkm2_completeness", "checkm2_contamination",
            "checkm2_genome_size_bp", "md5_fna", "alignments"]
    d = meta.copy()
    for c in ("citation", "doi", "genome_deposit"):
        d[c] = d["dataset"].map(cites[c])
    d = d[cols].sort_values(["GTDB_tk_genus", "stable_genome_id"]).reset_index(drop=True)
    assert len(d) == 435 and d["GTDB_tk_genus"].nunique() == 14, (len(d), d["GTDB_tk_genus"].nunique())
    return d.astype(str).replace({"nan": "", "None": ""})


def column_dictionary():
    """The run table's column dictionary, from benchmark/BENCHMARK_SCHEMA.md, restricted to columns present."""
    runs_cols = list(read_tsv(os.path.join(REPO, "benchmark", "runs", "RUNS.tsv")).columns)
    rows, group = [], ""
    for line in open(os.path.join(REPO, "benchmark", "BENCHMARK_SCHEMA.md")):
        m = re.match(r"^### ([A-M])\. (.*)$", line.strip())
        if m: group = f"{m.group(1)}. {m.group(2)}"
        m = re.match(r"^\| ([a-z_0-9]+) \| ([^|]*) \| ([^|]*) \| (.*) \|\s*$", line.rstrip("\n"))
        if m and m.group(1) in runs_cols:
            rows.append(dict(column=m.group(1), schema_group=group, source=m.group(2).strip(), status=m.group(3).strip(), definition=m.group(4).strip()))
    seen = {r["column"] for r in rows}
    for c in runs_cols:
        if c not in seen:
            rows.append(dict(column=c, schema_group="", source="", status="", definition="(present in RUNS.tsv; see benchmark/BENCHMARK_SCHEMA.md)"))
    return pd.DataFrame(rows, dtype=str)


INT_RX = re.compile(r"^-?\d+$")
FLOAT_RX = re.compile(r"^-?\d+\.\d+$|^-?\d+(\.\d+)?[eE][+-]?\d+$")


def _sig_digits(v):
    m = re.sub(r"[-+]", "", v.split("e")[0].split("E")[0]).replace(".", "").lstrip("0")
    return len(m.rstrip("0")) if m else 0

def to_number(col):
    """Numbers only when Excel can hold every value exactly (at most 15 significant digits)."""
    vals = [v for v in col if v != ""]
    if not vals: return None
    if any(_sig_digits(v) > 15 for v in vals): return None
    if all(INT_RX.match(v) for v in vals) and all(len(v) < 16 for v in vals) and not any(v != "0" and v.startswith("0") for v in vals):
        return col.map(lambda v: int(v) if v != "" else None).astype("object")
    if all(INT_RX.match(v) or FLOAT_RX.match(v) for v in vals):
        return col.map(lambda v: float(v) if v != "" else None).astype("object")
    return None


def main():
    B = os.path.join(REPO, "benchmark", "runs"); P = os.path.join(REPO, "benchmark", "profiling")
    items = [
        (0, "MAG_SOURCES", "Source genomes", "The 435 metagenome-assembled genomes (MAGs) across 14 bacterial genera from which every amino-acid alignment was built: one row per genome with its GTDB genus and species, the study it was published in with the citation and DOI, the public deposit the file was taken from, the source accession, sample, BioProject, BioSample and per-MAG assembly accession where the deposit supplies them, host, CheckM2 quality, file checksum, and which benchmark alignments carry it. Column meanings in benchmark/data/genomes/PROVENANCE.md.", mag_sources_table(), "benchmark/data/genomes/genome_metadata.tsv; benchmark/data/genomes/study_citations.tsv", "Methods (data); Supplementary Note 2"),
        (1, "RUNS", "Run table", "One row per paired run (226: the 182 runs of the reported build and the 44 runs of an earlier build, identified by binary_sha256, whose checksum begins 07c70cae): dataset, workflow, thread count, card, node, host CPU, driver, binary checksum, selected model, both arms' timings and log-likelihoods, the equivalence outcome, the engine's engagement counters and the sampled device metrics. Column dictionary in the last sheet.", read_tsv(f"{B}/RUNS.tsv"), "benchmark/runs/RUNS.tsv", "Methods; Results; Note 2 §1, §11.1"),
        (2, "RUNS_summary", "Run-table summary", "Mean and s.d. of speedup and device utilisation per (alphabet, model treatment, thread count, card) stratum, with engagement and gate-pass counts.", read_tsv(f"{B}/RUNS_summary.tsv"), "benchmark/runs/RUNS_summary.tsv", "Note 2 §11.1"),
        (3, "KEY_NUMBERS", "Key numbers", "Every number quoted in the Abstract, Results and Discussion as key-value pairs, regenerated from the run table by benchmark/figures_tables/tables.py (key_numbers); no number in the manuscript is hand-transcribed.", read_tsv(f"{B}/KEY_NUMBERS.tsv"), "benchmark/runs/KEY_NUMBERS.tsv", "Note 2 §10, §11.2"),
        (4, "PHASE_STEPS", "Phase steps", "Per-step wall and CPU time for both arms of all 226 recorded cells, with a level column separating the two top-level phases from the setup steps nested inside them.", read_tsv(f"{B}/PHASE_STEPS.tsv"), "benchmark/runs/PHASE_STEPS.tsv", "Results (Fig. 3); Note 2 §11.1"),
        (5, "REPLICATES_20T", "Replicate sets", "Replicate sets by alphabet and size at 20 threads: median, mean, s.d., coefficient of variation and the contributing speedups.", read_tsv(f"{B}/REPLICATES_20T.tsv"), "benchmark/runs/REPLICATES_20T.tsv", "Results; Supplementary Fig. S1"),
        (6, "ARENA_REPORTS", "Device-arena reports", "The engine's own arena report for every recorded run, parsed from gpu.stdout with the parser the key numbers use: selected and allocated rate-category counts, whether the capacity was reduced (the six guard runs) and from what, slot count, per-slot partial-likelihood and counter sizes, arena footprint, the need at the requested capacity, the usable device memory and the 70% budget, with the sampled device peak and the job's peak host memory from the run table.", arena_table(), "benchmark/runs/paired/<cell>/gpu.stdout; benchmark/runs/RUNS.tsv", "Results (device-memory demand); Methods (implementation); Note 2 §7"),
        (7, "LAUNCH_LATENCY", "Empty-kernel launch latency", "Empty-kernel launch cost on the L40S: sync (launch plus synchronise, timed per launch) and async (back-to-back launches, one synchronisation): n, median, quartiles, mean and minimum in microseconds.", read_tsv(f"{P}/launch_latency_l40s.tsv"), "benchmark/profiling/launch_latency_l40s.tsv", "Methods (profiling); Note 2 §8"),
        (8, "NSYS_KERNEL_SUMMARY", "Nsight Systems kernel attribution", "One row per alphabet from the profiled run: wall, kernel and memcpy seconds, launch count, the top kernel and its share.", read_tsv(f"{P}/nsys_kernel_summary.tsv"), "benchmark/profiling/nsys_kernel_summary.tsv", "Methods (profiling); Note 2 §8"),
        (9, "NSIGHT_SUMMARY", "Nsight Compute summary (header only)", "Header-only by design: the Nsight Compute hardware counters were administratively restricted on the cluster, so no kernel rows were collected; the empty table is distributed so that its emptiness is visible.", read_tsv(f"{P}/nsight_summary.tsv"), "benchmark/profiling/nsight_summary.tsv", "Methods (profiling); Note 2 §11.1"),
    ]
    coldict = column_dictionary()
    readme = pd.DataFrame([dict(supplementary_data=f"Supplementary Data {n}", sheet=f"Data {n}", title=t, contents=c, rows=len(d), columns=len(d.columns), source_file=src, cited_in=cit)
                           for n, s, t, c, d, src, cit in items])
    with pd.ExcelWriter(supplementary_data_OUT, engine="openpyxl") as xw:
        head = pd.DataFrame({"Supplementary Data": [
            "Machine-readable tables behind the main text and Supplementary Note 2 of the IQ-TREE3-GPU manuscript, one sheet per item, numbered Supplementary Data 0-9.",
            "Each sheet is the archived record's file without alteration: a cell holds a number only when every value in its column is a number; identifiers, checksums, model strings and command lines are text. Empty cells are empty in the source as well.",
            "The last sheet, RUNS_columns, defines every column of Data 1 (from benchmark/BENCHMARK_SCHEMA.md in the record).",
        ]})
        head.to_excel(xw, sheet_name="README", index=False)
        readme.to_excel(xw, sheet_name="README", index=False, startrow=len(head) + 2)
        for n, s, t, c, d, src, cit in items:
            e = d.copy()
            for col in e.columns:
                v = to_number(e[col])
                if v is not None: e[col] = v
            e.to_excel(xw, sheet_name=f"Data {n}", index=False)
            xw.sheets[f"Data {n}"].freeze_panes = "A2"
        coldict.to_excel(xw, sheet_name="RUNS_columns", index=False)
        xw.sheets["README"].column_dimensions["C"].width = 34; xw.sheets["README"].column_dimensions["D"].width = 110
    # verify
    import openpyxl
    wb = openpyxl.load_workbook(supplementary_data_OUT, read_only=True)
    expected = ["README"] + [f"Data {n}" for n, *_ in items] + ["RUNS_columns"]
    # `n` is 0 for the first sheet, so sheet names are tested with `is not None`, never truthiness.
    problems = [] if wb.sheetnames == expected else [f"sheets {wb.sheetnames}"]
    cells = 0
    for n, s, t, c, d, src, cit in items + [(None, None, None, None, coldict, None, None)]:
        name = f"Data {n}" if n is not None else "RUNS_columns"
        rows = list(wb[name].iter_rows(values_only=True))
        hdr = [("" if v is None else str(v)) for v in rows[0]]
        if hdr != list(d.columns): problems.append(f"{name}: header"); continue
        if len(rows) - 1 != len(d): problems.append(f"{name}: {len(rows)-1} rows vs {len(d)}"); continue
        for i, r in enumerate(rows[1:]):
            for j, col in enumerate(d.columns):
                cells += 1
                sv = d.iat[i, j]; x = r[j] if j < len(r) else None
                if x is None:
                    if sv != "": problems.append(f"{name} r{i+2} {col}: empty vs {sv!r}"); break
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    if float(sv) != float(x): problems.append(f"{name} r{i+2} {col}: {x} vs {sv}"); break
                elif str(x) != sv: problems.append(f"{name} r{i+2} {col}: {str(x)[:30]!r} vs {sv[:30]!r}"); break
            if len(problems) > 10: break
    if problems:
        print("  VERIFY FAILED: " + "; ".join(problems[:10])); sys.exit(1)
    print(f"  wrote {os.path.relpath(supplementary_data_OUT, REPO)}: {len(items)} data sheets + README + RUNS_columns; verified {cells:,} cells against the sources")
    import subprocess
    subprocess.run([sys.executable, __file__, "style_workbooks", supplementary_data_OUT], check=True)


supplementary_data_main = main


# ================================================================================================
# ==== section: style_workbooks ====
# ================================================================================================
# style_workbooks.py - one presentation for every supplementary workbook, values untouched.
#
#     python3 benchmark/figures_tables/tables.py style_workbooks <workbook.xlsx> [...]
#
# Applies to each workbook: a README sheet with a title, wrapped introductory text and a contents
# table whose sheet names link to the sheets; on every data sheet a bold shaded header row with a
# filter, frozen at the header, column widths fitted to the content (capped), and a sheet-tab colour
# by group (read from the README's group/contents columns). No cell value is changed: every value is
# recorded before styling and compared after the workbook is reloaded; a difference fails the run.

HEAD_FILL = PatternFill("solid", fgColor="D9E1F2")
HEAD_FONT = Font(bold=True)
TITLE_FONT = Font(bold=True, size=14)
BODY_FONT = Font(size=11)
THIN = Side(style="thin", color="8EA9DB")
TAB = {"Supplementary Data": "2F5597", "README": "1F3864", "RUNS_columns": "BF9000"}


def snapshot(wb):
    out = {}
    for ws in wb.worksheets:
        out[ws.title] = [[c.value for c in row] for row in ws.iter_rows()]
    return out


def _key(v):
    return str(v or "").strip().lower().replace(" ", "_")


def _header_row(ws):
    """The contents table's header: below the title, first cell a known label, second cell filled."""
    return next((r for r in range(2, ws.max_row + 1)
                 if _key(ws.cell(r, 1).value) in ("table", "supplementary_data", "sheet", "item") and ws.cell(r, 2).value), None)


def style_readme(ws, wb):
    """Title, wrapped intro paragraphs merged across the table width, contents table with links."""
    max_col = max(ws.max_column, 6)
    # the contents table starts at the first row whose first cell is a header keyword
    for rng in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(rng))
    hdr_row = _header_row(ws)
    ws["A1"].font = TITLE_FONT
    for r in range(2, (hdr_row or ws.max_row + 1)):
        v = ws.cell(r, 1).value
        if v:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=max_col)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(r, 1).font = BODY_FONT
            ws.row_dimensions[r].height = max(30, 15 * (len(str(v)) // 130 + 1))
    if hdr_row:
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(hdr_row, c)
            if cell.value is not None:
                cell.value = str(cell.value).replace("_", " ").capitalize() if str(cell.value).islower() else cell.value
                cell.font = HEAD_FONT; cell.fill = HEAD_FILL; cell.border = Border(bottom=THIN)
                cell.alignment = Alignment(vertical="center", wrap_text=True)
        headers = [_key(ws.cell(hdr_row, c).value) for c in range(1, ws.max_column + 1)]
        sheet_col = next((i + 1 for i, h in enumerate(headers) if h in ("sheet", "table")), 1)
        for r in range(hdr_row + 1, ws.max_row + 1):
            name = ws.cell(r, sheet_col).value
            if name and name in wb.sheetnames:
                cell = ws.cell(r, sheet_col)
                from openpyxl.worksheet.hyperlink import Hyperlink
                cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{name}'!A1", display=str(name))
                cell.font = Font(color="0563C1", underline="single")
            for c in range(1, ws.max_column + 1):
                ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")
        ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(ws.max_column)}{ws.max_row}"
        ws.freeze_panes = ws.cell(hdr_row + 1, 1)
    widths = {}
    for c in range(1, ws.max_column + 1):
        h = headers[c - 1] if hdr_row else ""
        widths[c] = {"contents": 90, "content": 90, "title": 30, "group": 34, "cited_in": 42, "source_file": 40}.get(h, 16)
    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.sheet_properties.tabColor = TAB["README"]


def style_table(ws, group):
    if ws.max_row < 1: return
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(1, c)
        cell.font = HEAD_FONT; cell.fill = HEAD_FILL; cell.border = Border(bottom=THIN)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
    for c in range(1, ws.max_column + 1):
        col = get_column_letter(c)
        longest = len(str(ws.cell(1, c).value or ""))
        for r in range(2, min(ws.max_row, 400) + 1):
            v = ws.cell(r, c).value
            if v is not None:
                longest = max(longest, len(str(v)))
        ws.column_dimensions[col].width = max(9, min(60, int(longest * 0.95) + 2))
    ws.row_dimensions[1].height = 30
    for key, colour in TAB.items():
        if group.startswith(key):
            ws.sheet_properties.tabColor = colour; break


def main():
    for path in sys.argv[1:]:
        wb = openpyxl.load_workbook(path)
        before = snapshot(wb)
        groups = {}
        if "README" in wb.sheetnames:
            ws = wb["README"]
            hdr_row = _header_row(ws)
            if hdr_row:
                headers = [_key(ws.cell(hdr_row, c).value) for c in range(1, ws.max_column + 1)]
                sc = next((i + 1 for i, h in enumerate(headers) if h in ("sheet", "table")), 1)
                gc = next((i + 1 for i, h in enumerate(headers) if h == "group"), None)
                for r in range(hdr_row + 1, ws.max_row + 1):
                    name = ws.cell(r, sc).value
                    if name: groups[name] = str(ws.cell(r, gc).value or "") if gc else "Supplementary Data"
            style_readme(ws, wb)
        for ws in wb.worksheets:
            if ws.title == "README": continue
            style_table(ws, groups.get(ws.title, ws.title))
        after = snapshot(wb)                    # checked before saving: nothing but README header labels may differ
        diffs = 0
        for name, rows in before.items():
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    w = after[name][i][j] if i < len(after[name]) and j < len(after[name][i]) else None
                    if v != w and not (name == "README" and isinstance(v, str) and _key(w) == _key(v)):
                        diffs += 1
        if diffs:
            sys.exit(f"  {path}: {diffs} cell(s) would change; not saved")
        wb.save(path)
        print(f"  styled {path.split('/')[-1]}: {len(wb.sheetnames)} sheets, values unchanged")


style_workbooks_main = main


# ================================================================================================
# ==== section: benchmark_map ====
# ================================================================================================
# benchmark_map.py - the figure and table map in benchmark/README.md, generated from the pipeline's own records.
#
#     python3 benchmark/figures_tables/tables.py benchmark_map          # rewrite the generated block of benchmark/README.md
#     python3 benchmark/figures_tables/tables.py benchmark_map --check  # exit 1 if that block is stale
#
# WHY GENERATED. The map answers "which script makes this figure, from which table?" for every figure
# and table the manuscript carries. Typed by hand it would drift the first time a panel moved between
# figures. So it is derived from the three places that already decide those facts: FIGURE_MANIFEST.tsv
# (panel -> generating script -> source table, written by the panel writer), composites.py (figure ->
# panels -> composite file, which IS the manuscript's figure numbering) and the item list of
# supplementary_data.py (Supplementary Data n -> sheet -> source file). `crosscheck.py` fails the build
# if the block in the README differs from what this script would write.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

BENCH = os.path.join(REPO, "benchmark")
HERE = os.path.dirname(os.path.abspath(__file__))


def _figures():
    """figures.py, loaded once: its composites section holds the figure layouts."""
    if "_FIGURES_MOD" not in globals():
        globals()["_FIGURES_MOD"] = _load("figures", os.path.join(HERE, "figures.py"))
    return globals()["_FIGURES_MOD"]
README = os.path.join(BENCH, "README.md")
BEGIN = "<!-- map:begin  generated by benchmark/figures_tables/tables.py (benchmark_map) from FIGURE_MANIFEST.tsv and the composites and supplementary_data sections; do not edit by hand -->"
END = "<!-- map:end -->"

# Which script writes each table that a Supplementary Data sheet or a panel is read from. A table
# absent here fails the build, so a new table cannot enter the workbook without naming its producer.
PRODUCER = {
    "RUNS.tsv": "`scripts/aggregate_runs.py`",
    "RUNS_summary.tsv": "`scripts/aggregate_runs.py`",
    "KEY_NUMBERS.tsv": "`figures_tables/tables.py key_numbers`",
    "PHASE_STEPS.tsv": "`scripts/phase_extract.py`",
    "REPLICATES_20T.tsv": "`figures_tables/figures.py supp_ported`",
    "genome_metadata.tsv": "curated source-genome metadata (`data/genomes/PROVENANCE.md`); citations hand-maintained in `data/genomes/study_citations.tsv`",
    "gpu.stdout": "`harness/paired_benchmark.sh` (the engine's own arena report per run)",
    "launch_latency_l40s.tsv": "`harness/launch_latency.sh`",
    "nsys_kernel_summary.tsv": "`harness/nsys_kernel_time.sh`",
    "nsight_summary.tsv": "`harness/roofline_profile.sh`",
}




def manifest():
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def producer(path):
    base = os.path.basename(path.split(";")[0].strip())
    if base not in PRODUCER:
        raise SystemExit(f"benchmark_map.py: no producer recorded for {path!r}; add it to PRODUCER")
    return PRODUCER[base]


def figure_rows():
    """One row per figure: label, composite file, panels with their letters, generating scripts, source tables."""
    CF = _figures()
    by_stem = {r["file"]: r for r in manifest()}
    out = []
    for book, pre, kind in ((CF.FIGURES, "", "Figure"), (CF.SUPP, "S", "Supplementary Figure")):
        for n, spec in sorted(book.items()):
            items = CF.all_items(spec)
            stems = CF.all_stems(spec)
            lettered = CF.lettered(spec, items)
            missing = [s for s in stems if s not in by_stem]
            if missing:
                raise SystemExit(f"benchmark_map.py: composite {pre}{n} uses panels absent from the manifest: {missing}")
            label = f"{kind} {pre}{n}" if not pre else f"Supplementary Fig. {pre}{n}"
            comp = f"Figure_{n}" if not pre else f"Supplementary_Figure_S{n}"
            if lettered:
                panels = ", ".join(f"{CF.LETTERS[k]}: " + ", ".join(f"`{s}`" for s in CF.stems_of(it))
                                   for k, it in enumerate(items))
            elif len(stems) == 1:
                panels = f"`{stems[0]}`"
            else:
                panels = "pages " + ", ".join(f"`{s}`" for s in stems)
            scripts = sorted({by_stem[s]["source_script"] for s in stems})
            tables = sorted({by_stem[s]["source_table"] for s in stems})
            out.append(dict(label=label, title=spec["title"], composite=comp, panels=panels,
                            scripts=", ".join(f"`{x}`" for x in scripts),
                            tables=", ".join(f"`{os.path.relpath(os.path.join(REPO, t), BENCH)}`" for t in tables)))
    return out


def data_rows():
    """Supplementary Data n -> sheet -> source file -> producer, read from supplementary_data.py's item list."""
    src = open(__file__).read()
    rx = re.compile(r'^\s*\((\d+), "([^"]*)", "([^"]*)", "(?:[^"\\]|\\.)*", .*, "([^"]*)", "([^"]*)"\),?\s*$', re.M)
    rows = [dict(n=int(m.group(1)), sheet=m.group(2), title=m.group(3), source=m.group(4), cited=m.group(5))
            for m in rx.finditer(src)]
    if len(rows) < 10:
        raise SystemExit(f"benchmark_map.py: parsed only {len(rows)} Supplementary Data items from supplementary_data.py")
    for r in rows:
        r["producer"] = producer(r["source"])
        r["source"] = "; ".join(f"`{os.path.relpath(os.path.join(REPO, p.strip()), BENCH)}`" for p in r["source"].split(";"))
    return sorted(rows, key=lambda r: r["n"])


def render():
    L = [BEGIN, ""]
    L += ["### Figures", "",
          "Every panel is written by one generator to `figures/panels/` (PDF + PNG) with a row in `figures/FIGURE_MANIFEST.tsv`; "
          "`figures.py composites` then sets the panels into the lettered figure the journal receives, in `figures/composites/`. "
          "Source tables are relative to `benchmark/`.", "",
          "| Figure | Title | Composite (`figures/composites/`) | Panels (`figures/panels/`) | Generated by (`figures_tables/`, section) | Source table |",
          "|---|---|---|---|---|---|"]
    for r in figure_rows():
        L.append(f"| {r['label']} | {r['title']} | `{r['composite']}.pdf` | {r['panels']} | {r['scripts']} | {r['tables']} |")
    L += ["", "### Tables", "",
          "The manuscript's machine-readable tables are the sheets of `runs/Supplementary_Data_benchmark.xlsx`, "
          "assembled by `tables.py supplementary_data` from the files below without alteration (every cell is read back and compared). "
          "Column dictionary for the run table: `BENCHMARK_SCHEMA.md`.", "",
          "| Supplementary Data | Sheet | Contents | Source file (relative to `benchmark/`) | Source produced by | Cited in |",
          "|---|---|---|---|---|---|"]
    for r in data_rows():
        L.append(f"| Data {r['n']} | `{r['sheet']}` | {r['title']} | {r['source']} | {r['producer']} | {r['cited']} |")
    L += ["", "### Numbers, Source Data and the other products", "",
          "| Product | Where | Written by |",
          "|---|---|---|",
          "| Every number quoted in the text, as key-value pairs and as a readable page | `runs/KEY_NUMBERS.tsv`, `runs/KEY_NUMBERS.md` | `tables.py key_numbers` |",
          "| The run table (one row per paired run) and its per-stratum summary | `runs/RUNS.tsv` (+ `.csv`), `runs/RUNS_summary.tsv` | `scripts/aggregate_runs.py`, from `runs/paired/<cell>/RESULT.tsv` |",
          "| Per-step wall and CPU time, both arms, every cell | `runs/PHASE_STEPS.tsv` | `scripts/phase_extract.py`, from each cell's IQ-TREE logs |",
          "| Source Data: the values behind every mark of every panel | `figures/source_data/<panel>.csv`, `figures/Source_Data.xlsx` | the panel writer (`lib/figstyle.py`) at panel time; the workbook by `tables.py source_data` |",
          "| Panel inventory: file, figure, panel, title, script, source table, rows used, data date | `figures/FIGURE_MANIFEST.tsv` | the panel writer; audited by `figures.py audit` |",
          "| Workbook presentation (README sheet, headers, widths; values untouched and verified) | both `.xlsx` | `tables.py style_workbooks` |",
          "| This map | `README.md` (this block) | `tables.py benchmark_map` |",
          "", END]
    return "\n".join(L) + "\n"


def splice(text, block):
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0:
        raise SystemExit("benchmark_map.py: benchmark/README.md has no map markers")
    return text[:i] + block + text[j + len(END) + 1:]


def stale(readme):
    """-> [] if the README's generated block equals a fresh render, else one message per difference."""
    text = open(readme).read()
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0:
        return ["benchmark/README.md has no generated map block"]
    have = text[i:j + len(END)].rstrip("\n").splitlines()
    want = render().rstrip("\n").splitlines()
    if have == want:
        return []
    import difflib
    d = [l for l in difflib.unified_diff(have, want, "README.md", "fresh", lineterm="", n=0)
         if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    return [f"map block differs from a fresh render ({len(d)} line(s)); run python3 benchmark/figures_tables/tables.py benchmark_map"] + d[:6]


def main():
    if "--check" in sys.argv:
        bad = stale(README)
        for b in bad:
            print("  " + b)
        print("  map is current" if not bad else "  map is STALE")
        return 1 if bad else 0
    text = open(README).read()
    new = splice(text, render())
    if new != text:
        open(README, "w").write(new)
    nf = len(figure_rows()); nd = len(data_rows())
    print(f"  wrote the map into benchmark/README.md: {nf} figures, {nd} Supplementary Data items")
    return 0


benchmark_map_main = main


# ================================================================================================
# ==== section: crosscheck ====
# ================================================================================================
# crosscheck.py - cross-file checks on the public tree: figures, docs, scripts and the data.
#
# Six checks, every one of them on files that ship in this repository; a failure is caught by
# `run_tests.sh` rather than by a reader:
#
#   1. no public document or live script references material archived out of the public tree;
#   2. benchmark/figures/ holds nothing loose beyond the panels, composites, Source Data,
#      the README and the manifest;
#   3. no panel title in the manifest is phrased as a question;
#   4. no panel title asserts a number that KEY_NUMBERS.tsv does not hold;
#   5. the per-entry arena figure is computed against the allocated capacity;
#   6. the figure and table map in benchmark/README.md is current.
#
# Exit status is non-zero if any check fails. Run: python3 benchmark/figures_tables/tables.py crosscheck
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

KEYNUM = os.path.join(REPO, "benchmark", "runs", "KEY_NUMBERS.tsv")
# The panel generators whose titles are audited against KEY_NUMBERS.tsv (main figures, then supplementary).
FIGURE_SCRIPTS = ("figures.py", "tables.py")           # every panel title is set in these two files

# The PUBLIC tree. These files are what a reader sees first, and a stale figure citation or a
# dangling pointer into archived material in any of them ships with the repository.
PUBLIC_DOCS = ["README.md", "docs/USER_GUIDE.md", "docs/DEPENDENCIES.md", "docs/CONTRIBUTING.md",
               "CHANGELOG.md", "examples/README.md", "scripts/dev/README.md", "validation/README.md",
               "benchmark/README.md", "benchmark/REPRODUCING.md", "benchmark/BENCHMARK_SCHEMA.md",
               "benchmark/figures/README.md", "benchmark/FIGURE_MAP.md", "benchmark/data/README.md",
               "benchmark/data/protein/README.md", "benchmark/data/dna/README.md",
               "benchmark/scripts/harness/README.md", "benchmark/scripts/dev/README.md",
               "benchmark/scripts/README.md", "benchmark/figures_tables/README.md"]

# Everything archived out of the public tree. A reference to any of these is a dangling pointer.
RETIRED_SCRIPTS = ["replicate_variance", "mechanism_figures", "per_step_figures",
                   "phase_decomposition", "complexity_scaling", "benchmark_figures",
                   "status_dashboard", "cpu_production_figure", "all_trees",
                   "tree_figures", "model_coverage", "interim_report",
                   "card_comparison", "paired_alphabet"]

ARCHIVED_PATHS = {
    r"\btrees_all\b": "earlier per-run tree record; replaced by FigS02a_p01..p06",
    r"\bTREE_INDEX\b": "index of the earlier tree record; not distributed",
    r"\bPRODUCTION_READINESS\b": "readiness assessment; not distributed",
    r"INVALIDATED_": "record of a resolved failure; not distributed",
    r"xx_archive_": "archived material; a public document must not link into it",
}


def load_manifest():
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check_public_tree():
    """No live file may reference anything that was archived out of the public tree.

    This is the check that makes the archive safe: a reintroduced reference now fails the build
    instead of shipping. It sweeps the public documents AND the live scripts, because the last
    sweep found dangling names in shell comments and docstrings as well as in prose.
    """
    pats = dict(ARCHIVED_PATHS)
    # Retired generators are matched as script file names, so a live script or figure whose name
    # merely contains the same words (submit_model_coverage.sh, FigS03a_model_coverage_data) is not flagged.
    pats.update({r"(?<![A-Za-z0-9_])" + re.escape(n) + r"\.(?:py|sh|R)\b": "retired generator; archived out of the public tree"
                 for n in RETIRED_SCRIPTS})

    targets = [os.path.join(REPO, f) for f in PUBLIC_DOCS]
    for top in (os.path.join(REPO, "scripts"), os.path.join(REPO, "benchmark", "scripts"),
                os.path.join(REPO, "benchmark", "figures_tables")):
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            targets += [os.path.join(root, f) for f in files
                        if f.endswith((".py", ".sh", ".R", ".md", ".tsv"))]

    # Exemptions, each for a stated reason rather than a blanket skip.
    exempt = {
        os.path.abspath(__file__),                           # this file lists the names (crosscheck section)
        os.path.join(REPO, "benchmark", "REPRODUCING.md"),   # names the archives to explain an absence
    }
    bad = []
    for t in targets:
        if t in exempt or not os.path.exists(t):
            continue
        try:
            text = open(t, errors="ignore").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, why in pats.items():
                if re.search(pat, line):
                    bad.append((os.path.relpath(t, REPO), i, pat, why))
    return bad


def check_figures_dir():
    """benchmark/figures/ must hold nothing loose. This is how the old set accumulated."""
    allowed = {"README.md", "FIGURE_MANIFEST.tsv", "panels", "composites", "source_data", "Source_Data.xlsx", "arial"}
    d = os.path.join(REPO, "benchmark", "figures")
    return sorted(set(os.listdir(d)) - allowed) if os.path.isdir(d) else []


def check_panel_literals():
    """No panel title may assert a number that no table holds.

    A review once found nine quoted values disagreeing with the panels that cite them. Four were a
    stratum drift between key_numbers and the panel generators; the rest were NUMERIC LITERALS typed
    into panel titles -- "the campaign aggregate, 60x" (computed: 68.8), "the six hosts differ by up
    to 1.67x" (computed max: 2.00), "19x in fp64, 3.7x in wall" on a panel that already annotated
    the computed spread, and "the guard degraded 6 runs", a number that appeared in no table at all.

    This walks the AST of every panel generator, pulls the STRING-LITERAL parts of set_title and
    suptitle calls (an f-string's computed parts are exempt by construction -- that is the point),
    and requires each numeric token to exist in KEY_NUMBERS.tsv at some accepted precision.
    """
    import ast
    vals = _keynumber_values()
    ALLOW = {"1", "2", "3", "4", "5", "6", "10", "20", "95", "100"}   # axis words, "20 threads"
    bad = []
    for f in sorted(os.path.join(REPO, "benchmark", "figures_tables", n) for n in FIGURE_SCRIPTS):
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("set_title", "suptitle")):
                continue
            lits = []
            for a in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    lits.append(a.value)
                elif isinstance(a, ast.JoinedStr):
                    lits += [v.value for v in a.values
                             if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            for text in lits:
                for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text):
                    if tok in ALLOW or tok in vals:
                        continue
                    bad.append((os.path.basename(f), node.lineno, tok,
                                " ".join(text.split())[:58]))
    return bad


def _keynumber_values():
    """Every KEY_NUMBERS value at the precisions a title might print it at."""
    vals = set()
    with open(KEYNUM) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 2:
                continue
            v = row[1].strip()
            vals.add(v); vals.add(v.lstrip("+"))
            try:
                fl = float(v)
            except ValueError:
                continue
            for fmt in ("%g", "%.0f", "%.1f", "%.2f", "%.3f"):
                vals.add(fmt % fl)
    return vals


def check_memory_denominator():
    """The per-entry arena figure must be computed against the allocated capacity (R13)."""
    vals = dict(csv.reader(open(KEYNUM), delimiter="\t"))
    v = float(vals.get("arena_bytes_per_entry_median", "nan"))
    return [] if 7.9 <= v <= 8.4 else [f"arena_bytes_per_entry_median = {v} (expected 8.0 +- 0.4)"]


def check_benchmark_map():
    """benchmark/README.md's figure and table tables are generated by benchmark_map.py; they must be current."""
    import importlib.util as ilu
    m = sys.modules[__name__]                           # the benchmark_map section of this file
    readme = os.path.join(REPO, "benchmark", "README.md")
    if not os.path.exists(readme):
        return ["benchmark/README.md is missing"]
    return m.stale(readme)


def check_titles(manifest_rows):
    return [r["file"] for r in manifest_rows if r["title"].strip().endswith("?")]


def main():
    rows = load_manifest()
    fails = 0

    bad = check_public_tree()
    if bad:
        fails += 1
        print(f"  FAIL  live files reference archived material: {len(bad)}")
        for f, i, pat, why in bad[:25]:
            print(f"          {f}:{i}  /{pat}/  -- {why}")
    else:
        print(f"  PASS  no live file references archived material "
              f"({len(PUBLIC_DOCS)} public docs + every script)")

    bad = check_figures_dir()
    if bad:
        fails += 1
        print(f"  FAIL  loose entries directly under benchmark/figures/: {bad}")
    else:
        print("  PASS  benchmark/figures/ holds only panels/, composites/, source_data/, "
              "README, the manifest, Source Data and the optional arial/ fonts")

    bad = check_titles(rows)
    if bad:
        fails += 1
        print(f"  FAIL  panel titles phrased as questions: {bad}")
    else:
        print("  PASS  no panel title is phrased as a question")

    bad = check_panel_literals()
    if bad:
        fails += 1
        print(f"  FAIL  panel titles assert {len(bad)} number(s) absent from KEY_NUMBERS.tsv:")
        for f, ln, tok, ctx in bad[:12]:
            print(f"          {f}:{ln}  '{tok}'  in  \"{ctx}\"")
    else:
        print("  PASS  every number in a panel title is traceable to KEY_NUMBERS.tsv")

    for name, fn in (("arena bytes per entry uses the allocated capacity", check_memory_denominator),
                     ("benchmark/README.md figure and table map is current", check_benchmark_map)):
        bad = fn()
        if bad:
            fails += 1
            print(f"  FAIL  {name}: {len(bad)}")
            for b in bad[:12]:
                print(f"          {b}")
        else:
            print(f"  PASS  {name}")

    print(f"\n  {'CROSS-CHECK FAILED' if fails else 'CROSS-CHECK PASSED'} ({fails} of 6 checks failing)")
    return 1 if fails else 0


crosscheck_main = main


# ================================================================================================
# ==== dispatch ====
# ================================================================================================
SECTIONS = {m: globals()[m + "_main"] for m in ['key_numbers', 'source_data', 'supplementary_data', 'style_workbooks', 'benchmark_map', 'crosscheck']}
ORDER = ["key_numbers", "source_data", "supplementary_data", "benchmark_map", "crosscheck"]
TAKES_OUT = {"source_data": "source_data_OUT", "supplementary_data": "supplementary_data_OUT"}


def _dispatch(argv):
    """`tables.py [section [args...]]`: one section with its arguments, or every section of ORDER."""
    if not argv:
        code = 0
        for n in ORDER:
            r = SECTIONS[n]()
            code = code or (r or 0)
        return code
    n, rest = argv[0], argv[1:]
    if n not in SECTIONS:
        sys.exit(f"tables.py: unknown section {n!r}; choose from {sorted(SECTIONS)}")
    if n in TAKES_OUT and rest:
        globals()[TAKES_OUT[n]] = rest[0]
    sys.argv = [sys.argv[0]] + rest           # the sections read their own arguments from sys.argv
    return SECTIONS[n]() or 0


if __name__ == "__main__":
    sys.exit(_dispatch(sys.argv[1:]))
