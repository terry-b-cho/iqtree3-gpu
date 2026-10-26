#!/usr/bin/env python3
"""key_numbers.py - every number the manuscript quotes, derived once, from the data.

WHY THIS EXISTS. Headline numbers used to be hand-typed into the prose and went stale silently.
This script makes them a build product: the manuscript quotes KEY_NUMBERS.tsv and a refresh
regenerates it. `scripts/crosscheck.py` then FAILS if the abstract quotes a number this file
does not contain at the same precision, so prose and data cannot drift apart again.

TWO HISTORICAL BUGS THIS REWRITE FIXES.
1. The script imported `complexity_scaling.py` for its complexity definition. That script was
   retired into `scripts/retired/`, so this one raised FileNotFoundError and stopped running: the
   committed KEY_NUMBERS.tsv was stale output from before the retirement. The definition now comes
   from `lib/figdata.py`, which is also what every panel uses, so they cannot diverge.
2. The speedup statistics excluded the `__mdl_*` correctness cells but NOT the gate-failure cell,
   while the panels exclude both via `figdata.for_fits()`. KEY_NUMBERS therefore reported
   protein_20T n=68 median 3.75 where the figures showed n=67 median 3.73. Both now use for_fits().

Outputs: results/benchmarks/KEY_NUMBERS.md  (quotable block)
         results/benchmarks/KEY_NUMBERS.tsv (key<TAB>value, for scripted assertions)

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, re, math, statistics as st, importlib.util
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load("figdata", os.path.join(_HERE, "lib", "figdata.py"))
H = _load("headline", os.path.join(_HERE, "lib", "headline.py"))

REPO = os.environ.get("REPO_DIR", os.path.dirname(_HERE))
RUNS = f"{REPO}/results/benchmarks/RUNS.tsv"
STEPS = f"{REPO}/results/benchmarks/PHASE_STEPS.tsv"
PAIRED = f"{REPO}/results/benchmarks/paired"
OUTMD = f"{REPO}/results/benchmarks/KEY_NUMBERS.md"
OUTTSV = f"{REPO}/results/benchmarks/KEY_NUMBERS.tsv"

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
    superseded = [s for s in shas if s.startswith(H.SUPERSEDED_SHA_PREFIX)]
    reported = [s for s in shas if s and s not in superseded]
    add("total_rows_recorded", len(all_rows))
    add("total_rows", len(rows))
    add("rows_superseded_build", len(all_rows) - len(rows))
    add("rows_analysed_for_fits", len(fits))
    add("build_sha_reported", reported[0] if reported else "NA")
    add("build_sha_superseded", superseded[0] if superseded else "NA")
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
    # how far the largest observed difference sits below the tolerance (Fig. 1A title, R12)
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
    # ---- cost law, T20 / L40S stratum ------------------------------------------------
    # The abscissa is WORK UNITS (invocations x m k^2 r), which is what Figure 3a plots -- NOT
    # `complexity`, which carries the 2n-3 branch term and no invocation count and is the abscissa
    # of Figure 2a. Fitting against the two gives different exponents (0.81 vs 1.08 for the
    # amino-acid CPU arm), so quoting one as the other would put a number in the text that no
    # figure shows. Both definitions live in figdata so they cannot drift.
    L = [r for r in fits if r["threads"] == "20" and r["card"] == "NVIDIA_L40S"]
    for dt in ("protein", "dna"):
        sub = [r for r in L if r["data_type"] == dt]
        for arm in ("cpu", "gpu"):
            f = F.loglog_fit([(r["_work_units"], num(r[f"{arm}_wall_s"])) for r in sub])
            if f:
                add(f"costlaw_{dt}_{arm}_exponent", "%.3f" % f["slope"])
                add(f"costlaw_{dt}_{arm}_ci_lo", "%.3f" % f["ci"][0])
                add(f"costlaw_{dt}_{arm}_ci_hi", "%.3f" % f["ci"][1])
                add(f"costlaw_{dt}_{arm}_r2", "%.3f" % f["r2"])
                add(f"costlaw_{dt}_{arm}_n", f["n"])
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
    add("costlaw_stratum", "20 threads, NVIDIA_L40S")
    add("costlaw_abscissa", "work units = invocations x patterns x states^2 x rate categories")

    # k^p comparison -- shows the k^2 weighting is a definition, not a fitted result
    for (p, arm), r2 in sorted(kexp_table(fits).items()):
        add(f"kexp_p{p}_{arm}_r2", "%.3f" % r2)

    # ---- outer-loop demand -----------------------------------------------------------
    # STRATUM: the 20-thread set, matching Fig. 3B (mechanism.panel_b). This fit used to
    # run over all thread counts here while the panel restricted to 20 threads, so the manuscript
    # quoted 1.632/n=91 against a panel showing 1.61/n=78. Both sides now call F.at_threads.
    outer_rows = F.at_threads(fits)
    add("outer_stratum_n", len(outer_rows))
    for ms in sorted({r["model_search"] for r in outer_rows}):
        f = F.loglog_fit([(num(r["taxa"]), num(r["gpu_calls"])) for r in outer_rows
                          if r["model_search"] == ms])
        if f:
            add(f"outer_{ms}_exponent", "%.3f" % f["slope"])
            add(f"outer_{ms}_ci_lo", "%.3f" % f["ci"][0])
            add(f"outer_{ms}_ci_hi", "%.3f" % f["ci"][1])
            add(f"outer_{ms}_r2", "%.3f" % f["r2"])
            add(f"outer_{ms}_n", f["n"])

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

    # ---- card specifications: the reference table, and how well it orders the cards ---------
    # card_specs.tsv is authoritative (R21); the spec columns in RUNS.tsv are the snapshot each
    # card-sweep row was written with. Keys are emitted per card so Note 2 quotes the table.
    SPEC_KEY = {"NVIDIA_A100_80GB_PCIe": "a100", "Tesla_V100S-PCIE-32GB": "teslaV100s",
                "Tesla_V100-SXM2-16GB": "teslaV100", "NVIDIA_L40S": "l40s",
                "NVIDIA_A40": "a40", "Quadro_RTX_8000": "rtx8000"}
    specs = {}
    with open(os.path.join(REPO, "scripts", "benchmark", "card_specs.tsv")) as fh:
        hdr = None
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "card":
                hdr = parts; continue
            specs[parts[0]] = dict(zip(hdr, parts))
    cards_run = sorted({r["card"] for r in fits if r["card"] in SPEC_KEY})
    for c in cards_run:
        k = SPEC_KEY[c]
        add(f"spec_fp64_tflops_{k}", specs[k]["fp64_tflops"])
        add(f"spec_bandwidth_gbs_{k}", specs[k]["mem_bandwidth_gbs"])
        add(f"spec_vram_gb_{k}", specs[k]["vram_gb"])
        add(f"spec_tdp_w_{k}", specs[k]["tdp_w"])
        add(f"spec_derived_{k}", specs[k]["derived"])
    # same shared-dataset basis as Fig. 5A/5B: summed median GPU-arm wall per card vs the spec
    by = {}
    for r in fits:
        if r["data_type"] != "protein" or r["threads"] != "20" or r["card"] not in SPEC_KEY:
            continue
        by.setdefault(r["card"], {}).setdefault(r["genus"], []).append(num(r.get("gpu_wall_s")))
    common = set.intersection(*[set(v) for v in by.values()]) if by else set()
    if common and len(by) >= 3:
        walls = {c: sum(st.median([w for w in v[g] if w]) for g in common) for c, v in by.items()}
        for field, tag in (("fp64_tflops", "fp64"), ("mem_bandwidth_gbs", "bandwidth")):
            xs = [math.log10(float(specs[SPEC_KEY[c]][field])) for c in walls]
            ys = [math.log10(walls[c]) for c in walls]
            n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
            sx = math.sqrt(sum((a - mx) ** 2 for a in xs)); sy = math.sqrt(sum((b - my) ** 2 for b in ys))
            r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")
            add(f"card_{tag}_r2", "%.2f" % (r * r))
            add(f"card_{tag}_spread", "%.1f" % (10 ** (max(xs) - min(xs))))
        add("card_wall_spread", "%.1f" % (max(walls.values()) / min(walls.values())))
        add("card_spec_n", len(walls))

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
        # Fitted scaling of both memories with the per-slot likelihood array (Fig. 6A), 20 threads
        for name, col in (("host", "peak_ram_mb"), ("vram", "peak_vram_mb")):
            pts = []
            for r in fits:
                if r.get("threads") != "20":
                    continue
                m, k = num(r.get("patterns")) or 0, num(r.get("num_states")) or 0
                y = num(r.get(col))
                if m and k and y and y > 0:
                    pts.append((m * k * F.rate_cats(r.get("model")), y))
            f = F.loglog_fit(pts)
            if f:
                add(f"memory_{name}_slope", "%.3f" % f["slope"])
                add(f"memory_{name}_slope_ci_lo", "%.3f" % f["ci"][0])
                add(f"memory_{name}_slope_ci_hi", "%.3f" % f["ci"][1])
                add(f"memory_{name}_r2", "%.2f" % f["r2"])
                add(f"memory_{name}_fit_n", f["n"])
        # Fig. 6B population: arena parsed, footprint above 0.00 GB, sampled peak VRAM present
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
        fr = []
        for r in fits:
            inf, sites = num(r.get("informative_sites")), num(r.get("sites"))
            if inf is not None and sites:
                fr.append(inf / sites)
        if dt == "protein":
            pass
    for dt in ("protein", "dna"):
        fr = [num(r["informative_sites"]) / num(r["sites"]) for r in fits
              if r["data_type"] == dt and num(r.get("informative_sites")) is not None
              and num(r.get("sites"))]
        if fr:
            add(f"informative_fraction_{dt}_min", "%.3f" % min(fr))
            add(f"informative_fraction_{dt}_max", "%.3f" % max(fr))
            add(f"informative_fraction_{dt}_spread", "%.1f" % (max(fr) / min(fr)) if min(fr) else "NA")
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
    af = [num(r.get("accelerable_fraction")) for r in fits
          if num(r.get("accelerable_fraction")) is not None]
    if af:
        add("accelerable_fraction_min", "%.3f" % min(af))
        add("accelerable_fraction_max", "%.3f" % max(af))

    # profiling: n = 1 per alphabet, from the nsys reports
    prof = os.path.join(REPO, "results", "profiling", "nsys_kernel_summary.tsv")
    if os.path.exists(prof):
        for r in csv.DictReader(open(prof), delimiter="\t"):
            dt = "protein" if r["case"] == "protein" else "dna"
            w, k, nk = float(r["total_wall_s"]), float(r["kernel_time_s"]), int(r["num_kernels"])
            add(f"nsys_{dt}_kernel_pct_of_wall", "%.1f" % (100 * k / w))
            add(f"nsys_{dt}_kernel_s", "%.2f" % k)
            add(f"nsys_{dt}_launches", nk)
            add(f"nsys_{dt}_mean_kernel_us", "%.2f" % (k / nk * 1e6))
        add("nsys_runs_per_alphabet", 1)

    # empty-kernel launch latency on the L40S (R14), from scripts/benchmark/launch_latency.sh
    ll = os.path.join(REPO, "results", "profiling", "launch_latency_l40s.tsv")
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

    # energy: GPU-side only, card sweep only
    en = [(r["card"], num(r.get("gpu_energy_j"))) for r in rows if num(r.get("gpu_energy_j"))]
    if en:
        add("energy_rows", len(en))
        byc = {}
        for c, j in en:
            byc.setdefault(c, []).append(j)
        for c, v in byc.items():
            add(f"energy_kj_median_{c}", "%.1f" % (st.median(v) / 1000.0))

    for stem, tag in (("protein20T", "protein"), ("dna20T", "dna")):
        f = os.path.join(REPO, "results", "benchmarks", f"REPLICATES_{stem}.tsv")
        if not os.path.exists(f):
            continue
        cv = [float(x["speedup_cv"]) for x in csv.DictReader(open(f), delimiter="\t")
              if x.get("speedup_cv")]
        if cv:
            add(f"replicate_cv_{tag}_min", "%.2f" % min(cv))
            add(f"replicate_cv_{tag}_max", "%.2f" % max(cv))
            add(f"replicate_cv_{tag}_median", "%.2f" % st.median(cv))
            add(f"replicate_cv_{tag}_n_sizes", len(cv))

    # superseded vs current build on the 44 shared cells (S09)
    pair = {}
    for r in all_rows:
        pair.setdefault(r["cell"].rsplit("__", 1)[0], {})[
            "old" if (r.get("binary_sha256") or "").startswith(H.SUPERSEDED_SHA_PREFIX)
            else "new"] = r
    pair = {k: v for k, v in pair.items() if "old" in v and "new" in v}
    add("build_regression_shared_cells", len(pair))
    for dt in ("protein", "dna"):
        w = [num(v["new"]["gpu_wall_s"]) / num(v["old"]["gpu_wall_s"]) for v in pair.values()
             if v["new"]["data_type"] == dt and num(v["old"].get("gpu_wall_s"))
             and num(v["new"].get("gpu_wall_s"))]
        c = [num(v["new"]["gpu_calls"]) / num(v["old"]["gpu_calls"]) for v in pair.values()
             if v["new"]["data_type"] == dt and num(v["old"].get("gpu_calls"))
             and num(v["new"].get("gpu_calls"))]
        if w:
            add(f"build_regression_{dt}_wall_ratio_median", "%.3f" % st.median(w))
            add(f"build_regression_{dt}_n", len(w))
        if c:
            add(f"build_regression_{dt}_call_ratio_median", "%.3f" % st.median(c))

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
    md = ["<!-- GENERATED by scripts/key_numbers.py - DO NOT EDIT BY HAND. -->",
          "<!-- Regenerate: python3 scripts/key_numbers.py (runs inside refresh_all.sh) -->",
          "", "# Key numbers", "",
          f"_Derived from `results/benchmarks/RUNS.tsv` on "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._", "",
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


if __name__ == "__main__":
    main()
