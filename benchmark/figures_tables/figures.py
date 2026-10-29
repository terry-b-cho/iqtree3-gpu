#!/usr/bin/env python3
"""figures.py - every figure of the benchmark, in one file.

Sections, each a former generator script: equivalence (Fig. 1B, 1C), decision (Fig. 2A),
mechanism (Fig. 2B), phases_cost (Fig. 3B-D), hardware (Supplementary Fig. S1D), supp_ported
(S1C and the replicate table), supplementary (Fig. 1A, Fig. 3A, S1A, S1B, S4), trees (S2, S3),
composites (the lettered figures) and panels (the figure-contract audit). The shared style and
data helpers are in lib/ (figstyle, figdata, headline).

Run one section or all, from the repository root:

    python3 benchmark/figures_tables/figures.py                 # every section, in order
    python3 benchmark/figures_tables/figures.py decision audit  # named sections

Each section writes its panels to benchmark/figures/panels/ (PDF + PNG), its Source Data CSV to
benchmark/figures/source_data/ and its rows of figures/FIGURE_MANIFEST.tsv; `composites` sets the
panels into the lettered figures; `audit` enforces the figure contract (naming, 5-8 pt type, no
DejaVu, manifest agreement). Nothing here runs IQ-TREE: the inputs are the measured record in
benchmark/runs/. The benchmark harness that produced that record lives in benchmark/scripts/.

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
# The publication style (fonts, 6 pt type, spines) is applied by _dispatch() before the first
# figure section draws; importing this file (the audit, the cross-check, the README map) needs
# no font at all.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, MaxNLocator
from matplotlib.patches import Patch
try:
    from Bio import Phylo
except ImportError:                       # only the tree montages need Biopython
    Phylo = None


# ================================================================================================
# ==== section: equivalence ====
# ================================================================================================
# equivalence.py - Figure 1B and 1C: the GPU returns the same answer.
#
#   c  the relative log-likelihood difference between the two arms as an ECDF per alphabet, the
#      difference on the vertical axis against the cumulative fraction of runs, with the 1e-4
#      tolerance as a rule. The bit-identical runs are the level each curve starts at, drawn at a
#      labelled '0' position below the decades: 100 of 182 runs are BIT-IDENTICAL and the rest sit
#      about 50-fold inside the tolerance.
#   b  the per-run maximum UFBoot support difference over shared splits as a dot plot, one dot per
#      run stacked at its value, colour = alphabet, marker = thread count, carrying the topology
#      result (RF = 0 in every run) in one line. No tolerance or threshold is drawn.
#   (Fig. 1A, the campaign-coverage bars, is drawn by supplementary.py : coverage_bars.)
#
# WHAT "MAX UFBOOT SUPPORT DIFFERENCE OVER SHARED SPLITS" MEANS. Each arm writes a consensus tree
# (<arm>.contree) whose internal branches carry ultrafast-bootstrap support (percent of 1000
# bootstrap trees containing that split). The harness (paired_benchmark.sh, equivalence block) takes
# every split present in BOTH arms' consensus trees -- all of them, since RF = 0 -- and the absolute
# difference of the two support values on it; the run's value is the LARGEST of those differences,
# in support points. A run at 0 has identical support on every branch; a run at 2 has one branch
# where the arms differ by two points (say 98 against 100) and no branch differing by more. Panel b
# re-reads the per-split supports from the stored consensus trees with the harness's own parser
# (split_supports) and asserts that the maximum it finds equals the support_max_absdiff column of
# every run before drawing.
#
# MARKER SHAPE IN PANEL B. Elsewhere marker shape carries alphabet beside hue (figstyle). Here the
# two alphabets are already separated by hue and shape is given to THREAD COUNT (triangle = 20
# threads, circle = 1 thread), so the panel shows both strata without a second hue.
#
# Histogram forms of both panels and five further drafts of panel b were retired.
#
# THE GATE-FAILURE CELL IS RETAINED HERE. It is excluded from fits and speedup summaries everywhere
# else, but removing it from the figure whose purpose is to display worst cases would be circular.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

equivalence_SRC = "benchmark/runs/RUNS.tsv"
YLAB = "relative log-likelihood difference\n|$\\Delta$logL| / |logL|"
equivalence_ALPHA = {"protein": "amino acid", "dna": "nucleotide"}
THREAD_MARKER = {"20": "^", "1": "o"}


def _threads(rows):
    return {r["threads"] for r in rows}


def _reldiff(rows):
    """(row, relative difference) for every row that has one."""
    out = []
    for r in rows:
        x = FD.num(r.get("logl_reldiff"))
        if x is not None:
            out.append((r, x))
    return out


def _zero_axis(ax, zx, axis="y", top=FD.LOGL_CONTRACT):
    """A log axis with the zero position labelled '0' below (or left of) the decades."""
    dec = [10.0 ** k for k in range(-12, 0, 2) if zx < 10.0 ** k <= top * 10]
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator([zx] + dec))
    a.set_major_formatter(FixedFormatter(["0"] + [f"10$^{{{int(np.log10(d))}}}$" for d in dec]))
    a.set_minor_formatter(NullFormatter())


def equivalence_panel_a(rows):
    """ECDF of the relative difference, one curve per alphabet, the difference on the vertical
    axis: the curve at cumulative fraction f is the f-quantile of the difference. The bit-identical
    runs are the level each curve starts at, drawn at a labelled '0' position below the decades."""
    fig, ax = FS.square_axes()
    pairs = _reldiff(rows)
    zy = min(x for _, x in pairs if x > 0) * 0.1
    for dt in ("protein", "dna"):
        v = sorted(x for r, x in pairs if r["data_type"] == dt)
        n, nz = len(v), sum(1 for x in v if x == 0)
        fr = [0.0, nz / n] + [(nz + k + 1) / n for k in range(n - nz)]
        vals = [zy, zy] + [x for x in v if x > 0]
        col = FS.ALPHABET[dt]
        # where="pre": the interval (f_{k-1}, f_k] takes the k-th value, i.e. the quantile function
        ax.step(fr, vals, where="pre", color=col, lw=0.9, zorder=3, label=equivalence_ALPHA[dt])
        ax.scatter([nz / n], [zy], s=10, color=col, marker=FS.alphabet_marker(dt), zorder=4)
        # the two notes share the floor; the one anchored at the right is set a line higher
        right = nz / n > 0.5
        ax.annotate(f"{nz} of {n} bit-identical", xy=(nz / n, zy),
                    xytext=(-5 if right else 5, 14 if right else 4), textcoords="offset points",
                    ha="right" if right else "left", va="bottom", color=col)
    ax.axhline(FD.LOGL_CONTRACT, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.annotate("tolerance 1e-4", xy=(0.98, FD.LOGL_CONTRACT), xytext=(0, -2),
                textcoords="offset points", ha="right", va="top", color=FS.MUTED)
    ax.set_yscale("log"); ax.set_ylim(zy * 0.4, FD.LOGL_CONTRACT * 3); ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _zero_axis(ax, zy)
    ax.set_xlabel("cumulative fraction of runs"); ax.set_ylabel(YLAB)
    ax.set_title(FS.stratum(_threads(rows), len(pairs)), loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)   # the curves span the box
    return fig, len(pairs)


# ------------------------------------------------------- per-split supports behind panel b
def _parse_contree(path):
    """(taxa, {canonical split: support}) of a consensus tree. This is the harness's own parser
    (paired_benchmark.sh, equivalence block), ported verbatim so the splits are the same ones
    support_max_absdiff was computed from."""
    if not os.path.exists(path):
        return None, None
    s = open(path, errors="ignore").read().strip()
    if not s:
        return None, None
    if s.endswith(";"):
        s = s[:-1]
    i = 0; leaves = []; splits = []
    def parse():
        nonlocal i
        clade = set(); sup = None
        if s[i] == "(":
            i += 1
            while True:
                child, _ = parse(); clade |= child
                if s[i] == ",": i += 1; continue
                if s[i] == ")": i += 1; break
            j = i
            while i < len(s) and s[i] not in ",():": i += 1
            lab = s[j:i]
            if lab:
                try: sup = float(lab)
                except Exception: sup = None
            if i < len(s) and s[i] == ":":
                i += 1
                while i < len(s) and s[i] not in ",()": i += 1
            splits.append((frozenset(clade), sup)); return clade, sup
        j = i
        while i < len(s) and s[i] not in ",()": i += 1
        leaves.append(s[j:i].split(":")[0]); return {leaves[-1]}, None
    try:
        parse()
    except (IndexError, RecursionError):
        return None, None
    taxa = set(leaves); n = len(taxa)
    if not taxa:
        return None, None
    first = sorted(taxa)[0]; d = {}
    for cl, sup in splits:
        if len(cl) <= 1 or len(cl) >= n - 1:
            continue
        side = cl if first not in cl else frozenset(taxa - cl); d[side] = sup
    return taxa, d


def split_supports(rows):
    """[(row, [(CPU support, GPU support)])] over the splits shared by the two arms' consensus
    trees. The maximum absolute difference per run is asserted equal to the run's
    support_max_absdiff column, so the panel cannot disagree with the run table."""
    out = []
    for r in rows:
        d = os.path.join(PAIRED, r["cell"])
        _, sa = _parse_contree(os.path.join(d, "cpu.contree"))
        _, sb = _parse_contree(os.path.join(d, "gpu.contree"))
        if not sa or not sb:
            continue
        shared = set(sa) & set(sb)
        pairs = [(sa[k], sb[k]) for k in shared if sa[k] is not None and sb[k] is not None]
        if not pairs:
            continue
        mx = max(abs(a - b) for a, b in pairs)
        col = FD.num(r.get("support_max_absdiff"))
        assert col is None or abs(mx - col) < 1e-9, f"{r['cell']}: re-read max {mx} != column {col}"
        out.append((r, pairs))
    return out


def equivalence_panel_b(rows):
    """The per-run maximum as a dot plot: one dot per run, stacked at its value. Colour is the
    alphabet; marker is the thread count (triangle 20, circle 1)."""
    sp = split_supports(rows)
    fig, ax = FS.square_axes()
    vals = collections.defaultdict(list)
    for r, pairs in sp:
        vals[max(abs(a - b) for a, b in pairs)].append(r)
    cols = 5
    order = lambda r: (r["data_type"] != "protein", r["threads"] != "20")
    for v, rs in sorted(vals.items()):
        for k, r in enumerate(sorted(rs, key=order)):
            ax.scatter([v + (k % cols - (cols - 1) / 2) * 0.12], [k // cols + 1], s=6,
                       color=FS.ALPHABET[r["data_type"]], marker=THREAD_MARKER[r["threads"]],
                       zorder=3, linewidths=0)
        ax.annotate(str(len(rs)), xy=(v, (len(rs) - 1) // cols + 1), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", color=FS.MUTED)
    for dt in ("protein", "dna"):
        for th in ("20", "1"):
            n_s = sum(1 for r, _ in sp if r["data_type"] == dt and r["threads"] == th)
            ax.scatter([], [], s=6, color=FS.ALPHABET[dt], marker=THREAD_MARKER[th],
                       label=f"{equivalence_ALPHA[dt]}, {th}T ({n_s})")
    rf0 = sum(1 for r in rows if (FD.num(r.get("rf_dist")) or 0) == 0)
    ax.annotate(f"every topology identical:\nRF = 0 in {rf0} of {len(rows)}",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                color=FS.INK, style="italic")
    ax.set_xticks(sorted(vals)); ax.set_ylim(0, max(len(d) for d in vals.values()) / cols + 4)
    ax.set_yticks([])
    ax.set_xlabel("max UFBoot support difference\nover shared splits (points)")
    ax.set_ylabel("runs (one dot each)")
    ax.set_title(FS.stratum(_threads(rows), len(sp)), loc="left")
    ax.legend(loc="center right")
    return fig, len(sp)


def main():
    rows = FD.for_equivalence(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    # Fig. 1: a = campaign coverage (supplementary.py: coverage_bars), b = support dot plot,
    # c = log-likelihood ECDF
    for fn, letter, slug, title in (
            (equivalence_panel_b, "b", "support_diff_dotplot",
             "UFBoot support difference per run, with the RF result"),
            (equivalence_panel_a, "c", "logl_reldiff_ecdf", "Relative log-likelihood difference vs tolerance, ECDF")):
        fig, n = fn(rows)
        stem = FS.save_panel(fig, 1, letter, slug, title, "figures.py (equivalence)", equivalence_SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


equivalence_main = main


# ================================================================================================
# ==== section: decision ====
# ================================================================================================
# decision.py - Figure 2A: when to offload (left: 20 threads; right: one thread).
#
# This is the figure a practitioner came for. It answers "should I add -gpu to this analysis?" from a
# quantity they can compute before submitting the job.
#
# THE TWO PANELS SHARE ONE SCALE. Both are drawn with the same x range, the same y range, the same
# ticks and the same top axis (`limits`, computed once from every fitted row), so the 20-thread and
# the one-thread case can be read side by side in the composite at the same length per decade.
#
# TWO THINGS THE EARLIER DRAFT GOT WRONG, both fixed here:
#   * The crossover was quoted from a DISCRETE estimator (geometric midpoint of the parity gap) whose
#     point estimate sat on its own lower bound -- an artefact of granularity. The estimator here is
#     OLS of log10(speedup) on log10(complexity) with the crossover read off the fitted line, named in
#     the caption, with a paired bootstrap CI.
#   * The crossover is NOT nucleotide-only. Two amino-acid runs at 20 threads fall below parity, and
#     the amino-acid crossover is estimable ~2 decades below the nucleotide one. A protein user with a
#     small alignment is exactly the reader who needs this figure.
#
# Each panel marks where each workflow's fitted line crosses parity and writes that abscissa (the
# tree complexity at which the GPU arm starts to win) at the foot of the axis; the shaded band is the
# 95% bootstrap interval of that crossover.
#
# Points are NEVER joined by lines: each is a different dataset, and a line between them would imply a
# trajectory that does not exist.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

decision_SRC = "benchmark/runs/RUNS.tsv"

# Alphabet is PERFECTLY ALIASED with model treatment in this campaign (85/85 amino-acid runs use
# ModelFinder, 97/97 nucleotide runs pin a model). Nothing here can separate the two, so every label
# names the WORKFLOW. See figdata.workflow().
decision_WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}

# Secondary abscissa: complexity as dataset shapes, so the axis is usable without arithmetic.
# Labels are single-line and terse; the third is raised a line because it abuts the second.
EXAMPLES = [(17 * 1_000 * 4 * 400, "10$\\times$1k"),
            (97 * 10_000 * 4 * 400, "50$\\times$10k"),
            (397 * 50_000 * 4 * 400, "200$\\times$50k\n")]
RATIO_TICKS = (0.5, 1, 2, 5, 10, 20)


def decision_limits(rows):
    """One x range and one y range for BOTH panels, from every fitted row of either thread count."""
    pts = [(r["_complexity"], r["_speedup"]) for r in rows if r["_complexity"] and r["_speedup"]]
    xs, ys = [a for a, _ in pts], [b for _, b in pts]
    return (min(xs) / 2.5, max(xs) * 2.5), (min(ys) / 1.4, max(ys) * 1.4)


def decision__sci(v):
    """1.1e+07 -> 1.1 x 10^7, as the journal sets it."""
    m, e = f"{v:.1e}".split("e")
    return f"{m}$\\times$10$^{{{int(e)}}}$"


def _decision_panel(ax, rows, threads, xlim, ylim):
    """Speedup vs complexity with the parity line, the fitted line per workflow, its crossover
    marked and labelled at the axis foot, and the shaded 95% bootstrap CI of that crossover."""
    stats, handles = {}, []
    for dt in ("protein", "dna"):
        pts = [(r["_complexity"], r["_speedup"]) for r in rows
               if r["data_type"] == dt and r["threads"] == threads
               and r["_complexity"] and r["_speedup"]]
        if len(pts) < 5:
            continue
        col = FS.ALPHABET[dt]
        handles.append(ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=col,
                                  zorder=3, marker=FS.alphabet_marker(dt),
                                  label=f"{decision_WF_SHORT[dt]} (n={len(pts)})"))
        c = FD.crossover(pts)
        if c:
            stats[dt] = c
            xr = np.logspace(np.log10(min(a for a, _ in pts)), np.log10(max(a for a, _ in pts)), 50)
            ax.plot(xr, 10 ** (np.log10(xr) * c["slope"] +
                               (np.log10(c["crossover"]) * -c["slope"])), color=col, lw=0.9, zorder=2)
            ax.axvspan(c["ci"][0], c["ci"][1], color=col, alpha=0.10, lw=0, zorder=0)
            ax.axvline(c["crossover"], color=col, lw=0.7, ls=":", zorder=1)
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # a bare log axis labels only decades; label the ratios readers use
    tk = [t for t in RATIO_TICKS if ylim[0] <= t <= ylim[1]]
    ax.yaxis.set_major_locator(FixedLocator(tk))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}x" for t in tk]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # The break-even: the fitted line's crossing of parity, ringed, with its abscissa written at
    # the foot of the axis beside the dotted rule, so the complexity value can be read directly.
    for dt, c in stats.items():
        col = FS.ALPHABET[dt]
        ax.scatter([c["crossover"]], [1.0], s=20, facecolors="white", edgecolors=col,
                   linewidths=0.9, zorder=6)
        ax.annotate(decision__sci(c["crossover"]), xy=(c["crossover"], ylim[0]), xytext=(2, 2),
                    textcoords="offset points", rotation=90, ha="left", va="bottom", color=col)
    # Region labels in the two corners that are empty in BOTH panels under the shared ranges:
    # top-left for the GPU-faster half, bottom-right for the CPU-faster half. Beside the parity
    # line at the left edge they sat on the smallest nucleotide runs of the 20-thread panel.
    ax.text(0.03, 0.97, "IQ-TREE3-GPU faster", transform=ax.transAxes, color=FS.MUTED, style="italic",
            ha="left", va="top")
    ax.text(0.97, 0.03, "IQ-TREE3 faster", transform=ax.transAxes, color=FS.MUTED, style="italic",
            ha="right", va="bottom")
    top = ax.secondary_xaxis("top")
    top.set_xticks([c for c, _ in EXAMPLES]); top.set_xticklabels([l for _, l in EXAMPLES])
    top.set_xlabel("example amino-acid analyses, taxa x patterns (LG+G4)", labelpad=1)
    ax.set_xlabel("tree complexity   branches x patterns x rate cat x states$^2$")
    ax.set_ylabel("speedup  (IQ-TREE3 wall / IQ-TREE3-GPU wall)")
    handles.append(Patch(facecolor=FS.MUTED, alpha=0.18, lw=0,
                         label="shaded: 95% bootstrap CI of the break-even"))
    # The legend hangs UNDER the axes: in a square plotting box the point clouds fill every corner.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    n = sum(1 for r in rows if r["threads"] == threads and r["_complexity"] and r["_speedup"])
    ax.set_title(FS.stratum(int(threads), n), loc="left", pad=18)     # pad clears the top axis
    return stats, n


def panel(rows, threads, xlim, ylim):
    fig, ax = FS.square_axes()
    st, n = _decision_panel(ax, rows, threads, xlim, ylim)
    return fig, n, st


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    xlim, ylim = decision_limits(rows)
    out = []
    # both boxes are Fig. 2A (left: 20 threads, right: one thread), so both carry the letter a
    fa, na, st20 = panel(rows, "20", xlim, ylim); out.append((fa, "a", "decision_t20", "Speedup vs complexity, 20 threads", na))
    fb, nb, st1 = panel(rows, "1", xlim, ylim);   out.append((fb, "a", "decision_t1", "Speedup vs complexity, 1 thread", nb))
    for fig, letter, slug, title, n in out:
        stem = FS.save_panel(fig, 2, letter, slug, title, "figures.py (decision)", decision_SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for th, st in (("20T", st20), ("1T", st1)):
        for dt, s in st.items():
            print(f"    {th} {dt:8s} crossover {s['crossover']:.3e} "
                  f"CI [{s['ci'][0]:.2e}, {s['ci'][1]:.2e}] slope {s['slope']:.3f} "
                  f"R2 {s['r2']:.2f} n={s['n']}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


decision_main = main


# ================================================================================================
# ==== section: mechanism ====
# ================================================================================================
# mechanism.py - Figure 2B: runtime scaling with total likelihood workload.
#
# Two panels, one message: GPU wall time grows more slowly than CPU wall time as the total likelihood
# workload increases. THE TWO PANELS SHARE ONE SCALE (`limits`, from every fitted row of either
# thread count), so the 20-thread and the one-thread cost law can be read side by side at the same
# length per decade.
#
# ENCODING. Marks are circles only: open + dashed = IQ-TREE3 (the CPU arm), filled + solid =
# IQ-TREE3-GPU (the GPU arm); hue = alphabet. The legend under each panel has one entry per series,
# drawn as the series is and carrying its slope at the smallest and the largest work units; R^2 and the paired slope difference
# sit in the table under the legend. Nothing is written on the plot itself.
#
# THE FIT IS QUADRATIC IN LOG-LOG, NOT A STRAIGHT LINE. Every series is concave-up on this scale
# (the curvature term is positive with a bootstrap interval excluding zero in seven of the eight
# series): small jobs scale more gently than large ones and the local exponent rises towards 1 at
# the largest workloads. A straight line would report one average exponent and hide that, so the
# curve drawn is figdata.quadratic_fit and the exponent is reported at the SMALLEST and the LARGEST
# measured workload ("0.28 to 0.91"). The straight-line average is still in KEY_NUMBERS
# (costlaw_*_exponent) beside the end exponents (costlaw_*_exponent_smallest / _largest).
#
#   a  the cost law at 20 threads: wall time of each arm against work units over every 20-thread
#      run, all six cards pooled, fitted log-log, with the explained variation (R^2) of every fit
#      stated on the panel. Thread count was already stratified.
#   b  the same law at one thread, the stratum where the largest speedups occur.
#
# WORK UNITS = likelihood calls x m k^2 r: the number of node-level likelihood calls the search made,
# times the per-call work (patterns x states^2 x rate categories). The definition lives in
# lib/figdata.work_units() and nowhere else. Every exponent carries n and a paired bootstrap CI, and
# whether the GPU exponent really is below the CPU exponent is tested as a PAIRED difference on the
# same cells (figdata.slope_difference), not by comparing two marginal CIs, which would wrongly say
# "no" for nucleotide.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

mechanism_SRC = "benchmark/runs/RUNS.tsv"
XLABEL = "likelihood calls  x  m k$^2$ r   (work units)"
FIT_HEADER = ("slope: of the quadratic log-log fit, at the smallest\n"
              "to the largest work units\n"
              "R$^2$: explained variation, IQ-TREE3 / IQ-TREE3-GPU\n"
              "difference: IQ-TREE3 $-$ IQ-TREE3-GPU slope, paired")


ARM_LABEL = {"cpu": "IQ-TREE3", "gpu": "IQ-TREE3-GPU"}


XLIM = (1e7, 1e17)       # work units: fixed, the same decades in both panels
YMAX = 1e8               # wall time (s): fixed top; the bottom follows the data


def mechanism_limits(rows):
    """One x range and one y range for BOTH panels: x and the top of y are fixed (XLIM, YMAX) so
    the two thread counts share the same decades; the bottom of y comes from the data."""
    ys = [FD.num(r.get(k)) for r in rows for k in ("cpu_wall_s", "gpu_wall_s")]
    ys = [y for y in ys if y]
    return XLIM, (min(ys) / 2.5, YMAX)


def _arm_fits(ax, sub, dt, fits):
    """Scatter + fitted line per arm for one alphabet; records the fits in `fits[dt]`.
    Circles only: open + dashed for IQ-TREE3, filled + solid for IQ-TREE3-GPU; hue = alphabet."""
    col = FS.ALPHABET[dt]
    d = fits.setdefault(dt, {})
    for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
        pts = [(FD.work_units(r), FD.num(r.get(key))) for r in sub]
        pts = [(a, b) for a, b in pts if a and b]
        if len(pts) < 5:
            continue
        st = FS.arm_style(arm, col)
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=4, zorder=3, marker="o",
                   facecolors=st["markerfacecolor"], edgecolors=col, linewidths=0.5)
        f = FD.quadratic_fit(pts)
        xr, yr = FD.quadratic_curve(f)
        ax.plot(xr, yr, color=col, lw=0.9, ls=st["linestyle"], zorder=2)
        d[arm] = f           # the exponents are keyed in the legend, not written on the plot
    # Is the GPU exponent really below the CPU exponent? Test the PAIRED difference on the same
    # cells -- comparing the two marginal CIs above would say "no" for nucleotide, wrongly.
    pc = [(FD.work_units(r), FD.num(r.get("cpu_wall_s"))) for r in sub]
    pg = [(FD.work_units(r), FD.num(r.get("gpu_wall_s"))) for r in sub]
    keep = [i for i in range(len(pc)) if pc[i][0] and pc[i][1] and pg[i][1]]
    if len(keep) >= 5:
        d["diff"] = FD.end_slope_difference([pc[i] for i in keep], [pg[i] for i in keep])


def _arm_legend(ax, fits):
    """Under the axes: one entry per series, drawn exactly as the series is (hue = alphabet,
    open + dashed = IQ-TREE3, filled + solid = IQ-TREE3-GPU) and carrying that series' local
    exponent at the smallest and the largest workload. The legend is therefore the key AND the
    exponent table; nothing is written on the plot itself."""
    from matplotlib.lines import Line2D
    h = []
    for dt in ("protein", "dna"):
        if dt not in fits:
            continue
        col = FS.ALPHABET[dt]
        for arm in ("cpu", "gpu"):
            f = fits[dt].get(arm)
            if not f:
                continue
            st = FS.arm_style(arm, col)
            h.append(Line2D([], [], color=col, marker="o", markerfacecolor=st["markerfacecolor"],
                            markeredgecolor=col, ls=st["linestyle"], lw=0.9, markersize=2.5,
                            label=f"{'amino acid' if dt == 'protein' else 'nucleotide'}, "
                                  f"{ARM_LABEL[arm]}: slope {f['slope_lo']:.2f} to {f['slope_hi']:.2f}"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.8)


def _footnote(fig, ax, fits):
    """The fit summary as a two-column strip UNDER the axes. Inside the axes it has nowhere to go:
    a log-log cost law fills its diagonal and a seven-line block collides with data in every
    corner. The strip hangs 6 pt below the x-axis label, measured from a draw, so it keeps its
    place whatever the tick labels take. Returns the summary lines for the console."""
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    r = fig.canvas.get_renderer()
    # left edge: the y-axis decorations (tick labels and label), so the strip spans the panel;
    # top edge: 6 pt under the legend, which hangs under the x-axis label
    x0 = ax.yaxis.get_tightbbox(r).x0 / fig.dpi / fw
    low = min(ax.xaxis.label.get_window_extent(r).y0,
              ax.get_legend().get_window_extent(r).y0 if ax.get_legend() else 1e9)
    y = low / fig.dpi / fh - 6.0 / 72 / fh
    hdr = fig.text(x0, y, FIT_HEADER, va="top", ha="left", color=FS.MUTED)
    y_body = hdr.get_window_extent(r).y0 / fig.dpi / fh - 4.0 / 72 / fh    # under the header
    out = []
    x = x0
    for k, dt in enumerate(d for d in ("protein", "dna") if d in fits):
        d = fits[dt]
        lines = [f"{'amino acid' if dt == 'protein' else 'nucleotide'} (n = {d['cpu']['n']})",
                 f"R$^2$ {100 * d['cpu']['r2']:.1f}% / {100 * d['gpu']['r2']:.1f}%"]
        if "diff" in d:
            sd = d["diff"]
            z = lambda v: round(v, 2) + 0.0          # no "-0.00"
            lines.append(f"difference {z(sd['diff_lo']):+.2f} to {z(sd['diff_hi']):+.2f}")
        t = fig.text(x, y_body, "\n".join(lines), va="top", ha="left")
        # the next column starts 8 pt right of this one's measured extent
        x = t.get_window_extent(r).x1 / fig.dpi / fw + 8.0 / 72 / fw
        out += [f"{'AA' if dt == 'protein' else 'NT'} {l}" for l in lines[1:]]
    return out


def mechanism_panel_a(rows, xlim, ylim):
    """Cost law at 20 threads: one fit per arm and alphabet over every run, all six cards pooled."""
    fig, ax = FS.square_axes()
    fits = {}
    for dt in ("protein", "dna"):
        # every 20-thread run, all six cards pooled: one fit per arm and alphabet
        _arm_fits(ax, [r for r in rows if r["data_type"] == dt and r["threads"] == "20"], dt, fits)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("wall time (s)")
    n = sum(1 for r in rows if r["threads"] == "20")
    ax.set_title(FS.stratum(20, n), loc="left")
    _arm_legend(ax, fits)
    return fig, n, _footnote(fig, ax, fits)


def mechanism_panel_b(rows, xlim, ylim):
    """Cost law at one thread: does the sub-unity GPU exponent hold where the speedups are largest?"""
    fig, ax = FS.square_axes()
    fits = {}
    n = 0
    for dt in ("protein", "dna"):
        sub = [r for r in rows if r["data_type"] == dt and r["threads"] == "1"]
        n += len(sub)
        _arm_fits(ax, sub, dt, fits)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(XLABEL); ax.set_ylabel("wall time (s)")
    ax.set_title(FS.stratum(1, n), loc="left")
    _arm_legend(ax, fits)
    return fig, n, _footnote(fig, ax, fits)


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    xlim, ylim = mechanism_limits(rows)
    fa, na, la = mechanism_panel_a(rows, xlim, ylim)
    fb, nb, lb = mechanism_panel_b(rows, xlim, ylim)
    # both boxes are Fig. 2B (left: 20 threads, right: one thread), so both carry the letter b
    out = [(fa, "b", "cost_law_t20", "Cost law at 20 threads, all cards pooled", na),
           (fb, "b", "cost_law_t1", "Cost law at one thread", nb)]
    for fig, letter, slug, title, n in out:
        stem = FS.save_panel(fig, 2, letter, slug, title, "figures.py (mechanism)", mechanism_SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for tag, lines in (("2b-left", la), ("2b-right", lb)):
        for l in lines:
            print(f"    {tag} {l}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


mechanism_main = main


# ================================================================================================
# ==== section: phases_cost ====
# ================================================================================================
# phases_cost.py - Figure 3B-D: where the time goes, and what it costs (3A, the phase-step
# heatmap, is drawn by supplementary.py).
#
# Carries the campaign's single most explanatory fact: model selection is ~88% of amino-acid wall time
# and is absent for nucleotide runs, which is most of why the two alphabets behave so differently.
#
#   a  two half-width boxes that together fill one 40 mm square: left, wall share by phase
#      (IQ-TREE3) per alphabet as stacked bars; right, the per-run processor time consumed,
#      IQ-TREE3 / IQ-TREE3-GPU, as a violin with jittered circles per alphabet
#   b  per-phase speedup against tree complexity: model selection (amino acid only, since nucleotide
#      runs pin a model), tree search on amino acid, tree search on nucleotide, each with its OLS
#      fit, its break-even on the parity line written at the axis foot, and the shaded 95% bootstrap
#      CI of that break-even, in the style of Fig. 2
#   c  cores kept busy (processor time / wall time) per build and alphabet as violins with jittered
#      circles (the former 4D).
#
# COLOUR. Panels a and b colour the PHASE (figstyle.PHASE_COLORS: bluish green = model selection,
# reddish purple = tree search, grey = untimed), the same in both panels and distinct from the blue /
# vermillion pair that means alphabet everywhere else. Panel c colours the alphabet. Marks are
# circles only: filled = amino acid in b; in c filled = IQ-TREE3-GPU, open = IQ-TREE3.
#
# TWO HONESTY CONSTRAINTS, both enforced here:
#   * Device time CANNOT be shown. `gpu_kernel_time_s` is NA in all 182 rows. What is
#     reportable is the bound from the profiling kernel fraction, and its DIRECTION matters: profiling
#     inflates host wall more than device kernel time, so the profiled fraction is a LOWER bound on the
#     unprofiled one, and the absolute kernel seconds are the usable number.
#   * The compute saving is POTENTIAL, not realised. The allocation was not node-exclusive but did
#     request -c 20 for both arms, so the cores the GPU arm leaves idle were reserved to our own job
#     and unavailable to others. The saving is realisable only if a GPU user requests fewer cores.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

phases_cost_SRC = "benchmark/runs/PHASE_STEPS.tsv"
phases_cost_ALPHA = {"protein": "amino acid", "dna": "nucleotide"}
BUILD = {"cpu": "IQ-TREE3", "gpu": "IQ-TREE3-GPU"}
PH = {"modelfinder": ("model selection", FS.PHASE_COLORS["modelfinder"]),
      "tree_search": ("tree search", FS.PHASE_COLORS["tree_search"]),
      "unattributed": ("untimed by IQ-TREE", FS.PHASE_COLORS["other"])}


def phases(mod, keep):
    """(cell, arm) -> {step: (wall, cpu)} for top-level steps, restricted to `keep` cells."""
    out = collections.defaultdict(dict)
    with open(os.path.join(mod, "benchmark", "runs", "PHASE_STEPS.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["cell"] not in keep:
                continue
            out[(r["cell"], r["arm"])][r["step"]] = (FD.num(r["wall_s"]), FD.num(r["cpu_s"]))
    return out


def _log_violin(ax, x, v, col, filled=True, rng=None):
    """Violin + jittered circles of a RATIO on a log axis: the density is estimated on log10
    values (a KDE on the raw ratios would be distorted by the scale) and drawn in log10 space,
    the axis then labelled in ratios."""
    lv = np.log10(v)
    vp = ax.violinplot([lv], positions=[x], widths=0.8, showextrema=False, showmedians=False)
    for body in vp["bodies"]:
        body.set_facecolor(col); body.set_edgecolor("none"); body.set_alpha(0.22)
    ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), lv, s=6, marker="o", alpha=0.55,
               facecolors=col if filled else "none", edgecolors=col, linewidths=0.6, zorder=3)
    med = st.median(v)
    ax.hlines(np.log10(med), x - 0.3, x + 0.3, color=FS.INK, lw=1.0, zorder=4)
    return med


def _ratio_axis(ax, values=(0.25, 0.5, 1, 2, 5, 10, 20, 50, 100)):
    lo, hi = ax.get_ylim()
    tk = [t for t in values if lo <= np.log10(t) <= hi]
    ax.set_yticks([np.log10(t) for t in tk]); ax.set_yticklabels([f"{t:g}x" for t in tk])


def phases_cost_panel_a(ph, rows):
    """Left box: wall share by phase (IQ-TREE3 build), per alphabet. Right box: per-run processor
    time consumed, IQ-TREE3 / IQ-TREE3-GPU, per alphabet."""
    fig, (ax, bx) = FS.split_axes(2, gap=0.72)      # two half boxes = one 40 mm square
    axes = np.array([[ax, bx]], dtype=object)
    rng = np.random.default_rng(7)
    labs, n_used = [], 0
    for i, dt in enumerate(("protein", "dna")):
        cells = [r["cell"] for r in rows if r["data_type"] == dt and r["threads"] == "20"]
        shares = collections.defaultdict(list)
        ratio = []
        for c in cells:
            v = ph.get((c, "cpu"), {})
            tot = (v.get("total") or (None,))[0]
            if tot:
                for s in PH:
                    shares[s].append(((v.get(s) or (0,))[0] or 0) / tot * 100)
            a = (v.get("total") or (None, None))[1]
            b = (ph.get((c, "gpu"), {}).get("total") or (None, None))[1]
            if a and b and b > 0:
                ratio.append(a / b)
        bottom = 0
        for s, (lab, col) in PH.items():
            m = st.mean(shares[s]) if shares[s] else 0.0
            ax.bar(i, m, 0.5, bottom=bottom, color=col, edgecolor="white", linewidth=0.5)
            if m > 6:
                ax.text(i, bottom + m / 2, f"{m:.0f}%", ha="center", va="center", color=FS.INK,
                        fontweight="bold")
            bottom += m
        short = {"protein": "amino\nacid", "dna": "nucleo-\ntide"}[dt]     # half-width boxes
        labs.append(f"{short}\n(n={len(cells)})")
        n_used += len(cells)
        # right box: the ratio as a violin, its median written above the violin
        med = _log_violin(bx, i, ratio, FS.ALPHABET[dt], rng=rng)
        bx.annotate(f"median {med:.1f}x", xy=(i, np.log10(max(ratio))), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", color=FS.ALPHABET[dt])
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs)
    ax.set_ylabel("share of IQ-TREE3 wall time (%)"); ax.set_ylim(0, 108)
    for s, (lab, col) in PH.items():
        ax.bar([0], [0], color=col, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    bx.axhline(0.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    bx.set_xticks([0, 1]); bx.set_xticklabels(labs); bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(bx.get_ylim()[0], bx.get_ylim()[1] + 0.25)     # headroom for the median labels
    _ratio_axis(bx)
    bx.set_ylabel("processor time consumed\nIQ-TREE3 / IQ-TREE3-GPU")
    FS.grid_note(fig, axes, FS.stratum(20, n_used))
    return fig, n_used


def phases_cost__sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}$\\times$10$^{{{int(e)}}}$"


def phases_cost_panel_b(ph, rows):
    """Per-phase speedup against tree complexity: model selection (amino acid), tree search on
    amino acid, tree search on nucleotide, each its own series with an OLS fit of log speedup on
    log complexity, the break-even ringed on the parity line, written at the axis foot and keyed
    in the legend, and
    the shaded 95% bootstrap CI of that break-even (figdata.crossover, as in Fig. 2). Hue =
    phase (as in a); fill = alphabet (filled amino acid, open nucleotide)."""
    from matplotlib.patches import Patch
    fig, ax = FS.square_axes()
    n_used, handles, stats = 0, [], []
    by = {r["cell"]: r for r in rows if r["threads"] == "20"}
    for s, dt, tag in (("modelfinder", "protein", "MS, AA"), ("tree_search", "protein", "TS, AA"),
                       ("tree_search", "dna", "TS, NT")):
        lab, col = PH[s]
        pts = []
        for c, r in by.items():
            if r["data_type"] != dt:
                continue
            a = (ph.get((c, "cpu"), {}).get(s) or (None,))[0]
            b = (ph.get((c, "gpu"), {}).get(s) or (None,))[0]
            if a and b and b > 0 and r["_complexity"]:
                pts.append((r["_complexity"], a / b))
        if not pts:
            continue
        n_used += len(pts)
        h = ax.scatter([x for x, _ in pts], [y for _, y in pts], s=7, marker="o", alpha=0.7,
                       zorder=3, facecolors=col if dt == "protein" else "none", edgecolors=col,
                       linewidths=0.6, label=f"{lab}, {phases_cost_ALPHA[dt]} (n={len(pts)})")
        handles.append(h)
        cr = FD.crossover(pts)
        if cr:
            # the break-even value is keyed in the legend entry of its series: three values
            # written on the plot collided with each other and with the data
            h.set_label(f"{lab}, {phases_cost_ALPHA[dt]} (n={len(pts)}): break-even {phases_cost__sci(cr['crossover'])}")
            xr = np.logspace(np.log10(min(x for x, _ in pts)), np.log10(max(x for x, _ in pts)), 50)
            ax.plot(xr, 10 ** (np.log10(xr) * cr["slope"] - np.log10(cr["crossover"]) * cr["slope"]),
                    color=col, lw=0.9, ls="-" if dt == "protein" else "--", zorder=2)
            ax.axvspan(cr["ci"][0], cr["ci"][1], color=col, alpha=0.10, lw=0, zorder=0)
            ax.axvline(cr["crossover"], color=col, lw=0.7, ls=":", zorder=1)
            stats.append((cr, col, tag))
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    xlim = (min(r["_complexity"] for r in by.values() if r["_complexity"]) / 2.5,
            max(r["_complexity"] for r in by.values() if r["_complexity"]) * 2.5)
    ax.set_xlim(*xlim)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo / 2.4, hi)      # a band under the data for the labels
    FS.ratio_ticks(ax)
    # The break-even abscissa at the axis foot beside its dotted rule, as in Fig. 2: the value
    # alone (its series is the colour, and the legend repeats the value with the series name),
    # since the two amino-acid values sit 0.4 decade apart and a longer label collided.
    for cr, col, tag in stats:
        ax.scatter([cr["crossover"]], [1.0], s=20, facecolors="white", edgecolors=col,
                   linewidths=0.9, zorder=6)
        ax.annotate(phases_cost__sci(cr["crossover"]), xy=(cr["crossover"], ax.get_ylim()[0]), xytext=(2, 2),
                    textcoords="offset points", rotation=90, ha="left", va="bottom", color=col)
    ax.set_xlabel("tree complexity   branches x patterns x rate cat x states$^2$")
    ax.set_ylabel("phase speedup\nIQ-TREE3 wall / IQ-TREE3-GPU wall")
    handles.append(Patch(facecolor=FS.MUTED, alpha=0.18, lw=0,
                         label="shaded: 95% bootstrap CI of the break-even"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    ax.set_title(FS.stratum(20, n_used, "phase-pairs"), loc="left")
    return fig, n_used


def panel_c(ph, rows):
    """Cores kept busy per build and alphabet: violin + jittered circles, open = IQ-TREE3,
    filled = IQ-TREE3-GPU, hue = alphabet."""
    fig, ax = FS.square_axes()
    rng = np.random.default_rng(7)
    pos, n_used = {}, 0
    for i, dt in enumerate(("protein", "dna")):
        col = FS.ALPHABET[dt]
        cells = [r["cell"] for r in rows if r["data_type"] == dt and r["threads"] == "20"]
        busy = {}
        for c in cells:
            for arm in ("cpu", "gpu"):
                w, cp = ph.get((c, arm), {}).get("total") or (None, None)
                if w and cp and w > 0:
                    busy.setdefault(arm, []).append(cp / w)
        for j, arm in enumerate(("cpu", "gpu")):
            v = busy.get(arm, [])
            if not v:
                continue
            x = 2.4 * i + j
            pos[x] = BUILD[arm]
            vp = ax.violinplot([v], positions=[x], widths=0.8, showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_edgecolor("none"); body.set_alpha(0.22)
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=6, marker="o", alpha=0.55,
                       facecolors=col if arm == "gpu" else "none", edgecolors=col, linewidths=0.6,
                       zorder=3)
            ax.hlines(st.median(v), x - 0.3, x + 0.3, color=FS.INK, lw=1.0, zorder=4)
        n_used += len(cells)
        ax.text(2.4 * i + 0.5, 23.4, f"{phases_cost_ALPHA[dt]} (n={len(cells)})", ha="center", va="top",
                color=col)
    ax.axhline(20, color=FS.MUTED, lw=0.7, ls=":", zorder=1)
    ax.annotate("20 cores requested", xy=(1.7, 20), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", color=FS.MUTED)
    xs = sorted(pos)
    ax.set_xticks(xs); ax.set_xticklabels([pos[x] for x in xs], rotation=30, ha="right")
    ax.set_ylabel("cores kept busy  (processor time / wall time)")
    ax.set_xlim(-0.6, 4.0); ax.set_ylim(0, 24)
    ax.set_title(FS.stratum(20, n_used), loc="left")
    return fig, n_used


def main():
    rows = FD.for_fits(FD.load(REPO))
    ph = phases(REPO, {r["cell"] for r in rows})
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    # Fig. 3: a = phase-step heatmap (supplementary.py), b = cores kept busy, c = share + processor
    # time, d = per-phase speedup
    out = [(panel_c(ph, rows), "b", "cores_busy", "Cores kept busy per build"),
           (phases_cost_panel_a(ph, rows), "c", "phase_share_cpu_time",
            "Wall-time share by phase; processor time consumed, IQ-TREE3 / IQ-TREE3-GPU"),
           (phases_cost_panel_b(ph, rows), "d", "phase_speedup_complexity",
            "Per-phase speedup vs tree complexity, with break-even")]
    for (fig, n), letter, slug, title in out:
        stem = FS.save_panel(fig, 3, letter, slug, title, "figures.py (phases_cost)", phases_cost_SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


phases_cost_main = main


# ================================================================================================
# ==== section: hardware ====
# ================================================================================================
# hardware.py - Supplementary Fig. S1D: the GPU arm across six cards.
#
#   speedup against complexity WITHIN each card, faceted: cards sit on different hosts, so the
#   CPU-arm wall differs between them (card_host_cpu_spread_max in KEY_NUMBERS.tsv) and
#   cross-card speedup is never compared
#
# The panels that ranked the cards by nominal fp64 and memory bandwidth, the per-card wall facets
# and the device-energy bars were removed from the publication.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

hardware_SRC = "benchmark/runs/RUNS.tsv"
LABEL = {"NVIDIA_A100_80GB_PCIe": "A100", "Tesla_V100S-PCIE-32GB": "V100S",
         "Tesla_V100-SXM2-16GB": "V100", "NVIDIA_L40S": "L40S",
         "NVIDIA_A40": "A40", "Quadro_RTX_8000": "RTX 8000"}


def _facet_labels(axes, xlabel, ylabel):
    """Axis labels on the outer facets only (bottom row, left column)."""
    nrow, ncol = axes.shape
    for j in range(ncol):
        axes[nrow - 1, j].set_xlabel(xlabel)
    for i in range(nrow):
        axes[i, 0].set_ylabel(ylabel)


def panel_d(rows):
    """Speedup WITHIN each card only -- never compared across cards, since the hosts differ."""
    cards = [c for c in LABEL if any(r["card"] == c for r in rows)]
    fig, axes = FS.square_grid(2, 3)
    for ax, card in zip(axes.flat, cards):
        pts = [(r["_complexity"], r["_speedup"]) for r in rows
               if r["card"] == card and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=8, color=FS.CARD_HUE, zorder=3)
        ax.axhline(1.0, color=FS.INK, lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{LABEL[card]} (n={len(pts)})", pad=2)
    for ax in axes.flat[len(cards):]:
        ax.axis("off")
    FS.ratio_ticks(axes.flat[0])
    _facet_labels(axes, "tree complexity", "speedup within card\n(IQ-TREE3 wall / IQ-TREE3-GPU wall)")
    n = sum(1 for r in rows if r["threads"] == "20")
    # speedup is comparable only WITHIN a card: the hosts differ by card_host_cpu_spread_max
    FS.grid_note(fig, axes, FS.stratum(20, n))
    return fig, n


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    out = [(panel_d(rows), "d", "card_facet_speedup", "Speedup vs complexity, per card")]
    for (fig, n), letter, slug, title in out:
        stem = FS.save_panel(fig, 1, letter, slug, title, "figures.py (hardware)", hardware_SRC, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


hardware_main = main


# ================================================================================================
# ==== section: supp_ported ====
# ================================================================================================
# supp_ported.py - Supplementary Fig. S1C and the replicate table.
#
#   S1C replicate spread at 20 threads, both alphabets on one axis (the former sampled-utilisation
#       figure was withdrawn)
#
# Also writes runs/REPLICATES_20T.tsv (Supplementary Data 5): one row per alphabet x taxon count,
# from the same replicate sets the S4 panel draws, so the table and the figure cannot drift.
#
# Every panel has one provenance: current-build rows only, the shared style, the Fig{S}{NN}{p}_{slug}
# contract, and a manifest row.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

supp_ported_SRC = "benchmark/runs/RUNS.tsv"


# ------------------------------------------------------------------- S4 replicate dispersion
def replicate_sets(rows, dt):
    """Repeat runs of the same dataset on the same card class, current build only."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["data_type"] != dt or r["threads"] != "20" or not H.is_replicate(r):
            continue
        if r["_speedup"]:
            by[int(r["taxa"])].append(r["_speedup"])
    return {k: v for k, v in by.items() if len(v) >= 2}


def replicates(rows):
    """Replicate spread at 20 threads, both alphabets on one axis: every repeat run of a size as a
    circle, the range as a bar, the median as a tick. The two alphabets share the same sizes, so
    they are offset slightly about each size."""
    fig, ax = FS.square_axes()
    # the sizes are an ordinal ladder (4, 6, 8, ... 30): equal spacing keeps every label legible,
    # where a log axis ran 26, 28 and 30 into one another
    sets = {dt: replicate_sets(rows, dt) for dt in ("protein", "dna")}
    xs = sorted(set(sets["protein"]) | set(sets["dna"]))
    pos = {x: i for i, x in enumerate(xs)}
    n, handles = 0, []
    for dt, shift in (("protein", -0.16), ("dna", 0.16)):
        by = sets[dt]
        col = FS.ALPHABET[dt]
        for x in sorted(by):
            v = by[x]; q = pos[x] + shift
            ax.vlines(q, min(v), max(v), color=col, lw=0.8, zorder=2)
            ax.scatter([q] * len(v), v, s=8, color=col, marker="o", alpha=0.8, zorder=3, linewidths=0)
            ax.hlines(st.median(v), q - 0.14, q + 0.14, color=FS.INK, lw=1.0, zorder=4)
        k = sum(len(v) for v in by.values()); n += k
        handles.append(ax.scatter([], [], s=8, color=col, marker="o",
                                  label=f"{'amino acid' if dt == 'protein' else 'nucleotide'} "
                                        f"({len(by)} sizes, n={k})"))
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log"); FS.ratio_ticks(ax)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlim(-0.7, len(xs) - 0.3)
    ax.set_xlabel("taxa"); ax.set_ylabel("speedup  (IQ-TREE3 wall / IQ-TREE3-GPU wall)")
    # the CV range and median per alphabet are in REPLICATES_20T.tsv and KEY_NUMBERS (replicate_cv_*)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    ax.set_title(FS.stratum(20, n), loc="left")
    return fig, n


# --------------------------------------------------------------------------- S6 GPU utilisation
def write_replicate_table(rows):
    """REPLICATES_20T.tsv -- one table for both alphabets (Supplementary Data 5), defined by the
    same replicate_sets() the S4 panel draws, so the table and the figure cannot drift."""
    path = os.path.join(REPO, "benchmark", "runs", "REPLICATES_20T.tsv")
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["data_type", "taxa", "n_replicates", "speedup_median", "speedup_mean",
                    "speedup_sd", "speedup_cv", "speedups"])
        for dt in ("protein", "dna"):
            by = replicate_sets(rows, dt)
            for t in sorted(by):
                v = by[t]
                sd = st.stdev(v) if len(v) > 1 else 0.0
                w.writerow([dt, t, len(v), f"{st.median(v):.3f}", f"{st.mean(v):.3f}", f"{sd:.3f}",
                            f"{sd/st.mean(v):.3f}", ";".join(f"{x:.2f}" for x in v)])
                n += 1
    print(f"  wrote benchmark/runs/REPLICATES_20T.tsv ({n} alphabet x size rows)")


def main():
    rows = FD.for_fits(FD.load(REPO))
    write_replicate_table(rows)
    date = max((r.get("datetime_utc") or "")[:10] for r in FD.load(REPO))
    out = [(replicates(rows), 1, "c", "replicates", "Replicate spread at 20 threads, both alphabets")]
    for (fig, n), num, letter, slug, title in out:
        stem = FS.save_panel(fig, num, letter, slug, title, "figures.py (supp_ported)", supp_ported_SRC, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


supp_ported_main = main


# ================================================================================================
# ==== section: supplementary ====
# ================================================================================================
# supplementary.py - Fig. 1A, Fig. 3A, Supplementary Figs. S1A, S1B and S4.
#
#   1A   campaign coverage: runs per taxon count as stacked bars by workflow and thread count, so
#        every one of the 182 paired runs is counted once
#   3A   phase-step heatmap from PHASE_STEPS.tsv, with the mean share of every step written on each
#        alphabet's block
#   S1A  the L40S runs by taxon count; S1B the five sweep cards stacked in one box
#   S4   model coverage REGENERATED FROM DATA (the previous figure was a hand-drawn schematic, which
#        cannot stand in a manuscript whose authority rests on measurement)
#
# S2 and S3 (the trees) are written by trees.py; S1C by supp_ported.py; S1D by hardware.py.
# The former pattern-saturation (S5), build-regression (S7) and coverage dot-matrix panels were
# removed.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

# Alphabet is perfectly aliased with model treatment; label the WORKFLOW.
supplementary_WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}


def phase_step_heatmap(fit_rows):
    """Step x cell, CPU arm. Shows at a glance which steps exist for which alphabet.

    `fit_rows` restricts to current-build cells: PHASE_STEPS.tsv covers all 226 recorded runs,
    including the 44 on the earlier binary, which must not enter a published figure.
    """
    keep = {r["cell"] for r in fit_rows}
    threads = {r["threads"] for r in fit_rows}
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "benchmark", "runs",
                                                        "PHASE_STEPS.tsv")), delimiter="\t")
            if r["cell"] in keep]
    steps = ["parsimony_tree", "fast_ml_search", "modelfinder", "ml_distances", "rapidnj_tree",
             "initial_tree_ll", "tree_search", "unattributed"]
    # Order amino acid FIRST explicitly. sorted() puts "dna" before "protein", which silently
    # transposed the two blocks against their labels -- the figure showed model selection on the
    # nucleotide side, which is impossible.
    order = sorted({(r["data_type"], r["cell"]) for r in rows if r["data_type"] in ("protein", "dna")},
                   key=lambda dc: (0 if dc[0] == "protein" else 1, dc[1]))
    cells = [c for _, c in order]
    idx = {c: i for i, c in enumerate(cells)}
    M = np.full((len(steps), len(cells)), np.nan)
    tot = {}
    for r in rows:
        if r["arm"] != "cpu" or r["cell"] not in idx:
            continue
        if r["step"] == "total":
            tot[r["cell"]] = FD.num(r["wall_s"])
    for r in rows:
        if r["arm"] != "cpu" or r["cell"] not in idx or r["step"] not in steps:
            continue
        t = tot.get(r["cell"])
        w = FD.num(r["wall_s"])
        if t and w is not None and t > 0:
            M[steps.index(r["step"]), idx[r["cell"]]] = 100.0 * w / t
    fig, ax = FS.square_axes()
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([s.replace("_", " ") for s in steps])
    nprot = sum(1 for d, _ in order if d == "protein")
    ax.axvline(nprot - 0.5, color="white", lw=1.0)
    # the mean share of each step, written on each alphabet's block (white on the dark end of
    # viridis, black on the light end); a step absent for that alphabet gets no number
    for i in range(len(steps)):
        for lo, hi in ((0, nprot), (nprot, len(cells))):
            blk = M[i, lo:hi]
            if np.isnan(blk).all():
                continue
            m = float(np.nanmean(blk))
            ax.text((lo + hi - 1) / 2, i, f"{m:.1f}%", ha="center", va="center",
                    color=FS.INK if m > 55 else "white", fontweight="bold")
    ax.set_xticks([nprot / 2, nprot + (len(cells) - nprot) / 2])
    ax.set_xticklabels([f"amino acid (n={nprot})", f"nucleotide (n={len(cells)-nprot})"])
    # the colour bar is an inset outside the plotting box, so the box keeps its size
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.set_label("% of that run's IQ-TREE3 wall")
    # white gaps are steps that do not exist for that alphabet (no model selection for nucleotide)
    ax.set_title(FS.stratum(threads, len(cells)), loc="left")
    return fig, len(cells)


def model_coverage(rows):
    """REGENERATED FROM DATA, replacing a hand-drawn schematic.

    Takes ALL current-build rows, NOT for_fits(). The `__mdl_*` correctness cells are excluded from
    speedup statistics because they sit below break-even by construction -- but they are precisely
    the runs that exercise the alternative nucleotide models, so excluding them from a COVERAGE
    figure collapses the nucleotide side to GTR alone and understates the very thing the figure
    exists to show. The exclusion rule that is right for speed is wrong for coverage.
    """
    cnt = collections.Counter()
    for r in rows:
        m = (r.get("model") or "NA").split("+")[0]
        cnt[(r["data_type"], m)] += 1
    items = sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1]))
    fig, ax = FS.square_axes()
    ys = range(len(items))
    ax.barh(list(ys), [v for _, v in items],
            color=[FS.ALPHABET[k[0]] for k, _ in items], height=0.62)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{k[1]}  ({'AA' if k[0]=='protein' else 'NT'})" for k, _ in items])
    ax.invert_yaxis()
    for i, (_, v) in enumerate(items):
        ax.annotate(str(v), xy=(v, i), xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xlabel("runs")
    ax.margins(x=0.14)
    # counted from the model each run selected; the title slot carries the stratum only
    ax.set_title(FS.stratum({r["threads"] for r in rows}, sum(cnt.values())), loc="left")
    return fig, sum(cnt.values())


# ------------------------------------------------------------------ campaign coverage
STRATA = [("protein", "1"), ("protein", "20"), ("dna", "1"), ("dna", "20")]
STRATUM_LABEL = {("protein", "1"): "amino acid, 1T", ("protein", "20"): "amino acid, 20T",
                 ("dna", "1"): "nucleotide, 1T", ("dna", "20"): "nucleotide, 20T"}
CARD_LABEL = {"NVIDIA_A100_80GB_PCIe": "A100", "Tesla_V100S-PCIE-32GB": "V100S",
              "Tesla_V100-SXM2-16GB": "V100", "NVIDIA_L40S": "L40S",
              "NVIDIA_A40": "A40", "Quadro_RTX_8000": "RTX 8000"}


def _taxa_axis(ax, taxa):
    ax.set_xticks(range(len(taxa))); ax.set_xticklabels([str(t) for t in taxa], rotation=90)
    ax.set_xlim(-0.7, len(taxa) - 0.3)
    ax.set_xlabel("taxa in tree")


def _stacked_bars(ax, rows, taxa):
    """Runs per taxon count as stacked bars by workflow (hue) and thread count (hatch)."""
    cnt = collections.Counter((r["data_type"], r["threads"], int(r["taxa"])) for r in rows)
    bottom = np.zeros(len(taxa))
    for dt, th in (("protein", "20"), ("protein", "1"), ("dna", "20"), ("dna", "1")):
        h = np.array([cnt[(dt, th, t)] for t in taxa], dtype=float)
        ax.bar(range(len(taxa)), h, 0.7, bottom=bottom, color=FS.ALPHABET[dt],
               edgecolor="white", linewidth=0.3, hatch="////" if th == "1" else None,
               label=STRATUM_LABEL[(dt, th)])
        bottom += h
    for j, t in enumerate(bottom):
        if t:
            ax.annotate(f"{int(t)}", xy=(j, t), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", color=FS.MUTED)
    _taxa_axis(ax, taxa)
    ax.margins(y=0.12)


def coverage_bars(rows):
    """Fig. 1A: every paired run counted once, by taxon count, workflow and thread count."""
    taxa = sorted({int(r["taxa"]) for r in rows})
    fig, ax = FS.square_axes()
    _stacked_bars(ax, rows, taxa)
    ax.set_ylabel("paired runs")
    ax.set_title(FS.stratum({r["threads"] for r in rows}, len(rows)), loc="left")
    ax.legend(loc="upper right")
    return fig, len(rows)


# S1A/S1B colours. The five sweep cards take five Okabe-Ito hues that are neither the alphabet pair
# (blue / vermillion) nor orange; L40S is grey, dark for 20 threads and light for 1 thread, so no
# bar in this panel can be read as "amino acid" or "nucleotide" by colour: alphabet is the FILL
# PATTERN here (solid = amino acid, striped = nucleotide).
SWEEP_COLOR = {"NVIDIA_A100_80GB_PCIe": "#000000", "Tesla_V100S-PCIE-32GB": "#56B4E9",
               "Tesla_V100-SXM2-16GB": "#009E73", "NVIDIA_A40": "#F0E442",
               "Quadro_RTX_8000": "#CC79A7"}
L40S_COLOR = {"20": "#6E6E6E", "1": "#C4C4C4"}
HATCH = {"protein": None, "dna": "////"}


def _stack(ax, rows, taxa, groups):
    """Stacked bars per taxon count; `groups` = [(colour, hatch, predicate), ...] in stacking
    order. A thin dark edge keeps the light colours and the hatching visible."""
    bottom = np.zeros(len(taxa))
    for col, hatch, pred in groups:
        cnt = collections.Counter(int(r["taxa"]) for r in rows if pred(r))
        h = np.array([cnt[t] for t in taxa], dtype=float)
        ax.bar(range(len(taxa)), h, 0.7, bottom=bottom, color=col, hatch=hatch,
               edgecolor="#444444", linewidth=0.3)
        bottom += h
    for j, t in enumerate(bottom):
        if t:
            ax.annotate(f"{int(t)}", xy=(j, t), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", color=FS.MUTED)
    _taxa_axis(ax, taxa)
    ax.set_ylim(0, max(bottom) * 1.3)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _coverage_groups(rows, which):
    """Stacking order for the two coverage-by-card panels (same code as before the split)."""
    if which == "l40s":
        return [(L40S_COLOR[th], HATCH[dt], (lambda r, th=th, dt=dt: r["threads"] == th and r["data_type"] == dt))
                for th in ("20", "1") for dt in ("protein", "dna")]
    return [(SWEEP_COLOR[c], HATCH[dt], (lambda r, c=c, dt=dt: r["card"] == c and r["data_type"] == dt))
            for c in SWEEP_COLOR for dt in ("protein", "dna")]


def coverage_bars_l40s(rows):
    """S1A: the L40S runs by taxon count, dark grey 20 threads, light grey 1 thread; solid =
    amino acid, striped = nucleotide."""
    from matplotlib.patches import Patch
    taxa = sorted({int(r["taxa"]) for r in rows})
    l40 = [r for r in rows if r["card"] == "NVIDIA_L40S"]
    fig, ax = FS.square_axes()
    _stack(ax, l40, taxa, _coverage_groups(rows, "l40s"))
    ax.set_ylabel("paired runs")
    key = [Patch(facecolor=L40S_COLOR["20"], edgecolor="#444444", lw=0.3, label="L40S, 20 threads"),
           Patch(facecolor=L40S_COLOR["1"], edgecolor="#444444", lw=0.3, label="L40S, 1 thread"),
           Patch(facecolor="white", edgecolor="#444444", lw=0.3, label="solid: amino acid"),
           Patch(facecolor="white", edgecolor="#444444", lw=0.3, hatch="////", label="striped: nucleotide")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    ax.set_title(FS.stratum({r["threads"] for r in l40}, len(l40)), loc="left")
    return fig, len(l40)


def coverage_bars_sweep(rows):
    """S1B: the five sweep cards stacked in one box, one Okabe-Ito hue per card (all at 20
    threads); solid = amino acid, striped = nucleotide."""
    from matplotlib.patches import Patch
    taxa = sorted({int(r["taxa"]) for r in rows})
    sweep = [r for r in rows if r["card"] in SWEEP_COLOR]
    fig, ax = FS.square_axes()
    _stack(ax, sweep, taxa, _coverage_groups(rows, "sweep"))
    ax.set_ylabel("paired runs")
    key = [Patch(facecolor=SWEEP_COLOR[c], edgecolor="#444444", lw=0.3,
                 label=f"{CARD_LABEL[c]} (n={sum(1 for r in sweep if r['card'] == c)})")
           for c in SWEEP_COLOR]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    ax.set_title(FS.stratum({r["threads"] for r in sweep}, len(sweep)), loc="left")
    return fig, len(sweep)


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    SRC = "benchmark/runs/RUNS.tsv"
    # (figure, letter, supplementary?) per panel: Fig. 1A coverage, Fig. 3A heatmap, S1A/S1B
    # coverage by card, S4 model coverage
    out = [(coverage_bars(FD.load(REPO)), 1, "a", False, "coverage_stacked_bars",
            "Campaign coverage: runs per taxon count, stacked", SRC),
           (phase_step_heatmap(rows), 3, "a", False, "phase_step_heatmap",
            "Phase-step share per run, with the mean per alphabet", "benchmark/runs/PHASE_STEPS.tsv"),
           (coverage_bars_l40s(FD.load(REPO)), 1, "a", True, "coverage_l40s",
            "Campaign coverage on the L40S, by taxon count", SRC),
           (coverage_bars_sweep(FD.load(REPO)), 1, "b", True, "coverage_sweep",
            "Campaign coverage on the five sweep cards, by taxon count", SRC),
           (model_coverage(FD.load(REPO)), 4, "a", True, "model_coverage_data",
            "Model classes exercised", SRC)]
    for (fig, n), num, letter, supp, slug, title, src in out:
        stem = FS.save_panel(fig, num, letter, slug, title, "figures.py (supplementary)", src, n,
                             supp=supp, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


supplementary_main = main


# ================================================================================================
# ==== section: trees ====
# ================================================================================================
# trees.py - Supplementary Figs. S1 and S2: the inferred trees, under the naming contract.
#
# S1   one curated montage per alphabet, the 12-rung 20-thread size ladder, ORDERED BY TAXON COUNT
#      with the complexity printed in each panel label. (Ordering by complexity, as an earlier draft
#      did, produced a taxon order of 4, 6, 14, 10, 8, ... which reads as an error even though it is
#      deliberate; taxa order with complexity printed conveys the same fact without the double-take.)
#
# S2   the complete per-run record, every current-build cell, as a multipage montage using the
#      contract's `_p{NN}` suffix. Page 1 states that the 44 earlier-build runs are excluded.
#
# ONE TREE PER DATASET. The CPU and GPU arms returned identical topologies in all 182 runs, so a
# second tree would be a second copy of the same figure. That identity is the result and it belongs
# in the caption, not in duplicated ink.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.

trees_SRC = "benchmark/runs/RUNS.tsv"
# Every montage is one US Letter page: 4 rows x 4 columns of trees inside 0.8 in margins, with
# the stratum line in the top margin. Type is 6 pt, tip labels 5 pt (the journal floor), so a
# tree whose tips would not fit at 5 pt in its cell is drawn with its tips unlabelled (and says so).
# A labelled tree keeps a quarter of its cell for the labels; an unlabelled tree fills the cell.
PAGE_IN = (8.5, 11.0)
MARGIN_IN = 0.8
NROW, NCOL = 4, 4
PER_PAGE = NROW * NCOL
CELL_H_IN = (PAGE_IN[1] - 2 * MARGIN_IN - 0.25) / NROW      # 0.25 in kept for the stratum line
TIP_PT = 5.0            # the journal floor, permitted for tip labels only (figstyle)
MAX_LABELLED_TIPS = int((CELL_H_IN - 0.45) * 72 / (TIP_PT * 1.05))   # cell minus scale-bar room


def si(x):
    if not x:
        return "NA"
    e = int(math.floor(math.log10(x)))
    return f"{x/10**e:.1f}e{e}"


def short_label(name):
    """Compact, readable tip label, ported from a retired tree-plotting script.

    Genome names are Genus_species_..._ID and collapse to G.species_ID:
    Homo_sapiens_H_Ca23_CH0E -> H.sapiens_CH0E. Simulated nucleotide tips (T1..Tn) have no
    underscores and are returned unchanged. Shortening keys on the LABEL'S FORMAT, not the data
    type, because the paired dna_pt_* sets are nucleotide data simulated along a protein topology
    and so carry real genome names.
    """
    if not name:
        return ""
    p = name.split("_")
    if len(p) >= 3 and p[0][:1].isalpha() and p[1]:
        return f"{p[0][0]}.{p[1]}_{p[-1]}"
    return name


def nice_scale(depth):
    """A round scale-bar length ~1/4 of tree depth, in {1,2,5}x10^k."""
    if depth <= 0:
        return 0.1
    raw = depth / 4.0
    k = math.floor(math.log10(raw))
    for m in (5, 2, 1):
        v = m * 10 ** k
        if v <= raw:
            return v
    return 10 ** k


def draw(ax, treefile, dt, title):
    """Ported from a retired tree-plotting script, with the type floor enforced.

    THREE THINGS A NAIVE REIMPLEMENTATION LOSES, all of which I lost before porting this properly:
      * Bio.Phylo draws branches as LineCollections, NOT Line2D, so recolouring only
        `ax.get_lines()` silently does nothing and every tree renders black.
      * tip labels need collapsing, or genome names crowd the panel into illegibility;
      * a phylogram without a substitution-rate scale bar cannot be read quantitatively at all.
    """
    try:
        tree = Phylo.read(treefile, "newick")
    except Exception:
        ax.text(0.5, 0.5, "tree unreadable", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off"); return
    tree.ladderize()
    ntip = tree.count_terminals()
    col = FS.ALPHABET[dt]
    # Tip-label size scales with tip count but is CLAMPED AT THE 5 pt JOURNAL FLOOR. Beyond ~30
    # tips even 5 pt collides in a panel this size, so labels are dropped and the panel says so
    # rather than printing an unreadable smear.
    show_labels = ntip <= MAX_LABELLED_TIPS
    fs = TIP_PT
    old = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = fs
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False,
               label_func=(lambda c: short_label(c.name) if c.is_terminal() else "")
                          if show_labels else (lambda c: ""),
               label_colors=lambda _l: FS.INK)
    plt.rcParams["font.size"] = old
    for coll in ax.collections:          # the branches themselves
        coll.set_color(col); coll.set_linewidth(0.9)
    for ln in ax.get_lines():
        ln.set_color(col); ln.set_linewidth(0.9)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # substitution-rate scale bar, below the last tip (Bio.Phylo lays tips top-down on an inverted
    # y-axis, so the spare room is at larger y)
    depth = max((tree.distance(t) for t in tree.get_terminals()), default=0.1)
    # compact: an unlabelled tree spans the cell; a labelled tree keeps exactly the room its
    # longest tip label needs (6 pt Helvetica averages about 0.042 in per character), measured
    # against the cell's width on the page, so no label is cut and no room is wasted
    cell_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    if show_labels:
        longest = max((len(short_label(t.name or "")) for t in tree.get_terminals()), default=0)
        label_in = min(0.042 * longest + 0.08, 0.6 * cell_in)
        ax.set_xlim(-0.02 * depth, depth * cell_in / (cell_in - label_in))
    else:
        ax.set_xlim(-0.02 * depth, depth * 1.03)
    sb = nice_scale(depth)
    x0, x1 = ax.get_xlim()
    xs = x0 + 0.04 * (x1 - x0)
    yb = ntip + 0.8
    ax.set_ylim(ntip + 2.4, 0.2)
    ax.plot([xs, xs + sb], [yb, yb], color=FS.INK, lw=1.2, clip_on=False)
    ax.text(xs + sb / 2, yb + 0.7, f"{sb:g} subs/site" + ("" if show_labels else "  (tips unlabelled)"),
            ha="center", va="top", color=FS.INK)
    ax.set_title(title, color=FS.INK, pad=2)


def _priority(r):
    """Which row represents a dataset. Prefer the 20-thread headline run, then any headline run,
    then anything. Without an explicit order the LAST row read won, so a nucleotide dataset with
    both thread counts could be represented by its 1-thread run and vanish from a 20-thread ladder."""
    return (2 if (H.is_headline(r) and r["threads"] == "20") else
            1 if H.is_headline(r) else 0)


CARD_SHORT = {"NVIDIA_A100_80GB_PCIe": "A100", "Tesla_V100S-PCIE-32GB": "V100S",
              "Tesla_V100-SXM2-16GB": "V100", "NVIDIA_L40S": "L40S", "NVIDIA_A40": "A40",
              "Quadro_RTX_8000": "RTX 8000"}


def datasets(rows):
    """(dataset, thread count, card, model) -> the cell whose tree we draw, plus its annotation.

    Every distinct analysis is drawn once: a dataset re-run on another card, at one thread or
    under another substitution model is a different inference and gets its own tree. Only
    same-settings replicates (same dataset, threads, card and model) collapse to one tree, the
    headline run by preference (_priority); their trees are identical.
    """
    out, rank = {}, {}
    for r in rows:
        # The model is a SETTING only in the __mdl_ cells (a fixed model chosen for the run);
        # elsewhere it is ModelFinder's outcome, so two replicates that selected different
        # models are still the same analysis and collapse.
        model_setting = r.get("model", "") if "__mdl_" in r["cell"] else ""
        g = (r.get("genus") or r["cell"], r["threads"], r["card"], model_setting)
        p = _priority(r)
        if g in out and p <= rank[g]:
            continue
        rank[g] = p
        out[g] = dict(cell=r["cell"], dt=r["data_type"], taxa=int(r["taxa"]),
                      cx=r["_complexity"], sp=r["_speedup"], model=r.get("model", ""),
                      thr=r["threads"], card=CARD_SHORT.get(r["card"], r["card"]),
                      head=H.is_headline(r))
    return out


def montage(items, dt, page=None, npages=None, note=None):
    n = len(items)
    # A fixed 4 x 4 grid on a Letter page; a short last page keeps the grid and leaves cells empty
    fig = plt.figure(figsize=PAGE_IN)
    l, b = MARGIN_IN / PAGE_IN[0], MARGIN_IN / PAGE_IN[1]
    axes = fig.subplots(NROW, NCOL, gridspec_kw=dict(
        left=l, right=1 - l, bottom=b, top=1 - b - 0.35 / PAGE_IN[1], wspace=0.12, hspace=0.22))
    # np.atleast_1d(...).ravel() handles every shape subplots() can return. The previous
    # special-case wrapped the whole ndarray in a list when n == 1, so the last page -- which can
    # hold a single tree -- crashed.
    import numpy as _np
    axes = list(_np.atleast_1d(axes).ravel())
    for ax in axes:
        ax.axis("off")
    for ax, d in zip(axes, items):
        tf = os.path.join(PAIRED, d["cell"], "gpu.treefile")
        if not os.path.exists(tf):
            continue
        ax.axis("on")
        # the title names what distinguishes this inference: taxa and model always, the card
        # when it is not the L40S, and the thread count when it is not 20
        tag = "".join([f"  {d['card']}" if d["card"] != "L40S" else "",
                       f"  {d['thr']}T" if d["thr"] != "20" else ""])
        draw(ax, tf, d["dt"], f"n={d['taxa']}  {d.get('model','')}{tag}\ncomplexity {si(d['cx'])}"
                              + (f"  |  {d['sp']:.2f}x" if d.get("sp") else ""))
    # The montage carries the stratum line only (alphabet, thread count, n trees, page); the
    # ordering, the one-tree-per-dataset rule and the identical-topology result are legend text.
    kind = "amino acid" if dt == "protein" else ("nucleotide" if dt == "dna" else "all datasets")
    ttl = f"{kind}, " + FS.stratum({d["thr"] for d in items}, n, "trees")
    if page:
        ttl += f"   (page {page} of {npages})"
    fig.text(l, 1 - b + 0.05 / PAGE_IN[1], ttl, ha="left", va="bottom")   # the stratum line, in the top margin
    return fig


def main():
    rows = FD.load(REPO)
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    ds = datasets(rows)
    # S2: every amino-acid inference; S3: every nucleotide inference (one tree per dataset x
    # thread count x card x model; same-settings replicates collapse), 16 to a Letter page.
    for dt, num, tag in (("protein", 2, "aa"), ("dna", 3, "nt")):
        items = sorted([d for d in ds.values() if d["dt"] == dt],
                       key=lambda d: (d["taxa"], d["model"], d["thr"] != "20", d["card"] != "L40S", d["cell"]))
        pages = math.ceil(len(items) / PER_PAGE)
        for i in range(pages):
            chunk = items[i * PER_PAGE:(i + 1) * PER_PAGE]
            fig = montage(chunk, dt, page=i + 1, npages=pages)
            stem = FS.save_panel(fig, num, "a", f"trees_{tag}",
                                 f"Every {'amino-acid' if dt == 'protein' else 'nucleotide'} inference (dataset x threads x card x model)",
                                 "figures.py (trees)", trees_SRC, len(chunk), supp=True,
                                 page=i + 1, data_date=date, square=False, full_page=True)
            plt.close(fig)
        print(f"  wrote FigS{num:02d}a_trees_{tag}_p01..p{pages:02d}  ({len(items)} trees)")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


trees_main = main


# ================================================================================================
# ==== section: composites ====
# ================================================================================================
# composites.py - assemble the panels into one lettered composite per figure.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.
#
# WHAT THIS PRODUCES. One figure = one file, panels lettered A, B, C ... as a journal sets them,
# written to benchmark/figures/composites/ as a vector PDF plus a 300 dpi PNG rendered from that same
# PDF, so the two cannot disagree.
#
# WHY THE PANELS ARE NOT RE-RENDERED. Each panel's existing PDF is placed into the page with pypdf, so
# every mark stays vector and byte-for-byte what the panel generator produced. The letters are a vector
# overlay drawn on top.
#
# THE SCALE RULE, which is the whole design of this file. Every panel is authored at 6 pt (figstyle
# FONT_PT) around a fixed 40 x 40 mm plotting box (figstyle AX_MM; the decorations around it set the
# panel's outer size). Placing a panel at scale s therefore prints its type at 6*s pt --
# scaling a panel silently rewrites its font size. A naive "stretch each row to fill the page width"
# layout gave one figure 8 pt in its top row and 3.8 pt in the row below it, which is both illegible and
# obviously amateur. So:
#
#   * panels are never scaled relative to one another. Within a row they sit at natural size and are
#     TOP-aligned, which is why rows here may hold panels of unequal height;
#   * the finished page is then scaled ONCE, uniformly, to fit the text block. Type stays uniform
#     across the whole figure and the effective size is simply 6 * scale;
#   * that effective size is asserted against figstyle's permitted band (5-8 pt). A grouping that
#     cannot be printed in the band is refused with a message naming the figure, rather than shipped
#     at 3.8 pt.
#
# Row assignments below follow from that rule. The per-card facet grid in Supplementary Fig. S6 is
# 139 mm wide at natural size, so it stands alone.

OUT = os.path.join(REPO, "benchmark", "figures", "composites")

MM = 72.0 / 25.4
GAP = 3.0 * MM                   # gutter between panels, and between rows
LABEL_H = 4.0 * MM               # strip above each row holding the panel letters
MAX_W = 180.0 * MM               # double-column width
MAX_H = 240.0 * MM               # text-block height
MIN_PT, MAX_PT = min(FS.PERMITTED_PT), max(FS.PERMITTED_PT)

# figure number -> rows of panel stems. Panels keep their authored size; a row's height is the
# tallest panel in it. These layouts ARE the manuscript's figure numbering and lettering; the panel
# stems are named to match (FigS07b is Supplementary Fig. S7B).
# A row holds ITEMS; an item is one stem, or a tuple of stems that share one letter and sit side
# by side (Fig. 2A is the 20-thread and one-thread decision boxes; Fig. 2B the two cost-law boxes).
FIGURES = {
    1: dict(dir=PANELS, title="Campaign coverage and equivalence of the two builds", rows=[
            ["Fig01a_coverage_stacked_bars", "Fig01b_support_diff_dotplot", "Fig01c_logl_reldiff_ecdf"]]),
    2: dict(dir=PANELS, title="Acceleration and runtime scaling at 20 threads and one thread", rows=[
            [("Fig02a_decision_t20", "Fig02a_decision_t1")],
            [("Fig02b_cost_law_t20", "Fig02b_cost_law_t1")]]),
    3: dict(dir=PANELS, title="Where the time goes, and what it costs", rows=[
            ["Fig03a_phase_step_heatmap", "Fig03b_cores_busy"],
            ["Fig03c_phase_share_cpu_time", "Fig03d_phase_speedup_complexity"]]),
}

def _pages(stem):
    """The page stems of a multipage panel, in page order, from the files present."""
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(PANELS, stem + "_p*.pdf")))


def _spec_pages(spec):
    """The pages of a multipage spec, resolved when asked (the files exist only after `trees`)."""
    if "pages" in spec:
        return spec["pages"]
    if "pages_glob" in spec:
        return [[[s]] for s in _pages(spec["pages_glob"])]
    return None


# Supplementary figures. Every supplementary figure in the submission gets one file here, so the
# master supplementary-figures document can be assembled from this directory alone. Three kinds:
#   rows=[...]            the lettered single-page composite used for the main figures;
#   pages=[[rows], ...]   one page per entry at NATURAL size (scale 1.0), because the tree panels are
#                         ~300 mm wide and shrinking a tree plot is what destroyed the tip labels once
#                         already: S1 and S2 are multipage (no letters: their legends speak
#                         of pages, not panels);
#   a single panel        is placed without a letter; the legends of S3 and S5-S8 name no panel
#                         letter, and a lone "A" on a one-panel figure is a journal error.
# Numbering is the order of first citation (S1-S7); the greyscale proof sheet is a production check
# and is not a supplementary figure.
SUPP = {
    # the per-card facet grid is 139 mm wide at natural size and takes a row of its own
    1: dict(dir=PANELS, title="Coverage by card, replicate spread and per-card speedup", rows=[
            ["FigS01a_coverage_l40s", "FigS01b_coverage_sweep", "FigS01c_replicates"],
            ["FigS01d_card_facet_speedup"]]),
    2: dict(dir=PANELS, title="Every amino-acid inference", letters=False, pages_glob="FigS02a_trees_aa"),
    3: dict(dir=PANELS, title="Every nucleotide inference", letters=False, pages_glob="FigS03a_trees_nt"),
    4: dict(dir=PANELS, title="Model classes exercised, from the data", rows=[["FigS04a_model_coverage_data"]]),
}

LETTERS = "ABCDEFGHIJKL"



def reference_index():
    """-> {figure label: {panel letter: panel stem}} for every composite.

    These layouts ARE the manuscript's figure numbering: which panel is Fig. 1B is decided by the
    row assignments above. The panel stems are named to agree with them, but the cross-check
    resolves manuscript references against this index rather than against file names, so a
    renamed stem cannot silently move a reference.
    """
    idx = {}
    for book, pre in ((FIGURES, ""), (SUPP, "S")):
        for n, spec in book.items():
            items = all_items(spec)
            idx[f"{pre}{n}"] = ({LETTERS[k]: (stems_of(it)[0] if len(stems_of(it)) == 1 else stems_of(it))
                                 for k, it in enumerate(items)} if lettered(spec, items) else {})
    return idx


def all_rows(spec):
    """The panel rows of a spec, across its pages if it has them."""
    return [row for page in (_spec_pages(spec) or [spec.get("rows", [])]) for row in page]


def stems_of(item):
    """The stems of a row item: a lone stem, or the stems of a tuple that share one letter."""
    return list(item) if isinstance(item, tuple) else [item]


def all_items(spec):
    return [item for row in all_rows(spec) for item in row]


def all_stems(spec):
    return [s for item in all_items(spec) for s in stems_of(item)]


def lettered(spec, items):
    """Panels carry letters only when the figure has more than one ITEM and the spec does not opt
    out; the stems of a tuple item share the item's letter."""
    return spec.get("letters", True) and len(items) > 1


def page_size(src):
    from pypdf import PdfReader
    b = PdfReader(src).pages[0].mediabox
    return float(b.width), float(b.height)


def layout(name, spec, rows_in=None, first_letter=0, fixed_scale=None, letters_on=True):
    """-> (page_w, page_h, scale, [(stem, x, y, letter)]) in natural units.

    Coordinates come back unscaled; build() applies the single page scale. `rows_in` lays out one
    page of a multi-page spec; `fixed_scale` prints that page as authored instead of fitting it to
    the text block; `first_letter` continues the lettering across pages.
    """
    rows_in = spec["rows"] if rows_in is None else rows_in
    sizes = {s: page_size(os.path.join(spec["dir"], s + ".pdf")) for r in rows_in for it in r for s in stems_of(it)}
    # A page printed as authored (fixed_scale: the Letter-size tree montages) keeps its exact page
    # size; its letter is drawn inside the page's own top margin, so no strip is added.
    label_h = LABEL_H if (letters_on and fixed_scale is None) else 0.0
    flat = {id(row): [s for it in row for s in stems_of(it)] for row in rows_in}
    rows = [(row,
             sum(sizes[s][0] for s in flat[id(row)]) + GAP * (len(flat[id(row)]) - 1),
             max(sizes[s][1] for s in flat[id(row)])) for row in rows_in]

    W = max(w for _, w, _ in rows)
    H = sum(h + label_h for _, _, h in rows) + GAP * (len(rows) - 1)

    # one uniform scale for the whole page: fit the text block, never magnify past the band
    scale = fixed_scale if fixed_scale else min(MAX_W / W, MAX_H / H, MAX_PT / FS.FONT_PT)
    pt = FS.FONT_PT * scale
    if pt < MIN_PT:
        raise SystemExit(
            f"Figure {name}: this grouping prints at {pt:.1f} pt, below the {MIN_PT:.0f} pt floor "
            f"(natural {W/MM:.0f} x {H/MM:.0f} mm). Give its widest panels a row of their own, or "
            f"move them to the supplement.")

    placed, y = [], H
    k = first_letter
    for row, rw, rh in rows:
        y -= label_h
        x = (W - rw) / 2.0                     # rows narrower than the page are centred
        for it in row:
            for j, s in enumerate(stems_of(it)):
                w0, h0 = sizes[s]
                # the letter sits on the first stem of the item only
                placed.append((s, x, y - h0, LETTERS[k] if (letters_on and j == 0) else ""))
                x += w0 + GAP
            k += 1
        y -= rh + GAP
    return W, H, scale, placed


def overlay(W, H, placed, sizes, path):
    """A transparent vector page carrying just the panel letters, at natural page size."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    FS.apply(strict_font=False)
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    for s, x, y, letter in placed:
        if not letter:
            continue
        top = y + sizes[s][1]
        if top + 0.6 + 8.0 > H:          # no strip above this panel: letter inside its top margin
            fig.text((x + 0.25 * 72) / W, (top - 0.25 * 72) / H, letter, fontsize=8.0,
                     fontweight="bold", va="top", ha="left", color=FS.INK)
        else:
            fig.text(x / W, (top + 0.6) / H, letter, fontsize=8.0, fontweight="bold",
                     va="bottom", ha="left", color=FS.INK)
    # rcParams sets savefig.bbox='tight', which would crop this overlay to the letters alone and
    # then merge them at the page origin. The bbox is given explicitly to keep the full page.
    fig.savefig(path, transparent=True, pad_inches=0,
                bbox_inches=Bbox([[0, 0], [W / 72.0, H / 72.0]]))
    plt.close(fig)


def build_page(spec, rows, first_letter, fixed_scale, letters_on, out_stem):
    from pypdf import PdfReader, Transformation, PageObject
    W, H, scale, placed = layout(spec.get("title", ""), spec, rows, first_letter, fixed_scale, letters_on)
    sizes = {s: page_size(os.path.join(spec["dir"], s + ".pdf")) for s, *_ in placed}
    page = PageObject.create_blank_page(width=W * scale, height=H * scale)
    for s, x, y, _letter in placed:
        p = PdfReader(os.path.join(spec["dir"], s + ".pdf")).pages[0]
        page.merge_transformed_page(
            p, Transformation().translate(x, y).scale(scale, scale))
    if any(letter for *_, letter in placed):
        ov = out_stem + "._overlay.pdf"
        overlay(W, H, placed, sizes, ov)
        op = PdfReader(ov).pages[0]
        page.merge_transformed_page(op, Transformation().scale(scale, scale))
        os.remove(ov)
    return page, W * scale, H * scale, scale, len(placed)


def build(name, spec, out_stem):
    """-> (page_w, page_h, scale, n_panels, n_pages) of the first page; writes <out_stem>.pdf (all
    pages) and, per page, a 300 dpi PNG (<out_stem>.png, or <out_stem>_p1.png ... for a
    multi-page figure). PDF and PNG are the publication formats; README figures get an SVG copy
    in assets/readme/ (see README_SVG)."""
    from pypdf import PdfWriter
    pages = _spec_pages(spec) or [spec["rows"]]
    letters_on = lettered(spec, all_items(spec))
    fixed = 1.0 if _spec_pages(spec) else None    # pages print as authored (6 pt), never refitted
    w = PdfWriter()
    first = None
    k = 0
    for rows in pages:
        page, W, H, sc, n = build_page(spec, rows, k, fixed, letters_on, out_stem)
        w.add_page(page)
        first = first or (W, H, sc, n)
        k += sum(1 for row in rows for _ in row)        # letters advance per ITEM
    with open(out_stem + ".pdf", "wb") as fh:
        w.write(fh)
    for old in [f for f in os.listdir(os.path.dirname(out_stem))
                if re.fullmatch(re.escape(os.path.basename(out_stem)) + r"(_p\d+)?\.png", f)]:
        os.remove(os.path.join(os.path.dirname(out_stem), old))
    for i in range(1, len(pages) + 1):
        stem = out_stem if len(pages) == 1 else f"{out_stem}_p{i}"
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "-f", str(i), "-l", str(i),
                        out_stem + ".pdf", stem], check=True)
    if os.path.basename(out_stem) in README_SVG:
        # vector copy for the repository README only (text as outlines, renders sharp on GitHub)
        os.makedirs(README_SVG_DIR, exist_ok=True)
        subprocess.run(["pdftocairo", "-svg", "-f", "1", "-l", "1", out_stem + ".pdf",
                        os.path.join(README_SVG_DIR, os.path.basename(out_stem) + ".svg")], check=True)
    return (*first, len(pages))


# Composites the repository README embeds; each gets an SVG in assets/readme/ (not a publication format)
README_SVG = {"Figure_2"}
README_SVG_DIR = os.path.join(REPO, "assets", "readme")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"  composites -> {os.path.relpath(OUT, REPO)}")
    print(f"  {'figure':<34s} {'panels':>6s}  {'page (mm)':>12s}  {'scale':>5s}  {'type':>6s}")
    for old in os.listdir(OUT):            # a renumbered or retired figure must not linger
        if re.fullmatch(r"(Figure_\d+|Supplementary_Figure_S\d+)(_p\d+)?\.(pdf|png|svg)", old):
            n = re.search(r"(S?\d+)", old.split("_p")[0]).group(1)
            book = SUPP if old.startswith("Supplementary") else FIGURES
            if int(n.lstrip("S")) not in book:
                os.remove(os.path.join(OUT, old))
    rows = [(f"Figure_{n}", str(n), s) for n, s in sorted(FIGURES.items())]
    rows += [(f"Supplementary_Figure_S{n}", f"S{n}", s) for n, s in sorted(SUPP.items())]
    for stem, name, spec in rows:
        W, H, sc, k, npages = build(name, spec, os.path.join(OUT, stem))
        pg = f"{npages} pages" if npages > 1 else ""
        print(f"  {stem:<34s} {k:>6d}  {W/MM:5.0f} x {H/MM:4.0f}  {sc:5.2f}  "
              f"{FS.FONT_PT*sc:4.1f} pt   {spec['title']}  {pg}")
    return 0


composites_main = main


# ================================================================================================
# ==== section: panels ====
# ================================================================================================
# panels.py - enforce the figure contract instead of asserting it.
#
# WHY THIS EXISTS. After the previous pass I reported that every text element was 6 pt and the theme
# was uniform. Both were false: I had verified the embedded font FAMILY in the PDFs and never the
# SIZE, and 102 explicit `fontsize=` arguments and 16 `.grid(True)` calls were quietly overriding the
# shared style. rcParams set defaults; they do not bind explicit arguments. The only way to know is to
# read the emitted files, which is what this does.
#
# Checks, each of which fails the build:
#   1. every panel's Tf font sizes lie in the permitted set (mathtext sub/superscripts allowed)
#   2. no panel embeds DejaVu (the silent fallback when Arial/Helvetica/Nimbus are absent)
#   3. every emitted file matches the naming contract Fig{NN}{p}_{slug} / FigS{NN}{p}_{slug}
#   4. every panel has BOTH a .pdf and a .png
#   5. every panel appears in FIGURE_MANIFEST.tsv, and the manifest has no orphan rows
#
# Exit code 0 if all pass, 1 otherwise.
#
# Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
# Nothing is hardcoded, so the repository runs from any directory.


NAME = re.compile(r"^Fig(S?)(\d{2})([a-z])_([a-z0-9_]+?)(_p\d{2})?$")


def pdf_font_sizes(path):
    d = open(path, "rb").read()
    out = set()
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", d, re.S):
        try:
            t = zlib.decompress(m.group(1))
        except Exception:
            t = m.group(1)
        for g in re.finditer(rb"/F\d+\s+([\d.]+)\s+Tf", t):
            out.add(round(float(g.group(1)), 2))
    return out


def pdf_fonts(path):
    d = open(path, "rb").read()
    return set(re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", d))


def main():
    fails, checked = [], 0
    if not os.path.isdir(PANELS):
        print("  no panels directory yet - nothing to audit"); return 0
    perm = set(FS.permitted_sizes())
    stems = sorted({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{PANELS}/*")})
    for stem in stems:
        checked += 1
        if not NAME.match(stem):
            fails.append(f"naming: {stem} does not match Fig{{NN}}{{p}}_{{slug}}")
        pdf, png = f"{PANELS}/{stem}.pdf", f"{PANELS}/{stem}.png"
        if not (os.path.exists(pdf) and os.path.exists(png)):
            fails.append(f"format: {stem} is missing a .pdf or .png")
            continue
        off = sorted(s for s in pdf_font_sizes(pdf) if s not in perm)
        if off:
            fails.append(f"type: {stem} uses {off} pt, outside {sorted(perm)}")
        dv = [f.decode() for f in pdf_fonts(pdf) if b"DejaVu" in f]
        if dv:
            fails.append(f"font: {stem} embeds {dv} - the silent fallback")
    # manifest agreement
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            listed = {r["file"] for r in csv.DictReader(fh, delimiter="\t")}
        for s in stems:
            if s not in listed:
                fails.append(f"manifest: {s} is not listed")
        for s in listed - set(stems):
            fails.append(f"manifest: {s} is listed but no file exists")
    else:
        fails.append("manifest: FIGURE_MANIFEST.tsv missing")
    # source-side: the overrides that defeat the shared style
    whole = open(__file__).read()
    src = whole[:whole.index("# ==== section: composites ====")]      # the panel generators only
    for pat, msg in ((r"fontsize\s*=\s*[0-9]", "literal fontsize="),
                     (r"\.grid\(True", ".grid(True)")):
        if re.search(pat, src):
            fails.append(f"source: figures.py contains {msg} - style belongs in figstyle")
    print(f"  audited {checked} panel(s)")
    for f in fails:
        print(f"    FAIL  {f}")
    print(f"  {'PASS' if not fails else 'FAILED'}: {len(fails)} problem(s)")
    return 1 if fails else 0


panels_main = main


# ================================================================================================
# ==== dispatch ====
# ================================================================================================
SECTIONS = {m: globals()[m + "_main"] for m in ['equivalence', 'decision', 'mechanism', 'phases_cost', 'hardware', 'supp_ported', 'supplementary', 'trees', 'composites', 'panels']}
SECTIONS["audit"] = SECTIONS.pop("panels")
ORDER = ["equivalence", "decision", "mechanism", "phases_cost", "hardware", "supp_ported",
         "supplementary", "trees", "composites", "audit"]


def _dispatch(argv):
    names = argv or ORDER
    bad = [n for n in names if n not in SECTIONS]
    if bad:
        sys.exit(f"figures.py: unknown section(s) {bad}; choose from {ORDER}")
    if any(n not in ("audit",) for n in names):
        FS.apply()                      # the one publication style, once, before any drawing
    code = 0
    for n in names:
        r = SECTIONS[n]()
        code = code or (r or 0)
    return code


if __name__ == "__main__":
    sys.exit(_dispatch(sys.argv[1:]))
