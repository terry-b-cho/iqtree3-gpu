#!/usr/bin/env python3
"""decision.py - Figure 2: when to offload (a: 20 threads; b: one thread).

This is the figure a practitioner came for. It answers "should I add -gpu to this analysis?" from a
quantity they can compute before submitting the job.

THE TWO PANELS SHARE ONE SCALE. Both are drawn with the same x range, the same y range, the same
ticks and the same top axis (`limits`, computed once from every fitted row), so the 20-thread and
the one-thread case can be read side by side in the composite at the same length per decade.

TWO THINGS THE EARLIER DRAFT GOT WRONG, both fixed here:
  * The crossover was quoted from a DISCRETE estimator (geometric midpoint of the parity gap) whose
    point estimate sat on its own lower bound -- an artefact of granularity. The estimator here is
    OLS of log10(speedup) on log10(complexity) with the crossover read off the fitted line, named in
    the caption, with a paired bootstrap CI.
  * The crossover is NOT nucleotide-only. Two amino-acid runs at 20 threads fall below parity, and
    the amino-acid crossover is estimable ~2 decades below the nucleotide one. A protein user with a
    small alignment is exactly the reader who needs this figure.

Each panel marks where each workflow's fitted line crosses parity and writes that abscissa (the
tree complexity at which the GPU arm starts to win) at the foot of the axis; the shaded band is the
95% bootstrap interval of that crossover.

Points are NEVER joined by lines: each is a different dataset, and a line between them would imply a
trajectory that does not exist.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"

# Alphabet is PERFECTLY ALIASED with model treatment in this campaign (85/85 amino-acid runs use
# ModelFinder, 97/97 nucleotide runs pin a model). Nothing here can separate the two, so every label
# names the WORKFLOW. See figdata.workflow().
WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}

# Secondary abscissa: complexity as dataset shapes, so the axis is usable without arithmetic.
# Labels are single-line and terse; the third is raised a line because it abuts the second.
EXAMPLES = [(17 * 1_000 * 4 * 400, "10$\\times$1k"),
            (97 * 10_000 * 4 * 400, "50$\\times$10k"),
            (397 * 50_000 * 4 * 400, "200$\\times$50k\n")]
RATIO_TICKS = (0.5, 1, 2, 5, 10, 20)


def limits(rows):
    """One x range and one y range for BOTH panels, from every fitted row of either thread count."""
    pts = [(r["_complexity"], r["_speedup"]) for r in rows if r["_complexity"] and r["_speedup"]]
    xs, ys = [a for a, _ in pts], [b for _, b in pts]
    return (min(xs) / 2.5, max(xs) * 2.5), (min(ys) / 1.4, max(ys) * 1.4)


def _sci(v):
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
                                  label=f"{WF_SHORT[dt]} (n={len(pts)})"))
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
        ax.annotate(_sci(c["crossover"]), xy=(c["crossover"], ylim[0]), xytext=(2, 2),
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
    xlim, ylim = limits(rows)
    out = []
    fa, na, st20 = panel(rows, "20", xlim, ylim); out.append((fa, "a", "decision_t20", "Speedup vs complexity, 20 threads", na))
    fb, nb, st1 = panel(rows, "1", xlim, ylim);   out.append((fb, "b", "decision_t1", "Speedup vs complexity, 1 thread", nb))
    for fig, letter, slug, title, n in out:
        stem = FS.save_panel(fig, 2, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for th, st in (("20T", st20), ("1T", st1)):
        for dt, s in st.items():
            print(f"    {th} {dt:8s} crossover {s['crossover']:.3e} "
                  f"CI [{s['ci'][0]:.2e}, {s['ci'][1]:.2e}] slope {s['slope']:.3f} "
                  f"R2 {s['r2']:.2f} n={s['n']}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
