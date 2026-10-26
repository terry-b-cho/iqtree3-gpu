#!/usr/bin/env python3
"""decision.py - Figure 2: when to offload.

This is the figure a practitioner came for. It answers "should I add -gpu to this analysis?" from a
quantity they can compute before submitting the job.

TWO THINGS THE EARLIER DRAFT GOT WRONG, both fixed here:
  * The crossover was quoted from a DISCRETE estimator (geometric midpoint of the parity gap) whose
    point estimate sat on its own lower bound -- an artefact of granularity. The estimator here is
    OLS of log10(speedup) on log10(complexity) with the crossover read off the fitted line, named in
    the caption, with a paired bootstrap CI.
  * The crossover is NOT nucleotide-only. Two amino-acid runs at 20 threads fall below parity, and
    the amino-acid crossover is estimable ~2 decades below the nucleotide one. A protein user with a
    small alignment is exactly the reader who needs this figure.

Points are NEVER joined by lines: each is a different dataset, and a line between them would imply a
trajectory that does not exist.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, collections, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"
W, H = 2.75, 2.15

# Alphabet is PERFECTLY ALIASED with model treatment in this campaign (85/85 amino-acid runs use
# ModelFinder, 97/97 nucleotide runs pin a model). Nothing here can separate the two, so every label
# names the WORKFLOW. See figdata.workflow().
WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}

# The speedups a reader actually decides on. Parity answers "is it worth it at all"; 2x and 5x
# answer "is it worth reorganising my pipeline".
TARGETS = (2.0, 5.0)

# Absolute times, so the ratios are not abstract. Both are amino-acid runs; the nucleotide series
# has no run above 1.57x, which is itself the point that panel's title makes.
ANCHORS = (("g__Colicola__T20__l40s_cx", "largest run 17.6 h to 3.4 h"),
           ("n10_g__Coprovivens__T20__l40s_r2", "mid-sized run 14 min to 7 min"))
# Both labels live in ONE block in the sub-parity band at high complexity -- the only region of this
# panel with no data in it -- with leader lines out to their points. Annotating each point in place
# put the text straight through the clouds.
ANCHOR_BLOCK = (2.2e10, 0.52)


def _decision_panel(ax, rows, threads, annotate_examples=True):
    """Speedup vs complexity with the parity line, shaded regions and the fitted crossover."""
    stats = {}
    for dt in ("protein", "dna"):
        pts = [(r["_complexity"], r["_speedup"]) for r in rows
               if r["data_type"] == dt and r["threads"] == threads
               and r["_complexity"] and r["_speedup"]]
        if len(pts) < 5:
            continue
        col = FS.ALPHABET[dt]
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=col, zorder=3,
                   marker=FS.alphabet_marker(dt), label=f"{WF_SHORT[dt]} (n={len(pts)})")
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
    # a bare log axis labels only decades; speedup lives in 0.4-8, so label the values readers use
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    tk = [0.5, 1, 2, 5, 10]
    ax.yaxis.set_major_locator(FixedLocator(tk))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}x" for t in tk]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("tree complexity   branches x patterns x rate cat x states$^2$")
    ax.set_ylabel("speedup (CPU wall / GPU wall)")
    return stats


def panel_a(rows):
    """The decision panel. Beyond the parity crossover it also answers "where do I get 2x, 5x?",
    which is the question a reader who is already convinced actually has.

    A target is marked ONLY where the fitted line reaches it inside the measured complexity range.
    The nucleotide fit reaches neither 2x nor 5x anywhere we measured, so nothing is drawn for it and
    no extrapolated promise is made.
    """
    fig, ax = plt.subplots(figsize=(W, H))
    st = _decision_panel(ax, rows, "20")
    lo, hi = ax.get_ylim()
    # faint rules at the decision targets; the y axis already labels 2x and 5x
    for t in TARGETS:
        ax.axhline(t, color=FS.MUTED, lw=0.4, ls=":", zorder=0)
    # where each workflow's fitted curve reaches each target, if it does so within the data
    reached = {}
    for dt, c in st.items():
        span = [r["_complexity"] for r in rows
                if r["data_type"] == dt and r["threads"] == "20" and r["_complexity"]]
        xlo, xhi = min(span), max(span)
        hits = []
        for t in TARGETS:
            xt = c["crossover"] * 10 ** (np.log10(t) / c["slope"])
            if not (xlo <= xt <= xhi):
                continue                      # outside the measured range: promise nothing
            ax.scatter([xt], [t], s=16, facecolors="none", edgecolors=FS.ALPHABET[dt],
                       linewidths=0.8, zorder=5)
            hits.append(f"{t:g}x at {xt:.0e}".replace("e+0", "e").replace("e+", "e"))
        if hits:
            reached[dt] = ", ".join(hits)
    # absolute times: the ratio means nothing without the hours it saves
    idx = {r["cell"]: r for r in rows}
    bx, byy = ANCHOR_BLOCK
    shown = [(idx[c], lab) for c, lab in ANCHORS if idx.get(c) and idx[c]["_complexity"]]
    if shown:
        ax.annotate("\n".join(lab for _, lab in shown), xy=(bx, byy), ha="center", va="center",
                    color=FS.INK, zorder=6)
        for r, _ in shown:
            ax.annotate("", xy=(r["_complexity"], r["_speedup"]), xytext=(bx, byy),
                        textcoords="data",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=FS.MUTED,
                                        shrinkA=9, shrinkB=3))
    # Region labels hug the parity line at the LEFT edge, the only area free of data in both
    # halves. Centring them in each region put them straight through the point clouds.
    x0 = ax.get_xlim()[0] * 1.8
    ax.text(x0, 1.18, "GPU faster", color=FS.MUTED, style="italic", va="bottom")
    ax.text(x0, 0.86, "CPU faster", color=FS.MUTED, style="italic", va="top")
    # Secondary abscissa: complexity as dataset shapes, so the axis is usable without arithmetic.
    # Labels are single-line and terse -- two-line labels at these positions overlapped each other.
    ex = [(17 * 1_000 * 4 * 400, "10 x 1k"),
          (97 * 10_000 * 4 * 400, "50 x 10k"),
          (397 * 50_000 * 4 * 400, "200 x 50k")]
    top = ax.secondary_xaxis("top")
    top.set_xticks([c for c, _ in ex]); top.set_xticklabels([l for _, l in ex])
    top.set_xlabel("example amino-acid analyses, taxa x patterns (LG+G4)", labelpad=1)
    def _sci(v):                      # 1.1e+07 -> 1.1 x 10^7, as the journal sets it
        m, e = f"{v:.1e}".split("e"); return f"{m} $\\times$ 10$^{{{int(e)}}}$"
    sub = "; ".join(f"{'AA+MF' if d=='protein' else 'NT fixed'} {_sci(s['crossover'])}"
                    for d, s in st.items())
    ax.set_title(f"At 20 threads the two workflows break even at\n"
                 f"{sub}  (OLS fit, shaded = 95% bootstrap CI)", pad=24)
    # One block, upper left -- the only region of this panel free of data. The open rings on the
    # fitted curves are keyed here rather than labelled in place, which collided at every offset.
    for dt, txt in reached.items():
        ax.scatter([], [], s=16, facecolors="none", edgecolors=FS.ALPHABET[dt], linewidths=0.8,
                   label=f"reaches {txt}")
    ax.legend(loc="upper left", frameon=False, handlelength=0.9, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.25)
    return fig, sum(len(FD.for_fits(rows)) for _ in [0]), st


def panel_b(rows):
    fig, ax = plt.subplots(figsize=(W, H))
    st = _decision_panel(ax, rows, "1")
    n = sum(1 for r in rows if r["threads"] == "1")
    ax.set_title(f"With one host thread the break-even moves far to the left\n"
                 f"(n = {n})")
    ax.legend(loc="lower right")
    return fig, n, st


def panel_c(rows):
    """Slope plot: the single most important honest statement in the paper."""
    by = collections.defaultdict(dict)
    for r in rows:
        if r["_speedup"]:
            by[(r["genus"], r["data_type"])][r["threads"]] = r["_speedup"]
    both = {k: v for k, v in by.items() if "1" in v and "20" in v}
    fig, ax = plt.subplots(figsize=(W, H))
    for (g, dt), v in sorted(both.items()):
        col = FS.ALPHABET[dt]
        ax.plot([0, 1], [v["1"], v["20"]], color=col, lw=0.7, alpha=0.75, zorder=2)
        ax.scatter([0, 1], [v["1"], v["20"]], s=8, color=col, zorder=3,
                   marker=FS.alphabet_marker(dt))
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    tk = [0.25, 0.5, 1, 2, 5, 10, 20, 40]
    ax.yaxis.set_major_locator(FixedLocator(tk))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}x" for t in tk]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.25, 1.25); ax.set_xticks([0, 1]); ax.set_xticklabels(["1 thread", "20 threads"])
    ax.set_ylabel("speedup (CPU wall / GPU wall)")
    nprot = sum(1 for (g, dt) in both if dt == "protein")
    import statistics as _st
    coll = _st.median([v["1"] / v["20"] for v in both.values()])
    # ONLY datasets measured at both thread counts may be joined. The 12-genus 1-thread amino-acid
    # set and the 12-genus 20-thread ladder are DIFFERENT alignments, so the campaign's 34x-vs-3.75x
    # comparison is between two distributions, not within datasets. This panel is the within-dataset
    # measurement, and it is the defensible one.
    ax.set_title(f"Within a dataset, the single-thread advantage falls\n"
                 f"median {coll:.1f}x over {len(both)} datasets run at both ({nprot} amino acid)")
    for dt in ("protein", "dna"):
        ax.plot([], [], color=FS.ALPHABET[dt], lw=0.9, marker=FS.alphabet_marker(dt),
                ms=3, label=WF_SHORT[dt])
    ax.legend(loc="lower left")
    return fig, len(both)


def panel_d(rows):
    """Per-invocation cost by arm and workflow.

    THIS DECOMPOSITION IS EXACT BY CONSTRUCTION, AND THE CAPTION MUST SAY SO. There is no CPU-arm
    invocation count anywhere in the data: `cpu_calls` records GPU-arm host fallbacks (0 in every
    row), and `gpu_calls` is used as the denominator for BOTH arms. Per-invocation cost is therefore
    wall / gpu_calls for each arm, and the ratio of the two is algebraically the wall-time ratio.
    Nothing is tested here; the panel re-expresses the speedup on a per-call axis.

    What is NOT tautological, and is the finding to state: a 20-state kernel call costs the GPU no
    more than a 4-state one, so protein data are where the device is underused least. The CPU pays
    2.89x more per amino-acid call than per nucleotide call; the GPU pays 0.61x -- less.

    THE CONFOUND IS TOTAL AND NOT REDUCIBLE. Every one of the 85 amino-acid runs ran ModelFinder and
    every one of the 97 nucleotide runs pinned a model, with zero exceptions either way; the
    nucleotide series is additionally simulated and ungapped, against 19-49% gaps in the amino-acid
    alignments. Alphabet, model search, and real-versus-simulated data are perfectly aliased, so the
    contrast is between WORKFLOWS and no analysis of these data can attribute it to the alphabet.
    """
    fig, ax = plt.subplots(figsize=(W, H))
    for dt in ("protein", "dna"):
        col = FS.ALPHABET[dt]
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pts = []
            for r in rows:
                if r["data_type"] != dt or r["threads"] != "20" or not r["_complexity"]:
                    continue
                w, gc = FD.num(r.get(key)), FD.num(r.get("gpu_calls"))
                if w and gc and gc > 0:
                    pts.append((r["_complexity"], w / gc))
            if len(pts) < 5:
                continue
            stl = FS.arm_style(arm, col)
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=6, zorder=3,
                       marker=FS.alphabet_marker(dt), facecolors=stl["markerfacecolor"],
                       edgecolors=col, linewidths=0.5)
            f = FD.loglog_fit(pts)
            xr = np.logspace(np.log10(min(a for a, _ in pts)), np.log10(max(a for a, _ in pts)), 30)
            ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)), color=col, lw=0.9,
                    ls=stl["linestyle"], zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tree complexity")
    ax.set_ylabel("cost per kernel invocation (s)")
    # Round 8 (R15): this panel moved from Fig. 2D to Supplementary Fig. S18 and the title is
    # descriptive. Wall time is divided by the GPU-arm invocation count for BOTH arms, so the ratio
    # of the two per-call costs equals the speedup and the panel re-expresses Fig. 2A per call.
    ax.set_title("Wall time per device invocation, by arm and workflow\n"
                 "(one device-side invocation count for both arms)", pad=3)
    for dt in ("protein", "dna"):
        ax.plot([], [], color=FS.ALPHABET[dt], marker=FS.alphabet_marker(dt), ms=3, lw=0.9,
                label=WF_SHORT[dt])
    ax.plot([], [], color=FS.MUTED, ls="--", lw=0.9, label="open + dashed = CPU arm")
    ax.legend(loc="lower right")
    return fig, sum(1 for r in rows if r["threads"] == "20")


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    out = []
    fa, na, st20 = panel_a(rows); out.append((fa, "a", "decision_t20", "Speedup vs complexity, 20 threads", na))
    fb, nb, st1 = panel_b(rows);  out.append((fb, "b", "decision_t1", "Speedup vs complexity, 1 thread", nb))
    fc, nc = panel_c(rows);       out.append((fc, "c", "thread_slope", "Single- to 20-thread speedup per dataset", nc))
    for fig, letter, slug, title, n in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, 2, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    # Supplementary Fig. S8: per-invocation wall time by arm
    fd, nd = panel_d(rows)
    fd.tight_layout()
    stem = FS.save_panel(fd, 8, "a", "per_invocation_cost",
                         "Wall time per device invocation, by arm and workflow (shared denominator)",
                         os.path.basename(__file__), SRC, nd, supp=True, data_date=date)
    plt.close(fd); print(f"  wrote {stem}")
    for th, st in (("20T", st20), ("1T", st1)):
        for dt, s in st.items():
            print(f"    {th} {dt:8s} crossover {s['crossover']:.3e} "
                  f"CI [{s['ci'][0]:.2e}, {s['ci'][1]:.2e}] slope {s['slope']:.3f} "
                  f"R2 {s['r2']:.2f} n={s['n']}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
