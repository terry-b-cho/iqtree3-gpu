#!/usr/bin/env python3
"""phases_cost.py - Figure 4: where the time goes, and what it costs.

Carries the campaign's single most explanatory fact: model selection is ~88% of amino-acid wall time
and is absent for nucleotide runs, which is most of why the two alphabets behave so differently.

  a  two half-width boxes that together fill one 40 mm square: left, wall share by phase
     (IQ-TREE3) per alphabet as stacked bars; right, the per-run processor time consumed,
     IQ-TREE3 / IQ-TREE3-GPU, as a violin with jittered circles per alphabet
  b  per-phase speedup against tree complexity: model selection (amino acid only, since nucleotide
     runs pin a model), tree search on amino acid, tree search on nucleotide, each with its OLS
     fit, its break-even on the parity line written at the axis foot, and the shaded 95% bootstrap
     CI of that break-even, in the style of Fig. 2
  c  cores kept busy (processor time / wall time) per build and alphabet as violins with jittered
     circles (the former 4D).

COLOUR. Panels a and b colour the PHASE (figstyle.PHASE_COLORS: bluish green = model selection,
reddish purple = tree search, grey = untimed), the same in both panels and distinct from the blue /
vermillion pair that means alphabet everywhere else. Panel c colours the alphabet. Marks are
circles only: filled = amino acid in b; in c filled = IQ-TREE3-GPU, open = IQ-TREE3.

TWO HONESTY CONSTRAINTS the board required, both enforced here:
  * Device time CANNOT be shown. `gpu_kernel_time_s` is NA/TODO_nsight in all 182 rows. What is
    reportable is the bound from the profiling kernel fraction, and its DIRECTION matters: profiling
    inflates host wall more than device kernel time, so the profiled fraction is a LOWER bound on the
    unprofiled one, and the absolute kernel seconds are the usable number.
  * The compute saving is POTENTIAL, not realised. The allocation was not node-exclusive but did
    request -c 20 for both arms, so the cores the GPU arm leaves idle were reserved to our own job
    and unavailable to others. The saving is realisable only if a GPU user requests fewer cores.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

SRC = "benchmark/runs/PHASE_STEPS.tsv"
ALPHA = {"protein": "amino acid", "dna": "nucleotide"}
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


def panel_a(ph, rows):
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


def _sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}$\\times$10$^{{{int(e)}}}$"


def panel_b(ph, rows):
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
                       linewidths=0.6, label=f"{lab}, {ALPHA[dt]} (n={len(pts)})")
        handles.append(h)
        cr = FD.crossover(pts)
        if cr:
            # the break-even value is keyed in the legend entry of its series: three values
            # written on the plot collided with each other and with the data
            h.set_label(f"{lab}, {ALPHA[dt]} (n={len(pts)}): break-even {_sci(cr['crossover'])}")
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
        ax.annotate(_sci(cr["crossover"]), xy=(cr["crossover"], ax.get_ylim()[0]), xytext=(2, 2),
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
        ax.text(2.4 * i + 0.5, 23.4, f"{ALPHA[dt]} (n={len(cells)})", ha="center", va="top",
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
    out = [(panel_a(ph, rows), "a", "phase_share_cpu_time",
            "Wall-time share by phase; processor time consumed, IQ-TREE3 / IQ-TREE3-GPU"),
           (panel_b(ph, rows), "b", "phase_speedup_complexity",
            "Per-phase speedup vs tree complexity, with break-even"),
           (panel_c(ph, rows), "c", "cores_busy", "Cores kept busy per build")]
    for (fig, n), letter, slug, title in out:
        stem = FS.save_panel(fig, 4, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
