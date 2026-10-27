#!/usr/bin/env python3
"""mechanism.py - Figure 3: runtime scaling with total likelihood workload.

Two panels, one message: GPU wall time grows more slowly than CPU wall time as the total likelihood
workload increases. THE TWO PANELS SHARE ONE SCALE (`limits`, from every fitted row of either
thread count), so the 20-thread and the one-thread cost law can be read side by side at the same
length per decade.

ENCODING. Marks are circles only: open + dashed = IQ-TREE3 (the CPU arm), filled + solid =
IQ-TREE3-GPU (the GPU arm); hue = alphabet. The legend under each panel has one entry per series,
drawn as the series is and carrying its slope at the smallest and the largest work units; R^2 and the paired slope difference
sit in the table under the legend. Nothing is written on the plot itself.

THE FIT IS QUADRATIC IN LOG-LOG, NOT A STRAIGHT LINE. Every series is concave-up on this scale
(the curvature term is positive with a bootstrap interval excluding zero in seven of the eight
series): small jobs scale more gently than large ones and the local exponent rises towards 1 at
the largest workloads. A straight line would report one average exponent and hide that, so the
curve drawn is figdata.quadratic_fit and the exponent is reported at the SMALLEST and the LARGEST
measured workload ("0.28 to 0.91"). The straight-line average is still in KEY_NUMBERS
(costlaw_*_exponent) beside the end exponents (costlaw_*_exponent_smallest / _largest).

  a  the cost law at 20 threads: wall time of each arm against work units, fitted log-log, with the
     explained variation (R^2) of every fit stated on the panel. The headline fit is the L40S alone,
     because the GPU arm pooled six cards and a sub-unity exponent could otherwise reflect the card
     mixture rather than a property of the backend; the pooled six-card fit is drawn faint as a
     reference. Thread count was already stratified.
  b  the same law at one thread, the stratum where the largest speedups occur.

WORK UNITS = likelihood calls x m k^2 r: the number of node-level likelihood calls the search made,
times the per-call work (patterns x states^2 x rate categories). The definition lives in
lib/figdata.work_units() and nowhere else. Every exponent carries n and a paired bootstrap CI, and
whether the GPU exponent really is below the CPU exponent is tested as a PAIRED difference on the
same cells (figdata.slope_difference), not by comparing two marginal CIs, which would wrongly say
"no" for nucleotide.

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
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"
HEADLINE_CARD = "NVIDIA_L40S"
XLABEL = "likelihood calls  x  m k$^2$ r   (work units)"
FIT_HEADER = ("slope: of the quadratic log-log fit, at the smallest\n"
              "to the largest work units\n"
              "R$^2$: explained variation, IQ-TREE3 / IQ-TREE3-GPU\n"
              "difference: IQ-TREE3 $-$ IQ-TREE3-GPU slope, paired")


ARM_LABEL = {"cpu": "IQ-TREE3", "gpu": "IQ-TREE3-GPU"}


def limits(rows):
    """One x range and one y range for BOTH panels, from every fitted row of either thread count
    and both arms."""
    xs = [FD.work_units(r) for r in rows]
    ys = [FD.num(r.get(k)) for r in rows for k in ("cpu_wall_s", "gpu_wall_s")]
    xs = [x for x in xs if x]; ys = [y for y in ys if y]
    return (min(xs) / 2.5, max(xs) * 2.5), (min(ys) / 2.5, max(ys) * 2.5)


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


def _arm_legend(ax, fits, pooled=False):
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
    if pooled:
        h.append(Line2D([], [], color=FS.MUTED, lw=0.6, alpha=0.55, label="pooled six-card fit"))
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


def panel_a(rows, xlim, ylim):
    """Cost law at 20 threads: L40S headline fit per arm and alphabet, pooled six-card fit faint."""
    fig, ax = FS.square_axes()
    fits = {}
    n_card = 0
    for dt in ("protein", "dna"):
        base = [r for r in rows if r["data_type"] == dt and r["threads"] == "20"]
        n_card += sum(1 for r in base if r.get("card") == HEADLINE_CARD)
        # pooled six-card fit, drawn faint as a robustness reference (same quadratic form)
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pooled = [(FD.work_units(r), FD.num(r.get(key))) for r in base]
            pooled = [(a, b) for a, b in pooled if a and b]
            f = FD.quadratic_fit(pooled, B=200)
            if f:
                xr, yr = FD.quadratic_curve(f)
                ax.plot(xr, yr, color=FS.MUTED, lw=0.6, alpha=0.55, zorder=1)
        # headline: single card
        _arm_fits(ax, [r for r in base if r.get("card") == HEADLINE_CARD], dt, fits)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("wall time (s)")
    n = sum(1 for r in rows if r["threads"] == "20")
    ax.set_title(FS.stratum(20, n) + f"; L40S markers n = {n_card}", loc="left")
    _arm_legend(ax, fits, pooled=True)
    return fig, n, _footnote(fig, ax, fits)


def panel_b(rows, xlim, ylim):
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
    xlim, ylim = limits(rows)
    fa, na, la = panel_a(rows, xlim, ylim)
    fb, nb, lb = panel_b(rows, xlim, ylim)
    out = [(fa, "a", "cost_law", "Cost law at 20 threads, stratified by card", na),
           (fb, "b", "cost_law_t1", "Cost law at one thread", nb)]
    for fig, letter, slug, title, n in out:
        stem = FS.save_panel(fig, 3, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for tag, lines in (("3a", la), ("3b", lb)):
        for l in lines:
            print(f"    {tag} {l}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
