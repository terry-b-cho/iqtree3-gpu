#!/usr/bin/env python3
"""equivalence.py - Figure 1: the GPU returns the same answer.

Three panels. The board's largest gap was that we held `logl_reldiff` for all 182 runs and reported
only the worst case; the distribution is far stronger than the scalar, because 100 of 182 runs are
BIT-IDENTICAL and the rest sit about 50-fold inside the tolerance.

PANEL D WAS RETIRED. The Robinson-Foulds distribution was a single bar at zero (182 of 182), which
is a result but not a figure: a degenerate distribution drawn as a histogram invites the reader to
look for spread that does not exist. The result is now stated on panel c and in its caption.

THE GATE-FAILURE CELL IS RETAINED HERE. It is excluded from fits and speedup summaries everywhere
else, but removing it from the figure whose purpose is to display worst cases would be circular: it
fails the gate ON the support criterion itself. Both worst cases are reported, with and without it.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, collections, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"
W, H = 2.6, 2.0          # single column panel


def panel_a(rows):
    """logL relative difference against the 1e-4 tolerance, with an explicit zero bin."""
    v = [FD.num(r.get("logl_reldiff")) for r in rows]
    v = [x for x in v if x is not None]
    zero = [x for x in v if x == 0]
    pos = [x for x in v if x > 0]
    fig, ax = plt.subplots(figsize=(W, H))
    lo, hi = min(pos), FD.LOGL_CONTRACT
    edges = np.logspace(np.log10(lo) - 0.3, np.log10(hi) + 0.3, 22)
    ax.hist(pos, bins=edges, color=FS.PROTEIN, edgecolor="white", linewidth=0.3)
    # the 100 bit-identical runs cannot sit on a log axis; give them their own labelled bar
    zx = edges[0] * 0.35
    ax.bar([zx], [len(zero)], width=zx * 0.8, color=FS.MUTED, edgecolor="white", linewidth=0.3)
    # place the label to the RIGHT of the bar: centred above it, it overlaps the y-tick labels
    ax.annotate(f"bit-identical\n(n = {len(zero)})", xy=(zx, len(zero)), xytext=(6, 2),
                textcoords="offset points", ha="left", va="bottom")
    ax.axvline(FD.LOGL_CONTRACT, color=FS.DNA, lw=0.9, ls="--")
    ax.annotate("tolerance\n1e-4", xy=(FD.LOGL_CONTRACT, ax.get_ylim()[1] * 0.62),
                xytext=(-3, 0), textcoords="offset points", ha="right", color=FS.DNA)
    ax.set_xscale("log")
    ax.set_ylim(0, len(zero) * 1.42)          # headroom: the zero-bin label must clear the title
    ax.set_xlabel("relative log-likelihood difference  |$\\Delta$logL| / |logL|")
    ax.set_ylabel("runs")
    nt = sum(1 for r in rows if r["data_type"] == "dna" and FD.num(r.get("logl_reldiff")) == 0)
    # R12 (round-8 review): the title used to say "by four orders"; the largest difference is
    # 1.9e-6 against a 1e-4 tolerance, about 50-fold. The margin is a key, not a literal.
    margin = FD.key_number(REPO, "logl_margin_fold")
    ax.set_title(f"All {len(v)} runs inside the 1e-4 tolerance; the largest difference\n"
                 f"is {margin:.0f}-fold below it; {len(zero)} exactly 0 ({nt} of them nucleotide)")
    return fig, len(v)


def panel_b(rows):
    """Absolute difference against magnitude, over the NON-IDENTICAL runs only.

    The 100 bit-identical runs have an absolute difference of exactly 0 and cannot be drawn on a log
    axis; including them would either blank them silently or force a symlog floor that compresses the
    real spread. Panel a is where the reader learns there are 100 of them. This panel therefore shows
    the 82 runs that differ at all, and says so in its title so the n is never mistaken for 182.
    """
    fig, ax = plt.subplots(figsize=(W, H))
    for dt in ("protein", "dna"):
        pts = [(abs(FD.num(r.get("cpu_logl"))), FD.num(r.get("logl_absdiff")))
               for r in rows if r["data_type"] == dt
               and FD.num(r.get("cpu_logl")) and FD.num(r.get("logl_absdiff"))]
        pts = [(a, b) for a, b in pts if b > 0]
        if pts:
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=6, color=FS.ALPHABET[dt],
                       label="amino acid" if dt == "protein" else "nucleotide", zorder=3, marker=FS.alphabet_marker(dt))
    g = [r for r in rows if FD.is_gate_failure(r)]
    if g and FD.num(g[0].get("logl_absdiff")):
        ax.scatter([abs(FD.num(g[0]["cpu_logl"]))], [FD.num(g[0]["logl_absdiff"])], s=26,
                   facecolors="none", edgecolors=FS.MUTED, linewidths=0.9, zorder=4,
                   label="gate failure (support)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|log-likelihood|"); ax.set_ylabel("absolute difference")
    n = sum(1 for r in rows if (FD.num(r.get("logl_absdiff")) or 0) > 0)
    ax.set_title(f"Absolute difference grows with magnitude,\n"
                 f"consistent with floating-point rounding ({n} non-identical runs)")
    ax.legend(loc="upper left")
    return fig, n


def panel_c(rows):
    """Support-difference distribution. The 6.0 IS the gate-failure cell and is labelled as such.

    This panel also carries the topology result, because the panel that used to hold it (a single bar
    at RF = 0) was retired as a degenerate distribution.
    """
    v = [FD.num(r.get("support_max_absdiff")) or 0.0 for r in rows]
    cnt = collections.Counter(v)
    fig, ax = plt.subplots(figsize=(W, H))
    ks = sorted(cnt)
    ax.bar(ks, [cnt[k] for k in ks], width=0.5, color=FS.PROTEIN, edgecolor="white", linewidth=0.3)
    for k in ks:
        ax.annotate(str(cnt[k]), xy=(k, cnt[k]), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom")
    ax.axvline(FD.SUPPORT_GATE_TOL, color=FS.DNA, lw=0.9, ls="--")
    ax.annotate("gate tolerance\n5 points", xy=(FD.SUPPORT_GATE_TOL, max(cnt.values()) * 0.45),
                xytext=(-3, 0), textcoords="offset points", ha="right", color=FS.DNA)
    g = max(ks)
    ax.annotate("the one gate failure\n(fails on support only)", xy=(g, cnt[g]),
                xytext=(-4, 20), textcoords="offset points", ha="right", color=FS.MUTED,
                arrowprops=dict(arrowstyle="->", lw=0.5, color=FS.MUTED))
    ax.set_ylim(top=max(cnt.values()) * 6)    # symlog headroom for the bar labels
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("max UFBoot support difference over shared splits (points)")
    ax.set_ylabel("runs (log)")
    nz = sum(1 for x in v if x > 0)
    rf0 = sum(1 for r in rows if (FD.num(r.get("rf_dist")) or 0) == 0)
    # the retired panel d in one line, placed inside the axes where the bars are absent
    # left-aligned: right-aligned it ran across the dashed gate-tolerance rule at 5 points
    ax.annotate(f"every topology identical: RF = 0 in {rf0} of {len(rows)}",
                xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                color=FS.INK, style="italic")
    worst = FD.key_number(REPO, "support_max_absdiff"); worst_x = FD.key_number(REPO, "support_max_absdiff_excl_gate_failure")
    ax.set_title(f"{len(v)-nz} of {len(v)} runs identical; worst {worst:.1f} ({worst_x:.1f} excluding\nthe gate-failure cell)")
    return fig, len(v)


def main():
    rows = FD.for_equivalence(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    for fn, letter, slug, title in (
            (panel_a, "a", "logl_reldiff_hist", "Relative log-likelihood difference vs tolerance"),
            (panel_b, "b", "logl_absdiff_vs_magnitude", "Absolute logL difference vs magnitude"),
            (panel_c, "c", "support_diff_hist",
             "UFBoot support difference distribution, with the RF result")):
        fig, n = fn(rows)
        fig.tight_layout()
        stem = FS.save_panel(fig, 1, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
