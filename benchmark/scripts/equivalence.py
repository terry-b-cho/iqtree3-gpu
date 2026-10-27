#!/usr/bin/env python3
"""equivalence.py - Figure 1: the GPU returns the same answer.

  a  the relative log-likelihood difference between the two arms as an ECDF per alphabet, the
     difference on the vertical axis against the cumulative fraction of runs, with the 1e-4
     tolerance as a rule. The bit-identical runs are the level each curve starts at, drawn at a
     labelled '0' position below the decades: 100 of 182 runs are BIT-IDENTICAL and the rest sit
     about 50-fold inside the tolerance.
  b  the per-run maximum UFBoot support difference over shared splits as a dot plot, one dot per
     run stacked at its value, colour = alphabet, marker = thread count, carrying the topology
     result (RF = 0 in every run) in one line. No tolerance or threshold is drawn.

WHAT "MAX UFBOOT SUPPORT DIFFERENCE OVER SHARED SPLITS" MEANS. Each arm writes a consensus tree
(<arm>.contree) whose internal branches carry ultrafast-bootstrap support (percent of 1000
bootstrap trees containing that split). The harness (paired_benchmark.sh, equivalence block) takes
every split present in BOTH arms' consensus trees -- all of them, since RF = 0 -- and the absolute
difference of the two support values on it; the run's value is the LARGEST of those differences,
in support points. A run at 0 has identical support on every branch; a run at 2 has one branch
where the arms differ by two points (say 98 against 100) and no branch differing by more. Panel b
re-reads the per-split supports from the stored consensus trees with the harness's own parser
(split_supports) and asserts that the maximum it finds equals the support_max_absdiff column of
every run before drawing.

MARKER SHAPE IN PANEL B. Elsewhere marker shape carries alphabet beside hue (figstyle). Here the
two alphabets are already separated by hue and shape is given to THREAD COUNT (triangle = 20
threads, circle = 1 thread), so the panel shows both strata without a second hue.

Histogram forms of both panels and five further drafts of panel b were retired on 2026-10-26.

THE GATE-FAILURE CELL IS RETAINED HERE. It is excluded from fits and speedup summaries everywhere
else, but removing it from the figure whose purpose is to display worst cases would be circular.

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
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"
PAIRED = os.path.join(REPO, "benchmark", "runs", "paired")
YLAB = "relative log-likelihood difference\n|$\\Delta$logL| / |logL|"
ALPHA = {"protein": "amino acid", "dna": "nucleotide"}
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


def panel_a(rows):
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
        ax.step(fr, vals, where="pre", color=col, lw=0.9, zorder=3, label=ALPHA[dt])
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


def panel_b(rows):
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
                       label=f"{ALPHA[dt]}, {th}T ({n_s})")
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
    for fn, letter, slug, title in (
            (panel_a, "a", "logl_reldiff_ecdf", "Relative log-likelihood difference vs tolerance, ECDF"),
            (panel_b, "b", "support_diff_dotplot",
             "UFBoot support difference per run, with the RF result")):
        fig, n = fn(rows)
        stem = FS.save_panel(fig, 1, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
