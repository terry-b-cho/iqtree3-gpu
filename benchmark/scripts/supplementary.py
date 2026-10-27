#!/usr/bin/env python3
"""supplementary.py - Supplementary Figs. S3, S5 and S7.

  S3  model coverage REGENERATED FROM DATA (the previous figure was a hand-drawn schematic, which
      cannot stand in a manuscript whose authority rests on measurement)
  S5  phase-step heatmap from PHASE_STEPS.tsv, with the mean share of every step written on each
      alphabet's block
  S7  campaign coverage: runs per taxon count as stacked bars by workflow and thread count (a),
      and the same faceted by card (b), so every one of the 182 paired runs is counted once

S1 and S2 (the trees) are written by trees.py; S4 by supp_ported.py; S6 by hardware.py.
The former pattern-saturation (S5), build-regression (S7) and coverage dot-matrix panels were
removed on 2026-10-26.

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

# Alphabet is perfectly aliased with model treatment; label the WORKFLOW.
WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}


def phase_step_heatmap(fit_rows):
    """Step x cell, CPU arm. Shows at a glance which steps exist for which alphabet.

    `fit_rows` restricts to current-build cells: PHASE_STEPS.tsv covers all 226 recorded runs,
    including the 44 on the superseded binary, which must not enter a published figure.
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


# ------------------------------------------------------------------ S7 campaign coverage
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
    """S7a: every paired run counted once, by taxon count, workflow and thread count."""
    taxa = sorted({int(r["taxa"]) for r in rows})
    fig, ax = FS.square_axes()
    _stacked_bars(ax, rows, taxa)
    ax.set_ylabel("paired runs")
    ax.set_title(FS.stratum({r["threads"] for r in rows}, len(rows)), loc="left")
    ax.legend(loc="upper right")
    return fig, len(rows)


# S7B colours. The five sweep cards take five Okabe-Ito hues that are neither the alphabet pair
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


def coverage_bars_by_card(rows):
    """S7B, two boxes: L40S alone (left; dark grey 20 threads, light grey 1 thread) and the five
    sweep cards stacked in one box (right; one colour per card, all at 20 threads). In both,
    solid = amino acid and striped = nucleotide."""
    from matplotlib.patches import Patch
    taxa = sorted({int(r["taxa"]) for r in rows})
    fig, axes = FS.square_grid(1, 2, sharex=True, sharey=False, gap=0.5)
    ax, bx = axes[0, 0], axes[0, 1]
    l40 = [r for r in rows if r["card"] == "NVIDIA_L40S"]
    sweep = [r for r in rows if r["card"] in SWEEP_COLOR]
    groups = [(L40S_COLOR[th], HATCH[dt], (lambda r, th=th, dt=dt: r["threads"] == th and r["data_type"] == dt))
              for th in ("20", "1") for dt in ("protein", "dna")]
    _stack(ax, l40, taxa, groups)
    ax.set_title(f"L40S (n={len(l40)})", pad=2); ax.set_ylabel("paired runs")
    groups = [(SWEEP_COLOR[c], HATCH[dt], (lambda r, c=c, dt=dt: r["card"] == c and r["data_type"] == dt))
              for c in SWEEP_COLOR for dt in ("protein", "dna")]
    _stack(bx, sweep, taxa, groups)
    bx.set_title(f"card sweep, five cards (n={len(sweep)})", pad=2)
    bx.tick_params(axis="x", labelbottom=True)
    key = [Patch(facecolor=L40S_COLOR["20"], edgecolor="#444444", lw=0.3, label="L40S, 20 threads"),
           Patch(facecolor=L40S_COLOR["1"], edgecolor="#444444", lw=0.3, label="L40S, 1 thread"),
           Patch(facecolor="white", edgecolor="#444444", lw=0.3, label="solid: amino acid"),
           Patch(facecolor="white", edgecolor="#444444", lw=0.3, hatch="////", label="striped: nucleotide")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    key = [Patch(facecolor=SWEEP_COLOR[c], edgecolor="#444444", lw=0.3,
                 label=f"{CARD_LABEL[c]} (n={sum(1 for r in sweep if r['card'] == c)})")
           for c in SWEEP_COLOR]
    bx.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    FS.grid_note(fig, axes, FS.stratum({r["threads"] for r in rows}, len(rows)))
    return fig, len(rows)


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    SRC = "benchmark/runs/RUNS.tsv"
    out = [(model_coverage(FD.load(REPO)), 3, "a", "model_coverage_data",
            "Model classes exercised", SRC),
           (phase_step_heatmap(rows), 5, "a", "phase_step_heatmap",
            "Phase-step share per run, with the mean per alphabet", "benchmark/runs/PHASE_STEPS.tsv"),
           (coverage_bars(FD.load(REPO)), 7, "a", "coverage_stacked_bars",
            "Campaign coverage: runs per taxon count, stacked", SRC),
           (coverage_bars_by_card(FD.load(REPO)), 7, "b", "coverage_by_card",
            "Campaign coverage by card", SRC)]
    for (fig, n), num, letter, slug, title, src in out:
        stem = FS.save_panel(fig, num, letter, slug, title, os.path.basename(__file__), src, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
