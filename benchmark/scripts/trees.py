#!/usr/bin/env python3
"""trees.py - S04 and S05: the inferred trees, under the naming contract.

S04  one curated montage per alphabet, the 12-rung 20-thread size ladder, ORDERED BY TAXON COUNT
     with the complexity printed in each panel label. (Ordering by complexity, as an earlier draft
     did, produced a taxon order of 4, 6, 14, 10, 8, ... which reads as an error even though it is
     deliberate; taxa order with complexity printed conveys the same fact without the double-take.)

S05  the complete per-run record, every current-build cell, as a multipage montage using the
     contract's `_p{NN}` suffix. Page 1 states that the 44 superseded-build runs are excluded.

ONE TREE PER DATASET. The CPU and GPU arms returned identical topologies in all 182 runs, so a
second tree would be a second copy of the same figure. That identity is the result and it belongs
in the caption, not in duplicated ink.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, re, csv, math, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); H = _load("headline"); FS.apply()
import matplotlib.pyplot as plt
from Bio import Phylo

SRC = "benchmark/runs/RUNS.tsv"
PAIRED = os.path.join(REPO, "benchmark", "runs", "paired")
PANEL_IN = 3.0          # matches the retired generator; 2.4 crowded the labels
PER_PAGE = 12


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
    show_labels = ntip <= 30
    # DISCRETE sizes only. A continuous clamp emits arbitrary values (5.11, 5.75, ...) which are
    # not in the permitted type set and which the panel audit rejects, correctly: "roughly 6 pt"
    # is not a typographic specification.
    fs = 6.0 if ntip <= 8 else 5.0
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


def datasets(rows):
    """dataset -> the cell whose tree we draw, plus its annotation."""
    out, rank = {}, {}
    for r in rows:
        g = r.get("genus") or r["cell"]
        p = _priority(r)
        if g in out and p <= rank[g]:
            continue
        rank[g] = p
        out[g] = dict(cell=r["cell"], dt=r["data_type"], taxa=int(r["taxa"]),
                      cx=r["_complexity"], sp=r["_speedup"], model=r.get("model", ""),
                      thr=r["threads"], head=H.is_headline(r))
    return out


def montage(items, dt, page=None, npages=None, note=None):
    n = len(items)
    # Cap the column count at the number of items. Without this a final page holding a single tree
    # still built a 3-wide grid, so the tree rendered into one third of a page-wide figure with two
    # empty columns beside it -- which is what the last page of the S05 record looked like.
    ncol = min(4 if n > 6 else 3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_IN * ncol, PANEL_IN * nrow))
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
        draw(ax, tf, d["dt"], f"n={d['taxa']}  {d.get('model','')}\ncomplexity {si(d['cx'])}"
                              + (f"  |  {d['sp']:.2f}x" if d.get("sp") else ""))
    kind = "amino acid" if dt == "protein" else ("nucleotide" if dt == "dna" else "all runs")
    ttl = f"Inferred maximum-likelihood trees - {kind}"
    if page:
        ttl += f"   (page {page} of {npages})"
    ttl += "\nordered by taxon count; one tree per dataset - the CPU and GPU arms returned identical topologies"
    if note:
        ttl += f"\n{note}"
    fig.suptitle(ttl, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def main():
    rows = FD.load(REPO)
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    ds = datasets(rows)

    # ---- S04: the curated 20-thread ladder, one montage per alphabet -------------------------
    for dt, num in (("protein", 2), ("dna", 2)):
        items = sorted([d for d in ds.values()
                        if d["dt"] == dt and d["head"] and d["thr"] == "20"],
                       key=lambda d: d["taxa"])
        if not items:
            continue
        fig = montage(items, dt)
        letter = "a" if dt == "protein" else "b"
        stem = FS.save_panel(fig, num, letter, f"tree_ladder_{'aa' if dt=='protein' else 'nt'}",
                             f"Tree montage, {'amino acid' if dt=='protein' else 'nucleotide'} ladder",
                             os.path.basename(__file__), SRC, len(items), supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}  ({len(items)} trees)")

    # ---- S05: every current-build dataset, multipage ------------------------------------------
    allds = sorted(ds.values(), key=lambda d: (d["dt"], d["taxa"], d["cell"]))
    pages = math.ceil(len(allds) / PER_PAGE)
    for i in range(pages):
        chunk = allds[i * PER_PAGE:(i + 1) * PER_PAGE]
        note = ("every dataset measured on the reported build; the 44 runs made with the superseded "
                "binary are excluded" if i == 0 else None)
        fig = montage(chunk, "all", page=i + 1, npages=pages, note=note)
        stem = FS.save_panel(fig, 3, "a", "tree_record_all", "Complete per-run tree record",
                             os.path.basename(__file__), SRC, len(chunk), supp=True,
                             page=i + 1, data_date=date)
        plt.close(fig)
    print(f"  wrote FigS03a_tree_record_all_p01..p{pages:02d}  ({len(allds)} datasets)")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
