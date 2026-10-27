#!/usr/bin/env python3
"""trees.py - Supplementary Figs. S1 and S2: the inferred trees, under the naming contract.

S1   one curated montage per alphabet, the 12-rung 20-thread size ladder, ORDERED BY TAXON COUNT
     with the complexity printed in each panel label. (Ordering by complexity, as an earlier draft
     did, produced a taxon order of 4, 6, 14, 10, 8, ... which reads as an error even though it is
     deliberate; taxa order with complexity printed conveys the same fact without the double-take.)

S2   the complete per-run record, every current-build cell, as a multipage montage using the
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
    # S1: every amino-acid inference; S2: every nucleotide inference (one tree per dataset x
    # thread count x card x model; same-settings replicates collapse), 16 to a Letter page.
    for dt, num, tag in (("protein", 1, "aa"), ("dna", 2, "nt")):
        items = sorted([d for d in ds.values() if d["dt"] == dt],
                       key=lambda d: (d["taxa"], d["model"], d["thr"] != "20", d["card"] != "L40S", d["cell"]))
        pages = math.ceil(len(items) / PER_PAGE)
        for i in range(pages):
            chunk = items[i * PER_PAGE:(i + 1) * PER_PAGE]
            fig = montage(chunk, dt, page=i + 1, npages=pages)
            stem = FS.save_panel(fig, num, "a", f"trees_{tag}",
                                 f"Every {'amino-acid' if dt == 'protein' else 'nucleotide'} inference (dataset x threads x card x model)",
                                 os.path.basename(__file__), SRC, len(chunk), supp=True,
                                 page=i + 1, data_date=date, square=False, full_page=True)
            plt.close(fig)
        print(f"  wrote FigS{num:02d}a_trees_{tag}_p01..p{pages:02d}  ({len(items)} trees)")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
