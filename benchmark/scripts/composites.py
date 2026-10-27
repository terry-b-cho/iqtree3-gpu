#!/usr/bin/env python3
"""composites.py - assemble the panels into one lettered composite per figure.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.

WHAT THIS PRODUCES. One figure = one file, panels lettered A, B, C ... as a journal sets them,
written to benchmark/figures/composites/ as a vector PDF plus a 300 dpi PNG rendered from that same
PDF, so the two cannot disagree.

WHY THE PANELS ARE NOT RE-RENDERED. Each panel's existing PDF is placed into the page with pypdf, so
every mark stays vector and byte-for-byte what the panel generator produced. The letters are a vector
overlay drawn on top.

THE SCALE RULE, which is the whole design of this file. Every panel is authored at 6 pt (figstyle
FONT_PT) around a fixed 40 x 40 mm plotting box (figstyle AX_MM; the decorations around it set the
panel's outer size). Placing a panel at scale s therefore prints its type at 6*s pt --
scaling a panel silently rewrites its font size. A naive "stretch each row to fill the page width"
layout gave one figure 8 pt in its top row and 3.8 pt in the row below it, which is both illegible and
obviously amateur. So:

  * panels are never scaled relative to one another. Within a row they sit at natural size and are
    TOP-aligned, which is why rows here may hold panels of unequal height;
  * the finished page is then scaled ONCE, uniformly, to fit the text block. Type stays uniform
    across the whole figure and the effective size is simply 6 * scale;
  * that effective size is asserted against figstyle's permitted band (5-8 pt). A grouping that
    cannot be printed in the band is refused with a message naming the figure, rather than shipped
    at 3.8 pt.

Row assignments below follow from that rule. The per-card facet grid in Supplementary Fig. S6 is
139 mm wide at natural size, so it stands alone.
"""
import os, re, subprocess, sys
import importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_p = os.path.join(REPO, "benchmark", "scripts", "lib", "figstyle.py")
_s = ilu.spec_from_file_location("figstyle", _p); FS = ilu.module_from_spec(_s); _s.loader.exec_module(FS)

PANELS = os.path.join(REPO, "benchmark", "figures", "panels")
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
FIGURES = {
    1: dict(dir=PANELS, title="IQ-TREE3-GPU returns the same answer as IQ-TREE3", rows=[
            ["Fig01a_logl_reldiff_ecdf", "Fig01b_support_diff_dotplot"]]),
    2: dict(dir=PANELS, title="Acceleration against workload and thread count", rows=[
            ["Fig02a_decision_t20", "Fig02b_decision_t1"]]),
    3: dict(dir=PANELS, title="Runtime scaling with total likelihood workload", rows=[
            ["Fig03a_cost_law", "Fig03b_cost_law_t1"]]),
    4: dict(dir=PANELS, title="Phases and cost", rows=[
            ["Fig04a_phase_share_cpu_time", "Fig04b_phase_speedup_complexity"],
            ["Fig04c_cores_busy"]]),
}

def _pages(stem):
    """The page stems of a multipage panel, in page order, from the files present."""
    import glob
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(PANELS, stem + "_p*.pdf")))


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
    1: dict(dir=PANELS, title="Every amino-acid inference", letters=False,
            pages=[[[s]] for s in _pages("FigS01a_trees_aa")]),
    2: dict(dir=PANELS, title="Every nucleotide inference", letters=False,
            pages=[[[s]] for s in _pages("FigS02a_trees_nt")]),
    3: dict(dir=PANELS, title="Model classes exercised, from the data", rows=[["FigS03a_model_coverage_data"]]),
    4: dict(dir=PANELS, title="Replicate spread at 20 threads", rows=[["FigS04a_replicates"]]),
    5: dict(dir=PANELS, title="Phase-step decomposition, step by cell", rows=[["FigS05a_phase_step_heatmap"]]),
    # the per-card facet grid is 139 mm wide at natural size
    6: dict(dir=PANELS, title="IQ-TREE3-GPU across six cards", rows=[["FigS06a_card_facet_speedup"]]),
    7: dict(dir=PANELS, title="Campaign coverage", rows=[
            ["FigS07a_coverage_stacked_bars"],
            ["FigS07b_coverage_by_card"]]),
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
            stems = [stem for row in all_rows(spec) for stem in row]
            idx[f"{pre}{n}"] = ({LETTERS[k]: stem for k, stem in enumerate(stems)}
                                if lettered(spec, stems) else {})
    return idx


def all_rows(spec):
    """The panel rows of a spec, across its pages if it has them."""
    return [row for page in spec.get("pages", [spec.get("rows", [])]) for row in page]


def lettered(spec, stems):
    """Panels carry letters only when the figure has more than one and the spec does not opt out."""
    return spec.get("letters", True) and len(stems) > 1


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
    sizes = {s: page_size(os.path.join(spec["dir"], s + ".pdf")) for r in rows_in for s in r}
    # A page printed as authored (fixed_scale: the Letter-size tree montages) keeps its exact page
    # size; its letter is drawn inside the page's own top margin, so no strip is added.
    label_h = LABEL_H if (letters_on and fixed_scale is None) else 0.0
    rows = [(row,
             sum(sizes[s][0] for s in row) + GAP * (len(row) - 1),
             max(sizes[s][1] for s in row)) for row in rows_in]

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
        for s in row:
            w0, h0 = sizes[s]
            placed.append((s, x, y - h0, LETTERS[k] if letters_on else ""))   # TOP-aligned
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
    pages) and <out_stem>.png, or <out_stem>_p1.png ... for a multi-page figure."""
    from pypdf import PdfWriter
    pages = spec.get("pages") or [spec["rows"]]
    stems = [stem for row in all_rows(spec) for stem in row]
    letters_on = lettered(spec, stems)
    fixed = 1.0 if "pages" in spec else None      # pages print as authored (6 pt), never refitted
    w = PdfWriter()
    first = None
    k = 0
    for rows in pages:
        page, W, H, sc, n = build_page(spec, rows, k, fixed, letters_on, out_stem)
        w.add_page(page)
        first = first or (W, H, sc, n)
        k += n
    with open(out_stem + ".pdf", "wb") as fh:
        w.write(fh)
    for old in [f for f in os.listdir(os.path.dirname(out_stem))
                if re.fullmatch(re.escape(os.path.basename(out_stem)) + r"(_p\d+)?\.png", f)]:
        os.remove(os.path.join(os.path.dirname(out_stem), old))
    if len(pages) == 1:
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                        out_stem + ".pdf", out_stem], check=True)
    else:
        for i in range(1, len(pages) + 1):
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "-f", str(i), "-l", str(i),
                            out_stem + ".pdf", f"{out_stem}_p{i}"], check=True)
    return (*first, len(pages))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"  composites -> {os.path.relpath(OUT, REPO)}")
    print(f"  {'figure':<34s} {'panels':>6s}  {'page (mm)':>12s}  {'scale':>5s}  {'type':>6s}")
    for old in os.listdir(OUT):            # a renumbered or retired figure must not linger
        if re.fullmatch(r"(Figure_\d+|Supplementary_Figure_S\d+)(_p\d+)?\.(pdf|png)", old):
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


if __name__ == "__main__":
    sys.exit(main())
