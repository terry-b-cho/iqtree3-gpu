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
FONT_PT) at a fixed 2.75 x 2.15 in. Placing a panel at scale s therefore prints its type at 6*s pt --
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

Row assignments below follow from that rule. The two per-card facet grids in Figure 6 are 139 mm
wide at natural size, so they take a row each; pairing them would force the page to 0.64 scale.
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
# tallest panel in it.
FIGURES = {
    1: dict(dir=PANELS, title="The GPU returns the same answer", rows=[
            ["Fig01a_logl_reldiff_hist", "Fig01b_logl_absdiff_vs_magnitude",
             "Fig01c_support_diff_hist"]]),
    # Round 8 (R15): the per-invocation panel (old 2D) restates 2A per call and moved to
    # Supplementary Fig. S18; the composite title is descriptive rather than a conclusion.
    2: dict(dir=PANELS, title="Acceleration against workload and thread count", rows=[
            ["Fig02a_decision_t20", "Fig02b_decision_t1"],
            ["Fig02c_thread_slope"]]),
    3: dict(dir=PANELS, title="Launch overhead, not arithmetic", rows=[
            ["Fig03a_cost_law", "Fig03b_outer_demand"],
            ["Fig03c_amdahl", "Fig03d_nsys_profile"]]),
    4: dict(dir=PANELS, title="Phases and cost", rows=[
            ["Fig04a_phase_share", "Fig04b_phase_speedup"],
            ["Fig04c_cpu_time_ratio", "Fig04d_parallel_efficiency"]]),
    # The three one-point-per-card summaries share the top row; the two per-card facet grids are
    # 139 mm wide at natural size and take a row each. This regroups the panels relative to the
    # old standalone order -- energy was 5e and is now 6C -- so the text references move with it.
    5: dict(dir=PANELS, title="GPU-arm performance across six cards", rows=[
            ["Fig05a_wall_vs_fp64", "Fig05b_wall_vs_bandwidth", "Fig05e_energy_per_card"],
            ["Fig05c_card_facet_wall"],
            ["Fig05d_card_facet_speedup"]]),
    6: dict(dir=PANELS, title="Memory and alignment properties", rows=[
            ["Fig06a_memory_sizing", "Fig06b_vram_guard"],
            ["Fig06c_gap_ambiguity", "Fig06d_informativeness"]]),
}

# Supplementary figures. Every supplementary figure in the submission gets one file here, so the
# master supplementary-figures document can be assembled from this directory alone. Three kinds:
#   rows=[...]            the lettered single-page composite used for the main figures;
#   pages=[[rows], ...]   one page per entry at NATURAL size (scale 1.0), because the tree panels are
#                         ~300 mm wide and shrinking a tree plot is what destroyed the tip labels once
#                         already: S2 is two pages (A, B), S3 six pages (no letters: its legend speaks
#                         of pages, not panels);
#   a single panel        is placed without a letter; the legends of S1, S4-S8 and S10-S13 name
#                         no panel letter, and a lone "A" on a one-panel figure is a journal error.
# Numbering is the order of first citation (S1-S13); the greyscale proof sheet is a production check
# and is not a supplementary figure.
SUPP = {
    1: dict(dir=PANELS, title="Campaign coverage", rows=[["FigS01a_coverage_grid"]]),
    2: dict(dir=PANELS, title="Inferred trees, size ladder",
            pages=[[["FigS02a_tree_ladder_aa"]], [["FigS02b_tree_ladder_nt"]]]),
    3: dict(dir=PANELS, title="Complete tree record, pages 1 to 6", letters=False,
            pages=[[[f"FigS03a_tree_record_all_p{i:02d}"]] for i in range(1, 7)]),
    4: dict(dir=PANELS, title="Model classes exercised, from the data", rows=[["FigS04a_model_coverage_data"]]),
    5: dict(dir=PANELS, title="Production-scale phylogenomic cost", rows=[["FigS05a_production_memory"]]),
    6: dict(dir=PANELS, title="Replicate spread, amino acid", rows=[["FigS06a_replicates_protein"]]),
    7: dict(dir=PANELS, title="Replicate spread, nucleotide", rows=[["FigS07a_replicates_nucleotide"]]),
    8: dict(dir=PANELS, title="Wall time per device invocation, by arm and workflow", rows=[
             ["FigS08a_per_invocation_cost"]]),
    9: dict(dir=PANELS, title="Cost law at one thread, and node effects", rows=[
             ["FigS09a_cost_law_t1", "FigS09b_node_effects"]]),
    10: dict(dir=PANELS, title="Occupancy does not explain the sub-parity runs", rows=[["FigS10a_pattern_saturation"]]),
    11: dict(dir=PANELS, title="Sampled device utilisation", rows=[["FigS11a_gpu_utilisation"]]),
    12: dict(dir=PANELS, title="Superseded build against current build", rows=[["FigS12a_build_regression"]]),
    13: dict(dir=PANELS, title="Phase-step decomposition, step by cell", rows=[["FigS13a_phase_step_heatmap"]]),
}

LETTERS = "ABCDEFGHIJKL"


def reference_index():
    """-> {figure label: {panel letter: panel stem}} for every composite.

    These layouts ARE the manuscript's figure numbering: which panel is Fig. 6C is decided by the
    row assignments above, not by the panel file names, which still carry the old Fig05e-style
    stems. The cross-check resolves manuscript references against this rather than keeping a second
    copy that could drift out of step.
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
    label_h = LABEL_H if letters_on else 0.0
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
        fig.text(x / W, min(top + 0.6, H - 0.5) / H, letter, fontsize=8.0, fontweight="bold",
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
