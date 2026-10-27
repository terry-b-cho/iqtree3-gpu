"""figstyle.py -- the ONE publication style for every figure in this repository.

WHY THIS EXISTS. Thirty-plus figures were each styled by whatever its own generator happened to set,
so type size, typeface and colour meaning drifted between them. A reader who has learned that blue
means "protein" in one panel should not have to relearn it in the next. this repository holds the style
once; every figure script imports it and calls `apply()` before plotting.

THE ENCODING CONTRACT -- the rule that makes the figure set readable as a whole:

    hue          means ALPHABET   (protein / DNA)
    fill + line  means ARM        (CPU = open marker + dashed, GPU = filled + solid)
    lightness    means SIZE       (ordinal ramp, small -> large)

Nothing else is given a hue. Arm is deliberately NOT a hue: the obvious grey-vs-blue encoding FAILS
colour-vision checks (grey has zero chroma, and normal-vision separation is dE 13.3, under the 15
floor), so arm is carried by fill and line style, which survives greyscale printing as well.

PALETTE PROVENANCE. Every colour below was checked with the dataviz skill's `validate_palette.js`
against a light surface, not chosen by eye:

  alphabet pair  #0072B2 / #D55E00   ALL PASS   CVD dE 21.9 (protan), 31.2 normal
  phase 3-way    #0072B2 #E69F00 #009E73        ALL PASS
  phase 4-way    + #CC79A7                      PASS, CVD dE 7.6 WARN -> direct labels REQUIRED
  size ramp      #6BAED6 .. #08306B             ALL PASS (monotone L, light end 2.36:1)

  A 6-hue categorical set for the six GPUs only reached dE 7.6 (deutan), inside the "floor band"
  that is legal only with secondary encoding -- so cards are FACETED or drawn as ordered bars in a
  single hue with direct labels, never as six overlaid hues.

  #E69F00 (2.19:1) and #56B4E9 (2.25:1) fall below 3:1 against white. That is not dismissable: any
  mark using them must carry a visible direct label.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- type -------------------------------------------------------------------------------------
# Arial and Helvetica are not installed on this cluster. Nimbus Sans is URW's metrically identical
# Helvetica clone and is the correct substitute; Arial/Helvetica are listed first so a machine that
# HAS them uses them. DejaVu is matplotlib's silent fallback and is what assert_font() exists to
# catch -- a missing font otherwise degrades quietly and every figure ships in the wrong typeface.
FONT_STACK = ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "Helvetica Neue", "FreeSans"]
FONT_PT = 6.0
# Arial itself, when its TrueType files are placed in benchmark/figures/arial/ (regular, bold,
# italic, bold italic). The files are Monotype-licensed and are not distributed with the
# repository; without them the stack falls through to Nimbus Sans, which is metrically identical.
import os as _os
_REPO = _os.environ.get("REPO_DIR", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
FONT_DIR = _os.path.join(_REPO, "benchmark", "figures", "arial")
ARIAL_FILES = ("ARIAL.TTF", "ARIALBD.TTF", "ARIALI.TTF", "ARIALBI.TTF")


def register_arial():
    """Register the Arial faces from FONT_DIR with matplotlib, if the directory is present.
    Returns the number of faces registered."""
    n = 0
    for f in ARIAL_FILES:
        p = _os.path.join(FONT_DIR, f)
        if _os.path.exists(p):
            font_manager.fontManager.addfont(p); n += 1
    return n

# ---- colour -----------------------------------------------------------------------------------
PROTEIN = "#0072B2"      # blue        -- amino acid
DNA     = "#D55E00"      # vermillion  -- nucleotide
INK     = "#000000"
MUTED   = "#444444"

ALPHABET = {"protein": PROTEIN, "dna": DNA, "aa": PROTEIN, "nt": DNA}

# Alphabet needs a SECOND channel as well as hue. #0072B2 and #D55E00 separate well in colour and
# under every CVD simulation, but they have similar LUMINANCE, so they collapse to the same grey in
# a greyscale print -- the proof sheet showed the two series becoming indistinguishable. Marker
# shape carries alphabet alongside hue so the encoding survives greyscale as the arm encoding does.
ALPHABET_MARKER = {"protein": "o", "dna": "s", "aa": "o", "nt": "s"}


def alphabet_marker(dt):
    return ALPHABET_MARKER.get(dt, "o")

# Phase / step hues, in fixed order. Never cycled; a 5th step folds into "other".
# Phases must NOT reuse the alphabet pair: a reader who has learned blue = amino acid would read a
# blue "model selection" bar as amino acid. The two phase hues are the remaining Okabe-Ito
# colours with the widest separation from #0072B2 / #D55E00 and from each other: bluish green
# (model selection) and reddish purple (tree search); they also differ in lightness, so the pair
# survives greyscale. "other" (untimed) is a neutral grey.
PHASE_ORDER  = ["modelfinder", "tree_search", "initial", "other"]
PHASE_COLORS = {"modelfinder": "#009E73", "tree_search": "#CC79A7",
                "initial": "#56B4E9", "other": "#D9D9D9"}
LABEL_REQUIRED = {"#E69F00", "#56B4E9"}   # below 3:1 on white -> must carry a visible label

# Ordinal ramp for SIZE (taxa / complexity). Validated monotone in lightness.
SIZE_RAMP = ["#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]

# Cards are faceted, not hued (see the docstring at the top of this file). One hue for every card panel.
CARD_HUE = "#0072B2"


def arm_style(arm, color):
    """Arm is fill + line style, never hue. Returns kwargs for plot()/scatter()."""
    if str(arm).lower().startswith("c"):          # CPU: open marker, dashed line
        return dict(color=color, markerfacecolor="none", markeredgecolor=color,
                    linestyle="--", markeredgewidth=0.8)
    return dict(color=color, markerfacecolor=color, markeredgecolor=color,
                linestyle="-", markeredgewidth=0.8)


def size_color(i, n):
    """Ordinal colour for the i-th of n size steps (small -> large = light -> dark)."""
    if n <= 1:
        return SIZE_RAMP[-1]
    k = round(i * (len(SIZE_RAMP) - 1) / (n - 1))
    return SIZE_RAMP[int(k)]


def resolved_font():
    """The family matplotlib will ACTUALLY render with, after fallback.

    NOTE: pass the family as a LIST. A bare "sans-serif" string is parsed as a fontconfig pattern
    (the hyphen trips the parser) and raises instead of resolving.
    """
    fam = list(plt.rcParams.get("font.sans-serif") or FONT_STACK)
    path = font_manager.findfont(font_manager.FontProperties(family=fam))
    try:
        return font_manager.FontProperties(fname=path).get_name()
    except Exception:
        return path


def assert_font(strict=True):
    """Fail loudly if we fell back to DejaVu. Silent fallback is the failure mode this guards."""
    name = resolved_font()
    if "dejavu" in name.lower():
        msg = (f"figstyle: resolved font is {name!r} -- Arial/Helvetica/Nimbus Sans are all absent, "
               f"so every figure would ship in the wrong typeface. Install one of {FONT_STACK}.")
        if strict:
            raise RuntimeError(msg)
        print("  WARNING: " + msg)
    return name


def apply(strict_font=True):
    """Apply the publication style. Call once, before any plotting."""
    register_arial()
    plt.rcParams.update({
        # type: 6 pt everywhere -- labels, ticks, legend, title, annotations
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT, "legend.title_fontsize": FONT_PT,
        "figure.titlesize": FONT_PT,
        # supxlabel/supylabel read figure.labelsize, which defaults to "large" (1.2x = 7.2 pt) and
        # is NOT covered by font.size -- the facet panels were shipping 7.2 pt axis labels.
        "figure.labelsize": FONT_PT,
        # egg::theme_article -- white ground, NO grid, full black panel border on all four spines
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        # thin marks
        "lines.linewidth": 0.8, "lines.markersize": 3.0, "patch.linewidth": 0.5,
        # legend: no frame
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3, "legend.handletextpad": 0.5,
        # journals want embedded TrueType, not outlined text
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    name = assert_font(strict=strict_font)
    # MATHTEXT IS A SEPARATE FONT SETTING. Any "$^2$" or "$R^2$" renders through mathtext, which
    # defaults to DejaVu regardless of font.sans-serif -- so five figures were shipping DejaVu
    # superscripts inside otherwise-Helvetica panels. Point mathtext at the SAME resolved family.
    plt.rcParams.update({
        # A custom fontset must fill EVERY slot. Leaving cal/sf/tt unset makes matplotlib resolve
        # them through the generic 'cursive'/'monospace' families, which are absent here, so it
        # falls back to DejaVu and prints a findfont warning on every figure.
        "mathtext.fontset": "custom",
        "mathtext.rm": name, "mathtext.it": f"{name}:italic", "mathtext.bf": f"{name}:bold",
        "mathtext.cal": name, "mathtext.sf": name, "mathtext.tt": name,
        "mathtext.default": "regular",
    })
    return name


# ================================================================================================
#  PANEL GEOMETRY: one plotting box, the same physical size in every panel
# ================================================================================================
#  Every panel's plotting area (the box the four spines enclose, excluding tick labels, axis
#  labels, legends and the stratum line) is a square of AX_MM x AX_MM millimetres. Titles, labels and
#  legends hang off that box; the canvas is cropped to them on save (savefig.bbox = "tight" crops,
#  it never rescales), so the box keeps its size whatever decorations a panel carries. A facet grid
#  places every facet at the same AX_MM square. The only panels outside this rule are the tree
#  montages (Supplementary Figs. S1 and S2), which save_panel accepts with square=False.
#
#  The box is PRE-POSITIONED, never fitted afterwards: tight_layout() and constrained layout resize
#  axes to fill a canvas, which is exactly what this rule forbids, so no generator calls them.
AX_MM = 40.0
AX_IN = AX_MM / 25.4
_MARGIN_IN = (0.85, 0.65, 0.35, 0.40)     # left, bottom, right, top: generous, the save crops them
_GRID_GAP_IN = 0.16                       # between facets of a grid


def square_axes():
    """-> (fig, ax) with the plotting box exactly AX_IN x AX_IN inches."""
    l, b, r, t = _MARGIN_IN
    fig = plt.figure(figsize=(l + AX_IN + r, b + AX_IN + t))
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([l / fw, b / fh, AX_IN / fw, AX_IN / fh])
    ax._square_box = True
    return fig, ax


def square_grid(nrow, ncol, sharex=True, sharey=True, gap=_GRID_GAP_IN):
    """-> (fig, axes[nrow][ncol]); every facet is an AX_IN square, gaps fixed in inches.
    `gap` is the space between facets: the default suits shared axes; a grid whose boxes carry
    their own y-axis labels needs about 0.9 in."""
    import numpy as np
    l, b, r, t = _MARGIN_IN
    gw = ncol * AX_IN + (ncol - 1) * gap
    gh = nrow * AX_IN + (nrow - 1) * gap
    fig = plt.figure(figsize=(l + gw + r, b + gh + t))
    fw, fh = fig.get_size_inches()
    axes = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            x = l + j * (AX_IN + gap)
            y = b + (nrow - 1 - i) * (AX_IN + gap)
            kw = {}
            if sharex and (i or j):
                kw["sharex"] = axes[0, 0]
            if sharey and (i or j):
                kw["sharey"] = axes[0, 0]
            ax = fig.add_axes([x / fw, y / fh, AX_IN / fw, AX_IN / fh], **kw)
            ax._square_box = True
            if sharex and i < nrow - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if sharey and j:
                ax.tick_params(axis="y", labelleft=False)
            axes[i, j] = ax
    return fig, axes


def split_axes(ncol=2, gap=0.75):
    """-> (fig, [ax, ...]): `ncol` boxes side by side that TOGETHER fill one AX_IN square, each
    AX_IN / ncol wide and AX_IN tall, `gap` inches apart so every box can carry its own y axis.
    For a panel that shows two quantities of the same stratum in one square of page area."""
    l, b, r, t = _MARGIN_IN
    w = AX_IN / ncol
    fig = plt.figure(figsize=(l + ncol * w + (ncol - 1) * gap + r, b + AX_IN + t))
    fw, fh = fig.get_size_inches()
    axes = []
    for j in range(ncol):
        ax = fig.add_axes([(l + j * (w + gap)) / fw, b / fh, w / fw, AX_IN / fh])
        ax._square_box = True; ax._box_fraction = 1.0 / ncol
        axes.append(ax)
    return fig, axes


def grid_note(fig, axes, text):
    """The stratum line of a facet grid, set above its top-left facet (grids have no title)."""
    top = max(ax.get_position().y1 for ax in axes.flat)
    left = min(ax.get_position().x0 for ax in axes.flat)
    fh = fig.get_size_inches()[1]
    fig.text(left, top + 0.26 / fh, text, ha="left", va="bottom", color=INK)


def stratum(threads, n, unit="runs"):
    """The ONLY text a panel carries in its title slot: thread count and n.

    `threads` is a thread count or an iterable of the thread counts the panel pools. The n passed
    here is the n recorded in the manifest, so the two cannot disagree.
    """
    if isinstance(threads, (str, int)):
        threads = [threads]
    th = sorted({int(t) for t in threads})
    if not th:
        lab = ""
    elif len(th) == 1:
        lab = f"{th[0]} thread{'s' if th[0] != 1 else ''}, "
    else:
        lab = " and ".join(str(t) for t in th) + " threads, "
    return f"{lab}n = {n}" + (f" {unit}" if unit and unit != "runs" else "")


def assert_square(fig):
    """Every plotting box of the figure is an AX_IN square, to a tenth of a point."""
    boxes = [ax for ax in fig.axes if getattr(ax, "_square_box", False)]
    assert boxes, "no square plotting box: build the panel with square_axes() or square_grid()"
    fw, fh = fig.get_size_inches()
    for ax in boxes:
        x0, y0, w, h = ax.get_position().bounds
        wi, hi = w * fw, h * fh
        want = AX_IN * getattr(ax, "_box_fraction", 1.0)      # split boxes share one square
        assert abs(wi - want) < 1e-3 and abs(hi - AX_IN) < 1e-3, \
            f"plotting box is {wi*25.4:.2f} x {hi*25.4:.2f} mm, not {want*25.4:.0f} x {AX_MM}"


# ================================================================================================
#  PANEL OUTPUT: naming contract + manifest
# ================================================================================================
#  The production office identifies panels by filename alone, so the pattern is fixed and enforced:
#
#      Fig{NN}{p}_{slug}.{pdf,png}        main figures
#      FigS{NN}{p}_{slug}.{pdf,png}       supplementary
#      ...{slug}_p{NN}.{pdf,png}          multipage montages
#
#  Numbers are zero-padded so alphabetical order in the portal IS figure order. No dates, no
#  versions, no "final" -- versioning belongs in git and the data date is inside the figure. A facet
#  grid is ONE file with ONE letter; facets are never split into separately lettered files.
#
#  Permitted type sizes. The journal floor is 5 pt; we work at 6-8 pt. This set is what the audit in
#  run_tests.sh enforces against the Tf operators of every emitted PDF, which is the only way to
#  catch a stray `fontsize=` argument -- rcParams defaults do not bind explicit arguments.
# 5 pt is the journal's absolute minimum and is permitted ONLY for dense tip labels in the tree
# montages, where a 6 pt label collides. Everything else lives at 6-8 pt.
PERMITTED_PT = (5.0, 6.0, 7.0, 8.0)
ABSOLUTE_MIN_PT = 5.0
# Mathtext renders superscripts and subscripts at 0.7x the surrounding size, so a "10^3" tick label
# at 6 pt legitimately emits a 4.2 pt glyph run for the exponent. That is ordinary typesetting, not
# a styling fault, and the audit must permit it while still catching a stray fontsize= argument.
MATHTEXT_SUB_FACTOR = 0.7
def permitted_sizes():
    base = set(PERMITTED_PT)
    return sorted(base | {round(p * MATHTEXT_SUB_FACTOR, 2) for p in base}
                        | {round(p * MATHTEXT_SUB_FACTOR ** 2, 2) for p in base})

_MANIFEST = []


def save_panel(fig, figure, panel, slug, title, source_script, source_table, n_rows,
               outdir=None, supp=False, page=None, data_date=None, square=True, full_page=False):
    """Write one panel as vector PDF + 300 dpi PNG under the naming contract, and record it.

    `figure` is an int, `panel` a single lowercase letter. Returns the stem actually written.
    `square` enforces the AX_MM plotting box (assert_square); only the tree montages pass False.
    `full_page` writes the figure at its own canvas size instead of cropping to the artists: the
    tree montages are laid out as whole US Letter pages and must keep their margins.
    """
    import os, re, datetime
    if square:
        assert_square(fig)
    assert isinstance(figure, int) and 1 <= figure <= 99, f"bad figure number {figure!r}"
    assert re.fullmatch(r"[a-z]", panel or ""), f"panel must be one lowercase letter, got {panel!r}"
    assert re.fullmatch(r"[a-z0-9_]+", slug or ""), f"bad slug {slug!r}"
    stem = f"Fig{'S' if supp else ''}{figure:02d}{panel}_{slug}"
    if page is not None:
        stem += f"_p{page:02d}"
    outdir = outdir or os.environ.get("PANEL_DIR", "benchmark/figures/panels")
    os.makedirs(outdir, exist_ok=True)
    # The tight crop measures every axis LABEL with its width collapsed to the axis centre
    # (matplotlib's for_layout_only rule), so a label wider than its axis was cut at the canvas
    # edge. Naming the labels as extra artists restores their full extent to the crop. The list
    # REPLACES the default extra-artist list of every axes (annotations, legends, texts), so
    # those defaults are carried along explicitly or they would be cropped instead.
    extra = [t for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label) if t.get_text()]
    extra += [a for ax in fig.axes for a in ax.get_default_bbox_extra_artists()]
    extra += list(fig.get_default_bbox_extra_artists())
    if full_page:       # bbox_inches=None would fall back to rcParams' "tight": give the page bbox
        from matplotlib.transforms import Bbox
        fw, fh = fig.get_size_inches()
        kw = dict(bbox_inches=Bbox([[0, 0], [fw, fh]]), pad_inches=0)
    else:
        kw = dict(bbox_extra_artists=extra)
    fig.savefig(os.path.join(outdir, stem + ".pdf"), **kw)
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=300, **kw)
    write_source_data(fig, stem, os.path.join(os.path.dirname(outdir.rstrip("/")), "source_data"))
    _MANIFEST.append(dict(
        file=stem, figure=("S%d" % figure) if supp else str(figure), panel=panel, title=title,
        source_script=source_script, source_table=source_table, rows_used=n_rows,
        data_date=data_date or datetime.date.today().isoformat()))
    return stem


def write_source_data(fig, stem, outdir):
    """Source Data (the journal asks for it): the values behind every mark in a panel, read back from
    the drawn artists rather than re-derived, so the file cannot disagree with the figure.

    One CSV per panel: axes index and title, artist kind (scatter, line, bar, image, segment), the
    series label where the artist has one, and x, y (and for bars the width and height; for images
    the cell value with its row and column). Empty and legend-only artists are skipped.
    """
    import os, csv
    import numpy as np
    import matplotlib as mpl
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for ai, ax in enumerate(fig.axes):
        title = (ax.get_title() or "").replace("\n", " ")
        xl, yl = ax.get_xlabel().replace("\n", " "), ax.get_ylabel().replace("\n", " ")
        for col in ax.collections:
            lab = col.get_label() or ""
            if lab.startswith("_") and not isinstance(col, mpl.collections.PathCollection): lab = ""
            if isinstance(col, mpl.collections.PathCollection):
                off = np.asarray(col.get_offsets())
                for x, y in off:
                    rows.append((ai, title, "scatter", "" if lab.startswith("_") else lab, xl, yl, x, y, "", ""))
            elif isinstance(col, mpl.collections.LineCollection):
                for seg in col.get_segments():
                    for x, y in seg:
                        rows.append((ai, title, "segment", "" if lab.startswith("_") else lab, xl, yl, x, y, "", ""))
            elif isinstance(col, mpl.collections.PolyCollection):
                for path in col.get_paths():
                    for x, y in path.vertices:
                        rows.append((ai, title, "band", "" if lab.startswith("_") else lab, xl, yl, x, y, "", ""))
        for ln in ax.lines:
            xs, ys = np.asarray(ln.get_xdata(), dtype=float), np.asarray(ln.get_ydata(), dtype=float)
            if xs.size == 0: continue
            lab = ln.get_label() or ""
            for x, y in zip(xs, ys):
                rows.append((ai, title, "line", "" if lab.startswith("_") else lab, xl, yl, x, y, "", ""))
        for pt in ax.patches:
            if isinstance(pt, mpl.patches.Rectangle) and pt.get_width() and pt.get_height():
                lab = pt.get_label() or ""
                rows.append((ai, title, "bar", "" if lab.startswith("_") else lab, xl, yl, pt.get_x(), pt.get_y(), pt.get_width(), pt.get_height()))
        for im in ax.images:
            arr = np.asarray(im.get_array())
            if arr.ndim != 2 or arr.size > 200000: continue
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    rows.append((ai, title, "image", "", xl, yl, j, i, "", arr[i, j]))
    if not rows:
        return None
    path = os.path.join(outdir, stem + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axes", "axes_title", "kind", "series", "x_label", "y_label", "x", "y", "width", "height_or_value"])
        w.writerows(rows)
    return path


def write_manifest(path):
    """Append this run's panels to FIGURE_MANIFEST.tsv, generated -- never hand-maintained."""
    import csv, os
    cols = ["file", "figure", "panel", "title", "source_script", "source_table", "rows_used",
            "data_date"]
    prior = []
    panels = os.path.join(os.path.dirname(path), "panels")
    if os.path.exists(path):
        with open(path) as fh:
            prior = [r for r in csv.DictReader(fh, delimiter="\t")
                     if r["file"] not in {m["file"] for m in _MANIFEST}
                     # A renamed or removed panel must not linger as an orphan row. The audit
                     # fails on exactly this, so prune here rather than leave it to be reported.
                     and os.path.exists(os.path.join(panels, r["file"] + ".pdf"))]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in sorted(prior + _MANIFEST, key=lambda r: r["file"]):
            w.writerow({c: r.get(c, "") for c in cols})
    return len(prior) + len(_MANIFEST)


def ratio_ticks(ax, values=(0.25, 0.5, 1, 2, 5, 10, 20, 40), axis="y"):
    """Label a log RATIO axis at the values readers use, not at decades.

    A bare log axis labels 10^0 and 10^1 only, which leaves speedup -- the quantity the whole paper
    is about -- effectively unlabelled between 1x and 10x, and makes 0.74x visually indistinct from
    parity. Call this on every speedup axis.
    """
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    tk = [t for t in values if lo <= t <= hi] or list(values)
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(tk))
    a.set_major_formatter(FixedFormatter([f"{t:g}x" for t in tk]))
    a.set_minor_formatter(NullFormatter())
    return tk
