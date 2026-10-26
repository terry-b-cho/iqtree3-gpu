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
PHASE_ORDER  = ["modelfinder", "tree_search", "initial", "other"]
PHASE_COLORS = {"modelfinder": "#0072B2", "tree_search": "#E69F00",
                "initial": "#009E73", "other": "#CC79A7"}
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
               outdir=None, supp=False, page=None, data_date=None):
    """Write one panel as vector PDF + 300 dpi PNG under the naming contract, and record it.

    `figure` is an int, `panel` a single lowercase letter. Returns the stem actually written.
    """
    import os, re, datetime
    assert isinstance(figure, int) and 1 <= figure <= 99, f"bad figure number {figure!r}"
    assert re.fullmatch(r"[a-z]", panel or ""), f"panel must be one lowercase letter, got {panel!r}"
    assert re.fullmatch(r"[a-z0-9_]+", slug or ""), f"bad slug {slug!r}"
    stem = f"Fig{'S' if supp else ''}{figure:02d}{panel}_{slug}"
    if page is not None:
        stem += f"_p{page:02d}"
    outdir = outdir or os.environ.get("PANEL_DIR", "results/figures/panels")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, stem + ".pdf"))
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=300)
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
