#!/usr/bin/env python3
"""assemble.py - assembled previews per main figure, and the greyscale proof sheet.

The panels are the submission unit; these are for the board to see the intended layout and to verify
the accessibility claim. The greyscale sheet is not decorative: the whole reason arm is encoded by
fill and line style rather than by hue is that the encoding must survive greyscale, print and
colour-vision deficiency. Rendering it proves the claim instead of asserting it.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, re, glob, collections, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_p = os.path.join(REPO, "benchmark", "scripts", "lib", "figstyle.py")
_s = ilu.spec_from_file_location("figstyle", _p); FS = ilu.module_from_spec(_s); _s.loader.exec_module(FS)
FS.apply()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PANELS = os.path.join(REPO, "benchmark", "figures", "panels")
OUT = os.path.join(REPO, "benchmark", "figures", "previews")
NAME = re.compile(r"^Fig(S?)(\d{2})([a-z])_(.+?)(?:_p\d{2})?$")


def groups():
    g = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{PANELS}/*.png")):
        m = NAME.match(os.path.splitext(os.path.basename(f))[0])
        if m:
            g[(m.group(1), int(m.group(2)))].append((m.group(3), f))
    return g


def assemble(key, items, grey=False):
    supp, num = key
    items = sorted(items)
    ncol = 2 if len(items) <= 4 else 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (letter, path) in zip(axes, items):
        im = mpimg.imread(path)
        if grey:
            # luminance, the same transform a greyscale printer applies
            rgb = im[..., :3] if im.ndim == 3 else np.dstack([im] * 3)
            im = (rgb * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
        ax.imshow(im, cmap="gray" if grey else None, vmin=0 if grey else None,
                  vmax=1 if grey else None)
        ax.set_anchor("NW")
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
                    textcoords="offset points", fontweight="bold", va="bottom", ha="left")
    tag = f"Figure {'S' if supp else ''}{num}"
    fig.suptitle(tag + (" - greyscale proof" if grey else ""), y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    g = groups()
    # a retired or renumbered figure must not leave a stale preview behind
    live = {f"PREVIEW_Fig{'S' if k[0] else ''}{k[1]:02d}" for k in g}
    for old in os.listdir(OUT):
        stem = old.rsplit(".", 1)[0]
        if stem.startswith("PREVIEW_") and stem not in live:
            os.remove(os.path.join(OUT, old))
    n = 0
    for key, items in sorted(g.items()):
        fig = assemble(key, items)
        stem = f"PREVIEW_Fig{'S' if key[0] else ''}{key[1]:02d}"
        fig.savefig(os.path.join(OUT, stem + ".pdf"))
        fig.savefig(os.path.join(OUT, stem + ".png"), dpi=110)   # for on-screen proofing
        plt.close(fig)
        print(f"  wrote previews/{stem}.pdf  ({len(items)} panels)")
        n += 1
    # greyscale proof: main figures only, one page per figure
    main_keys = [k for k in sorted(g) if not k[0]]
    figs = []
    for key in main_keys:
        figs.append(assemble(key, g[key], grey=True))
    from matplotlib.backends.backend_pdf import PdfPages
    proof = os.path.join(OUT, "GREYSCALE_PROOF_main_figures.pdf")
    with PdfPages(proof) as pp:
        for f in figs:
            pp.savefig(f); plt.close(f)
    # The proof sheet is a production check, not a result: it stays in previews/ and is not a
    # supplementary panel (rev5, T5).
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))
    print(f"  wrote previews/GREYSCALE_PROOF_main_figures.pdf ({len(figs)} pages)")
    print(f"  {n} assembled previews")


if __name__ == "__main__":
    main()
