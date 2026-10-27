#!/usr/bin/env python3
"""hardware.py - Supplementary Fig. S6: the GPU arm across six cards.

  speedup against complexity WITHIN each card, faceted: cards sit on different hosts, so the
  CPU-arm wall differs between them (card_host_cpu_spread_max in KEY_NUMBERS.tsv) and
  cross-card speedup is never compared

The panels that ranked the cards by nominal fp64 and memory bandwidth, the per-card wall facets
and the device-energy bars were removed from the publication on 2026-10-26.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt

SRC = "benchmark/runs/RUNS.tsv"
LABEL = {"NVIDIA_A100_80GB_PCIe": "A100", "Tesla_V100S-PCIE-32GB": "V100S",
         "Tesla_V100-SXM2-16GB": "V100", "NVIDIA_L40S": "L40S",
         "NVIDIA_A40": "A40", "Quadro_RTX_8000": "RTX 8000"}


def _facet_labels(axes, xlabel, ylabel):
    """Axis labels on the outer facets only (bottom row, left column)."""
    nrow, ncol = axes.shape
    for j in range(ncol):
        axes[nrow - 1, j].set_xlabel(xlabel)
    for i in range(nrow):
        axes[i, 0].set_ylabel(ylabel)


def panel_d(rows):
    """Speedup WITHIN each card only -- never compared across cards, since the hosts differ."""
    cards = [c for c in LABEL if any(r["card"] == c for r in rows)]
    fig, axes = FS.square_grid(2, 3)
    for ax, card in zip(axes.flat, cards):
        pts = [(r["_complexity"], r["_speedup"]) for r in rows
               if r["card"] == card and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=8, color=FS.CARD_HUE, zorder=3)
        ax.axhline(1.0, color=FS.INK, lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{LABEL[card]} (n={len(pts)})", pad=2)
    for ax in axes.flat[len(cards):]:
        ax.axis("off")
    FS.ratio_ticks(axes.flat[0])
    _facet_labels(axes, "tree complexity", "speedup within card\n(IQ-TREE3 wall / IQ-TREE3-GPU wall)")
    n = sum(1 for r in rows if r["threads"] == "20")
    # speedup is comparable only WITHIN a card: the hosts differ by card_host_cpu_spread_max
    FS.grid_note(fig, axes, FS.stratum(20, n))
    return fig, n


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    out = [(panel_d(rows), "a", "card_facet_speedup", "Speedup vs complexity, per card")]
    for (fig, n), letter, slug, title in out:
        stem = FS.save_panel(fig, 6, letter, slug, title, os.path.basename(__file__), SRC, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
