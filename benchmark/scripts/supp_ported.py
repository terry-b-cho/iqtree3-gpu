#!/usr/bin/env python3
"""supp_ported.py - Supplementary Fig. S4 and the replicate table.

  S4  replicate spread at 20 threads, both alphabets on one axis (the former sampled-utilisation
      figure was withdrawn on 2026-10-26)

Also writes runs/REPLICATES_20T.tsv (Supplementary Data 5): one row per alphabet x taxon count,
from the same replicate sets the S4 panel draws, so the table and the figure cannot drift.

Every panel has one provenance: current-build rows only, the shared style, the Fig{S}{NN}{p}_{slug}
contract, and a manifest row.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); H = _load("headline"); FS.apply()
import matplotlib.pyplot as plt

SRC = "benchmark/runs/RUNS.tsv"


# ------------------------------------------------------------------- S4 replicate dispersion
def replicate_sets(rows, dt):
    """Repeat runs of the same dataset on the same card class, current build only."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["data_type"] != dt or r["threads"] != "20" or not H.is_replicate(r):
            continue
        if r["_speedup"]:
            by[int(r["taxa"])].append(r["_speedup"])
    return {k: v for k, v in by.items() if len(v) >= 2}


def replicates(rows):
    """Replicate spread at 20 threads, both alphabets on one axis: every repeat run of a size as a
    circle, the range as a bar, the median as a tick. The two alphabets share the same sizes, so
    they are offset slightly about each size."""
    fig, ax = FS.square_axes()
    # the sizes are an ordinal ladder (4, 6, 8, ... 30): equal spacing keeps every label legible,
    # where a log axis ran 26, 28 and 30 into one another
    sets = {dt: replicate_sets(rows, dt) for dt in ("protein", "dna")}
    xs = sorted(set(sets["protein"]) | set(sets["dna"]))
    pos = {x: i for i, x in enumerate(xs)}
    n, handles = 0, []
    for dt, shift in (("protein", -0.16), ("dna", 0.16)):
        by = sets[dt]
        col = FS.ALPHABET[dt]
        for x in sorted(by):
            v = by[x]; q = pos[x] + shift
            ax.vlines(q, min(v), max(v), color=col, lw=0.8, zorder=2)
            ax.scatter([q] * len(v), v, s=8, color=col, marker="o", alpha=0.8, zorder=3, linewidths=0)
            ax.hlines(st.median(v), q - 0.14, q + 0.14, color=FS.INK, lw=1.0, zorder=4)
        k = sum(len(v) for v in by.values()); n += k
        handles.append(ax.scatter([], [], s=8, color=col, marker="o",
                                  label=f"{'amino acid' if dt == 'protein' else 'nucleotide'} "
                                        f"({len(by)} sizes, n={k})"))
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log"); FS.ratio_ticks(ax)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlim(-0.7, len(xs) - 0.3)
    ax.set_xlabel("taxa"); ax.set_ylabel("speedup  (IQ-TREE3 wall / IQ-TREE3-GPU wall)")
    # the CV range and median per alphabet are in REPLICATES_20T.tsv and KEY_NUMBERS (replicate_cv_*)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    ax.set_title(FS.stratum(20, n), loc="left")
    return fig, n


# --------------------------------------------------------------------------- S6 GPU utilisation
def write_replicate_table(rows):
    """REPLICATES_20T.tsv -- one table for both alphabets (Supplementary Data 5), defined by the
    same replicate_sets() the S4 panel draws, so the table and the figure cannot drift."""
    path = os.path.join(REPO, "benchmark", "runs", "REPLICATES_20T.tsv")
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["data_type", "taxa", "n_replicates", "speedup_median", "speedup_mean",
                    "speedup_sd", "speedup_cv", "speedups"])
        for dt in ("protein", "dna"):
            by = replicate_sets(rows, dt)
            for t in sorted(by):
                v = by[t]
                sd = st.stdev(v) if len(v) > 1 else 0.0
                w.writerow([dt, t, len(v), f"{st.median(v):.3f}", f"{st.mean(v):.3f}", f"{sd:.3f}",
                            f"{sd/st.mean(v):.3f}", ";".join(f"{x:.2f}" for x in v)])
                n += 1
    print(f"  wrote benchmark/runs/REPLICATES_20T.tsv ({n} alphabet x size rows)")


def main():
    rows = FD.for_fits(FD.load(REPO))
    write_replicate_table(rows)
    date = max((r.get("datetime_utc") or "")[:10] for r in FD.load(REPO))
    out = [(replicates(rows), 4, "a", "replicates", "Replicate spread at 20 threads, both alphabets")]
    for (fig, n), num, letter, slug, title in out:
        stem = FS.save_panel(fig, num, letter, slug, title, os.path.basename(__file__), SRC, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
