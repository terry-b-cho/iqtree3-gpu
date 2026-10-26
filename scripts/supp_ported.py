#!/usr/bin/env python3
"""supp_ported.py - S01-S05, S07 and S10, rebuilt against the current pipeline.

These figures previously came from generators that wrote to results/figures/ under ad-hoc names,
outside the naming contract, outside the manifest, and outside the style and build filters. They are
rebuilt here so that every published figure has one provenance: current-build rows only, the shared
style, the Fig{S}{NN}{p}_{slug} contract, and a manifest row.

  S01 campaign coverage grid          S04 tree montage, ordered by taxa
  S02 replicate spread, amino acid    S05 complete per-run tree record (multipage)
  S03 replicate spread, nucleotide    S07 production memory sizing
                                      S10 sampled GPU utilisation

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, math, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); H = _load("headline"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

SRC = "results/benchmarks/RUNS.tsv"
W, H_ = 2.75, 2.15
PANEL_IN = 2.5
SAMPLE_INTERVAL_S = 0.5      # the nvidia-smi sampler's period, set in paired_benchmark.sh


# ----------------------------------------------------------------------------- S01 coverage grid
def s01_coverage(rows):
    """Which (dataset, thread count) cells exist. Replaces the old status dashboard, minus the
    project-management framing and the live timestamp."""
    cells = collections.defaultdict(set)
    for r in rows:
        cells[(r["data_type"], int(r["taxa"]))].add(r["threads"])
    dts = ["protein", "dna"]
    taxa = sorted({t for _, t in cells})
    M = np.zeros((len(dts) * 2, len(taxa)))
    for i, dt in enumerate(dts):
        for j, th in enumerate(("1", "20")):
            for k, t in enumerate(taxa):
                M[i * 2 + j, k] = len([1 for th2 in cells.get((dt, t), ()) if th2 == th])
    fig, ax = plt.subplots(figsize=(W * 2.0, H_ * 0.8))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest")
    for (i, j), v in np.ndenumerate(M):
        if v:
            ax.text(j, i, "y", ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["amino acid, 1T", "amino acid, 20T", "nucleotide, 1T", "nucleotide, 20T"])
    ax.set_xticks(range(len(taxa))); ax.set_xticklabels([str(t) for t in taxa])
    ax.set_xlabel("taxa in tree")
    ax.set_title(f"Campaign coverage: {len(rows)} paired runs on the reported build\n"
                 "a filled cell means at least one measured run at that size", pad=3)
    return fig, len(rows)


# ------------------------------------------------------------------- S02/S03 replicate dispersion
def replicate_sets(rows, dt):
    """Repeat runs of the same dataset on the same card class, current build only."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["data_type"] != dt or r["threads"] != "20" or not H.is_replicate(r):
            continue
        if r["_speedup"]:
            by[int(r["taxa"])].append(r["_speedup"])
    return {k: v for k, v in by.items() if len(v) >= 2}


def s02_s03(rows, dt):
    by = replicate_sets(rows, dt)
    fig, ax = plt.subplots(figsize=(W, H_))
    xs = sorted(by)
    col = FS.ALPHABET[dt]
    for x in xs:
        v = by[x]
        ax.vlines(x, min(v), max(v), color=col, lw=0.8, zorder=2)
        ax.scatter([x] * len(v), v, s=8, color=col, marker=FS.alphabet_marker(dt), zorder=3)
        ax.hlines(st.median(v), x * 0.93, x * 1.07, color=FS.INK, lw=1.0, zorder=4)
    cvs = [st.stdev(by[x]) / st.mean(by[x]) for x in xs if len(by[x]) > 1]
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); FS.ratio_ticks(ax)
    # A log x-axis labels only decades, so the taxon counts -- the whole point of the abscissa --
    # were invisible behind a lone "10^1". Label the sizes that actually exist.
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xs]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("taxa"); ax.set_ylabel("speedup")
    ax.set_title(f"Replicate spread, {'amino acid' if dt=='protein' else 'nucleotide'}, 20 threads\n"
                 f"{len(xs)} sizes, 2-5 runs each; CV {min(cvs):.2f}-{max(cvs):.2f}, "
                 f"median {st.median(cvs):.2f}", pad=3)
    return fig, sum(len(v) for v in by.values())


# ------------------------------------------------------------------------ S07 production sizing
def s07_production():
    """Peak RAM against genome count for the real 42-genus workload. The wall-time panel of the old
    figure is deliberately NOT carried over: it fitted an exponent at R^2 = 0.44."""
    path = os.path.join(REPO, "results", "benchmarks", "cpu_production_scaling.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    n = np.array([int(r["n_mags"]) for r in rows])
    ram = np.array([int(r["peak_pam_mb"]) if "peak_pam_mb" in r else int(r["peak_ram_mb"])
                    for r in rows]) / 1024.0
    slope, icpt = np.polyfit(n, ram, 1)
    r2 = 1 - ((ram - (slope * n + icpt)) ** 2).sum() / ((ram - ram.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(W, H_))
    ax.scatter(n, ram, s=9, color=FS.PROTEIN, zorder=3)
    xr = np.linspace(n.min(), n.max(), 30)
    ax.plot(xr, slope * xr + icpt, color=FS.PROTEIN, lw=0.9, ls="--", zorder=2)
    ax.set_xlabel("genomes in the genus"); ax.set_ylabel("peak host RAM (GB)")
    ax.set_title(f"Production memory sizing: {slope*1024:.0f} MB per genome\n"
                 f"R$^2$ = {r2:.2f}, n = {len(rows)} genera (CPU-only production runs)", pad=3)
    return fig, len(rows)


# --------------------------------------------------------------------------- S10 GPU utilisation
def s10_utilisation(rows):
    """Sampled SM utilisation, WITH the sampling interval stated -- without it the numbers cannot
    be interpreted against kernels of 3.6-29 microseconds."""
    fig, ax = plt.subplots(figsize=(W, H_))
    for dt in ("protein", "dna"):
        for th, mk in (("1", "^"), ("20", None)):
            pts = [(r["_complexity"], FD.num(r.get("util_sm_mean"))) for r in rows
                   if r["data_type"] == dt and r["threads"] == th]
            pts = [(a, b) for a, b in pts if a and b is not None]
            if len(pts) < 3:
                continue
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=FS.ALPHABET[dt],
                       marker=mk or FS.alphabet_marker(dt), zorder=3,
                       label=f"{'amino acid' if dt=='protein' else 'nucleotide'}, {th}T (n={len(pts)})")
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    ax.set_xlabel("tree complexity"); ax.set_ylabel("sampled SM utilisation (%)")
    ax.set_title(f"Sampled GPU utilisation\nnvidia-smi every {SAMPLE_INTERVAL_S} s against "
                 f"kernels of 3.6-29 µs: a coarse proxy", pad=3)
    ax.legend(loc="lower right", ncol=1)
    return fig, sum(1 for r in rows if FD.num(r.get("util_sm_mean")) is not None)


def write_replicate_tables(rows):
    """REPLICATES_{protein,dna}20T.tsv -- kept because the manuscript cites them as data tables.
    Previously produced by a figure script that is now retired; the definition now lives with the figure that
    uses it so the two cannot drift."""
    STEM = {"protein": "protein20T", "dna": "dna20T"}
    for dt, stem in STEM.items():
        by = replicate_sets(rows, dt)
        path = os.path.join(REPO, "results", "benchmarks", f"REPLICATES_{stem}.tsv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["taxa", "n_replicates", "speedup_median", "speedup_mean", "speedup_sd",
                        "speedup_cv", "speedups"])
            for t in sorted(by):
                v = by[t]
                sd = st.stdev(v) if len(v) > 1 else 0.0
                w.writerow([t, len(v), f"{st.median(v):.3f}", f"{st.mean(v):.3f}", f"{sd:.3f}",
                            f"{sd/st.mean(v):.3f}", ";".join(f"{x:.2f}" for x in v)])
        print(f"  wrote results/benchmarks/REPLICATES_{stem}.tsv ({len(by)} sizes)")


def main():
    rows = FD.for_fits(FD.load(REPO))
    write_replicate_tables(rows)
    allrows = FD.load(REPO)
    date = max((r.get("datetime_utc") or "")[:10] for r in allrows)
    out = [(s01_coverage(allrows), 1, "a", "coverage_grid", "Campaign coverage", SRC),
           (s02_s03(rows, "protein"), 6, "a", "replicates_protein",
            "Replicate spread, amino acid", SRC),
           (s02_s03(rows, "dna"), 7, "a", "replicates_nucleotide",
            "Replicate spread, nucleotide", SRC),
           (s07_production(), 5, "a", "production_memory", "Production memory sizing",
            "results/benchmarks/cpu_production_scaling.tsv"),
           (s10_utilisation(rows), 11, "a", "gpu_utilisation", "Sampled GPU utilisation", SRC)]
    for (fig, n), num, letter, slug, title, src in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, num, letter, slug, title, os.path.basename(__file__), src, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "results", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
