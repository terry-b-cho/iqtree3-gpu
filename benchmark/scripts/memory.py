#!/usr/bin/env python3
"""memory.py - Figure 6: how much memory to ask for, and what happens when it is not enough.

VRAM is the constraint users actually hit, and the board's letter assumed we had not logged it. We
had, twice over:
  * `peak_vram_mb`, from an nvidia-smi sampler -- device-wide, MiB, 0.5 s interval, GPU arm only.
    Because the allocation was NOT node-exclusive, this is an UPPER BOUND on our own process.
  * the engine's own arena footprint, printed in every gpu.stdout, which is a model-side number
    computed from the allocated slot capacity and is independent of the sampler.
Plotting the two against each other is the check: agreement between independent measurements is
evidence, and disagreement would itself be worth knowing.

PANEL B REPLACES the board's requested rebuilds/rebinds panel, with their agreement. `rebuilds` is 1
in every row (the arena is allocated once per process) so a panel of it plots a constant, and
`rebinds` counts re-points that were served WITHOUT touching device memory -- reallocations avoided,
not memory traffic. What is a real and unreported result is the VRAM guard.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, re, csv, glob, math, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

SRC = "benchmark/runs/RUNS.tsv"
W, H = 2.75, 2.15

# Alphabet is perfectly aliased with model treatment; label the WORKFLOW.
WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}
ARENA = re.compile(r"pooling (\d+) slots \(IQ-TREE arena (\d+) \+ 2 NNI\) -> ([\d.]+) GB")
GUARD = re.compile(r"rate-category capacity reduced from (\d+) to (\d+): (\d+) slots at ncat=(\d+) "
                   r"would need ([\d.]+) GB of the ([\d.]+) GB")


def arena_and_guard(mod, cells):
    arena, guard = {}, []
    for c in cells:
        p = os.path.join(mod, "benchmark", "runs", "paired", c, "gpu.stdout")
        try:
            txt = open(p, errors="ignore").read()
        except OSError:
            continue
        m = ARENA.search(txt)
        if m:
            arena[c] = dict(slots=int(m.group(1)), gb=float(m.group(3)))
        g = GUARD.search(txt)
        if g:
            guard.append(dict(cell=c, frm=int(g.group(1)), to=int(g.group(2)),
                              need=float(g.group(5)), have=float(g.group(6))))
    return arena, guard


def entries(r):
    """patterns x states x rate categories of the SELECTED model -- conditional-likelihood entries
    per slot as the analysis needs them. The engine allocates for its category CAPACITY (ten under
    ModelFinder), which is why the per-entry bytes in KEY_NUMBERS use the capacity instead (R13)."""
    m, k = FD.num(r.get("patterns")), FD.num(r.get("num_states"))
    return m * k * FD.rate_cats(r.get("model")) if (m and k) else None


def panel_a(rows):
    """Sizing rule. Peak host RAM per job (sacct MaxRSS, one value per paired job, so it is not
    split by arm) and sampled device VRAM for the GPU arm. Until round 8 this panel drew two
    "host RAM" series, CPU arm and GPU arm, from the SAME column, so the two were identical."""
    fig, ax = plt.subplots(figsize=(W, H))
    txt = []
    # Hue is reserved for ALPHABET throughout this paper, and this panel compares memory TYPES,
    # not alphabets. Using the nucleotide hue for "device VRAM" would make orange mean two different
    # things in one figure set. Memory type is carried by marker shape, arm by fill, as elsewhere.
    series = [("host RAM, job peak (MaxRSS)", "peak_ram_mb", "cpu", FS.PROTEIN, "o"),
              ("device VRAM, GPU arm (sampled)", "peak_vram_mb", "gpu", FS.PROTEIN, "^")]
    slopes = {}
    for lab, col_name, arm, col, mk in series:
        pts = [(entries(r), FD.num(r.get(col_name))) for r in rows if r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b and b > 0]
        if len(pts) < 5:
            continue
        stl = FS.arm_style(arm, col)
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=8, zorder=3, marker=mk,
                   facecolors=stl["markerfacecolor"], edgecolors=col, linewidths=0.5, label=lab)
        f = FD.loglog_fit(pts)
        xr = np.logspace(np.log10(min(a for a, _ in pts)), np.log10(max(a for a, _ in pts)), 30)
        ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)), color=col, lw=0.8,
                ls=stl["linestyle"], zorder=2)
        slopes[lab] = f
        txt.append(f"{'host' if 'host' in lab else 'device'}: slope {f['slope']:.2f} "
                   f"[{f['ci'][0]:.2f}, {f['ci'][1]:.2f}], r$^2$ {f['r2']:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("conditional-likelihood entries\n(patterns × states × selected rate categories)")
    ax.set_ylabel("peak memory (MiB)")
    # R11 (round-6 review): the second clause used to read "and VRAM is the binding constraint",
    # which this panel does not establish -- it shows host RAM and device VRAM scaling on one axis,
    # i.e. that both are PREDICTABLE. The guard evidence that would bear on binding is in panel B.
    # R13 (round-8 review): "scale with" implied linear; both are sub-linear over the measured
    # range because the slot count and the fixed process footprint do not grow with the array.
    hs = FD.key_number(REPO, "memory_host_slope"); vs = FD.key_number(REPO, "memory_vram_slope")
    ax.set_title(f"Both memories grow sub-linearly with the likelihood array\n"
                 f"(log-log slopes {hs:.2f} host, {vs:.2f} device)", pad=3)
    ax.annotate("\n".join(txt), xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                color=FS.MUTED)
    ax.legend(loc="upper left")
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_b(rows, arena, guard):
    """Two independent device-memory measurements, and the guard events."""
    fig, ax = plt.subplots(figsize=(W, H))
    pts = [(arena[r["cell"]]["gb"] * 1024, FD.num(r.get("peak_vram_mb")))
           for r in rows if r["cell"] in arena and FD.num(r.get("peak_vram_mb"))]
    pts = [(a, b) for a, b in pts if a > 0 and b > 0]
    gcells = {g["cell"] for g in guard}
    reg = [(a, b) for (a, b), r in zip(pts, [r for r in rows if r["cell"] in arena
                                             and FD.num(r.get("peak_vram_mb"))])
           if r["cell"] not in gcells]
    ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=FS.DNA, zorder=3,
               label=f"all runs (n={len(pts)})")
    gp = [(arena[g["cell"]]["gb"] * 1024,
           FD.num(next(r for r in rows if r["cell"] == g["cell"]).get("peak_vram_mb")))
          for g in guard if g["cell"] in arena]
    if gp:
        ax.scatter([a for a, _ in gp], [b for _, b in gp], s=34, facecolors="none",
                   edgecolors=FS.INK, linewidths=1.0, zorder=4,
                   label=f"rate-category capacity reduced (n={len(gp)})")
    lim = [min(min(a for a, _ in pts), min(b for _, b in pts)),
           max(max(a for a, _ in pts), max(b for _, b in pts))]
    ax.plot(lim, lim, color=FS.MUTED, lw=0.7, ls="--", zorder=1)
    ax.annotate("y = x", xy=(lim[0] * 3, lim[0] * 3), xytext=(6, -2),
                textcoords="offset points", ha="left", va="top", color=FS.MUTED)
    # The sampler reads high on small arenas by a fixed ~450 MB -- it is device-wide and also sees
    # the CUDA context. Where the constraint actually binds (multi-GB arenas) the two agree to ~1.0.
    ax.annotate("sampler also sees the\nCUDA context (~450 MiB)", xy=(lim[0] * 4, 500),
                xytext=(0, 26), textcoords="offset points", ha="left", color=FS.MUTED,
                arrowprops=dict(arrowstyle="->", lw=0.5, color=FS.MUTED))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("engine's own arena footprint (MiB)")
    ax.set_ylabel("sampled peak VRAM (MiB, upper bound)")
    ax.set_title("Two independent device-memory measurements;\n"
                 f"capacity reduced in {int(FD.key_number(REPO, 'guard_events_n'))} runs, none failed", pad=3)
    ax.legend(loc="upper left")
    return fig, len(pts)


def _vs(rows, field, xlabel, title, only=None):
    fig, ax = plt.subplots(figsize=(W, H))
    for dt in ("protein", "dna"):
        if only and dt != only:
            continue
        pts = [(FD.num(r.get(field)), r["_speedup"]) for r in rows
               if r["data_type"] == dt and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a is not None and b]
        if pts:
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=FS.ALPHABET[dt],
                       marker=FS.alphabet_marker(dt), zorder=3,
                       label=f"{WF_SHORT[dt]} (n={len(pts)})")
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log"); FS.ratio_ticks(ax)
    ax.set_xlabel(xlabel); ax.set_ylabel("speedup")
    ax.set_title(title, pad=3)
    ax.legend(loc="lower right")
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_d_informativeness(rows):
    """Does an alignment's information content predict how well it accelerates?

    It appears to: the parsimony-informative fraction correlates about +0.73 with log speedup in
    BOTH workflows, and it varies widely in both (0.025-0.514 amino acid, 0.020-0.898 nucleotide),
    so this is not a one-workflow curiosity.

    IT IS ENTIRELY A SIZE EFFECT, AND THE PANEL SAYS SO. Informative fraction correlates +0.89 and
    +0.91 with log complexity -- informative alignments are simply bigger. Once complexity is
    partialled out the correlation collapses to -0.11 and -0.29, i.e. nothing, with a sign opposite
    to the marginal one. This panel therefore forecloses a plausible alternative predictor rather
    than adding one, and supports complexity as the quantity the decision figure should use.

    It replaces the retired `per_pattern_macs` panel, which was states^2 x rate categories exactly
    and so could only restate its own abscissa.
    """
    fig, ax = plt.subplots(figsize=(W, H))
    note = []
    n_tot = 0
    for dt in ("protein", "dna"):
        sub = [r for r in rows if r["data_type"] == dt and r["threads"] == "20"
               and FD.num(r.get("informative_sites")) is not None and FD.num(r.get("sites"))
               and r["_speedup"] and r["_complexity"]]
        if len(sub) < 5:
            continue
        n_tot += len(sub)
        frac = np.array([FD.num(r["informative_sites"]) / FD.num(r["sites"]) for r in sub])
        sp = np.log10([r["_speedup"] for r in sub])
        cx = np.log10([r["_complexity"] for r in sub])
        ax.scatter(frac, 10 ** sp, s=7, color=FS.ALPHABET[dt], marker=FS.alphabet_marker(dt),
                   zorder=3, label=f"{WF_SHORT[dt]} (n={len(sub)})")

        def _resid(y, x):
            A = np.column_stack([x, np.ones_like(x)])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - A @ b
        marg = float(np.corrcoef(frac, sp)[0, 1])
        part = float(np.corrcoef(_resid(frac, cx), _resid(sp, cx))[0, 1])
        note.append(f"{'AA' if dt == 'protein' else 'NT'}: r = {marg:+.2f}, "
                    f"{part:+.2f} given complexity")
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log"); FS.ratio_ticks(ax)
    ax.set_xlabel("parsimony-informative fraction of sites")
    ax.set_ylabel("speedup (CPU wall / GPU wall)")
    ax.text(0.03, 0.97, "\n".join(note), transform=ax.transAxes, va="top", ha="left")
    ax.set_title("Informative alignments accelerate better only\n"
                 "because they are larger, not because they are informative", pad=3)
    ax.legend(loc="lower right", frameon=False)
    return fig, n_tot


def main():
    rows = FD.for_fits(FD.load(REPO))
    arena, guard = arena_and_guard(REPO, [r["cell"] for r in rows])
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    print(f"  arena footprint parsed for {len(arena)} cells; guard events: {len(guard)}")
    for g in guard:
        print(f"    {g['cell']:30s} ncat cap {g['frm']} -> {g['to']}  "
              f"(needed {g['need']:.1f} GB of {g['have']:.1f})")
    out = [(panel_a(rows), "a", "memory_sizing", "Peak memory vs likelihood-array size"),
           (panel_b(rows, arena, guard), "b", "vram_guard", "Arena footprint vs sampled VRAM"),
           (_vs(rows, "gap_ambiguity_pct", "gaps and ambiguity (%)",
                "Gap content does not drive speedup", only="protein"), "c", "gap_ambiguity",
            "Speedup vs gap and ambiguity, amino acid"),
           (panel_d_informativeness(rows), "d", "informativeness",
            "Speedup vs parsimony-informative fraction")]
    for (fig, n), letter, slug, title in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, 6, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
