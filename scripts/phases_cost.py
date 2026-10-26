#!/usr/bin/env python3
"""phases_cost.py - Figure 4: where the time goes, and what it costs.

Carries the campaign's single most explanatory fact: model selection is ~91% of amino-acid wall time
and is absent for nucleotide runs, which is most of why the two alphabets behave so differently.

TWO HONESTY CONSTRAINTS the board required, both enforced here:
  * Device time CANNOT be shown. `gpu_kernel_time_s` is NA/TODO_nsight in all 182 rows. What is
    reportable is the bound from the profiling kernel fraction, and its DIRECTION matters: profiling
    inflates host wall more than device kernel time, so the profiled fraction is a LOWER bound on the
    unprofiled one, and the absolute kernel seconds are the usable number.
  * The compute saving is POTENTIAL, not realised. The allocation was not node-exclusive but did
    request -c 20 for both arms, so the cores the GPU arm leaves idle were reserved to our own job
    and unavailable to others. The saving is realisable only if a GPU user requests fewer cores.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt

SRC = "results/benchmarks/PHASE_STEPS.tsv"
W, H = 2.75, 2.15


def phases(mod, keep):
    """(cell, arm) -> {step: (wall, cpu)} for top-level steps, restricted to `keep` cells."""
    out = collections.defaultdict(dict)
    with open(os.path.join(mod, "results", "benchmarks", "PHASE_STEPS.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["cell"] not in keep:
                continue
            out[(r["cell"], r["arm"])][r["step"]] = (FD.num(r["wall_s"]), FD.num(r["cpu_s"]))
    return out


def panel_a(ph, meta):
    """Wall share by phase, per alphabet -- split as the board asked."""
    fig, ax = plt.subplots(figsize=(W, H))
    labs, bots = [], []
    for i, dt in enumerate(("protein", "dna")):
        cells = [c for c, (d, t) in meta.items() if d == dt and t == "20"]
        shares = collections.defaultdict(list)
        for c in cells:
            v = ph.get((c, "cpu"), {})
            tot = (v.get("total") or (None,))[0]
            if not tot:
                continue
            for s in ("modelfinder", "tree_search", "unattributed"):
                shares[s].append(((v.get(s) or (0,))[0] or 0) / tot * 100)
        bottom = 0
        # One hue at three lightnesses: this is a parts-of-a-whole stack, and the categorical
        # alphabet hues are already in use elsewhere in this very figure.
        for s, col in (("modelfinder", FS.SIZE_RAMP[4]),
                       ("tree_search", FS.SIZE_RAMP[1]),
                       ("unattributed", "#D9D9D9")):
            m = st.mean(shares[s]) if shares[s] else 0.0
            ax.bar(i, m, 0.5, bottom=bottom, color=col, edgecolor="white", linewidth=0.5)
            if m > 6:
                ax.text(i, bottom + m / 2, f"{m:.0f}%", ha="center", va="center",
                        color="white" if s == "modelfinder" else FS.INK, fontweight="bold")
            bottom += m
        labs.append(f"{'amino acid' if dt=='protein' else 'nucleotide'}\n(n={len(cells)})")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs)
    ax.set_ylabel("share of CPU-arm wall time (%)"); ax.set_ylim(0, 108)
    for s, col, lab in (("modelfinder", FS.SIZE_RAMP[4], "model selection"),
                        ("tree_search", FS.SIZE_RAMP[1], "tree search (incl. bootstrap)"),
                        ("unattributed", "#D9D9D9", "untimed by IQ-TREE")):
        ax.bar([0], [0], color=col, label=lab)
    # -0.62 put the legend's first row on top of the second line of the two-line tick labels
    # ("amino acid" / "(n=67)"). The anchor clears both lines.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    ax.set_title("Model selection dominates amino-acid runtime\nand does not exist for nucleotide", pad=3)
    return fig, len(meta)


def panel_b(ph, meta):
    fig, ax = plt.subplots(figsize=(W, H))
    for s, col, lab in (("modelfinder", FS.PHASE_COLORS["modelfinder"], "model selection"),
                        ("tree_search", FS.PHASE_COLORS["tree_search"], "tree search")):
        pts = []
        for (c, arm), v in ph.items():
            if arm != "cpu" or meta.get(c, ("", ""))[1] != "20":
                continue
            a = (v.get(s) or (None,))[0]
            b = (ph.get((c, "gpu"), {}).get(s) or (None,))[0]
            if a and b and b > 0:
                pts.append((FD.num(meta[c][2]), a / b))
        pts = [(x, y) for x, y in pts if x]
        if pts:
            ax.scatter([x for x, _ in pts], [y for _, y in pts], s=7, color=col, zorder=3,
                       label=f"{lab} (n={len(pts)})")
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("taxa"); ax.set_ylabel("phase speedup")
    FS.ratio_ticks(ax)
    ax.set_title("Both phases accelerate, model selection most", pad=3)
    ax.legend(loc="lower right")
    return fig, len(meta)


def panel_c(ph, meta):
    """PER-RUN distribution of the CPU-time ratio -- an aggregate is dominated by the largest cells."""
    fig, ax = plt.subplots(figsize=(W, H))
    data = []
    for dt in ("protein", "dna"):
        v = []
        for c, (d, t, _) in meta.items():
            if d != dt or t != "20":
                continue
            a = (ph.get((c, "cpu"), {}).get("total") or (None, None))[1]
            b = (ph.get((c, "gpu"), {}).get("total") or (None, None))[1]
            if a and b and b > 0:
                v.append(a / b)
        data.append((dt, v))
    for i, (dt, v) in enumerate(data):
        if not v:
            continue
        col = FS.ALPHABET[dt]
        # jitter: without it ~70 points stack on one abscissa and the density is unreadable
        import random as _r
        rng = _r.Random(7)
        ax.scatter([i + rng.uniform(-0.16, 0.16) for _ in v], v, s=7, color=col, alpha=0.7,
                   zorder=3, marker=FS.alphabet_marker(dt))
        ax.hlines(st.median(v), i - 0.26, i + 0.26, color=FS.INK, lw=1.1, zorder=4)
        # %.0f printed the nucleotide median of 1.6 as "2x", which is what made the panel
        # disagree with the manuscript. One decimal is the resolution this ratio actually has.
        ax.annotate(f"median {st.median(v):.1f}x", xy=(i + 0.26, st.median(v)), xytext=(3, 6),
                    textcoords="offset points", va="bottom", ha="left", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{'amino acid' if d=='protein' else 'nucleotide'}\n(n={len(v)})"
                        for d, v in data])
    ax.set_ylabel("CPU-time consumed:  CPU arm / GPU arm")
    ax.set_xlim(-0.5, len(data) - 0.05)
    # The aggregate was hardcoded as "60x" while the computed value is corehours_aggregate_ratio
    # = 68.8. It is read from KEY_NUMBERS now, so a panel title cannot assert a number no table
    # holds -- and crosscheck's existing ban on "60-fold" finally bites here too.
    agg = FD.key_number(REPO, "corehours_aggregate_ratio")
    ax.set_title("Compute consumed, per run, not aggregated\n"
                 f"(the campaign aggregate, {agg:.0f}x, is dominated by the largest cells)", pad=3)
    return fig, sum(len(v) for _, v in data)


def panel_d(ph, meta):
    fig, ax = plt.subplots(figsize=(W, H))
    for i, arm in enumerate(("cpu", "gpu")):
        for dt in ("protein", "dna"):
            v = []
            for c, (d, t, _) in meta.items():
                if d != dt or t != "20":
                    continue
                w, cp = (ph.get((c, arm), {}).get("total") or (None, None))
                if w and cp and w > 0:
                    v.append(cp / w)
            if v:
                x = i + (-0.14 if dt == "protein" else 0.14)
                ax.scatter([x] * len(v), v, s=7, color=FS.ALPHABET[dt], alpha=0.8, zorder=3,
                           marker=FS.alphabet_marker(dt))
                ax.hlines(st.median(v), x - 0.10, x + 0.10, color=FS.INK, lw=1.0, zorder=4)
    ax.axhline(20, color=FS.MUTED, lw=0.7, ls=":", zorder=1)
    ax.annotate("20 cores requested", xy=(1.45, 20), xytext=(0, 2), textcoords="offset points",
                ha="right", va="bottom", color=FS.MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CPU arm", "GPU arm"])
    ax.set_ylabel("cores kept busy  (CPU-time / wall-time)")
    ax.set_xlim(-0.45, 1.55); ax.set_ylim(0, 24)
    ax.set_title("GPU arm: 2 of 20 cores on amino acid, 10 on nucleotide\n"
                 "cores stay allocated, so the saving is potential", pad=3)
    return fig, len(meta)


def main():
    rows = FD.for_fits(FD.load(REPO))
    meta = {r["cell"]: (r["data_type"], r["threads"], r["taxa"]) for r in rows}
    ph = phases(REPO, set(meta))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    out = [(panel_a(ph, {k: (v[0], v[1]) for k, v in meta.items()}), "a", "phase_share",
            "Wall-time share by phase"),
           (panel_b(ph, meta), "b", "phase_speedup", "Per-phase speedup vs taxa"),
           (panel_c(ph, meta), "c", "cpu_time_ratio", "Per-run CPU-time ratio"),
           (panel_d(ph, meta), "d", "parallel_efficiency", "Parallel efficiency by arm")]
    for (fig, n), letter, slug, title in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, 4, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "results", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
