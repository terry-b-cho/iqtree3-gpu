#!/usr/bin/env python3
"""mechanism.py - Figure 3: why the speedup behaves as it does.

PANEL COUNT CHANGED. `pattern_saturation` was retired to Supplementary Fig. S13: it is a negative
result (occupancy does not explain the sub-parity runs) and the board asked the main-text figure to
carry the positive mechanism only. The profiled decomposition moved up from panel e to panel d.

The cost law is the strongest result in the campaign and had a confound the board required fixing:
the GPU arm pooled six hardware cards, so a sub-unity exponent could have reflected the card mixture
rather than a property of the backend. (Thread count was ALREADY stratified -- only the card pooling
was real.) The headline fit is therefore T20 on the L40S alone, with the pooled six-card fit shown
as a faint reference, and every exponent carries n and a paired bootstrap CI.

WHY AN EXPONENT BELOW 1 IS POSSIBLE with invocation count already on the abscissa: per-invocation
cost FALLS with problem size, because fixed launch and transfer overheads amortise over more work per
call. The profiling panel is the direct evidence -- a nucleotide kernel averages 3.6 us, so dispatch
is a large fraction of each call at small sizes and a small one at large.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, math, importlib.util as ilu

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
HEADLINE_CARD = "NVIDIA_L40S"


def panel_a(rows):
    fig, ax = plt.subplots(figsize=(W + 0.35, H))
    lines = []
    for dt in ("protein", "dna"):
        base = [r for r in rows if r["data_type"] == dt and r["threads"] == "20"]
        col = FS.ALPHABET[dt]
        # pooled six-card fit, drawn faint as a robustness reference
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pooled = [(FD.work_units(r), FD.num(r.get(key))) for r in base]
            f = FD.loglog_fit([(a, b) for a, b in pooled if a and b])
            if f:
                xr = np.logspace(*np.log10([min(a for a, b in pooled if a and b),
                                            max(a for a, b in pooled if a and b)]), 30)
                ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)),
                        color=FS.MUTED, lw=0.6, alpha=0.55, zorder=1)
        # headline: single card
        one = [r for r in base if r.get("card") == HEADLINE_CARD]
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pts = [(FD.work_units(r), FD.num(r.get(key))) for r in one]
            pts = [(a, b) for a, b in pts if a and b]
            if len(pts) < 5:
                continue
            st = FS.arm_style(arm, col)
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=6, zorder=3,
                       facecolors=st["markerfacecolor"], edgecolors=col, linewidths=0.5,
                       marker=FS.alphabet_marker(dt))
            f = FD.loglog_fit(pts)
            xr = np.logspace(np.log10(min(a for a, _ in pts)), np.log10(max(a for a, _ in pts)), 30)
            ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)), color=col, lw=0.9,
                    ls=st["linestyle"], zorder=2)
            lines.append(f"{'AA' if dt=='protein' else 'NT'} {arm.upper()}: "
                         f"{f['slope']:.2f} [{f['ci'][0]:.2f},{f['ci'][1]:.2f}] n={f['n']}")
        # Is the GPU exponent really below the CPU exponent? Test the PAIRED difference on the same
        # cells -- comparing the two marginal CIs above would say "no" for nucleotide, wrongly.
        pc = [(FD.work_units(r), FD.num(r.get("cpu_wall_s"))) for r in one]
        pg = [(FD.work_units(r), FD.num(r.get("gpu_wall_s"))) for r in one]
        keep = [i for i in range(len(pc)) if pc[i][0] and pc[i][1] and pg[i][1]]
        if len(keep) >= 5:
            sd = FD.slope_difference([pc[i] for i in keep], [pg[i] for i in keep])
            lines.append(f"   CPU-GPU {sd['diff']:+.2f} [{sd['ci'][0]:+.2f},{sd['ci'][1]:+.2f}]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("invocations  x  m k$^2$ r   (work units)")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"Wall time follows invocations x per-invocation work\n"
                 f"L40S only; grey = pooled six-card fit", pad=3)
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left")
    return fig, sum(1 for r in rows if r["threads"] == "20"), lines


def panel_b(rows):
    """Invocations vs taxa, stratified by whether ModelFinder ran -- the variance pooling lost."""
    fig, ax = plt.subplots(figsize=(W, H))
    for ms, lab, mk in (("full_ModelFinder_TESTNEW", "with model selection", "o"),
                        ("fixed_model", "fixed model", "s")):
        pts = [(FD.num(r.get("taxa")), FD.num(r.get("gpu_calls"))) for r in rows
               if r.get("model_search") == ms and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        if len(pts) < 4:
            continue
        col = FS.PROTEIN if ms.startswith("full") else FS.DNA
        f = FD.loglog_fit(pts)
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=col, marker=mk, zorder=3,
                   label=f"{lab}: n$^{{{f['slope']:.2f}}}$, R$^2$={f['r2']:.2f} (n={f['n']})")
        xr = np.linspace(min(a for a, _ in pts), max(a for a, _ in pts), 30)
        ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)), color=col, lw=0.9, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("taxa (n)"); ax.set_ylabel("likelihood-kernel invocations")
    ax.set_title("Outer-loop demand, split by whether\nmodel selection ran", pad=3)
    ax.legend(loc="lower right")
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_c(rows):
    """Measured speedup against the Amdahl bound implied by each run's accelerable fraction."""
    fig, ax = plt.subplots(figsize=(W, H))
    for dt in ("protein", "dna"):
        pts = [(FD.num(r.get("accelerable_fraction")), r["_speedup"]) for r in rows
               if r["data_type"] == dt and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        if pts:
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=FS.ALPHABET[dt],
                       marker=FS.alphabet_marker(dt), zorder=3,
                       label=f"{'amino acid' if dt=='protein' else 'nucleotide'} (n={len(pts)})")
    f = np.linspace(0.30, 0.999, 200)
    ax.plot(f, 1 / (1 - f), color=FS.INK, lw=0.9, ls="--", zorder=2, label="Amdahl bound 1/(1-f)")
    ax.set_yscale("log")
    ax.set_xlabel("accelerable fraction f  (share of wall in the offloaded phase)")
    ax.set_ylabel("speedup")
    ax.set_ylim(0.3, 300)
    FS.ratio_ticks(ax, (0.5, 1, 2, 5, 10, 50, 200))
    ax.set_title("Every run sits below its own Amdahl ceiling\nand far from it", pad=3)
    ax.legend(loc="upper left")
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_d():
    """Profiled decomposition. BARS, not a stack -- see the note in the plan: kernel and memcpy
    come from two different nsys reports, are device-timeline durations that can overlap, and the
    denominator is host wall of the instrumented process."""
    path = os.path.join(REPO, "benchmark", "profiling", "nsys_kernel_summary.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    fig, ax = plt.subplots(figsize=(W, H))
    labs = []
    for i, r in enumerate(rows):
        dt = "protein" if r["case"] == "protein" else "dna"
        col = FS.ALPHABET[dt]
        wall = float(r["total_wall_s"]); kern = float(r["kernel_time_s"]); mem = float(r["memcpy_time_s"])
        nk = int(r["num_kernels"])
        ax.bar(i - 0.17, kern, 0.32, color=col, edgecolor=col, linewidth=0.6)
        ax.bar(i + 0.17, mem, 0.32, facecolor="none", edgecolor=col, hatch="///", linewidth=0.6)
        ax.hlines(wall, i - 0.42, i + 0.42, color=FS.INK, lw=1.0, ls="--")
        ax.annotate(f"wall {wall:.0f}s", xy=(i, wall), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom")
        ax.annotate(f"{kern/wall*100:.0f}%", xy=(i - 0.17, kern), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom")
        labs.append(f"{'amino acid' if dt=='protein' else 'nucleotide'}\n"
                    f"{nk:,} launches\nmean {kern/nk*1e6:.1f} µs")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labs)
    ax.set_ylabel("seconds")
    ax.set_ylim(0, max(float(r["total_wall_s"]) for r in rows) * 1.32)
    ax.set_title("Device kernels are a small part of wall time\n"
                 "solid = kernel, hatched = memcpy; n = 1 profiled run each", pad=3)
    return fig, len(rows)


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    fa, na, lines = panel_a(rows)
    fb, nb = panel_b(rows)
    fc, nc = panel_c(rows)
    fd, nd = panel_d()
    # Each panel is built ONCE. The previous form called panel_b/c/d twice -- once for the figure
    # and once for its n -- which built and leaked a second copy of every one of them.
    out = [(fa, "a", "cost_law", "Cost law, stratified by card", na),
           (fb, "b", "outer_demand", "Kernel invocations vs taxa", nb),
           (fc, "c", "amdahl", "Speedup vs accelerable fraction", nc),
           (fd, "d", "nsys_profile", "Profiled wall decomposition", nd)]
    for fig, letter, slug, title, n in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, 3, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for l in lines:
        print(f"    {l}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
