#!/usr/bin/env python3
"""supplementary.py - the supplementary figures the board asked for that did not exist.

S6  phase-step heatmap from PHASE_STEPS.tsv
S8  model coverage REGENERATED FROM DATA (the previous figure was a hand-drawn schematic, which
    cannot stand in a manuscript whose authority rests on measurement)
S9  superseded-build vs current-build regression on the 44 shared cells
S12a single-thread cost law -- the stratum Figure 3a no longer shows, and where the largest
    speedups occur
S12b cost-law residuals by compute node, with the exponent refitted using node fixed effects.
    Nodes DO shift the level (~2.9x across node medians) and move the nucleotide exponents by
    ~0.10; what survives is the CPU-minus-GPU exponent gap, which is what Figure 3a claims
S13  pattern saturation, retired from Figure 3d -- a negative result kept in the record

S1-S5, S7, S10 are re-emissions of existing generators under the naming contract; S11 (greyscale
proof sheet) is produced by assemble.py.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, csv, collections, statistics as st, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(n):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m
FS = _load("figstyle"); FD = _load("figdata"); FS.apply()
import matplotlib.pyplot as plt
import numpy as np

W, H = 2.75, 2.15

# Alphabet is perfectly aliased with model treatment; label the WORKFLOW.
WF_SHORT = {"protein": "amino acid + ModelFinder", "dna": "nucleotide, fixed model"}


def s6_phase_heatmap(keep):
    """Step x cell, CPU arm. Shows at a glance which steps exist for which alphabet.

    `keep` restricts to current-build cells: PHASE_STEPS.tsv covers all 226 recorded runs, including
    the 44 on the superseded binary, which must not enter a published figure.
    """
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "benchmark", "runs",
                                                        "PHASE_STEPS.tsv")), delimiter="\t")
            if r["cell"] in keep]
    steps = ["parsimony_tree", "fast_ml_search", "modelfinder", "ml_distances", "rapidnj_tree",
             "initial_tree_ll", "tree_search", "unattributed"]
    # Order amino acid FIRST explicitly. sorted() puts "dna" before "protein", which silently
    # transposed the two blocks against their labels -- the figure showed model selection on the
    # nucleotide side, which is impossible.
    order = sorted({(r["data_type"], r["cell"]) for r in rows if r["data_type"] in ("protein", "dna")},
                   key=lambda dc: (0 if dc[0] == "protein" else 1, dc[1]))
    cells = [c for _, c in order]
    idx = {c: i for i, c in enumerate(cells)}
    M = np.full((len(steps), len(cells)), np.nan)
    tot = {}
    for r in rows:
        if r["arm"] != "cpu" or r["cell"] not in idx:
            continue
        if r["step"] == "total":
            tot[r["cell"]] = FD.num(r["wall_s"])
    for r in rows:
        if r["arm"] != "cpu" or r["cell"] not in idx or r["step"] not in steps:
            continue
        t = tot.get(r["cell"])
        w = FD.num(r["wall_s"])
        if t and w is not None and t > 0:
            M[steps.index(r["step"]), idx[r["cell"]]] = 100.0 * w / t
    fig, ax = plt.subplots(figsize=(W * 2.1, H))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([s.replace("_", " ") for s in steps])
    nprot = sum(1 for d, _ in order if d == "protein")
    ax.axvline(nprot - 0.5, color="white", lw=1.0)
    ax.set_xticks([nprot / 2, nprot + (len(cells) - nprot) / 2])
    ax.set_xticklabels([f"amino acid (n={nprot})", f"nucleotide (n={len(cells)-nprot})"])
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("% of that run's CPU-arm wall")
    ax.set_title("Where wall time sits, per run (CPU arm). White gaps are steps that do not exist\n"
                 "for that alphabet -- nucleotide runs have no model selection", pad=3)
    return fig, len(cells)


def s8_model_coverage(rows):
    """REGENERATED FROM DATA, replacing a hand-drawn schematic.

    Takes ALL current-build rows, NOT for_fits(). The `__mdl_*` correctness cells are excluded from
    speedup statistics because they sit below break-even by construction -- but they are precisely
    the runs that exercise the alternative nucleotide models, so excluding them from a COVERAGE
    figure collapses the nucleotide side to GTR alone and understates the very thing the figure
    exists to show. The exclusion rule that is right for speed is wrong for coverage.
    """
    cnt = collections.Counter()
    for r in rows:
        m = (r.get("model") or "NA").split("+")[0]
        cnt[(r["data_type"], m)] += 1
    items = sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1]))
    fig, ax = plt.subplots(figsize=(W * 1.5, H))
    ys = range(len(items))
    ax.barh(list(ys), [v for _, v in items],
            color=[FS.ALPHABET[k[0]] for k, _ in items], height=0.62)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{k[1]}  ({'AA' if k[0]=='protein' else 'NT'})" for k, _ in items])
    ax.invert_yaxis()
    for i, (_, v) in enumerate(items):
        ax.annotate(str(v), xy=(v, i), xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xlabel("runs")
    ax.margins(x=0.14)
    ax.set_title(f"Model classes actually exercised ({len(items)} classes, "
                 f"{sum(cnt.values())} runs)\ncounted from the model each run selected", pad=3)
    return fig, sum(cnt.values())


def s9_build_regression():
    """The 44 shared cells, superseded build against current. Equivalence is unaffected; speed is."""
    allrows = FD.load(REPO, current_build_only=False)
    by = {r["cell"]: r for r in allrows}
    pairs = []
    for c, r in by.items():
        if not c.endswith("__l40s_b2"):
            continue
        old = by.get(c[:-3])
        if old:
            pairs.append((old, r))
    fig, ax = plt.subplots(figsize=(W, H))
    for dt, mk in (("protein", "o"), ("dna", "s")):
        p = [(FD.num(o.get("speedup")), FD.num(n.get("speedup")))
             for o, n in pairs if o["data_type"] == dt]
        p = [(a, b) for a, b in p if a and b]
        if p:
            ax.scatter([a for a, _ in p], [b for _, b in p], s=12, color=FS.ALPHABET[dt],
                       marker=FS.alphabet_marker(dt), zorder=3,
                       label=f"{'amino acid' if dt=='protein' else 'nucleotide'} (n={len(p)})")
    lim = [0.1, 60]
    ax.plot(lim, lim, color=FS.MUTED, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("speedup, superseded build (07c70cae)")
    ax.set_ylabel("speedup, current build (edf783c3)")
    ax.set_title("The two builds differ on nucleotide data only;\n"
                 "every run of both returns the identical tree", pad=3)
    ax.legend(loc="upper left")
    return fig, len(pairs)


def s12_t1_cost_law(rows):
    """Does the sub-unity exponent hold at one thread, where the biggest speedups are?"""
    fig, ax = plt.subplots(figsize=(W, H))
    lines = []
    for dt in ("protein", "dna"):
        sub = [r for r in rows if r["data_type"] == dt and r["threads"] == "1"]
        col = FS.ALPHABET[dt]
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pts = [(FD.work_units(r), FD.num(r.get(key))) for r in sub]
            pts = [(a, b) for a, b in pts if a and b]
            if len(pts) < 4:
                continue
            stl = FS.arm_style(arm, col)
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, zorder=3,
                       facecolors=stl["markerfacecolor"], edgecolors=col, linewidths=0.5)
            f = FD.loglog_fit(pts)
            xr = np.logspace(np.log10(min(a for a, _ in pts)), np.log10(max(a for a, _ in pts)), 30)
            ax.plot(xr, 10 ** (f["intercept"] + f["slope"] * np.log10(xr)), color=col, lw=0.9,
                    ls=stl["linestyle"], zorder=2)
            lines.append(f"{'AA' if dt=='protein' else 'NT'} {arm.upper()}: {f['slope']:.2f} "
                         f"[{f['ci'][0]:.2f},{f['ci'][1]:.2f}] n={f['n']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("invocations x m k$^2$ r"); ax.set_ylabel("wall time (s)")
    ax.set_title("Cost law at ONE thread (n = 30; wide intervals)\n"
                 "filled + solid = GPU arm", pad=3)
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left")
    return fig, sum(1 for r in rows if r["threads"] == "1"), lines


def s12b_node_effects(rows):
    """Is the cost law a property of the workload, or of the machines that happened to run it?

    The Figure 3a stratum (20 threads, L40S) was not one node: it spans a dozen compute nodes and
    two host CPU models, assigned by the scheduler and not by us. Residuals about the Figure 3a fit
    are plotted per node, and the exponent is refitted with node FIXED EFFECTS so the reader can see
    what absorbing the node identity does to it.

    THE ANSWER IS NOT "NO EFFECT", AND THE PANEL MUST NOT SAY SO. Node medians span roughly 2.9x in
    level, and absorbing node identity moves the nucleotide exponents by about 0.10. What survives
    is the quantity Figure 3a actually claims -- the CPU-minus-GPU exponent gap -- which stays
    positive in both workflows (about +0.13 amino acid, +0.06 nucleotide) after node effects are
    removed. The level shifts are absorbed into the intercept and do not reverse the comparison.

    Residuals are log10 wall-seconds, so 0.1 is about 26%. `cpu_model` is used rather than
    `host_cpu_model`, which is populated in only 48 of 175 rows.
    """
    sub = [r for r in rows if r["threads"] == "20" and r.get("card") == "NVIDIA_L40S"
           and r["_work_units"]]
    fig, ax = plt.subplots(figsize=(W + 0.35, H))
    resid, models, gaps = collections.defaultdict(list), {}, {}
    for dt in ("protein", "dna"):
        exps = {}
        for arm, key in (("cpu", "cpu_wall_s"), ("gpu", "gpu_wall_s")):
            pts = [(r, FD.num(r.get(key))) for r in sub if r["data_type"] == dt]
            pts = [(r, w) for r, w in pts if w]
            if len(pts) < 5:
                continue
            f = FD.loglog_fit([(r["_work_units"], w) for r, w in pts])
            for r, w in pts:
                e = np.log10(w) - (f["intercept"] + f["slope"] * np.log10(r["_work_units"]))
                resid[r["node"]].append((e, dt, arm))
                models[r["node"]] = r.get("cpu_model") or "NA"
            # refit with one indicator per node: the exponent after node identity is absorbed
            nodes = sorted({r["node"] for r, _ in pts})
            D = np.zeros((len(pts), len(nodes)))
            for i, (r, _) in enumerate(pts):
                D[i, nodes.index(r["node"])] = 1.0
            A = np.column_stack([np.log10([r["_work_units"] for r, _ in pts]), D])
            beta, *_ = np.linalg.lstsq(A, np.log10([w for _, w in pts]), rcond=None)
            exps[arm] = (f["slope"], beta[0])
        if "cpu" in exps and "gpu" in exps:
            gaps[dt] = (exps["cpu"][0] - exps["gpu"][0], exps["cpu"][1] - exps["gpu"][1])
    order = sorted(resid, key=lambda n: (models[n], n))
    cpus = sorted({models[n] for n in order})
    for i, node in enumerate(order):
        for e, dt, arm in resid[node]:
            stl = FS.arm_style(arm, FS.ALPHABET[dt])
            ax.scatter([i + (-0.14 if arm == "cpu" else 0.14)], [e], s=6, zorder=3,
                       marker=FS.alphabet_marker(dt), facecolors=stl["markerfacecolor"],
                       edgecolors=FS.ALPHABET[dt], linewidths=0.5)
        ax.hlines(st.median([e for e, _, _ in resid[node]]), i - 0.30, i + 0.30,
                  color=FS.INK, lw=1.0, zorder=4)
    ax.axhline(0.0, color=FS.MUTED, lw=0.7, ls="--", zorder=1)
    if len(cpus) > 1:
        ax.axvline(sum(1 for n in order if models[n] == cpus[0]) - 0.5,
                   color=FS.MUTED, lw=0.5, ls=":", zorder=1)
    ax.set_xticks(range(len(order)))
    # node names share the prefix "compute-g-"; the distinguishing part is the tail
    ax.set_xticklabels([n.replace("compute-", "") for n in order], rotation=90)
    ax.set_xlabel(f"compute node ({len(order)} nodes, {len(cpus)} host CPU models)")
    ax.set_ylabel("residual about the Fig. 3a fit\n(log$_{10}$ wall seconds)")
    meds = [st.median([e for e, _, _ in resid[n]]) for n in order]
    spread = max(meds) - min(meds)
    txt = "\n".join(f"{'AA' if d == 'protein' else 'NT'} CPU-GPU exponent gap: "
                     f"{g0:+.2f} pooled, {g1:+.2f} with node effects"
                     for d, (g0, g1) in gaps.items())
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left")
    ax.set_title(f"Nodes shift the level ({10 ** spread:.1f}x across node medians)\n"
                 f"but the CPU-GPU exponent gap survives absorbing them", pad=3)
    return fig, sum(len(v) for v in resid.values())


def s13_pattern_saturation(rows):
    """Retired from Figure 3d. A negative result: occupancy does not explain the sub-parity runs.

    It stays in the record because it rules out the most natural alternative mechanism -- that small
    alignments are slow on the device because they cannot fill it -- and a reader who suspects that
    is owed the measurement.
    """
    fig, ax = plt.subplots(figsize=(W, H))
    n = 0
    for dt in ("protein", "dna"):
        pts = [(FD.num(r.get("pattern_saturation")), r["_speedup"]) for r in rows
               if r["data_type"] == dt and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        if pts:
            n += len(pts)
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=7, color=FS.ALPHABET[dt],
                       marker=FS.alphabet_marker(dt), zorder=3,
                       label=f"{WF_SHORT[dt]} (n={len(pts)})")
    ax.axhline(1.0, color=FS.INK, lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("pattern saturation (occupancy proxy)")
    ax.set_ylabel("speedup (CPU wall / GPU wall)")
    FS.ratio_ticks(ax)
    ax.set_title("Occupancy does not explain the sub-parity runs", pad=3)
    ax.legend(loc="upper left", frameon=False)
    return fig, n


def main():
    rows = FD.for_fits(FD.load(REPO))
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    SRC = "benchmark/runs/RUNS.tsv"
    f12, n12, l12 = s12_t1_cost_law(rows)
    out = [(s6_phase_heatmap({r["cell"] for r in rows}), 13, "a", "phase_step_heatmap",
            "Phase-step share per run",
            "benchmark/runs/PHASE_STEPS.tsv"),
           (s8_model_coverage(FD.load(REPO)), 4, "a", "model_coverage_data",
            "Model classes exercised", SRC),
           (s9_build_regression(), 12, "a", "build_regression", "Superseded vs current build", SRC),
           ((f12, n12), 9, "a", "cost_law_t1", "Cost law at one thread", SRC),
           (s12b_node_effects(rows), 9, "b", "node_effects",
            "Cost-law residuals by compute node", SRC),
           (s13_pattern_saturation(rows), 10, "a", "pattern_saturation",
            "Speedup vs pattern saturation", SRC)]
    for (fig, n), num, letter, slug, title, src in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, num, letter, slug, title, os.path.basename(__file__), src, n,
                             supp=True, data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    for l in l12:
        print(f"    S9 {l}")
    FS.write_manifest(os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
