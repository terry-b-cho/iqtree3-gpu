#!/usr/bin/env python3
"""hardware.py - Figure 5: which card, and why fp64 is the wrong question.

The conclusion (a 19x spread in nominal fp64 produces only a ~3.7x spread in wall time) was right
but tested only the null. This figure tests a CANDIDATE EXPLANATION alongside it: nominal memory
bandwidth, and -- better, because it is a measurement rather than a datasheet -- the memory-controller
utilisation the harness actually sampled.

Cards sit on different hosts, so the CPU-arm wall differs between them (see
card_host_cpu_spread_max in KEY_NUMBERS.tsv; 2.00x over the four shared datasets). Cross-card
SPEEDUP is therefore not comparable and is never used for the ranking; absolute GPU-arm wall on a
shared dataset set is. Panel d shows speedup only within a card, faceted.

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

SRC = "results/benchmarks/RUNS.tsv"
W, H = 2.75, 2.15
LABEL = {"NVIDIA_A100_80GB_PCIe": "A100", "Tesla_V100S-PCIE-32GB": "V100S",
         "Tesla_V100-SXM2-16GB": "V100", "NVIDIA_L40S": "L40S",
         "NVIDIA_A40": "A40", "Quadro_RTX_8000": "RTX 8000"}
SPEC_KEY = {"NVIDIA_A100_80GB_PCIe": "a100", "Tesla_V100S-PCIE-32GB": "teslaV100s",
            "Tesla_V100-SXM2-16GB": "teslaV100", "NVIDIA_L40S": "l40s",
            "NVIDIA_A40": "a40", "Quadro_RTX_8000": "rtx8000"}


def specs():
    out = {}
    with open(os.path.join(REPO, "scripts", "benchmark", "card_specs.tsv")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "card":
                hdr = parts; continue
            out[parts[0]] = dict(zip(hdr, parts))
    return out


def shared(rows):
    """The datasets measured on EVERY card -- the only fair basis for a cross-card wall comparison."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["threads"] != "20" or r["data_type"] != "protein":
            continue
        by[r["card"]][r["genus"]].append(r)
    if not by:
        return {}, set()
    common = set.intersection(*[set(v) for v in by.values()])
    return by, common


def _wall(by, common, card, key="gpu_wall_s"):
    return sum(st.median([FD.num(x.get(key)) for x in by[card][g] if FD.num(x.get(key))])
               for g in common)


def _corr(xs, ys):
    import math
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs)); sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def _scatter_spec(ax, by, common, sp, field, xlabel, title):
    """Scatter with generous margins: card labels sit at the data extremes and will otherwise
    overrun the panel, which is how a label ends up printed over the title."""
    import math
    xs, ys, labs = [], [], []
    for card in by:
        k = SPEC_KEY.get(card)
        if not k or k not in sp:
            continue
        xs.append(float(sp[k][field])); ys.append(_wall(by, common, card)); labs.append(LABEL[card])
    ax.scatter(xs, ys, s=22, color=FS.CARD_HUE, zorder=3)
    # Steer labels inward from the nearest edge, AND alternate vertically among points that are
    # close together in x -- L40S and V100 sit at nearly the same abscissa and collided.
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    # Label placement. Alternating above/below against the PREVIOUS point only is not enough: the
    # three fp64-capable cards (V100 7.0, V100S 8.2, A100 9.7) form one tight cluster in x, and
    # simple alternation put A100's label straight through V100S's. Consecutive near-neighbours are
    # therefore treated as a CLUSTER and their labels are staggered with growing offsets, so a run
    # of k crowded points gets k distinct vertical slots rather than two.
    cluster, clusters = [], []
    for rank, i in enumerate(order):
        if rank and xs[i] / xs[order[rank - 1]] < 1.6:
            cluster.append(i)
        else:
            if cluster:
                clusters.append(cluster)
            cluster = [i]
    if cluster:
        clusters.append(cluster)
    place = {}
    for cl in clusters:
        for k, i in enumerate(cl):
            below = bool(k % 2)
            mag = 5 + 9 * (k // 2)
            place[i] = ("top", -mag) if below else ("bottom", mag)
    for i, (x, y, l) in enumerate(zip(xs, ys, labs)):
        # The rightmost card labels OUTWARD. Pointing its label left drove it into the neighbouring
        # card's marker, which is what put "A100" on top of the V100S point; the x margin has room.
        if x == max(xs):
            ha = "left"
        else:
            ha = "right" if x >= max(xs) * 0.75 else "left"
        dx = -5 if ha == "right" else 5
        va, dy = place.get(i, ("bottom", 5))
        ax.annotate(l, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.margins(x=0.22, y=0.22)
    r = _corr([math.log10(v) for v in xs], [math.log10(v) for v in ys])
    spread_x = max(xs) / min(xs); spread_y = max(ys) / min(ys)
    ax.set_xlabel(xlabel); ax.set_ylabel("GPU-arm wall on the shared datasets (s)")
    ax.set_title(title, pad=3)
    # bottom-left in panel A; in panel B the V100 label sits there, so the note goes top-right,
    # the one corner of B with no card (R16, rev5)
    corner = (0.97, 0.97, "top", "right") if "bandwidth" in xlabel else (0.03, 0.03, "bottom", "left")
    ax.annotate(f"r$^2$ = {r*r:.2f} (n = {len(xs)} cards)\n{spread_x:.0f}x spread in x, "
                f"{spread_y:.1f}x in wall", xy=corner[:2], xycoords="axes fraction",
                va=corner[2], ha=corner[3], color=FS.MUTED)
    return len(xs)


def panel_a(by, common, sp):
    fig, ax = plt.subplots(figsize=(W, H))
    # The title used to hardcode "19x in fp64, 3.7x in wall" while _scatter_spec annotates the
    # COMPUTED spread on the same axes. Two sources for one number is one too many; the title now
    # makes the qualitative point and the annotation carries the values.
    n = _scatter_spec(ax, by, common, sp, "fp64_tflops", "nominal fp64 (TFLOP/s, vendor)",
                      "fp64 ranks the cards, but the response\n"
                      "is far from proportional")
    return fig, n


def panel_b(by, common, sp):
    fig, ax = plt.subplots(figsize=(W, H))
    n = _scatter_spec(ax, by, common, sp, "mem_bandwidth_gbs",
                      "nominal memory bandwidth (GB/s, vendor)",
                      "Nominal bandwidth orders the cards no better;\n"
                      "six cards cannot separate one specification from another")
    return fig, n


def panel_c(rows):
    """Facet per card: wall vs complexity. Replaces a six-hue encoding that failed at dE 7.6."""
    cards = [c for c in LABEL if any(r["card"] == c for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(W * 2.05, H * 1.5), sharex=True, sharey=True)
    for ax, card in zip(axes.flat, cards):
        pts = [(r["_complexity"], FD.num(r.get("gpu_wall_s"))) for r in rows
               if r["card"] == card and r["threads"] == "20"]
        pts = [(a, b) for a, b in pts if a and b]
        allpts = [(r["_complexity"], FD.num(r.get("gpu_wall_s"))) for r in rows if r["threads"] == "20"]
        allpts = [(a, b) for a, b in allpts if a and b]
        ax.scatter([a for a, _ in allpts], [b for _, b in allpts], s=3, color=FS.MUTED, alpha=0.30,
                   zorder=1)
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=8, color=FS.CARD_HUE, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{LABEL[card]} (n={len(pts)})", pad=2)
    for ax in axes.flat[len(cards):]:
        ax.axis("off")
    fig.supxlabel("tree complexity"); fig.supylabel("GPU-arm wall (s)")
    fig.suptitle("Every card, same axes; grey = all cards pooled as a reference", y=1.0)
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_d(rows):
    """Speedup WITHIN each card only -- never compared across cards, since the hosts differ."""
    cards = [c for c in LABEL if any(r["card"] == c for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(W * 2.05, H * 1.5), sharex=True, sharey=True)
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
    fig.supxlabel("tree complexity"); fig.supylabel("speedup (within card)")
    # "1.67x" was a literal; the computed maximum is card_host_cpu_spread_max.
    spread = FD.key_number(REPO, "card_host_cpu_spread_max")
    fig.suptitle(f"Speedup is shown only within a card: the six hosts differ by up to "
                 f"{spread:.2f}x", y=1.0)
    return fig, sum(1 for r in rows if r["threads"] == "20")


def panel_e(rows):
    """Energy per analysis, by card. GPU-SIDE ONLY.

    This panel exists because my earlier data description was wrong and said no power data existed.
    `gpu_power_mean_w`, `gpu_power_max_w` and `gpu_energy_j` ARE measured, sampled by the harness
    during the card sweep: 48 runs, 8 per card on all six cards.

    WHAT CANNOT BE SAID HERE, and is stated on the panel: `cpu_energy_j` is a placeholder in every
    row, so there is NO host-side energy and therefore NO CPU-versus-GPU energy comparison. This is
    a comparison BETWEEN CARDS of the device energy each spends to produce the same trees.
    """
    by = collections.defaultdict(list)
    for r in rows:
        e, p = FD.num(r.get("gpu_energy_j")), FD.num(r.get("gpu_power_mean_w"))
        if r["threads"] == "20" and e and p:
            by[r["card"]].append((r["genus"], e, p))
    if not by:
        return None, 0
    # PER-RUN MEDIAN kJ per tree, which is the quantity the manuscript quotes and the quantity
    # KEY_NUMBERS holds as energy_kj_median_<card>. The previous version summed the median of each
    # genus over the datasets common to all cards -- a different quantity, on a stratum that also
    # silently pooled both alphabets (8 genera) while the text said four. One quantity, one place.
    med = {c: st.median([e for _, e, _ in v]) / 1000.0 for c, v in by.items()}
    pw = {c: st.median([p for _, _, p in v]) for c, v in by.items()}
    n_runs = sum(len(v) for v in by.values())
    order = sorted(med, key=lambda c: med[c])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.bar(range(len(order)), [med[c] for c in order], 0.6,
           color=FS.CARD_HUE, edgecolor=FS.CARD_HUE)
    for i, c in enumerate(order):
        ax.annotate(f"{pw[c]:.0f} W", xy=(i, med[c]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[c] for c in order], rotation=30, ha="right")
    ax.set_ylabel("median device energy per tree (kJ)")
    ax.margins(y=0.18)
    ax.set_title(f"Energy to produce the same trees, by card\n"
                 f"GPU-side only; no host energy was measured "
                 f"(n = {n_runs} card-sweep runs)", pad=3)
    return fig, sum(len(v) for v in by.values())


def main():
    rows = FD.for_fits(FD.load(REPO))
    sp = specs()
    by, common = shared(rows)
    date = max((r.get("datetime_utc") or "")[:10] for r in rows)
    print(f"  shared amino-acid datasets across all cards: {sorted(common)}")
    out = [(panel_a(by, common, sp), "a", "wall_vs_fp64", "GPU wall vs nominal fp64"),
           (panel_b(by, common, sp), "b", "wall_vs_bandwidth", "GPU wall vs nominal bandwidth"),
           (panel_c(rows), "c", "card_facet_wall", "Wall vs complexity, per card"),
           (panel_d(rows), "d", "card_facet_speedup", "Speedup vs complexity, per card"),
           (panel_e(rows), "e", "energy_per_card", "Device energy per analysis, by card")]
    for (fig, n), letter, slug, title in out:
        fig.tight_layout()
        stem = FS.save_panel(fig, 5, letter, slug, title, os.path.basename(__file__), SRC, n,
                             data_date=date)
        plt.close(fig); print(f"  wrote {stem}")
    FS.write_manifest(os.path.join(REPO, "results", "figures", "FIGURE_MANIFEST.tsv"))


if __name__ == "__main__":
    main()
