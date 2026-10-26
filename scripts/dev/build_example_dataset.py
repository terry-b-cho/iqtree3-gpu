#!/usr/bin/env python3
"""build_example_dataset.py - build the tiny bundled dataset for examples/.

The example dataset exists so a NEW USER can run the tool end-to-end in under ~2 minutes and watch it
work, before touching their own data. It must therefore be REAL (so the run is meaningful and the tree
is interpretable) but SMALL (so it is fast).

Construction: take a real the upstream pipeline genus alignment and cut it down on BOTH axes -
  * taxa   : a seeded, deterministic subset (default 8) -> keeps informative host-species tip labels
  * columns: a contiguous slice (default 4000) taken from the middle of the alignment, where marker
             genes are well covered (a leading slice can land in a gap-heavy region)
Columns that are all-gap across the retained taxa are dropped, and any taxon left effectively empty is
reported, so the demo alignment always carries real phylogenetic signal.

The sequence data is real, public (GTDB-classified MAGs; see examples/data/PROVENANCE.md) - only the
size is reduced. System python3, stdlib only.

Usage: build_example_dataset.py [source_genus] [n_taxa] [n_sites]

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, sys, random, textwrap

# THREE dirname levels: this file lives in scripts/dev/, so the repository root is three
# levels up. Two levels resolved to scripts/ and silently wrote into scripts/examples/.
REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Several genera are provided so users can try more than one tree, and so no single topology is
# presented as the "expected" one - real microbial clades may or may not track host relatedness.
DEFAULT_GENERA = ["g__CAG-269", "g__Nanosyncoccus", "g__Lachnospira"]
N_TAXA = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_SITES = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
SEED = 1
OUTDIR = f"{REPO}/examples/data"


def read_fasta(p):
    recs, name, seq = [], None, []
    for ln in open(p):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = ln[1:], []
        else:
            seq.append(ln)
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def build_one(src_genus, is_default):
    SRC = f"{REPO}/benchmark/data/protein/{src_genus}/concat_alignment.faa"
    short = src_genus.replace("g__", "")
    OUT = f"{OUTDIR}/example_{short}.faa"
    if not os.path.exists(SRC):
        print(f"  SKIP {src_genus}: no source alignment"); return None
    recs = read_fasta(SRC)
    n_src, len_src = len(recs), len(recs[0][1])

    # 1) seeded taxon subset (deterministic)
    rnd = random.Random(SEED)
    keep_i = sorted(rnd.sample(range(n_src), min(N_TAXA, n_src)))
    sub = [recs[i] for i in keep_i]

    # 2) contiguous column slice from the middle (better marker coverage than the alignment edges)
    start = max(0, (len_src - N_SITES) // 2)
    end = min(len_src, start + N_SITES)
    sliced = [(n, s[start:end]) for n, s in sub]

    # 3) drop columns that are all-gap across the retained taxa (they carry no signal)
    ncol = end - start
    keep_cols = [c for c in range(ncol) if any(s[c] not in "-?xX" for _, s in sliced)]
    final = [(n, "".join(s[c] for c in keep_cols)) for n, s in sliced]

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w") as f:
        for n, s in final:
            f.write(f">{n}\n")
            for i in range(0, len(s), 60):
                f.write(s[i:i+60] + "\n")

    n_out, sites_out = len(final), len(final[0][1])
    # sanity: report per-taxon non-gap fraction so we never ship an empty/degenerate demo
    occ = [(n, sum(1 for ch in s if ch not in "-?xX") / max(1, len(s))) for n, s in final]
    worst = min(occ, key=lambda t: t[1])

    print(f"  wrote {os.path.basename(OUT)}: taxa {n_src}->{n_out}, columns {len_src}->{sites_out}, "
          f"{os.path.getsize(OUT)/1024:.0f} KB, min occupancy {worst[1]*100:.1f}%")
    return dict(genus=src_genus, short=short, file=os.path.basename(OUT), n_src=n_src, n=n_out,
                len_src=len_src, sites=sites_out, occ=worst[1] * 100, worst=worst[0],
                default=is_default)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    genera = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_GENERA
    built = [b for b in (build_one(g, i == 0) for i, g in enumerate(genera)) if b]
    if not built:
        sys.exit("ERROR: no example datasets built")
    rows = "\n".join(
        f"| `{b['file']}` | {b['genus'].replace('g__','')} | {b['n']} | {b['sites']:,} | "
        f"{b['n_src']} taxa x {b['len_src']:,} cols | {b['occ']:.0f}% |" for b in built)
    with open(f"{OUTDIR}/PROVENANCE.md", "w") as f:
        f.write(textwrap.dedent(f"""\
            # examples/data — provenance and citation

            **What these are:** deliberately tiny amino-acid alignments used by
            `examples/run_example.sh` so a new user can run IQ-TREE3-GPU end to end in ~1-2 minutes,
            watch real output, and confirm the installation works. **Several genera are provided** so
            you can try more than one tree.

            | file | source genus | taxa | columns | reduced from | min occupancy |
            |---|---|---|---|---|---|
            {rows}

            **Source (real, public data):** concatenated marker-gene alignments from this repository's
            benchmark set (`benchmark/data/protein/<genus>/concat_alignment.faa`), produced by the upstream marker-gene pipeline
            (Prodigal gene calling -> DIAMOND marker search -> MAFFT alignment -> concatenation) on
            metagenome-assembled genomes (MAGs) classified with **GTDB-Tk** against the **GTDB**
            taxonomy and quality-controlled with **CheckM2**. Tip labels record the host species and
            sample the genome came from, e.g. `Alces_alces_W_Sv17_PPQH`.

            **How they were reduced** (`scripts/build_example_dataset.py`, fully deterministic):
            seeded taxon subset (`random.Random({SEED}).sample`) plus a contiguous column slice taken
            from the alignment midpoint, after which all-gap columns are dropped.

            **Validity:** the residues are unmodified real sequence data; only the number of taxa and
            columns is reduced. These are *demonstration* datasets for learning the interface and
            verifying an installation. The paper's benchmarks use the full-size alignments in
            `benchmark/data/protein/`.

            **A note on the trees:** these are **bacterial** phylogenies whose tips happen to be
            labelled by host. Do not expect tips to group by host species: gut symbiont clades may or
            may not track host relatedness, and recent host switching or genuinely non-codiversifying
            lineages are biologically real. Whether codiversification occurs is a *research question*,
            not an assumption - and 8 genomes could not settle it either way. Use these runs to confirm
            the software works, not to draw biological conclusions.

            **Citation:** please cite GTDB (Parks et al.), CheckM2 (Chklovski et al.), IQ-TREE 3
            (Wong et al. 2026, https://doi.org/10.1093/molbev/msag117), and this work (Cho and Kostic 2026,
            IQ-TREE3-GPU) - see the repository `README.md` Citation section and `assets/CITATION.cff`.
            """))
    print(f"  built {len(built)} example dataset(s); default = {built[0]['file']}")


if __name__ == "__main__":
    main()
