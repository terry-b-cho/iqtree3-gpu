#!/usr/bin/env python3
"""build_capped_genomes.py - build a benchmark set SPREAD ACROSS tree sizes up to N_MAX.

WHY A SPREAD, NOT A CAP
-----------------------
This is a TOOL-DEVELOPMENT benchmark: it must show the GPU returns the same tree as CPU IQ-TREE and
is faster, ACROSS PROBLEM SIZES, cheaply enough to re-run whenever the engine changes.

Simply capping every genus at N_MAX is wrong for that purpose: the large genera all collapse onto
n = N_MAX and the set ends up clustered at one size (an earlier version produced six datasets at
n=30 and nothing below n=10). Scaling behaviour cannot be extrapolated from a cluster.

So instead each genus is subsampled to a DIFFERENT target, giving evenly spaced datapoints across
n = 4..N_MAX. Twelve distinct sizes support a real fit of runtime and speedup versus taxon count,
while every cell stays small enough to run in hours.

Note this caps only the BENCHMARK. The tool itself has no size limit: the device arena is sized from
IQ-TREE's own max_lh_slots, and if it will not fit in VRAM the run degrades to the CPU kernel rather
than failing. Users can analyse trees of any size.

Subsampling is seeded and deterministic, sources from this repository's own staged data/protein/, and
drops columns that become all-gap once taxa are removed (they would otherwise inflate the pattern
count with sites that are not really constant).

Usage: build_capped_genomes.py [n_max]

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, sys, random, glob

# THREE dirname levels: this file lives in scripts/dev/, so the repository root is three
# levels up. Two levels resolved to scripts/ and silently wrote into scripts/examples/.
REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = 1
GEN = f"{REPO}/data/protein"


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


def main():
    # Evenly spaced targets across the range. Twelve distinct sizes, so speedup-vs-taxa can be fitted
    # rather than inferred from a cluster of identical n.
    targets = [4, 6, 8, 10, 12, 14, 17, 20, 23, 26, 28, N_MAX]

    # Pair the smallest genus with the smallest target, so every assignment is feasible
    # (a genus can only be subsampled DOWN) and different genera land at different sizes.
    src = []
    for d in sorted(glob.glob(f"{GEN}/g__*/")):
        a = os.path.join(d, "concat_alignment.faa")
        if os.path.exists(a):
            src.append((os.path.basename(d.rstrip("/")), a, sum(1 for l in open(a) if l.startswith(">"))))
    src.sort(key=lambda t: t[2])

    built = []
    for (genus, aln, n), k in zip(src, targets):
        if n < k:
            print(f"  SKIP {genus}: n={n} < target {k}"); continue
        out_dir = f"{GEN}/n{k:02d}_{genus}"
        os.makedirs(out_dir, exist_ok=True)
        out_aln = f"{out_dir}/concat_alignment.faa"
        recs = read_fasta(aln)
        rnd = random.Random(SEED)
        keep = sorted(rnd.sample(range(n), k))
        sub = [recs[i] for i in keep]
        L = len(sub[0][1])
        cols = [c for c in range(L) if any(s[c] not in "-?xX" for _, s in sub)]
        with open(out_aln, "w") as f:
            for nm, sq in sub:
                f.write(f">{nm}\n")
                t = "".join(sq[c] for c in cols)
                for i in range(0, len(t), 60):
                    f.write(t[i:i + 60] + "\n")
        built.append((genus, n, k, L, len(cols)))

    print(f"  target spread: {targets}\n")
    for g, n, k, L, lc in built:
        print(f"  n{k:02d}_{g:24s} {n:3d} -> {k:2d} taxa   columns {L} -> {lc}")
    print(f"\n  {len(built)} datasets at {len(set(b[2] for b in built))} distinct sizes "
          f"(n = {min(b[2] for b in built)}..{max(b[2] for b in built)})")


if __name__ == "__main__":
    main()
