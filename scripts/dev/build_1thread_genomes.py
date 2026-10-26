#!/usr/bin/env python3
"""build_1thread_genomes.py - build the SMALL, MATCHED input set for the 1-thread benchmark.

WHY THIS EXISTS
---------------
The benchmark makes only like-for-like comparisons: CPU-20T vs GPU-20T, and CPU-1T vs GPU-1T. A
cross-thread comparison is never made. The two comparisons have very different compute budgets:

  * 20-THREAD comparison -> the FULL genus set (n = 10-66 taxa, ~10^5 sites). Cheap enough (~1-11 h per
    run) to span small -> large real trees, so it probes production scale.
  * 1-THREAD comparison  -> a run on one core does all the site-parallel work itself and is ~19x slower
    in wall time. On the full set that makes these cells IMPOSSIBLE, not merely slow: measured from the
    CPU-20T runs' own CPU-time, CPU-1T projects to 0.8-9+ DAYS per genus (CAG-269 18.5 h,
    Brotaphodocola 108 h, Caccovivens 141 h, Coprococcus_C 217 h), and GPU-1T on the larger genera would
    exceed the 5-day wall cap of EVERY available GPU partition - guaranteed timeouts that can never
    yield a number. Submitting jobs certain to fail is waste.

MATCHED DESIGN (the important property)
---------------------------------------
This set has the SAME NUMBER OF DATAPOINTS as the 20-thread set - one dataset per genus, all 14 genera -
so both comparisons are built from the same genera and the same number of trees. The ONLY difference is
input size: every genus is subsampled to n = 6 taxa. That isolates the variable of interest (threads,
CPU vs GPU) and keeps the two halves of the study directly parallel.

Subsampling to n = 6 (not 8-10) is deliberate. 1-thread cost is NOT predictable from size: CAG-269 and
Brotaphodocola both have n = 10 and differ only 1.37x in sites, yet Brotaphodocola is 5.9x slower
because ModelFinder selected a far costlier model (+R9 vs +I+R5). Model complexity and search path
dominate, so projections carry ~5x uncertainty. n = 6 projects to ~6-14 h and retains ample headroom
under the wall limit even if a genus behaves like the worst observed outlier.

HOW IT IS BUILT
---------------
Source = this repository's OWN staged alignments (`data/protein/g__<genus>/concat_alignment.faa`) - already
copied into the repo, self-contained, and intact (the upstream D_C_I arm has since been cleared, so
these copies are the only remaining record). Copies only; nothing upstream is modified or moved.
Subsampling is a seeded, deterministic taxon subset (`random.Random(1).sample`). The sequence data is
real the upstream pipeline output in every case; only the number of tips is reduced. This is valid for a
performance comparison because CPU and GPU receive byte-identical input - the only requirement.

Outputs data/protein/1Thread_g__<genus>_n6/{concat_alignment.faa, PROVENANCE.md}. System python3, stdlib only.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, random, textwrap, glob

# THREE dirname levels: this file lives in scripts/dev/, so the repository root is three
# levels up. Two levels resolved to scripts/ and silently wrote into scripts/examples/.
REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GEN = f"{REPO}/data/protein"
SEED = 1
TARGET_N = int(os.environ.get("TARGET_N", "6"))

# The SAME 14 genera as the 20-thread (full) set -> matched number of datapoints.
GENERA = ["g__Brotaphodocola", "g__CAG-269", "g__Caccovivens", "g__Colicola", "g__Coprococcus_C",
          "g__Coprovivens", "g__Enterousia", "g__Hominicoprocola", "g__Lachnospira",
          "g__Nanosyncoccus", "g__Onthovivens", "g__Roseburia", "g__Ruminococcoides",
          "g__Ruminococcus"]


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


def write_fasta(p, recs):
    with open(p, "w") as f:
        for n, s in recs:
            f.write(f">{n}\n")
            for i in range(0, len(s), 60):
                f.write(s[i:i+60] + "\n")


def main():
    # clear any earlier 1Thread_ build so the set is exactly the matched 14
    for d in glob.glob(f"{GEN}/1Thread_*"):
        for f in glob.glob(f"{d}/*"):
            os.remove(f)
        os.rmdir(d)
    made = []
    for genus in GENERA:
        src = f"{GEN}/{genus}/concat_alignment.faa"
        if not os.path.exists(src):
            print(f"  SKIP {genus}: no staged alignment"); continue
        recs = read_fasta(src)
        n_src = len(recs); sites = len(recs[0][1]) if recs else 0
        if n_src <= TARGET_N:
            sel, mode = recs, f"full staged alignment, unmodified (n={n_src} <= target {TARGET_N})"
        else:
            rnd = random.Random(SEED)
            keep = sorted(rnd.sample(range(n_src), TARGET_N))
            sel = [recs[i] for i in keep]
            mode = f"seeded taxon subsample {n_src} -> {TARGET_N} (random.Random({SEED}).sample)"
        name = f"1Thread_{genus}_n{len(sel)}"
        d = f"{GEN}/{name}"
        os.makedirs(d, exist_ok=True)
        write_fasta(f"{d}/concat_alignment.faa", sel)
        with open(f"{d}/PROVENANCE.md", "w") as f:
            f.write(textwrap.dedent(f"""\
                # {name}

                Member of the **small, matched input set for the 1-thread comparison** (CPU-1T vs GPU-1T).
                The 20-thread comparison (CPU-20T vs GPU-20T) uses the full-size counterpart
                `data/protein/{genus}/` (n = {n_src}). Both sets contain **one dataset per genus for the same
                14 genera**, so the two comparisons use the same genera and the same number of trees;
                the only difference is input size.

                **Why smaller for 1 thread:** a 1-thread run is ~19x slower in wall time (one core does
                all the site-parallel work). On the full-size genera CPU-1T projects to 0.8-9+ days each,
                and GPU-1T would exceed the 5-day wall cap of every available GPU partition - those cells
                could never complete. Sizing this set to n = {TARGET_N} keeps even CPU-1T to ~6-14 h, which
                makes the 1T-vs-1T comparison obtainable while leaving the 20-thread comparison free to
                explore the large genomes.

                **Source (copied, never moved):** `data/protein/{genus}/concat_alignment.faa`
                (this repository's own staged copy of the upstream alignment:
                Prodigal -> DIAMOND -> MAFFT -> concatenate).
                **Construction:** {mode}
                **Result:** n = {len(sel)} taxa, {sites} alignment columns.
                **Validity:** real sequence data; only the number of tips is reduced. CPU and GPU receive
                byte-identical input, which is what the performance comparison requires.
                """))
        made.append((name, len(sel), sites))
        print(f"  built {name:42s} n={len(sel):<3d} sites={sites:<7d} (from n={n_src})")
    print(f"\n  {len(made)} datasets (matched to the {len(GENERA)}-genus 20-thread set)")


if __name__ == "__main__":
    main()
