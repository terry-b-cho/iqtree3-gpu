# Benchmark datasets

Inputs for the IQ-TREE3-GPU benchmark and validation suites. Three sets, each with a distinct job.
**These constrain the benchmark only — the tool itself has no size limit** (the device arena is sized
from IQ-TREE's own `max_lh_slots`, and falls back to the CPU kernel if it will not fit in VRAM).

All amino-acid data are concatenated marker-gene alignments of metagenome-assembled genomes,
classified with GTDB-Tk against GTDB and quality-controlled with CheckM2. Tip labels record host
species and sample. The `concat_alignment.faa` files are **ordinary git files, present in every
clone** (~126 MB total). The raw source genomes (`*.fna`, ~830 MB) are in `../genomes/`, stored
through Git LFS, with their study, host and accession metadata in `../genomes/genome_metadata.tsv`
(see `../genomes/PROVENANCE.md` and `../README.md`).

---

## 1. `n<NN>_g__<genus>/` — the size sweep (12 datasets) — **the benchmark set**

Twelve genera subsampled to **twelve distinct tree sizes**, so speedup versus taxon count can be
*fitted* rather than inferred. An earlier version capped everything at n=30 and produced six datasets
at the same size, from which no scaling behaviour can be extrapolated. Alignment length varies
independently of n, so the taxa and sites axes are not confounded.

| dataset | taxa | columns |
|---|---|---|
| `n04_g__Brotaphodocola` | 4 | 195,456 |
| `n06_g__CAG-269` | 6 | 143,055 |
| `n08_g__Hominicoprocola` | 8 | 169,245 |
| `n10_g__Coprovivens` | 10 | 122,439 |
| `n12_g__Caccovivens` | 12 | 114,107 |
| `n14_g__Nanosyncoccus` | 14 | 99,220 |
| `n17_g__Lachnospira` | 17 | 205,341 |
| `n20_g__Coprococcus_C` | 20 | 206,383 |
| `n23_g__Onthovivens` | 23 | 134,971 |
| `n26_g__Ruminococcus` | 26 | 212,930 |
| `n28_g__Roseburia` | 28 | 233,159 |
| `n30_g__Enterousia` | 30 | 109,330 |

Built by [`../../scripts/dev/build_capped_genomes.py`](../../scripts/dev/build_capped_genomes.py)
(seeded, deterministic; all-gap columns dropped after subsampling).

## 2. `1Thread_g__<genus>_n6/` — the single-thread set (14 datasets)

n=6 subsets for the CPU-1T vs GPU-1T comparison. One thread is roughly 19x slower in wall time, so
full-size genera would take days per cell and time out; these keep that comparison obtainable.
14 subsets were built; the reported 1-thread distribution is the 12 that were run, matching the
12 sizes of the 20-thread sweep. Built by
[`../../scripts/dev/build_1thread_genomes.py`](../../scripts/dev/build_1thread_genomes.py).

## 3. `g__<genus>/` — the original full-size genera (14 datasets)

Kept unmodified as the source for the subsets above, and runnable individually for anyone who wants
a larger tree than the sweep covers:

Run this **from the repository root**, not from this directory: the harness writes its SLURM log into
`logs/`, which exists only there.

```bash
GENUS=g__Colicola THREADS=20 sbatch benchmark/scripts/harness/paired_benchmark.sh
```

| genus | taxa | columns |
|---|---|---|
| `g__Brotaphodocola` | 10 | 206,171 |
| `g__CAG-269` | 10 | 150,679 |
| `g__Caccovivens` | 17 | 117,218 |
| `g__Colicola` | 65 | 236,088 |
| `g__Coprococcus_C` | 30 | 214,187 |
| `g__Coprovivens` | 13 | 127,286 |
| `g__Enterousia` | 54 | 114,315 |
| `g__Hominicoprocola` | 12 | 173,758 |
| `g__Lachnospira` | 25 | 213,975 |
| `g__Nanosyncoccus` | 20 | 104,808 |
| `g__Onthovivens` | 34 | 150,333 |
| `g__Roseburia` | 41 | 245,386 |
| `g__Ruminococcoides` | 66 | 228,101 |
| `g__Ruminococcus` | 38 | 217,518 |

---

## Nucleotide data — `../dna/` (12 datasets)

Simulated with IQ-TREE's AliSim at the same twelve tree sizes, so protein and DNA are compared at
matched shapes. Simulated rather than real because this repository has no nucleotide alignments and
back-translating the amino-acid ones would need CDS we do not have. For a likelihood-engine benchmark
that is arguably better: byte-reproducible from a seed, and the **true generating model is known**
(`../dna/TRUE_MODEL.txt`), so model recovery is checkable.

| dataset | taxa | sites |
|---|---|---|
| `dna_n04` | 4 | 195456 |
| `dna_n06` | 6 | 143055 |
| `dna_n08` | 8 | 169245 |
| `dna_n10` | 10 | 122439 |
| `dna_n12` | 12 | 114107 |
| `dna_n14` | 14 | 99220 |
| `dna_n17` | 17 | 205341 |
| `dna_n20` | 20 | 206383 |
| `dna_n23` | 23 | 134971 |
| `dna_n26` | 26 | 212930 |
| `dna_n28` | 28 | 233159 |
| `dna_n30` | 30 | 109330 |

**Sites are not the unit of work for DNA.** IQ-TREE evaluates the likelihood per distinct site
*pattern*, and DNA patterns are bounded by 4^taxa — `dna_n04` has 195,456 sites but only **183
distinct patterns**. That is a real property of nucleotide phylogenetics, not a defect in the data,
and it means the small-n DNA cells measure launch overhead rather than throughput.

Built by [`../../scripts/dev/build_dna_datasets.sh`](../../scripts/dev/build_dna_datasets.sh).

---

## Citation

GTDB (Parks et al.), CheckM2 (Chklovski et al.), IQ-TREE 3 (Wong et al. 2026,
https://doi.org/10.1093/molbev/msag117) and AliSim (Ly-Trong
et al.) for the simulated nucleotide data.

## Complexity-extension datasets (added 2026-08-31)

The original ladders varied only TAXA, which left DNA and protein on almost disjoint complexity
ranges (complexity = branches x patterns x rate categories x states^2, and states^2 is 16 for DNA vs
400 for protein). These sets close that gap. All are built by
[`../../scripts/dev/build_complexity_datasets.sh`](../../scripts/dev/build_complexity_datasets.sh)
and every benchmark row they produce is namespaced `__l40s_cx`, so no earlier measurement is touched.

| family | count | what it is |
|---|---|---|
| `p_lc_*` (in `benchmark/data/protein/`) | 5 | LOW-complexity protein: real staged alignments with alignment COLUMNS seeded-subsampled to 200-50,000 sites, reaching complexity 1.13e6 - a regime the full-length ladder could never enter |
| `dna_pt_n<NN>` | 7 | DNA simulated with AliSim along that protein cell's OWN inferred topology (same taxa, same tree, same branch lengths), n=14..30 - the paired test in which the alphabet is the only variable |
| `dna_hc_*` | 5 | HIGH-complexity DNA on random Yule-Harding trees, extending the two axes never explored: taxa > 30 (to n=100) and sites > 233k (to 4M) |
| `dna_ux_*` | 2 | ULTRA-high-complexity DNA (n=60 at 6M sites, n=100 at 4M sites) reaching past the protein maximum |

**Why the paired sets start at n=14.** DNA site patterns are bounded by 4^n, so the maximum DNA
complexity is 8.2e4 at n=4, 2.4e6 at n=6 and 5.5e7 at n=8. **DNA cannot reach protein complexity below
n ~= 14 at any alignment length** - that is a property of a 4-letter alphabet, not a gap in the design.

The large simulated alignments (`dna_pt_*`, `dna_hc_*`, `dna_ux_*`; ~2.4 GB) are gitignored and
regenerate byte-identically from the seeded builder; each carries its own `PROVENANCE.md`.
