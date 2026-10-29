# Reproducing the benchmarks

Everything in `benchmark/figures/` and `benchmark/runs/` can be reproduced. There are three tiers;
tier 2 needs nothing beyond a `git clone` and Python. All commands are run from the repository
root; the map of which section produces which figure and table is in [`README.md`](README.md), and
the function behind every panel in [`FIGURE_MAP.md`](FIGURE_MAP.md).

| Tier | What it reproduces | Needs | Time |
|---|---|---|---|
| **1. Quick test** | that your build is exact | clone + a GPU | ~2 min |
| **2. All figures and tables** | every figure in the paper | clone only, **no GPU** | ~4 min |
| **3. The measurements themselves** | every row of `RUNS.tsv` | clone + the Zenodo inputs + a GPU | days |

---

## Tier 1 — Quick test (ships in the repository)

Confirms your build produces the same tree as the CPU path.

All commands below are run **from the repository root**.

```bash
bash scripts/iqtree3-gpu-verify.sh ./iqtree3_gpu_build/iqtree3   # ~30 s: engagement + RF=0 + logL
( cd examples && ./run_example.sh )                              # ~2 min: worked example, 8 taxa
```

The parentheses matter: `run_example.sh` must be started from `examples/`, and the subshell puts you
back at the repository root afterwards so the commands that follow work as written.

For a slightly larger but still fast run, fetch the alignments (Tier 3 below) and use one of the small
nucleotide sets; `dna_n04` through `dna_n12` each complete in under 20 seconds per arm:

```bash
bash benchmark/scripts/fetch_benchmark_data.sh alignments
GENUS=dna_n06 THREADS=20 CARD=mytest MODEL=GTR+F+G4 \
  ALN_OVERRIDE="$PWD/benchmark/data/dna/dna_n06/aln.phy" \
  sbatch benchmark/scripts/harness/paired_benchmark.sh
```

This one needs **SLURM and a module system**, not just a GPU: it is the real benchmark harness. On a
workstation set `NO_LMOD=1` in `config/site.env` and run the underlying commands directly, or stay
with the two checks above.

## Tier 2 — Regenerate every figure and table (no GPU required)

The complete measured record ships in the repository: `benchmark/runs/RUNS.tsv` (226 runs x 122
columns, of which the 182 measured with the released binary are the ones analysed -- see
`binary_sha256`) plus, for all 226 cells, the per-cell `RESULT.tsv`, `PHASES.tsv`, `equivalence.kv` and both
arms' inferred trees. Every figure derives from those files alone.

```bash
bash benchmark/figures_tables/refresh_all.sh
```

This rewrites all of `benchmark/figures/` (the 25 panels as PDF+PNG, the 7 composites, the Source
Data and the manifest), regenerates `runs/KEY_NUMBERS.md` (every number quoted in the text), then
the Supplementary Data workbook and the map in `benchmark/README.md`. It needs Python with numpy,
matplotlib, pypdf and Biopython; the two workbooks additionally need pandas and openpyxl and are
skipped with a notice when those are absent. From a clone, the composite figures, the run tables and
the manifest regenerate byte-identically; individual panel PNGs and Source Data CSVs can differ in the
last floating-point digit or with another font environment, and the PDFs differ only in their creation
date. One exception: the device-arena report (Supplementary Data 6 and the `arena_*` key numbers) is
parsed from each run's `gpu.stdout`, which is in neither git nor the Zenodo record, so those regenerate
only on the authors' copy.

## Tier 3 — Re-run the benchmarks themselves

Requires a CUDA GPU and the input alignments, which are not in git.

**The inputs.** The 35 amino-acid and 26 nucleotide alignments (and the 435 source genomes, not needed
to run anything) are on Zenodo, doi:10.5281/zenodo.22942159, one archive per `benchmark/data/` folder. From the
repository root:

```bash
bash benchmark/scripts/fetch_benchmark_data.sh alignments   # downloads, verifies and unpacks into benchmark/data/
```

The DOI resolves once the record is published with the paper. The 14 large nucleotide alignments can
also be regenerated: each `benchmark/data/dna/<set>/PROVENANCE.md` records the taxa, sites, guide tree,
exact model string, seed and the literal AliSim command, and the guide trees are in git. To rebuild them all:

```bash
bash benchmark/scripts/dev/build_complexity_datasets.sh
sha256sum -c benchmark/data/dna/CHECKSUMS.sha256
```

Simulated with **IQ-TREE 3.1.3** — AliSim is deterministic given a seed, but only within a version, so
check yours first if a checksum does not match.

**Then submit the benchmark cells:**

```bash
bash benchmark/scripts/harness/submit.sh                  # the headline size sweep
bash benchmark/scripts/harness/submit_complexity.sh go   # the complexity extension
```

Each job runs both arms in one job, on one node, from one binary, differing only by `-gpu`, and
appends one row to `RUNS.tsv`. The harness refuses to overwrite a cell that already holds a successful
measurement unless `FORCE=1`.

---

## What is not in the repository

The manuscript text is not distributed here. **Nothing in
`benchmark/figures/` or `benchmark/runs/` depends on them**: every figure, table and number
regenerates from `benchmark/runs/RUNS.tsv` and the per-cell record with
`benchmark/figures_tables/`.
