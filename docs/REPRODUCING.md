# Reproducing the benchmarks

Everything in `results/figures/` and `results/benchmarks/` can be reproduced. There are three tiers,
and the first two need **nothing beyond a `git clone`**.

| Tier | What it reproduces | Needs | Time |
|---|---|---|---|
| **1. Quick test** | that your build is exact | clone + a GPU | ~2 min |
| **2. All figures and tables** | every figure in the paper | clone only, **no GPU** | ~4 min |
| **3. The measurements themselves** | every row of `RUNS.tsv` | clone + Zenodo inputs + a GPU | days |

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

For a slightly larger but still fast run, the repository ships **12 complete DNA alignments**
(`data/dna/dna_n04` … `dna_n30`, 0.8–6.3 MB each) and **all 35 real amino-acid alignments** used by
the benchmark. `dna_n04` through `dna_n12` each complete in under 20 seconds per arm:

```bash
GENUS=dna_n06 THREADS=20 CARD=mytest MODEL=GTR+F+G4 \
  ALN_OVERRIDE="$PWD/data/dna/dna_n06/aln.phy" \
  sbatch scripts/benchmark/paired_benchmark.sh
```

This one needs **SLURM and a module system**, not just a GPU: it is the real benchmark harness. On a
workstation set `NO_LMOD=1` in `config/site.env` and run the underlying commands directly, or stay
with the two checks above.

## Tier 2 — Regenerate every figure and table (no GPU required)

The complete measured record ships in the repository: `results/benchmarks/RUNS.tsv` (226 runs x 123
columns, of which the 182 measured with the released binary are the ones analysed -- see
`binary_sha256`) plus, for all 226 cells, the per-cell `RESULT.tsv`, `PHASES.tsv`, `equivalence.kv` and both
arms' inferred trees. Every figure derives from those files alone.

```bash
bash scripts/refresh_all.sh
```

This rewrites all of `results/figures/` (129 files: 44 panels as PDF+PNG, the assembled previews,
the manifest and the README) and regenerates `KEY_NUMBERS.md`, from which every
number quoted in the documentation is taken. It needs only Python, matplotlib and numpy.

## Tier 3 — Re-run the benchmarks themselves

Requires a CUDA GPU and, for the large DNA cells, one extra download.

**The inputs that do not fit on GitHub.** Fourteen simulated DNA alignments total ~2.4 GB and eleven
exceed GitHub's hard 100 MB per-file limit. They are deposited on Zenodo as a single folder:

> **Zenodo DOI: _PLACEHOLDER — add on deposit_**

```bash
# download and unpack the deposit, then:
bash iqtree3-gpu-benchmark-inputs-v0.1.0/install_into_repo.sh /path/to/this/clone
```

The installer copies each alignment into the matching `data/dna/<set>/` and verifies every file
against `data/dna/CHECKSUMS.sha256`.

**Or skip the download entirely.** Every one of those alignments is *simulated and regenerable*. Each
`data/dna/<set>/PROVENANCE.md` records the taxa, sites, guide tree, exact model string, seed and
the literal AliSim command, and the guide trees themselves are in the repository. To rebuild them all:

```bash
bash scripts/dev/build_complexity_datasets.sh
sha256sum -c data/dna/CHECKSUMS.sha256
```

Simulated with **IQ-TREE 3.1.3** — AliSim is deterministic given a seed, but only within a version, so
check yours first if a checksum does not match.

**Then submit the benchmark cells:**

```bash
bash scripts/benchmark/submit.sh                  # the headline size sweep
bash scripts/benchmark/submit_complexity.sh go   # the complexity extension
```

Each job runs both arms in one job, on one node, from one binary, differing only by `-gpu`, and
appends one row to `RUNS.tsv`. The harness refuses to overwrite a cell that already holds a successful
measurement unless `FORCE=1`.

---

## What is not in the repository

Manuscript drafts, progress ledgers, development history and the literature scoping review
(Supplementary Note 1) are team-internal and are not distributed here. **Nothing in
`results/figures/` or `results/benchmarks/` depends on them.** The three composite figures built
from the scoping review's panels (Supplementary Figures S14 to S16) ship as files and are left
untouched by `scripts/refresh_all.sh`, which skips them when their panels are absent.
