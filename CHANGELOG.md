# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Repository split into the tool (root: `build_iqtree_gpu.sh`, `src/`, `examples/`, `scripts/`,
  `docs/`) and the published benchmark (`benchmark/`: inputs, measured record, figures, tables and
  the scripts that produce them). `benchmark/README.md` maps every figure and table to the script
  that generates it. The validation harnesses write to `validation/`.
- README leads with the two-part layout and a quick start as a new user meets it (clone without
  the benchmark genomes, load the toolchain, build, verify on a GPU node).
- `build_iqtree_gpu.sh` fetches the Eigen 3.4.0 and Boost 1.87.0 headers (checksum-verified) when
  none are installed; a fresh clone on a cluster without them used to stop in CMake. Every script
  with a shebang is now executable in git, so `./build_iqtree_gpu.sh` works from a fresh clone.
- `scripts/portability.py` ignores CMake build trees, so `run_tests.sh` passes after a build.
- Containers: both recipes now build from the checkout (`%files` / `COPY`) instead of cloning
  GitHub, so they work in any clone or fork; the Apptainer recipe previously had its build
  commented out and the Docker image lacked `rf_logl_check.py`. CI builds both images and runs
  the worked example inside each on the CPU.
- Benchmark figures and tables cut to the core story. Fig. 1 keeps the log-likelihood histogram
  alone; Fig. 2 drops the thread-slope panel; Fig. 3 is now the cost law at 20 threads and at one
  thread, with the explained variation of every fit on the panel; Fig. 6 keeps memory sizing and
  informativeness. The secondary mechanistic analyses (old Fig. 1B-C, 2C, 3B-D, 6B-C, Supplementary
  Figs. S1, S5, S8, S9B) are removed together with their derived quantities, the run-table column
  they used, and the harness experiment behind them. Supplementary figures renumbered S1-S8 (the two
  replicate-spread figures are one two-panel figure); the two replicate tables are one Supplementary
  Data sheet; a new Supplementary Data 0 lists source accession and citation for all 435 MAGs. Panel
  file stems of Figs. 5 and 6 now match their manuscript letters. `benchmark/FIGURE_MAP.md` maps
  every panel to the function that draws it, its row filter and its inputs.
- No behavioural change to the binary or the patch.

## [0.1.0] - 2026-10-25

First public release: the version described in the preprint *IQ-TREE3-GPU: GPU-accelerated
maximum-likelihood phylogenetic tree inference for complex amino acid substitution models*
(Byeongyeon Cho and Aleksandar D. Kostic, 2026). A patch against [IQ-TREE 3](https://github.com/iqtree/iqtree3)
at commit `6401e06` (3.1.3). Primary repository
<https://github.com/kosticlab/iqtree3-gpu>, synchronized mirror <https://github.com/terry-b-cho/iqtree3-gpu>.
See [`benchmark/runs/KEY_NUMBERS.md`](benchmark/runs/KEY_NUMBERS.md) for the measured numbers
(regenerated from data, never typed by hand).

### Added
- GPU likelihood kernel for amino-acid and DNA single-matrix reversible models, applied to IQ-TREE 3
  as an ~400-line patch across 8 files plus 4 new CUDA sources (`src/`).
- `--gpu-strict` engagement gate: the run fails rather than silently falling back to the CPU.
- On-device Newton branch-length optimisation, removing per-branch kernel-launch overhead.
- Paired benchmark harness: both arms in one job, on one node, from one binary (`benchmark/scripts/harness/`).
- `scripts/lib/paths.sh` + `config/site.env.example` so the repository runs outside its original cluster.
- `benchmark/scripts/key_numbers.py`, making every number quoted in the docs a build product.

### Validated
- Topology identical (RF = 0) in every recorded run; worst log-likelihood relative difference
  1.893e-06 against a 1e-4 contract.
- Model generality: amino-acid models selected by ModelFinder, plus DNA under GTR+F+G4, HKY+F+G4,
  GTR+F+R4, GTR+F+I+G4 and JC.
- Six GPUs spanning roughly a 19x range in fp64 throughput.

### Known limitations
- At 20 threads the GPU is slower than the CPU for most DNA trees; the payoff case is protein, and
  DNA only above roughly 1e9 tree complexity. See [`benchmark/runs/KEY_NUMBERS.md`](benchmark/runs/KEY_NUMBERS.md).
- One recorded cell fails the bootstrap-support arm of the equivalence gate while remaining exact in
  topology and likelihood; the per-run evidence is distributed under `benchmark/runs/paired/`.
- Profile-mixture, codon, partitioned and `+ASC` models fall back to the CPU (still exact).
