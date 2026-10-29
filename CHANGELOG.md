# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `--gpu-newton host|device` selects where the per-branch Newton-Raphson branch-length loop runs
  under `-gpu`. `host` (the default) is IQ-TREE's own loop on the CPU with every derivative
  evaluated on the device; it is the configuration of every run in the published benchmark.
  `device` runs the whole loop on the GPU in one conditional CUDA graph with a single host
  round-trip per branch; same routine, same tolerances, unit-tested against the host loop, not
  benchmarked, hence opt-in. The choice is printed in the banner and recorded as `newton_mode=`
  in the GPUSTATS line. This replaces the undocumented environment switch that previously turned
  the device loop on by default, so a default `-gpu` run now behaves exactly as the benchmarked
  binary did.
- `scripts/iqtree3-gpu-verify.sh` now also gates on the exit accounting (calls on the device,
  none delegated, no device errors), not only on the start-up banner. A binary built for another
  compute capability prints the banner, falls back to the CPU on every call, and matched the CPU
  log-likelihood trivially; it now fails with the reason and the rebuild hint.
- `scripts/iqtree3-gpu-benchmark.sh`: optional wrapper that runs one command on the CPU and on the
  GPU (`--gpu-strict`) and reports wall times, speedup and the agreement verdict. A plain `-gpu` run
  has never done any of this; the start-up banner line that read "Equivalence target vs CPU
  IQ-TREE ..." is reworded to say so, and the README and user guide state it in one sentence.
- Public tree scrubbed of working notes: dated remarks and development history in comments and
  documents, the internal placeholder tokens in the run record (now `NA`), and the cluster path
  in the per-run logs and the run table (now `/path/to/iqtree3-gpu`). The earlier build whose 44
  rows are excluded from every figure is now called the prior build throughout
  (`rows_prior_build`, `build_sha_prior` in the key numbers).

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
  and the support-difference histogram; Fig. 2 drops the thread-slope panel; Fig. 3 is now the cost
  law at 20 threads and at one thread, with the explained variation of every fit on the panel; the
  six-card hardware figure moves to the supplement (Supplementary Fig. S9) and the memory figure
  becomes Fig. 5 (memory sizing, informativeness). The secondary mechanistic analyses (old Fig. 1B,
  2C, 3B-D, 6B-C, Supplementary Figs. S1, S5, S8, S9B) are removed together with their derived
  quantities, the run-table column they used, and the harness experiment behind them. Supplementary
  figures renumbered S1-S9 (the two replicate-spread figures are one two-panel figure); the two
  replicate tables are one Supplementary Data sheet; a new Supplementary Data 0 lists source
  accession and citation for all 435 MAGs. Panel file stems match their manuscript letters.
  `benchmark/FIGURE_MAP.md` maps every panel to the function that draws it, its row filter and its
  inputs.
- Figure presentation: no panel carries a narrative title any more; the title slot holds the
  thread count and n only (`figstyle.stratum`), and every plotting box is a 40 x 40 mm square
  (`figstyle.square_axes`, `square_grid`; the tree montages excepted), asserted on save. Fig. 1A
  is now an ECDF of the relative log-likelihood difference per alphabet (difference on the
  vertical axis) and Fig. 1B a dot plot of the per-run maximum support difference (colour =
  alphabet, marker = thread count), re-read from the stored consensus trees and asserted against
  the run table; neither draws a support tolerance. Both histograms and the other drafts were
  retired. Fig. 2A and 2B share one x and y range, mark each workflow's break-even on the parity
  line with its complexity value written at the axis foot, and key the shaded band as the 95%
  bootstrap CI; the absolute-time notes and the 2x/5x target rings are gone. Fig. 3A and 3B
  likewise share one x and y range; their marks are circles only (open + dashed = IQ-TREE3,
  filled + solid = IQ-TREE3-GPU, hue = alphabet) with the key and the fit table under each panel.
  The cost law is now a quadratic log-log fit: every series is concave-up on that scale, so the
  exponent is reported at the smallest and the largest measured workload (e.g. 0.28 to 0.91)
  instead of one straight-line average; `KEY_NUMBERS` carries both forms
  (`costlaw_*_exponent`, `costlaw_*_exponent_smallest`/`_largest`, curvature, paired
  differences at both ends). Every label that said CPU arm or GPU arm now says IQ-TREE3 or
  IQ-TREE3-GPU. Fig. 4 is three panels: the phase hues are now bluish green and reddish purple
  (Okabe-Ito, distinct from the alphabet pair); 4B separates tree search on amino acid from tree
  search on nucleotide, with work-unit and complexity abscissae as drafts; 4A is two boxes, the
  wall share by phase beside a violin of the per-run processor time consumed (IQ-TREE3 /
  IQ-TREE3-GPU, the old 4C) as two half-width boxes in one square; 4B is the per-phase speedup
  against tree complexity with each series' break-even and shaded 95% bootstrap CI; 4C is a
  violin-plus-jitter panel of cores kept busy per build (the old 4D). Fig. 5 is the memory-sizing
  panel alone. The supplement is trimmed to S1-S8: the pattern-saturation, build-regression,
  nominal-fp64, nominal-bandwidth and per-card wall panels are withdrawn with their key numbers
  and the CARD_SPECS sheet (Supplementary Data is now 0-9); the phase-step heatmap carries the
  mean share of every step per alphabet, and campaign coverage is stacked bars (all runs; then
  L40S beside the five sweep cards stacked by card, with the alphabet as fill pattern). The
  sampled-utilisation figure and the device-energy panel are withdrawn with the energy keys, and
  the two replicate-spread panels are one axis, so the supplement is S1-S7. Figure 5 (memory
  sizing) is withdrawn with its script and slope keys; the main text has four figures. The tree
  montages are US Letter pages, 4 rows x 4 columns inside 0.8 in margins, 6 pt type with 5 pt
  tip labels: S1 is every amino-acid tree and S2 every nucleotide tree, one per dataset x thread count x
  card x model (only same-settings replicates collapse). Figures embed Arial when its
  TrueType files are placed in `benchmark/figures/arial/` (not distributed; Nimbus Sans otherwise).
- Figures reorganised: Fig. 1 = coverage, support dot plot, log-likelihood ECDF; Fig. 2 = the
  decision boxes (A, 20 threads and one thread) over the cost-law boxes (B); Fig. 3 = phase-step
  heatmap, cores kept busy, phase share with processor time, per-phase speedup; Supplementary
  Fig. S1 = coverage by card (L40S; the five sweep cards), replicate spread, per-card speedup;
  S2 and S3 the tree montages; S4 model coverage. The presentation tier is consolidated into
  `benchmark/figures_tables/` (`figures.py`, `tables.py`, `lib/`, `refresh_all.sh`), separate from
  the benchmark harness in `benchmark/scripts/`; every output was verified byte-identical across
  the move. The assembled previews are retired. The README embeds a vector SVG of Figure 2
  (`assets/readme/`); publication figures stay PDF and PNG.
  Fig. 3A fits every 20-thread run with all six cards pooled (no separate L40S fit or grey
  reference), on fixed axes (10^7 to 10^17 work units, wall time up to 10^8 s) shared with 3B;
  the cost-law key numbers use the same stratum. The campaign-coverage figure returns as Supplementary Fig. S10 with one mark per run;
  four alternative presentations of it are written beside the panel as
  `FigS10a_alternative<k>_<slug>` for a choice to be made; none is in a composite.
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
